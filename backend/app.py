from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from enum import Enum
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pipeline.fused_breaths import extract_fused_resp

# Import the bad singing detection pipeline
import sys

# Add the directory containing bad_singing_onefile_fixed.py to the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.bad_singing import Config, run_pipeline as run_singing_pipeline
from pipeline.breath_feedback_pipeline import redistribute_bad_breaths, print_redistribution_summary

# -----------------------------
# Job store + executor (+ cancel)
# -----------------------------
class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"
    cancelled = "cancelled"


@dataclass
class JobState:
    status: JobStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result: dict | None = None
    cancel_requested: bool = False


JOBS: dict[str, JobState] = {}
JOB_TASKS: dict[str, asyncio.Task] = {}  # job_id -> asyncio task

# NOTE: keep this small; each process runs heavy analysis
EXECUTOR = ProcessPoolExecutor(max_workers=2)

# -----------------------------
# Paths / storage
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
SINGING_OUTPUT_DIR = BASE_DIR / "outputs" / "singing_analysis"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SINGING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Store analysis results in memory (or use a database in production)
# Maps video_id -> singing_analysis_results
SINGING_RESULTS_CACHE: Dict[str, Dict[str, Any]] = {}

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="BreathMap Backend", version="0.1.0")

# Serve uploaded videos and generated plots
app.mount("/media", StaticFiles(directory=str(UPLOAD_DIR)), name="media")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Dev-friendly CORS (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


# -----------------------------
# Helpers (frontend schema)
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _build_analysis_result(resp_out: Dict[str, Any], video_url: str) -> Dict[str, Any]:
    """Map pose_resp output -> frontend AnalysisResult-ish schema."""
    t: List[float] = resp_out.get("time", []) or []
    q: List[float] = resp_out.get("quality", []) or []
    inhalations: List[Dict[str, Any]] = resp_out.get("inhalations", []) or []
    bpm = resp_out.get("breath_rate_bpm", None)

    # timelineScores: per-second score 0..100 (using quality as proxy)
    if t and q:
        t0 = float(t[0])
        t_end = float(t[-1])
        n_secs = max(1, int(round(t_end - t0)))
        scores: List[Dict[str, Any]] = []
        for s in range(n_secs + 1):
            sec_start = t0 + s
            sec_end = sec_start + 1.0

            acc = 0.0
            cnt = 0
            for ti, qi in zip(t, q):
                if sec_start <= float(ti) < sec_end:
                    acc += float(qi)
                    cnt += 1

            q_avg = (acc / cnt) if cnt else 0.0
            score = int(round(_clamp(q_avg, 0.0, 1.0) * 100))
            scores.append({"time": float(s), "score": score})
    else:
        scores = []

    # annotations: inhalation durations
    annotations: List[Dict[str, Any]] = []
    for inh in inhalations[:50]:
        tp = float(inh.get("t_peak", 0.0))
        dur = float(inh.get("duration_s", 0.0))
        conf = float(inh.get("confidence", 0.0))
        annotations.append({"time": tp, "text": f"Inhale ~{dur:.2f}s (conf {conf:.2f})"})

    # summary
    if scores:
        avg_score = sum(s["score"] for s in scores) / len(scores)
        final_score = scores[-1]["score"]
        mean = avg_score
        var = sum((s["score"] - mean) ** 2 for s in scores) / len(scores)
        std = var**0.5
        stability = float(_clamp(1.0 - (std / 50.0), 0.0, 1.0))
        spike_count = sum(1 for s in scores if s["score"] < 30)
    else:
        avg_score = 0.0
        final_score = 0
        stability = 0.0
        spike_count = 0

    if bpm is None:
        bpm_text = "Breath rate estimate unavailable (low tracking quality)."
    else:
        bpm_text = f"Estimated breath rate: {float(bpm):.1f} bpm."

    short_expl = f"{bpm_text} Detected {len(inhalations)} inhalations. Tracking stability: {stability:.2f}."

    return {
        "videoUrl": video_url,
        "timelineScores": scores,
        "annotations": annotations,
        "summary": {
            "finalScore": int(round(final_score)),
            "avgScore": int(round(avg_score)),
            "stability": float(stability),
            "spikeCount": int(spike_count),
        },
        "shortExplanation": short_expl,
        # Raw breathing report used by the frontend waveform plot
        "report": resp_out,
    }


# -----------------------------
# Plotting helpers
# -----------------------------
def nearest_indices(times: List[float], query_times: List[float]) -> List[int]:
    """Map each query time to the nearest index in times (monotonic list)."""
    idx: List[int] = []
    j = 0
    n = len(times)
    for qt in query_times:
        while j + 1 < n and abs(times[j + 1] - qt) <= abs(times[j] - qt):
            j += 1
        idx.append(j)
    return idx


def make_breath_plot(out: Dict[str, Any], save_path: Path) -> None:
    t: List[float] = out.get("time", []) or []
    resp: List[float] = out.get("resp", []) or []
    if not t or not resp:
        return

    peaks = out.get("breath_events_video", []) or []
    inhalations = out.get("inhalations", []) or []

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t, resp, linewidth=1)

    if peaks:
        pt = [float(e.get("time_s", 0.0)) for e in peaks]
        idx = nearest_indices([float(x) for x in t], pt)
        py = [float(resp[i]) for i in idx]
        ax.scatter(pt, py, s=35, marker="o", label="inhale_peak")

    for inh in inhalations:
        try:
            ax.axvspan(float(inh["t_start"]), float(inh["t_peak"]), alpha=0.2)
        except Exception:
            pass

    ax.set_title("Breathing waveform with inhale peaks + inhalation durations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resp (a.u.)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# -----------------------------
# Bad Singing Analysis Helper
# -----------------------------
def run_singing_analysis(video_path: str, vid_id: str) -> Dict[str, Any]:
    """
    Run the bad singing detection pipeline on the uploaded video.
    Returns a dict with singing analysis results and stores it in SINGING_RESULTS_CACHE.
    """
    try:
        # Create output directory for this specific video
        out_dir = SINGING_OUTPUT_DIR / vid_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Configure the singing analysis
        cfg = Config(
            sr=16000,
            top_db_split=30.0,
            strain_thr=0.62,
            collapse_thr=0.62,
            top_k_strain=3,
            top_k_collapse=3,
            edge_ignore_s=1.0,
            phrase_rms_percentile=30.0,
            save_timeline=True,
            save_plot=True,
        )

        # Run the pipeline
        results = run_singing_pipeline(video_path, str(out_dir), cfg)

        # Convert file paths to URLs for frontend access
        if "artifacts" in results:
            if "timeline_csv" in results["artifacts"]:
                csv_path = Path(results["artifacts"]["timeline_csv"])
                if csv_path.exists():
                    rel_path = csv_path.relative_to(OUTPUT_DIR)
                    results["artifacts"]["timeline_csv_url"] = f"/outputs/{rel_path}"

            if "report_png" in results["artifacts"]:
                png_path = Path(results["artifacts"]["report_png"])
                if png_path.exists():
                    rel_path = png_path.relative_to(OUTPUT_DIR)
                    results["artifacts"]["report_png_url"] = f"/outputs/{rel_path}"

        # Store results in cache
        SINGING_RESULTS_CACHE[vid_id] = results

        return {"success": True, "video_id": vid_id, "results": results}

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return {"success": False, "error": str(e), "error_details": error_details}


# -----------------------------
# Heavy analysis (pure function for ProcessPoolExecutor)
# -----------------------------
def run_full_analysis(video_path: str, vid_id: str, suffix: str) -> dict:
    # Run breathing analysis
    resp_out = extract_fused_resp(video_path, outputs_dir=str(OUTPUT_DIR))

    # Generate breathing plot
    plot_name = f"breath_plot_{vid_id}.png"
    plot_path = OUTPUT_DIR / plot_name
    make_breath_plot(resp_out, plot_path)

    video_url = f"/media/{Path(video_path).name}"
    result = _build_analysis_result(resp_out, video_url)
    result["plot_url"] = f"/outputs/{plot_name}" if plot_path.exists() else None

    # Singing analysis
    singing_results = run_singing_analysis(video_path, vid_id)
    result["singing_analysis"] = singing_results
    result["video_id"] = vid_id

    # Breath feedback
    try:
        if singing_results.get("success") and "results" in singing_results:
            breath_feedback = redistribute_bad_breaths(resp_out, singing_results["results"], anchor_mode="before")
            # Optional: keep console logging for debugging
            try:
                print_redistribution_summary(breath_feedback)
            except Exception:
                pass
            result["breath_feedback"] = breath_feedback
        else:
            result["breath_feedback"] = {
                "error": "Singing analysis failed",
                "details": singing_results.get("error", "Unknown error"),
            }
    except Exception as e:
        import traceback

        result["breath_feedback"] = {"error": str(e), "traceback": traceback.format_exc()}

    return result


# -----------------------------
# Background runner (+ cancel support)
# -----------------------------
async def _run_job(job_id: str, video_path: str, vid_id: str, suffix: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return

    # cancelled before start
    if job.cancel_requested:
        job.status = JobStatus.cancelled
        job.finished_at = time.time()
        return

    job.status = JobStatus.running
    job.started_at = time.time()

    try:
        loop = asyncio.get_running_loop()
        fut = loop.run_in_executor(EXECUTOR, run_full_analysis, video_path, vid_id, suffix)
        result = await fut

        job = JOBS.get(job_id)
        if not job:
            return

        # If cancelled while process work was running, ignore result
        if job.cancel_requested:
            job.status = JobStatus.cancelled
            job.result = None
            return

        job.result = result
        job.status = JobStatus.done

    except asyncio.CancelledError:
        job = JOBS.get(job_id)
        if job:
            job.status = JobStatus.cancelled
            job.result = None
        raise

    except Exception as e:
        job = JOBS.get(job_id)
        if job:
            job.status = JobStatus.error
            job.error = str(e)

    finally:
        job = JOBS.get(job_id)
        if job and job.finished_at is None:
            job.finished_at = time.time()
        JOB_TASKS.pop(job_id, None)


# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ("video/mp4", "video/quicktime", "application/octet-stream"):
        return JSONResponse({"error": f"Unsupported content type: {file.content_type}"}, status_code=400)

    filename = (file.filename or "").lower()
    suffix = ".mp4" if filename.endswith(".mp4") else ".mov"

    vid_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{vid_id}{suffix}"
    out_path.write_bytes(await file.read())

    job_id = uuid.uuid4().hex
    JOBS[job_id] = JobState(status=JobStatus.queued, created_at=time.time())

    # fire-and-forget background task (tracked so we can cancel)
    task = asyncio.create_task(_run_job(job_id, str(out_path), vid_id, suffix))
    JOB_TASKS[job_id] = task

    return JSONResponse(
        {
            "job_id": job_id,
            "video_id": vid_id,
            "status_url": f"/api/jobs/{job_id}",
            "cancel_url": f"/api/jobs/{job_id}/cancel",
            "video_url": f"/media/{out_path.name}",
        }
    )


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    payload = asdict(job)
    # Don’t return full result until done
    if job.status != JobStatus.done:
        payload["result"] = None

    return JSONResponse(payload)


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "job not found"}, status_code=404)

    # already terminal
    if job.status in (JobStatus.done, JobStatus.error, JobStatus.cancelled):
        return JSONResponse({"job_id": job_id, "status": job.status})

    job.cancel_requested = True
    job.status = JobStatus.cancelled
    job.result = None
    job.finished_at = time.time()

    # Cancels the awaiting task (best effort; process may still finish in background)
    t = JOB_TASKS.get(job_id)
    if t:
        t.cancel()

    return JSONResponse({"job_id": job_id, "status": job.status})


@app.post("/api/analyze-singing")
async def analyze_singing(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint specifically for singing analysis without breathing analysis.
    Stores singing results in SINGING_RESULTS_CACHE.
    """
    if file.content_type not in ("video/mp4", "video/quicktime", "application/octet-stream"):
        return JSONResponse({"error": f"Unsupported content type: {file.content_type}"}, status_code=400)

    filename = (file.filename or "").lower()
    suffix = ".mp4" if filename.endswith(".mp4") else ".mov"

    vid_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{vid_id}{suffix}"

    data = await file.read()
    out_path.write_bytes(data)

    video_url = f"/media/{out_path.name}"

    # Run singing analysis (results stored in SINGING_RESULTS_CACHE)
    singing_results = run_singing_analysis(str(out_path), vid_id)

    return JSONResponse({"videoUrl": video_url, "video_id": vid_id, "singing_analysis": singing_results})


@app.get("/api/singing-results/{vid_id}")
async def get_singing_results(vid_id: str) -> JSONResponse:
    """
    Retrieve stored singing analysis results by video ID.
    """
    if vid_id not in SINGING_RESULTS_CACHE:
        return JSONResponse({"error": f"No singing analysis found for video ID: {vid_id}"}, status_code=404)

    return JSONResponse({"video_id": vid_id, "results": SINGING_RESULTS_CACHE[vid_id]})


@app.get("/api/singing-results")
async def list_singing_results() -> JSONResponse:
    """
    List all video IDs that have singing analysis results.
    """
    return JSONResponse({"video_ids": list(SINGING_RESULTS_CACHE.keys()), "count": len(SINGING_RESULTS_CACHE)})
