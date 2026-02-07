from __future__ import annotations

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

# -----------------------------
# Paths / storage
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    short_expl = (
        f"{bpm_text} Detected {len(inhalations)} inhalations. Tracking stability: {stability:.2f}."
    )

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
# API
# -----------------------------
@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type not in ("video/mp4", "video/quicktime", "application/octet-stream"):
        return JSONResponse(
            {"error": f"Unsupported content type: {file.content_type}"},
            status_code=400,
        )

    filename = (file.filename or "").lower()
    suffix = ".mp4" if filename.endswith(".mp4") else ".mov"

    vid_id = uuid.uuid4().hex
    out_path = UPLOAD_DIR / f"{vid_id}{suffix}"

    data = await file.read()
    out_path.write_bytes(data)

    resp_out = extract_fused_resp(str(out_path), outputs_dir=str(OUTPUT_DIR))

    # Generate plot file and return URL
    plot_name = f"breath_plot_{vid_id}.png"
    plot_path = OUTPUT_DIR / plot_name
    make_breath_plot(resp_out, plot_path)

    video_url = f"/media/{out_path.name}"
    result = _build_analysis_result(resp_out, video_url)
    result["plot_url"] = f"/outputs/{plot_name}" if plot_path.exists() else None

    return JSONResponse(result)
