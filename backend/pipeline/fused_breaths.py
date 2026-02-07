from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .pose_resp import extract_pose_resp
from .detecting_breath_audio import breath_timestamps_from_mp4


# -----------------------------
# Utilities
# -----------------------------
def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _noisy_or(a: float, b: float) -> float:
    # boosts confidence if both detectors agree
    return 1.0 - (1.0 - a) * (1.0 - b)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _iou(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = _overlap(a0, a1, b0, b1)
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def merge_close_intervals(events: List[Dict[str, Any]], merge_within: float = 0.10) -> List[Dict[str, Any]]:
    if not events:
        return []
    events = sorted(events, key=lambda e: e["start"])
    out = [events[0].copy()]
    for e in events[1:]:
        prev = out[-1]
        if e["start"] <= prev["end"] + merge_within:
            prev["end"] = max(prev["end"], e["end"])
            prev["confidence"] = max(prev["confidence"], e["confidence"])
            prev_sources = set(prev.get("sources", []))
            prev_sources.update(e.get("sources", []))
            prev["sources"] = sorted(prev_sources)
        else:
            out.append(e.copy())
    return out


# -----------------------------
# Normalize: pose/audio -> events
# -----------------------------
def pose_to_events(pose_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pose gives inhalations: {t_start, t_peak, duration_s, confidence}
    Use interval [t_start, t_peak] as "breath window".
    """
    events: List[Dict[str, Any]] = []
    for inh in (pose_out.get("inhalations", []) or []):
        try:
            s = float(inh["t_start"])
            e = float(inh["t_peak"])
            c = float(inh.get("confidence", 0.0))
            if e > s:
                events.append({"start": s, "end": e, "confidence": c, "sources": ["pose"]})
        except Exception:
            pass
    return events


def audio_to_events(audio_breaths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Audio gives breaths: {start, end, confidence}
    """
    events: List[Dict[str, Any]] = []
    for b in (audio_breaths or []):
        try:
            s = float(b["start"])
            e = float(b["end"])
            c = float(b.get("confidence", 0.0))
            if e > s:
                events.append({"start": s, "end": e, "confidence": c, "sources": ["audio"]})
        except Exception:
            pass
    return events


# -----------------------------
# Fuse events
# -----------------------------
def fuse_breath_events(
    pose_events: List[Dict[str, Any]],
    audio_events: List[Dict[str, Any]],
    *,
    max_center_diff_s: float = 0.35,
    min_iou: float = 0.02,
    pose_only_scale: float = 0.80,
    audio_only_scale: float = 0.70,
    merge_within: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Returns fused events with fields:
      {start, end, confidence, sources}

    Matching is by overlap (IoU) + center time closeness.
    """
    pose_events = sorted(pose_events, key=lambda e: e["start"])
    audio_events = sorted(audio_events, key=lambda e: e["start"])

    used_audio = set()
    fused: List[Dict[str, Any]] = []

    for p in pose_events:
        p0, p1, pc = float(p["start"]), float(p["end"]), float(p["confidence"])
        pc = _clamp(pc, 0.0, 1.0)
        p_center = 0.5 * (p0 + p1)

        best_j = None
        best_score = -1.0

        for j, a in enumerate(audio_events):
            if j in used_audio:
                continue
            a0, a1, ac = float(a["start"]), float(a["end"]), float(a["confidence"])
            ac = _clamp(ac, 0.0, 1.0)
            a_center = 0.5 * (a0 + a1)

            if abs(a_center - p_center) > max_center_diff_s:
                continue

            iou = _iou(p0, p1, a0, a1)
            if iou < min_iou:
                continue

            score = iou + 0.1 * min(pc, ac)
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is not None:
            a = audio_events[best_j]
            used_audio.add(best_j)

            start = min(p0, float(a["start"]))
            end = max(p1, float(a["end"]))
            conf = _noisy_or(pc, float(a["confidence"]))

            fused.append(
                {
                    "start": start,
                    "end": end,
                    "confidence": _clamp(conf, 0.0, 1.0),
                    "sources": ["pose", "audio"],
                }
            )
        else:
            fused.append(
                {
                    "start": p0,
                    "end": p1,
                    "confidence": _clamp(pc * pose_only_scale, 0.0, 1.0),
                    "sources": ["pose"],
                }
            )

    for j, a in enumerate(audio_events):
        if j in used_audio:
            continue
        a0, a1, ac = float(a["start"]), float(a["end"]), float(a["confidence"])
        fused.append(
            {
                "start": a0,
                "end": a1,
                "confidence": _clamp(ac * audio_only_scale, 0.0, 1.0),
                "sources": ["audio"],
            }
        )

    fused = merge_close_intervals(fused, merge_within=merge_within)
    return fused


# -----------------------------
# Convert fused events -> ONLY the interval objects you want
# -----------------------------
def fused_events_to_intervals_only(fused: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return ONLY:
      [{"t_start":..., "t_end":..., "duration_s":...}, ...]
    """
    out: List[Dict[str, Any]] = []
    for e in fused:
        s = float(e["start"])
        en = float(e["end"])
        if en > s:
            out.append(
                {
                    "t_start": s,
                    "t_end": en,
                    "duration_s": float(en - s),
                }
            )
    return out


# -----------------------------
# Main fused pipeline
# -----------------------------
def extract_fused_resp(
    video_path: str,
    *,
    outputs_dir: str | None = None,
    use_demucs: bool = True,
    audio_work_subdir: str = "audio_work",
) -> Dict[str, Any]:
    """
    Returns the SAME SHAPE as pose_resp output, except:

    - inhalations is replaced with ONLY:
        [{"t_start","t_end","duration_s"}, ...]
    - breath_events_video is emptied (no peaks list)
    - no extra debug fields added
    """
    # 1) Pose analysis (waveform + pose inhalations)
    pose_out = extract_pose_resp(video_path)

    # 2) Audio analysis (timestamp events)
    if outputs_dir is None:
        outputs_dir = str(Path(video_path).resolve().parent / "outputs")

    outputs_dir_p = Path(outputs_dir)
    outputs_dir_p.mkdir(parents=True, exist_ok=True)

    vid_stem = Path(video_path).stem
    work_dir = outputs_dir_p / f"{audio_work_subdir}_{vid_stem}"

    audio_breaths, _audio_paths = breath_timestamps_from_mp4(
        video_path,
        use_demucs=use_demucs,
        work_dir=str(work_dir),
        export_prefix="audio_breaths",
    )

    # 3) Fuse
    pose_events = pose_to_events(pose_out)
    audio_events = audio_to_events(audio_breaths)
    fused = fuse_breath_events(pose_events, audio_events)

    # 4) Convert fused -> ONLY the intervals you want
    fused_intervals = fused_events_to_intervals_only(fused)

    # 5) Final resp_out: keep pose waveform arrays, replace only the breath fields
    resp_out = dict(pose_out)
    resp_out["inhalations"] = fused_intervals
    resp_out["breath_events_video"] = []  # remove peak-only list entirely

    return resp_out
