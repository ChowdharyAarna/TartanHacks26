# pose_resp.py
"""
pose_resp.py — Video-side breathing waveform + breath events from pose motion.

Input:  video_path (mp4/mov)
Output: dict with:
  - fps
  - time: list[float]
  - resp: list[float]             (bandpassed breathing waveform)
  - resp_raw: list[float]         (pre-filter combined signal)
  - quality: list[float]          (0..1 pose visibility proxy)
  - breath_events_video: list[{time_s, confidence, type}]         # inhale peaks
  - breath_rate_bpm: float | None                                 # overall estimate
  - breath_rate_timeline: list[{t0,t1,bpm,confidence}] (optional)
  - inhalations: list[{
        t_start,        # estimated inhale start
        t_peak,         # inhale peak time
        duration_s,     # inhalation length = t_peak - t_start
        confidence
    }]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter

mp_pose = mp.solutions.pose


def _interp_nan(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs. If all NaN, returns as-is."""
    a = a.astype(float)
    n = np.isnan(a)
    if n.all():
        return a
    a[n] = np.interp(np.flatnonzero(n), np.flatnonzero(~n), a[~n])
    return a


def _zscore(a: np.ndarray) -> np.ndarray:
    mu = float(np.mean(a))
    sd = float(np.std(a)) + 1e-8
    return (a - mu) / sd


def _bandpass(x: np.ndarray, fs: float, lo: float = 0.1, hi: float = 0.7, order: int = 3) -> np.ndarray:
    """Butterworth bandpass + zero-phase filtering."""
    nyq = 0.5 * fs
    lo_n = lo / nyq
    hi_n = hi / nyq

    lo_n = max(1e-6, min(0.99, lo_n))
    hi_n = max(lo_n + 1e-6, min(0.999, hi_n))
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)


def _safe_fps(cap: cv2.VideoCapture, default: float = 30.0) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3 or math.isnan(float(fps)):
        return default
    return float(fps)


def _resize_for_speed(frame: np.ndarray, target_w: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= target_w:
        return frame
    scale = target_w / float(w)
    target_h = int(h * scale)
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _peak_confidence(prom: float, resp_std: float, q_local: float) -> float:
    """Combine signal prominence + pose quality into [0,1]."""
    s = prom / (resp_std + 1e-6)
    s = 1.0 - math.exp(-0.7 * max(0.0, s))
    conf = 0.65 * s + 0.35 * float(np.clip(q_local, 0.0, 1.0))
    return float(np.clip(conf, 0.0, 1.0))


def _q_local(q: np.ndarray, idx: int, r: int = 3) -> float:
    if len(q) == 0:
        return 0.0
    return float(np.mean(q[max(0, idx - r) : min(len(q), idx + r + 1)]))


def _odd(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return n if (n % 2 == 1) else (n + 1)


def _rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    """Simple rolling std with edge padding."""
    win = max(3, int(win))
    pad = win // 2
    xp = np.pad(x.astype(float), (pad, pad), mode="edge")
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        seg = xp[i : i + win]
        out[i] = float(np.std(seg))
    return out


def _derivative_inhale_start(resp: np.ndarray, p0: int, p1: int, fps: float) -> Optional[int]:
    """
    Find inhale start between p0 and p1 by:
      - smoothing
      - looking for negative->positive derivative transition
      - requiring sustained positive slope for a short run
    Returns index in [p0, p1] or None.
    """
    if p1 <= p0 + 6:
        return None

    seg = resp[p0 : p1 + 1]
    n = len(seg)

    # smooth segment (preserve peak timing)
    win = _odd(max(7, int(0.35 * fps)))
    win = min(win, _odd(max(7, n // 2 * 2 - 1)))  # keep < n and odd
    if win >= 7 and win < n:
        seg_s = savgol_filter(seg, window_length=win, polyorder=2, mode="interp")
    else:
        seg_s = seg

    d = np.diff(seg_s)
    # "sustained" positive slope for ~150ms
    run = max(2, int(0.15 * fps))
    # minimum slope relative to segment scale (avoid tiny sign flips)
    scale = float(np.std(seg_s)) + 1e-8
    dmin = 0.05 * scale

    # search from early in interval to avoid picking right at p0
    start_i = 1
    end_i = max(start_i + 1, len(d) - run)

    for i in range(start_i, end_i):
        if d[i - 1] < 0 and d[i] > 0:
            # require sustained rise
            if np.all(d[i : i + run] > dmin):
                return p0 + i

    return None


# -----------------------------
# Main extraction
# -----------------------------
def extract_pose_resp(
    video_path: str,
    *,
    target_width: int = 640,
    lo_hz: float = 0.1,
    hi_hz: float = 0.7,
    # (2) SavGol params (replaces moving average)
    savgol_win_s: float = 0.35,
    savgol_poly: int = 2,
    min_peak_distance_s: float = 1.0,
    # (3) Adaptive prominence parameters
    prom_win_s: float = 4.0,
    peak_prominence_k: float = 0.9,  # larger -> fewer peaks (more strict)
    compute_rate_timeline: bool = True,
    # Inhalation gating (conservative defaults)
    min_inhale_s: float = 0.25,
    max_inhale_s: float = 4.00,
    min_q_for_inhale: float = 0.25,
) -> Dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = _safe_fps(cap, default=30.0)

    torso_series: List[float] = []
    y_series: List[float] = []
    q_series: List[float] = []
    t_series: List[float] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _resize_for_speed(frame, target_w=target_width)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if res.pose_landmarks is None:
                torso_series.append(np.nan)
                y_series.append(np.nan)
                q_series.append(0.0)
            else:
                lm = res.pose_landmarks.landmark

                LS = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                RS = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                LH = lm[mp_pose.PoseLandmark.LEFT_HIP]
                RH = lm[mp_pose.PoseLandmark.RIGHT_HIP]

                vis = float(min(LS.visibility, RS.visibility, LH.visibility, RH.visibility))
                q_series.append(vis)

                # (1) Torso length normalized by hip width
                mid_sh_x = 0.5 * (LS.x + RS.x)
                mid_sh_y = 0.5 * (LS.y + RS.y)
                mid_hp_x = 0.5 * (LH.x + RH.x)
                mid_hp_y = 0.5 * (LH.y + RH.y)

                torso_len = math.hypot(mid_sh_x - mid_hp_x, mid_sh_y - mid_hp_y)
                hip_w = math.hypot(LH.x - RH.x, LH.y - RH.y) + 1e-6
                torso_norm = torso_len / hip_w

                # shoulder vertical proxy (still useful)
                shoulder_y = -mid_sh_y

                torso_series.append(torso_norm)
                y_series.append(shoulder_y)

            t_series.append(i / fps)
            i += 1

    cap.release()

    torso = _interp_nan(np.array(torso_series, dtype=float))
    y = _interp_nan(np.array(y_series, dtype=float))
    q = np.array(q_series, dtype=float)
    t = np.array(t_series, dtype=float)

    resp_raw = 0.7 * _zscore(torso) + 0.3 * _zscore(y)

    # (2) Savitzky–Golay smoothing (better timing than moving average)
    if len(resp_raw) >= 9:
        win = _odd(max(7, int(savgol_win_s * fps)))
        win = min(win, _odd(len(resp_raw) - 1))
        if win >= 7:
            try:
                resp_raw = savgol_filter(resp_raw, window_length=win, polyorder=savgol_poly, mode="interp")
            except Exception:
                pass

    resp = resp_raw.copy()
    if len(resp_raw) > int(3 * fps):
        try:
            resp = _bandpass(resp_raw, fs=fps, lo=lo_hz, hi=hi_hz, order=3)
        except Exception:
            resp = resp_raw

    min_dist = max(1, int(min_peak_distance_s * fps))

    # (3) Rolling/adaptive prominence instead of global std
    resp_std_global = float(np.std(resp)) + 1e-8
    prom_win = max(9, int(prom_win_s * fps))
    rolling = _rolling_std(resp, prom_win)
    # per-frame prominence threshold; clamp to avoid absurdly small thresholds
    prom_thr = np.clip(peak_prominence_k * rolling, 0.15 * resp_std_global, 3.0 * resp_std_global)

    # ---- Peak detection using adaptive prominence ----
    # We detect peaks with a "loose" threshold, then filter by local adaptive threshold.
    # This works because scipy's find_peaks doesn't accept per-sample prominence.
    peaks0, props0 = find_peaks(resp, distance=min_dist, prominence=0.15 * resp_std_global)

    peaks: List[int] = []
    prominences: List[float] = []
    if len(peaks0) > 0:
        p_prom = props0.get("prominences", np.zeros(len(peaks0), dtype=float))
        for j, p in enumerate(peaks0.astype(int)):
            if float(p_prom[j]) >= float(prom_thr[p]):
                peaks.append(int(p))
                prominences.append(float(p_prom[j]))
    peaks = np.array(peaks, dtype=int)

    breath_events_video: List[Dict] = []
    if len(peaks) > 0:
        for j, p in enumerate(peaks):
            ql = _q_local(q, int(p), r=3)
            conf = _peak_confidence(float(prominences[j]), resp_std_global, ql)
            breath_events_video.append({"time_s": float(t[p]), "confidence": conf, "type": "inhale_peak"})

    # ---- Inhalation lengths (start -> peak) using derivative-based start ----
    inhalations: List[Dict] = []
    if len(peaks) >= 2:
        peaks_sorted = np.sort(peaks).astype(int)
        for p0, p1 in zip(peaks_sorted[:-1], peaks_sorted[1:]):
            if p1 <= p0 + 6:
                continue

            # (2) inhale start via derivative sign change (fallback to argmin if needed)
            tr = _derivative_inhale_start(resp, p0, p1, fps=fps)
            if tr is None:
                seg = resp[p0 : p1 + 1]
                tr = int(p0 + np.argmin(seg))

            inhale_len = float(t[p1] - t[tr])
            if inhale_len <= 0:
                continue
            if inhale_len < float(min_inhale_s) or inhale_len > float(max_inhale_s):
                continue

            ql = float(np.mean(q[max(0, tr - 3) : min(len(q), p1 + 4)])) if len(q) else 0.0
            if ql < float(min_q_for_inhale):
                continue

            dur_mid = 0.5 * (min_inhale_s + max_inhale_s)
            dur_span = max(1e-6, (max_inhale_s - min_inhale_s))
            dur_score = 1.0 - min(1.0, abs(inhale_len - dur_mid) / (0.5 * dur_span))
            conf = float(np.clip(0.8 * ql + 0.2 * dur_score, 0.0, 1.0))

            inhalations.append(
                {
                    "t_start": float(t[tr]),
                    "t_peak": float(t[p1]),
                    "duration_s": float(inhale_len),
                    "confidence": float(conf),
                }
            )

    # ---- Overall breathing rate estimate (peak-to-peak median) ----
    breath_rate_bpm: Optional[float] = None
    if len(peaks) >= 2:
        intervals = np.diff(t[peaks])
        T = float(np.median(intervals)) if len(intervals) else None
        if T and T > 0:
            breath_rate_bpm = float(60.0 / T)

    # ---- Optional: rate timeline via sliding peak-to-peak in windows ----
    breath_rate_timeline: List[Dict] = []
    if compute_rate_timeline and len(resp) > int(6 * fps) and len(peaks) >= 2:
        win_s = min(10.0, max(6.0, (t[-1] - t[0]) * 0.6))
        step_s = 1.0
        win = int(win_s * fps)
        step = int(step_s * fps)

        peak_times = t[peaks]
        for start in range(0, len(t) - win, step):
            end = start + win
            t0, t1 = float(t[start]), float(t[end - 1])

            inside = peak_times[(peak_times >= t0) & (peak_times <= t1)]
            if len(inside) < 2:
                continue

            intervals = np.diff(inside)
            T = float(np.median(intervals))
            bpm = float(60.0 / T) if T > 0 else None

            q_win = float(np.mean(q[start:end])) if len(q) else 0.0
            conf = float(np.clip(0.5 * q_win + 0.5 * min(1.0, len(inside) / 4.0), 0.0, 1.0))

            if bpm is not None:
                breath_rate_timeline.append({"t0": t0, "t1": t1, "bpm": bpm, "confidence": conf})

    return {
        "fps": float(fps),
        "time": t.tolist(),
        "resp": resp.tolist(),
        "resp_raw": resp_raw.tolist(),
        "quality": q.tolist(),
        "breath_events_video": breath_events_video,
        "breath_rate_bpm": breath_rate_bpm,
        "breath_rate_timeline": breath_rate_timeline,
        "inhalations": inhalations,
        "video_notes": {
            "method": (
                "MediaPipe Pose torso motion (torso length normalized by hip width + shoulder y). "
                "SavGol smoothing, bandpass 0.1–0.7 Hz. Inhale peaks via adaptive-prominence find_peaks; "
                "inhale start via derivative-based onset; inhalation length = start -> next peak."
            ),
            "params": {
                "target_width": target_width,
                "bandpass_lo_hz": lo_hz,
                "bandpass_hi_hz": hi_hz,
                "savgol_win_s": savgol_win_s,
                "savgol_poly": savgol_poly,
                "min_peak_distance_s": min_peak_distance_s,
                "prom_win_s": prom_win_s,
                "peak_prominence_k": peak_prominence_k,
                "min_inhale_s": min_inhale_s,
                "max_inhale_s": max_inhale_s,
                "min_q_for_inhale": min_q_for_inhale,
            },
        },
    }
