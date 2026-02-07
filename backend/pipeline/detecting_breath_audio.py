import json
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import webrtcvad

# graphing
import matplotlib.pyplot as plt


# ============================================================
# Phase 1: MP4 -> audio array
# ============================================================
def load_audio_from_mp4(mp4_path: str, sr: int = 16000):
    """
    Extract audio from an MP4 using ffmpeg, load into numpy via librosa, return (y, sr).
    """
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()

    cmd = [
        "ffmpeg",
        "-y",
        "-i", mp4_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        tmp_wav.name
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        y, _ = librosa.load(tmp_wav.name, sr=sr, mono=True)
        return y, sr
    finally:
        if os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)


def mp4_to_wav(mp4_path: str, out_wav_path: str, sr: int = 16000):
    """
    Extract audio from MP4 and write a normalized WAV to disk (needed for Demucs).
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", mp4_path,
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        out_wav_path
    ]
    subprocess.run(cmd, check=True)


# ============================================================
# Phase 2: WAV -> vocals-only WAV (Demucs)
# ============================================================
def make_vocals_only(wav_path: str, out_dir: str = "separated") -> str:
    """
    Run Demucs and return the path to the generated vocals.wav.
    """
    subprocess.run(["demucs", "-o", out_dir, wav_path], check=True)

    stem_name = Path(wav_path).stem
    vocals_path = Path(out_dir) / "htdemucs" / stem_name / "vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(f"Could not find Demucs vocals output at: {vocals_path}")

    return str(vocals_path)


# ============================================================
# Phase 3: Candidate gaps (VAD + energy union)
# ============================================================
def vad_frames_from_audio(y: np.ndarray, sr: int = 16000, frame_ms: int = 30, aggressiveness: int = 0):
    """
    Run WebRTC VAD on audio array; return list of (t0, t1, voiced).

    NOTE: aggressiveness=0 is least strict and often works better for singing.
    """
    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError("webrtcvad supports sr in {8000, 16000, 32000, 48000}")
    if frame_ms not in (10, 20, 30):
        raise ValueError("webrtcvad supports frame_ms in {10, 20, 30}")

    y = np.asarray(y, dtype=np.float32)
    y = np.clip(y, -1.0, 1.0)
    pcm = (y * 32768.0).astype(np.int16).tobytes()

    vad = webrtcvad.Vad(int(aggressiveness))

    frame_len = int(sr * frame_ms / 1000)  # samples
    byte_len = frame_len * 2               # int16 => 2 bytes

    frames = []
    hop_s = frame_ms / 1000.0
    t = 0.0

    for i in range(0, len(pcm) - byte_len + 1, byte_len):
        frame = pcm[i:i + byte_len]
        voiced = vad.is_speech(frame, sr)
        frames.append((t, t + hop_s, voiced))
        t += hop_s

    return frames


def candidate_gaps_from_vad(frames, min_gap: float = 0.02, max_gap: float = 1.2):
    """
    Find unvoiced gaps that start right after voicing and end when voicing resumes.
    Returns list of (start, end).
    """
    gaps = []
    in_gap = False
    start = None
    prev_voiced = False

    for t0, t1, voiced in frames:
        if (not voiced) and prev_voiced and (not in_gap):
            start = t0
            in_gap = True

        if voiced and in_gap:
            end = t0
            dur = end - start
            if min_gap <= dur <= max_gap:
                gaps.append((float(start), float(end)))
            in_gap = False

        prev_voiced = voiced

    return gaps


def candidate_gaps_from_energy(
    y: np.ndarray,
    sr: int,
    frame_length: int = 1024,
    hop_length: int = 256,
    min_gap: float = 0.02,
    max_gap: float = 1.2,
    low_percentile: float = 15.0
):
    """
    Low-energy fallback gaps (good for singing / breaths that VAD misses).
    """
    y = np.asarray(y, dtype=np.float32)

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    thr = np.percentile(rms, low_percentile)  # tune 10-25

    low = rms < thr
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    gaps = []
    start = None

    for i, is_low in enumerate(low):
        if is_low and start is None:
            start = float(times[i])
        if (not is_low) and start is not None:
            end = float(times[i])
            dur = end - start
            if min_gap <= dur <= max_gap:
                gaps.append((start, end))
            start = None

    if start is not None:
        end = float(times[-1])
        dur = end - start
        if min_gap <= dur <= max_gap:
            gaps.append((start, end))

    return gaps


def merge_gap_ranges(gaps, merge_within: float = 0.08):
    """
    Merge overlapping / near-adjacent gap ranges.
    """
    if not gaps:
        return []

    gaps = sorted([(float(a), float(b)) for a, b in gaps], key=lambda x: x[0])
    merged = [list(gaps[0])]

    for a, b in gaps[1:]:
        prev = merged[-1]
        if a <= prev[1] + merge_within:
            prev[1] = max(prev[1], b)
        else:
            merged.append([a, b])

    return [(a, b) for a, b in merged]


# ============================================================
# Phase 4: Breath scoring
# ============================================================
def _breath_score(segment: np.ndarray, sr: int):
    """
    Breath-likeness score using:
      - spectral flatness (noise-like)
      - high-frequency energy ratio

    Returns (score, rms) [rms is informational only; not used for gating]
    """
    if len(segment) < int(0.03 * sr):
        return 0.0, 0.0

    n_fft = 1024 if sr >= 16000 else 512
    hop = n_fft // 4

    S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop)) + 1e-9
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    rms = float(np.mean(librosa.feature.rms(y=segment, frame_length=n_fft, hop_length=hop)))

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    hf_band = (freqs >= 2000) & (freqs <= 8000)
    hf_ratio = float(S[hf_band, :].sum() / (S.sum() + 1e-9))

    score = 0.55 * flat + 0.45 * hf_ratio
    return score, rms


def detect_breaths_from_gaps(
    y: np.ndarray,
    sr: int,
    gaps,
    score_thr: float = 0.07,
    win_ms: int = 60,
    hop_ms: int = 10
):
    """
    Scan short sub-windows inside each candidate gap and pick the most breath-like window.

    IMPORTANT: No RMS gating (Demucs can make breaths very quiet).
    """
    y = np.asarray(y, dtype=np.float32)

    win = max(1, int(sr * (win_ms / 1000.0)))
    hop = max(1, int(sr * (hop_ms / 1000.0)))

    breaths = []

    for (t0, t1) in gaps:
        i0 = max(0, int(t0 * sr))
        i1 = min(len(y), int(t1 * sr))
        seg = y[i0:i1]

        if len(seg) == 0:
            continue

        # If gap shorter than window, just score the entire gap
        if len(seg) < win:
            score, _ = _breath_score(seg, sr)
            if score >= score_thr:
                conf = max(0.0, min(1.0, (score - score_thr) / (1.0 - score_thr)))
                breaths.append({"start": float(t0), "end": float(t1), "confidence": float(conf)})
            continue

        best = None  # (score, start_sample, end_sample)
        for s in range(0, len(seg) - win + 1, hop):
            chunk = seg[s:s + win]
            score, _ = _breath_score(chunk, sr)
            if best is None or score > best[0]:
                best = (score, s, s + win)

        best_score, bs0, bs1 = best

        if best_score >= score_thr:
            breath_start = (i0 + bs0) / sr
            breath_end = (i0 + bs1) / sr
            conf = max(0.0, min(1.0, (best_score - score_thr) / (1.0 - score_thr)))
            breaths.append({"start": float(breath_start), "end": float(breath_end), "confidence": float(conf)})

    return breaths


# ============================================================
# Phase 5: Post-processing + export
# ============================================================
def merge_close_segments(segments, merge_within: float = 0.10):
    if not segments:
        return []

    segs = sorted(segments, key=lambda x: x["start"])
    merged = [segs[0].copy()]

    for s in segs[1:]:
        prev = merged[-1]
        if s["start"] - prev["end"] <= merge_within:
            prev["end"] = max(prev["end"], s["end"])
            prev["confidence"] = max(prev["confidence"], s["confidence"])
        else:
            merged.append(s.copy())

    return merged


def filter_by_duration(segments, min_dur: float = 0.03, max_dur: float = 1.2):
    out = []
    for s in segments:
        dur = s["end"] - s["start"]
        if min_dur <= dur <= max_dur:
            out.append(s)
    return out


def export_json(segments, out_path: str):
    with open(out_path, "w") as f:
        json.dump(segments, f, indent=2)


def export_timeline_txt(segments, out_path: str):
    """
    idx  start  end  duration  confidence
    """
    with open(out_path, "w") as f:
        for i, s in enumerate(segments, start=1):
            dur = s["end"] - s["start"]
            f.write(f"{i}\t{s['start']:.3f}\t{s['end']:.3f}\t{dur:.3f}\t{s['confidence']:.2f}\n")


def export_debug(frames, gaps, out_path: str):
    voiced_count = sum(1 for _, _, v in frames if v)
    total = len(frames)
    voiced_frac = (voiced_count / total) if total else 0.0

    data = {
        "num_frames": total,
        "voiced_frames": voiced_count,
        "voiced_fraction": voiced_frac,
        "num_gaps": len(gaps),
        "gaps_preview": gaps[:50],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def _srt_time(t: float) -> str:
    hours = int(t // 3600)
    t -= hours * 3600
    minutes = int(t // 60)
    t -= minutes * 60
    seconds = int(t)
    millis = int(round((t - seconds) * 1000))
    if millis == 1000:
        seconds += 1
        millis = 0
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def export_srt(segments, out_path: str, label: str = "BREATH"):
    with open(out_path, "w") as f:
        for idx, s in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n")
            f.write(f"{label} (conf={s['confidence']:.2f})\n\n")


def plot_breath_timeline(segments, duration_s: float, out_path: str, bin_ms: int = 100):
    bin_s = bin_ms / 1000.0
    n_bins = int(np.ceil(duration_s / bin_s))
    t = np.arange(n_bins) * bin_s
    activity = np.zeros(n_bins, dtype=np.float32)

    for s in segments:
        i0 = max(0, int(np.floor(s["start"] / bin_s)))
        i1 = min(n_bins - 1, int(np.ceil(s["end"] / bin_s)))
        activity[i0:i1 + 1] = 1.0

    plt.figure()
    plt.plot(t, activity)
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Time (s)")
    plt.ylabel("Breath activity (0/1)")
    plt.title("Detected breath timeline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# End-to-end convenience pipeline
# ============================================================
def breath_timestamps_from_mp4(
    mp4_path: str,
    use_demucs: bool = True,
    work_dir: str = "work",
    vad_frame_ms: int = 30,
    vad_aggressiveness: int = 0,
    gap_min: float = 0.02,
    gap_max: float = 1.2,
    score_thr: float = 0.07,
    merge_within: float = 0.10,
    dur_min: float = 0.03,
    dur_max: float = 1.2,
    export_prefix: str = "breaths"
):
    """
    Full pipeline:
      MP4 -> WAV -> (optional Demucs -> vocals)
      -> candidate gaps = union(VAD gaps, energy gaps) + merge
      -> breath scoring (short-window max)
      -> postprocess
      -> export JSON + SRT + TIMELINE + PLOT + DEBUG
    """
    os.makedirs(work_dir, exist_ok=True)
    base = Path(mp4_path).stem

    wav_path = str(Path(work_dir) / f"{base}_audio.wav")
    mp4_to_wav(mp4_path, wav_path, sr=16000)

    analysis_wav = wav_path
    vocals_path = None

    if use_demucs:
        vocals_path = make_vocals_only(wav_path, out_dir=str(Path(work_dir) / "separated"))
        analysis_wav = vocals_path

    y, sr = librosa.load(analysis_wav, sr=16000, mono=True)
    duration_s = len(y) / sr

    frames = vad_frames_from_audio(y, sr=sr, frame_ms=vad_frame_ms, aggressiveness=vad_aggressiveness)
    gaps_vad = candidate_gaps_from_vad(frames, min_gap=gap_min, max_gap=gap_max)
    gaps_energy = candidate_gaps_from_energy(y, sr, min_gap=gap_min, max_gap=gap_max)
    gaps = merge_gap_ranges(gaps_vad + gaps_energy, merge_within=0.08)

    breaths = detect_breaths_from_gaps(y, sr, gaps, score_thr=score_thr, win_ms=60, hop_ms=10)

    # One extra permissive pass if still empty
    if len(breaths) == 0 and len(gaps) > 0:
        breaths = detect_breaths_from_gaps(y, sr, gaps, score_thr=max(0.05, score_thr * 0.7), win_ms=50, hop_ms=10)

    breaths = merge_close_segments(breaths, merge_within=merge_within)
    breaths = filter_by_duration(breaths, min_dur=dur_min, max_dur=dur_max)

    json_path = str(Path(work_dir) / f"{export_prefix}_{base}.json")
    srt_path = str(Path(work_dir) / f"{export_prefix}_{base}.srt")
    timeline_path = str(Path(work_dir) / f"{export_prefix}_{base}_timeline.tsv")
    plot_path = str(Path(work_dir) / f"{export_prefix}_{base}_timeline.png")
    debug_path = str(Path(work_dir) / f"{export_prefix}_{base}_debug.json")

    export_json(breaths, json_path)
    export_srt(breaths, srt_path)
    export_timeline_txt(breaths, timeline_path)
    plot_breath_timeline(breaths, duration_s=duration_s, out_path=plot_path, bin_ms=100)
    export_debug(frames, gaps, debug_path)

    return breaths, {
        "json": json_path,
        "srt": srt_path,
        "timeline_tsv": timeline_path,
        "timeline_plot_png": plot_path,
        "debug_json": debug_path,
        "wav": wav_path,
        "vocals_wav": vocals_path
    }
