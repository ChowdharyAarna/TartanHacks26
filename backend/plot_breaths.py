# plot_breaths.py
import matplotlib.pyplot as plt

from pipeline.pose_resp import extract_pose_resp


def nearest_indices(times, query_times):
    """Map each query time to the nearest index in times (monotonic list)."""
    idx = []
    j = 0
    n = len(times)
    for qt in query_times:
        while j + 1 < n and abs(times[j + 1] - qt) <= abs(times[j] - qt):
            j += 1
        idx.append(j)
    return idx


def main():
    video_path = "samples/demo2.mp4"  # change me
    out = extract_pose_resp(video_path)

    t = out["time"]
    resp = out["resp"]

    peaks = out.get("breath_events_video", [])   # inhale peaks
    inhalations = out.get("inhalations", [])     # {t_start, t_peak, duration_s, confidence}

    fig, ax = plt.subplots(figsize=(12, 5))

    # waveform
    ax.plot(t, resp, linewidth=1)

    # inhale peaks
    if peaks:
        pt = [e["time_s"] for e in peaks]
        idx = nearest_indices(t, pt)
        py = [resp[i] for i in idx]
        ax.scatter(pt, py, s=35, marker="o", label="inhale_peak")

    # inhalation spans (t_start -> t_peak)
    for inh in inhalations:
        ax.axvspan(inh["t_start"], inh["t_peak"], alpha=0.2)

    ax.set_title("Breathing waveform with inhale peaks + inhalation durations")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Resp (a.u.)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
