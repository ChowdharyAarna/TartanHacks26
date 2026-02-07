# test_video.py
from pipeline.pose_resp import extract_pose_resp
import json

out = extract_pose_resp("samples/demo2.mp4")

print("bpm:", out["breath_rate_bpm"])
print("inhale_peaks:", len(out["breath_events_video"]))
print("inhalations:", len(out.get("inhalations", [])))

print("\nINHALE PEAKS:")
print(json.dumps(out["breath_events_video"], indent=2))

print("\nINHALATIONS:")
print(json.dumps(out.get("inhalations", []), indent=2))

# quick summary stats for inhalation duration
inhs = out.get("inhalations", [])
if inhs:
    durs = [x["duration_s"] for x in inhs]
    print(
        "\nINHALATION DURATION STATS (s):",
        {
            "mean": sum(durs) / len(durs),
            "min": min(durs),
            "max": max(durs),
        },
    )
