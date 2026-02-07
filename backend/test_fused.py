import json
from pprint import pprint
from pathlib import Path

from pipeline.fused_breaths import extract_fused_resp


def main():
    video_path = "uploads/7eff1fe162b844daa01b1d22ac7f6118.mp4"

    if not Path(video_path).exists():
        raise FileNotFoundError(video_path)

    resp_out = extract_fused_resp(
        video_path,
        outputs_dir="test_outputs",
        use_demucs=True
    )

    # ---- quick overview ----
    print("\nTop-level keys:")
    print(sorted(resp_out.keys()))

    print("\nNumber of fused inhalations:",
          len(resp_out.get("inhalations", [])))

    print("Number of fused breath events:",
          len(resp_out.get("breath_events_video", [])))

    print("\nFusion counts:")
    pprint(resp_out.get("fusion_notes", {}))

    # ---- print first few fused inhalations ----
    print("\nFirst fused inhalations:")
    for inh in resp_out.get("inhalations", [])[:5]:
        pprint(inh)

    # ---- print first few breath events ----
    print("\nFirst fused breath_events_video:")
    for ev in resp_out.get("breath_events_video", [])[:5]:
        pprint(ev)

    # ---- save full resp_out to json for inspection ----
    out_path = Path("test_outputs/resp_out_debug.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with open(out_path, "w") as f:
        json.dump(resp_out, f, indent=2)

    print("\nFull resp_out written to:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
