"""Invoke the real process_video pipeline on a clip using our keypoints as
calibration. The pipeline handles release/bounce/impact detection and returns
a clean trajectory list — no need to re-isolate the delivery ourselves.

Usage:
    python -m backend.pipeline.reconstruct3d.run_full_pipeline \
        --video     path/to/clip.mp4 \
        --keypoints path/to/clip_keypoints.json \
        --weights   D:/intership/may/model/Weights/red/red20.onnx \
        --out-dir   path/to/output_dir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def keypoints_to_calibration(keypoints_data: dict) -> dict:
    """Map our 6-point keypoints.json into the schema process_video expects.

    Our LEFT/RIGHT are as seen on screen; the pipeline's L/R follow the same
    on-screen convention (bat_L has smaller x than bat_R in the sample).
    """
    by_name = {p["name"]: tuple(p["uv"]) for p in keypoints_data["points"]}
    return {
        "bat_L":      list(by_name["popping_striker_leg"]),
        "bat_R":      list(by_name["popping_striker_off"]),
        "bat_stump":  list(by_name["stump_base_striker"]),
        "bowl_L":     list(by_name["popping_nonstriker_leg"]),
        "bowl_R":     list(by_name["popping_nonstriker_off"]),
        "bowl_stump": list(by_name["stump_base_nonstriker"]),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",     required=True, type=Path)
    p.add_argument("--keypoints", required=True, type=Path)
    p.add_argument("--weights",   required=True, type=Path)
    p.add_argument("--out-dir",   required=True, type=Path)
    args = p.parse_args(argv)

    kp = json.loads(args.keypoints.read_text(encoding="utf-8"))
    calibration = keypoints_to_calibration(kp)
    print("calibration mapped from keypoints:")
    for k, v in calibration.items():
        print(f"  {k:12s} = {v}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    from backend.pipeline import process_video
    from backend.pipeline.errors import PipelineError

    def _progress(f, tot):
        if f % 25 == 0:
            print(f"  frame {f}/{tot}")

    try:
        result = process_video(
            video_path=args.video,
            output_dir=args.out_dir,
            calibration=calibration,
            weights_path=args.weights,
            interp_method="parabola",
            progress_callback=_progress,
            video_filename=args.video.name,
        )
    except PipelineError as e:
        print(f"pipeline error [{e.code}]: {e}", file=sys.stderr)
        return 2

    tr = result.get("trajectory", [])
    ev = result.get("events", {})
    print("\n=== pipeline result ===")
    print(f"trajectory points: {len(tr)}")
    for t in tr:
        print(f"  frame {t['frame']:3d}  ({t['x']:4d},{t['y']:4d})  "
              f"conf={t.get('conf', 0.0):.2f}  interp={t.get('interpolated', False)}")
    print(f"\nevents:")
    for k, v in ev.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
