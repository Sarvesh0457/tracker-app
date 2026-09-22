"""Run the legacy ball detector on a video and dump per-frame ball detections.

This is a thin wrapper — no tracking, no Kalman, no filtering. Just:
  frame N → detect_objects → keep ball rows → write (frame, u, v, radius_px, conf).

Usage:
    python -m backend.pipeline.reconstruct3d.detect_ball \
        --video   path/to/clip.mp4 \
        --weights D:/intership/may/model/Weights/red/red20.onnx \
        --out     path/to/detections.json \
        [--legacy-dir D:/intership/tracker_app/backend/worker/legacy/Zone-div(final)]

Output JSON:
{
  "video": "clip.mp4",
  "fps": 30.0,
  "image_size": [W, H],
  "num_frames": N,
  "detections": [
    {"frame": 12, "u": 640.5, "v": 355.2, "radius_px": 4.8, "conf": 0.71},
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


def _prepare_legacy(legacy_dir: Path):
    """Point the pipeline package at the legacy tracker folder before importing."""
    os.environ["TRACKER_LEGACY_DIR"] = str(legacy_dir)
    if str(legacy_dir) not in sys.path:
        sys.path.insert(0, str(legacy_dir))


def _bbox_to_center_radius(x1, y1, x2, y2):
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    r = 0.25 * ((x2 - x1) + (y2 - y1))  # avg half-width; robust to slight aspect
    return float(u), float(v), float(r)


def run(video_path: Path, weights_path: Path):
    # Use the legacy pipeline's default detector at its stock config — no
    # threshold tampering, no filter bypass.
    from detector import detect_objects
    from filters import reset_filters
    from backend.pipeline.process import get_session

    reset_filters()
    session = get_session(weights_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dets_out = []
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detect_objects(session, frame, f)
        for d in dets:
            if d.get("class_name") != "ball":
                continue
            u, v, r = _bbox_to_center_radius(d["x1"], d["y1"], d["x2"], d["y2"])
            dets_out.append({
                "frame": f,
                "u": round(u, 2),
                "v": round(v, 2),
                "radius_px": round(r, 2),
                "conf": round(float(d.get("conf", 0.0)), 3),
            })
        if f % 25 == 0:
            print(f"  frame {f}/{total}   kept ball so far: {len(dets_out)}", flush=True)
        f += 1
    cap.release()

    return {
        "video": Path(video_path).name,
        "fps": fps,
        "image_size": [W, H],
        "num_frames": f,
        "detections": dets_out,
    }


def main(argv=None) -> int:
    default_legacy = Path(__file__).resolve().parents[2] / "worker" / "legacy" / "Zone-div(final)"
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--weights",    required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--legacy-dir", type=Path, default=default_legacy)
    args = p.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2
    if not args.legacy_dir.exists():
        print(f"legacy dir not found: {args.legacy_dir}", file=sys.stderr)
        return 2

    _prepare_legacy(args.legacy_dir)

    result = run(args.video, args.weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    n = len(result["detections"])
    print(f"wrote {args.out}   ({n} ball detections across {result['num_frames']} frames, {n/max(result['num_frames'],1)*100:.0f}% coverage)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
