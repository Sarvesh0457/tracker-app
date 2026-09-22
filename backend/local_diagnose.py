"""
Local ball-detection diagnostic.

Runs the red-ball ONNX (or any ONNX that the Zone-div(final) detector accepts)
over a video file and writes an annotated MP4 with a green bounding box + score
label on every ball detection the pipeline keeps.

No calibration / homography / release-bounce logic — this is purely to see
whether the model + thresholds are producing detections where you expect.

Usage (PowerShell):
    cd D:\intership\tracker_app\backend
    python local_diagnose.py `
        --video "D:\intership\Input files\Red_ball\4_025_05.mp4" `
        --weights "worker\models\red.onnx" `
        --legacy  "worker\legacy\Zone-div(final)" `
        --out     "D:\intership\OUTPUT\4_025_05_diag.mp4"

Optional flags:
    --imgsz 1280     letterbox size (default 1280, model must accept it)
    --debug          also print [DBG] lines to stdout as detection runs
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2


def _bind_legacy(legacy_dir: Path) -> None:
    legacy_dir = legacy_dir.resolve()
    if not legacy_dir.exists():
        raise SystemExit(f"legacy dir not found: {legacy_dir}")
    sys.path.insert(0, str(legacy_dir))
    os.environ["TRACKER_LEGACY_DIR"] = str(legacy_dir)


def _override_weights(legacy_dir: Path, weights: Path) -> None:
    """Poke both config.WEIGHTS_PATH AND detector.WEIGHTS_PATH — the detector
    module does `from config import WEIGHTS_PATH` at import time, which binds
    the *value*, not a live reference; reassigning config alone is a no-op."""
    from pathlib import Path as _P
    import config as legacy_config  # noqa: E402
    import detector as legacy_detector  # noqa: E402
    legacy_config.WEIGHTS_PATH = _P(weights)
    legacy_detector.WEIGHTS_PATH = _P(weights)


def _override_imgsz(imgsz: int) -> None:
    import config as legacy_config  # noqa: E402
    legacy_config.IMGSZ = imgsz
    import detector as legacy_detector  # noqa: E402
    legacy_detector.IMGSZ = imgsz


def _draw_ball_box(frame, det) -> None:
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    conf = det.get("conf", 0.0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
    label = f"ball {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1),
        (0, 255, 0), -1,
    )
    cv2.putText(
        frame, label, (x1 + 3, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def _put_hud(frame, text) -> None:
    cv2.putText(
        frame, text, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--legacy", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    if args.debug:
        os.environ["TRACKER_DEBUG"] = "1"

    _bind_legacy(args.legacy)

    # Legacy modules must import in this order — filters holds tracking_state.
    from filters import reset_filters  # noqa: E402
    from detector import detect_objects, load_model  # noqa: E402
    _override_weights(args.legacy, args.weights)
    _override_imgsz(args.imgsz)
    reset_filters()
    session = load_model()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[VIDEO] {w}x{h} @ {fps:.2f} fps, {total} frames")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"can't open writer: {args.out}")

    kept_frames = 0
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = detect_objects(session, frame, f)
        ball_dets = [d for d in dets if d.get("class_name") == "ball"]

        if ball_dets:
            kept_frames += 1
            for d in ball_dets:
                _draw_ball_box(frame, d)

        hud = f"f={f}  IMGSZ={args.imgsz}  balls_this_frame={len(ball_dets)}"
        _put_hud(frame, hud)
        writer.write(frame)

        if f % 25 == 0:
            print(f"[PROG] frame {f}/{total}  kept_so_far={kept_frames}")
        f += 1

    cap.release()
    writer.release()
    print(f"[DONE] wrote {args.out}")
    print(f"[STATS] frames_processed={f}  frames_with_ball={kept_frames} "
          f"({100.0 * kept_frames / max(1, f):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
