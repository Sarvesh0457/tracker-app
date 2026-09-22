"""V8m (YOLOv8) ball detector with pose-gated release detection.

Runs the ball model, YOLOv8-pose (to find the bowler), and V8m's own
`ReleaseDetector` (which requires ball to be near the bowler's shoulder and
above the bowling-stump top) — same logic as V8m's main pipeline. The
detected release frame is written into the output JSON so downstream stages
can pick it up.

Usage:
    "C:/Program Files/Python311/python.exe" -m backend.pipeline.reconstruct3d.detect_ball_v8m \
        --video     path/to/clip.mp4 \
        --weights   "D:/intership/July/Zone-div_V8m - Copy/best.pt" \
        --keypoints path/to/clip_keypoints.json \
        --out       path/to/detections.json \
        [--conf 0.30] [--imgsz 1280] [--ball-class 0]

Output JSON adds a `release_frame` field alongside the existing schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


V8M_DIR = Path(r"D:/intership/July/Zone-div_V8m - Copy")


def keypoints_to_markers(kp_json: dict) -> dict:
    """Map our 8-point keypoints.json to the 6-marker calibration dict that
    V8m's ReleaseDetector + PoseDetector expect."""
    by_name = {p["name"]: tuple(p["uv"]) for p in kp_json["points"]}
    return {
        "bat_L":      list(by_name["popping_striker_leg"]),
        "bat_R":      list(by_name["popping_striker_off"]),
        "bat_stump":  list(by_name["stump_base_striker"]),
        "bowl_L":     list(by_name["popping_nonstriker_leg"]),
        "bowl_R":     list(by_name["popping_nonstriker_off"]),
        "bowl_stump": list(by_name["stump_base_nonstriker"]),
    }


def run(video_path: Path, weights_path: Path, keypoints_path: Path,
        conf: float, imgsz: int, ball_class: int):
    # Make V8m's package importable — pose_detector + release_detector live there.
    if str(V8M_DIR) not in sys.path:
        sys.path.insert(0, str(V8M_DIR))
    from ultralytics import YOLO
    import torch
    from pose_detector import PoseDetector
    from release_detector import ReleaseDetector

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[detect_ball_v8m] device: {device}   cuda: {torch.cuda.is_available()}", flush=True)

    model = YOLO(str(weights_path))
    model.to(device)

    kp = json.loads(keypoints_path.read_text(encoding="utf-8"))
    markers = keypoints_to_markers(kp)

    pose_det = PoseDetector(conf_thresh=0.20)
    pose_det.set_calibration(markers)
    release_det = ReleaseDetector(window_frames=10, min_detections=3, calib_markers=markers)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dets_out: list = []
    release_frame = None
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Pose (runs until confirmed, then locks — cheap after that).
        pose_evt = pose_det.update(frame, f)
        wrist = pose_det.get_raised_wrist_coords(f)
        shoulder = pose_det.get_raised_shoulder_coords(f)
        pose_ok = pose_det.confirmed

        # Ball detection.
        results = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        frame_dets = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes   = results.boxes.xyxy.cpu().numpy()
            scores  = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), sc, cl in zip(boxes, scores, classes):
                if cl != ball_class:
                    continue
                d = {"x1": float(x1), "y1": float(y1),
                     "x2": float(x2), "y2": float(y2),
                     "conf": float(sc), "class_name": "ball"}
                frame_dets.append(d)
                u = 0.5 * (x1 + x2); v = 0.5 * (y1 + y2)
                r = 0.25 * ((x2 - x1) + (y2 - y1))
                dets_out.append({
                    "frame": f, "u": round(float(u), 2), "v": round(float(v), 2),
                    "radius_px": round(float(r), 2), "conf": round(float(sc), 3),
                })

        # ReleaseDetector — feed detections + pose data. Fires once when locked.
        if release_frame is None:
            evt = release_det.update(
                frame_dets, f,
                pose_confirmed=pose_ok,
                wrist_xy=wrist,
                shoulder_xy=shoulder,
                hand_raise_frame=(pose_det.event.frame if pose_det.event else None),
            )
            if evt is not None:
                release_frame = int(evt.release_frame)
                print(f"[detect_ball_v8m] release CONFIRMED at frame {release_frame}", flush=True)

        if f % 25 == 0:
            tag = "pose OK" if pose_ok else "no pose"
            rf  = release_frame if release_frame is not None else "-"
            print(f"  frame {f}/{total}   dets so far: {len(dets_out)}   {tag}   release: {rf}", flush=True)
        f += 1
    cap.release()

    return {
        "video": Path(video_path).name,
        "fps": fps,
        "image_size": [W, H],
        "num_frames": f,
        "release_frame": release_frame,
        "detector": {"backend": "ultralytics-YOLOv8", "weights": str(weights_path),
                     "conf": conf, "imgsz": imgsz, "ball_class": ball_class,
                     "pose_gated_release": True},
        "detections": dets_out,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--weights",    required=True, type=Path)
    p.add_argument("--keypoints",  required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    p.add_argument("--conf",       type=float, default=0.30)
    p.add_argument("--imgsz",      type=int,   default=1280)
    p.add_argument("--ball-class", type=int,   default=0, dest="ball_class")
    args = p.parse_args(argv)

    for f in (args.video, args.weights, args.keypoints):
        if not f.exists():
            print(f"not found: {f}", file=sys.stderr); return 2

    result = run(args.video, args.weights, args.keypoints, args.conf, args.imgsz, args.ball_class)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    n = len(result["detections"])
    print(f"wrote {args.out}   ({n} ball dets, release_frame={result['release_frame']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
