"""Diagnose why V8m's PoseDetector didn't lock the bowler earlier — runs
YOLOv8-pose on a given frame range and prints, for each detected person,
their wrist + shoulder coords + confidences and whether the raised-arm
check would fire.

Usage:
    "C:/Program Files/Python311/python.exe" -m backend.pipeline.reconstruct3d._pose_diag \
        --video path/to/clip.mp4 --start 190 --end 232
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import cv2
import numpy as np


L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST,    R_WRIST    = 9, 10


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--start", type=int, default=190)
    p.add_argument("--end",   type=int, default=232)
    p.add_argument("--conf",  type=float, default=0.10)   # very loose to see everything
    args = p.parse_args(argv)

    from ultralytics import YOLO
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n-pose.pt")
    model.to(device)

    cap = cv2.VideoCapture(str(args.video))
    for f in range(args.start, args.end + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue

        r = model.track(frame, imgsz=640, conf=args.conf, verbose=False,
                        persist=True, tracker="bytetrack.yaml")[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            print(f"f={f:3d}  NO PERSONS DETECTED")
            continue

        kxy   = r.keypoints.xy.cpu().numpy()          # (N, 17, 2)
        kconf = (r.keypoints.conf.cpu().numpy()
                 if r.keypoints.conf is not None else None)  # (N, 17)
        pscores = r.boxes.conf.cpu().numpy()
        ids     = r.boxes.id.int().cpu().tolist() if r.boxes.id is not None else [-1] * len(pscores)

        # Only show persons with box confidence ≥ 0.25 (i.e., legitimate people).
        for i, ps in enumerate(pscores):
            if ps < 0.25: continue
            kp = kxy[i]
            kc = kconf[i] if kconf is not None else np.ones(17)
            LS_ok = kc[L_SHOULDER] > 0.3 and kp[L_SHOULDER][0] > 0
            LW_ok = kc[L_WRIST]    > 0.3 and kp[L_WRIST][0]    > 0
            RS_ok = kc[R_SHOULDER] > 0.3 and kp[R_SHOULDER][0] > 0
            RW_ok = kc[R_WRIST]    > 0.3 and kp[R_WRIST][0]    > 0

            note = []
            if LS_ok and LW_ok:
                dy = kp[L_WRIST][1] - kp[L_SHOULDER][1]   # >0 means wrist BELOW shoulder
                raised = kp[L_WRIST][1] < kp[L_SHOULDER][1]
                note.append(f"L wrist({int(kp[L_WRIST][0])},{int(kp[L_WRIST][1])}) sh({int(kp[L_SHOULDER][0])},{int(kp[L_SHOULDER][1])}) dy={dy:+.0f} raised={raised}")
            if RS_ok and RW_ok:
                dy = kp[R_WRIST][1] - kp[R_SHOULDER][1]
                raised = kp[R_WRIST][1] < kp[R_SHOULDER][1]
                note.append(f"R wrist({int(kp[R_WRIST][0])},{int(kp[R_WRIST][1])}) sh({int(kp[R_SHOULDER][0])},{int(kp[R_SHOULDER][1])}) dy={dy:+.0f} raised={raised}")

            b = r.boxes.xyxy[i].cpu().numpy()
            print(f"f={f:3d} id={ids[i]:>3} pconf={ps:.2f} box=({int(b[0])},{int(b[1])})-({int(b[2])},{int(b[3])})  "
                  f"kconf[L_sh,L_wr,R_sh,R_wr]=({kc[L_SHOULDER]:.2f},{kc[L_WRIST]:.2f},{kc[R_SHOULDER]:.2f},{kc[R_WRIST]:.2f})  "
                  + "  ".join(note))
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
