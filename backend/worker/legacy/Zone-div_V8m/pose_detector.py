"""Geometry-only stub of pose_detector for the Cloud Run worker.

The full V8m pose_detector runs YOLOv8-pose (ultralytics + torch) to gate
release detection on a bowler run-up. In this deployment the pipeline
orchestrator (pipeline/process.py) invokes ReleaseDetector without pose
inputs — pose_confirmed defaults to True and wrist/shoulder args are None,
which disables those gates gracefully. We only need bowl_stump_top_y, a
pure-geometry helper that release_detector imports at module load time.
"""
from typing import Optional

STUMP_H_OVER_PITCH_L = 0.71 / 17.68


def bowl_stump_top_y(markers) -> Optional[float]:
    if not markers:
        return None
    bs = markers.get("bowl_stump")
    ts = markers.get("bat_stump")
    if not (bs and ts):
        return None
    pitch_dy = abs(float(ts[1]) - float(bs[1]))
    stump_h_px = pitch_dy * STUMP_H_OVER_PITCH_L
    return float(bs[1]) - stump_h_px
