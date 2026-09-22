"""3D reconstruction pipeline: single-camera calibration + physics-fit trajectory.

Stages:
  keypoints  — mark 6 stump points on frame 0 (manual click tool for v1)
  calibrate  — solvePnP → camera pose in Unity world coords
  backproject — 2D detections → 3D rays
  physics_fit — projectile + bounce fit, minimising reprojection error
  export     — track_3d.json for Unity WebGL playback
"""
