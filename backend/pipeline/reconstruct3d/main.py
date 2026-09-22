"""One-command end-to-end pipeline.

    python -m backend.pipeline.reconstruct3d.main --video path/to/clip.mp4

Runs all stages, deriving output names from the video's stem, and drops the
final overlay in D:\\intership\\OUTPUT\\ plus the Unity JSON in the project's
StreamingAssets folder. Skips stages whose output already exists unless
--force is passed.

Stages, in order:
  1. keypoints       — interactive 8-point picker (frame 0 by default)
  2. calibrate       — solvePnP → camera pose
  3. detect          — V8m ball detector (uses system Python for ultralytics)
  4. trajectory-fit  — physics fit with auto release + bat-impact detection
  5. overlay-video   — reprojected trajectory on source video
  6. copy-to-unity   — writes track_3d.json into StreamingAssets

Flags:
  --frame N              first-frame index for the keypoint picker
  --force                re-run every stage even if outputs already exist
  --skip-unity           don't copy JSON to Unity project
  --release-before N     window frames before release (default 0)
  --release-after  N     window frames after  release (default 22)
  --release-frame  N     override auto-detected release frame
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SAMPLES_DIR = Path(r"D:/intership/tracker_app/samples")
OUTPUT_DIR  = Path(r"D:/intership/OUTPUT")
UNITY_STREAMING = Path(r"C:/Internship/My project/Assets/StreamingAssets")
V8M_WEIGHTS = Path(r"D:/intership/July/Zone-div_V8m - Copy/best.pt")
V8M_CONF    = 0.30

VENV_PY   = Path(r"D:/intership/tracker_app/backend/.venv/Scripts/python.exe")
SYSTEM_PY = Path(r"C:/Program Files/Python311/python.exe")   # has ultralytics + torch+cuda
REPO_ROOT = Path(r"D:/intership/tracker_app")


def _run(cmd, cwd=REPO_ROOT):
    print(f"\n$ {' '.join(str(x) for x in cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(cwd))
    if r.returncode != 0:
        raise SystemExit(f"stage failed with exit {r.returncode}")


def _need(out_path: Path, force: bool) -> bool:
    if force or not out_path.exists():
        return True
    print(f"[skip] {out_path.name} already exists (use --force to redo)")
    return False


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",           required=True, type=Path)
    p.add_argument("--frame",           type=int, default=0)
    p.add_argument("--force",           action="store_true")
    p.add_argument("--skip-unity",      action="store_true")
    p.add_argument("--release-before",  type=int, default=0)
    p.add_argument("--release-after",   type=int, default=22)
    p.add_argument("--release-frame",   type=int, default=None)
    args = p.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr); return 2

    stem = args.video.stem
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keypoints_json = SAMPLES_DIR / f"{stem}_keypoints.json"
    pose_json      = SAMPLES_DIR / f"{stem}_camera_pose.json"
    calib_check    = SAMPLES_DIR / f"{stem}_calibration_check.png"
    detections     = SAMPLES_DIR / f"{stem}_detections_v8m.json"
    track_json     = SAMPLES_DIR / f"{stem}_track_3d.json"
    track_check    = SAMPLES_DIR / f"{stem}_track_check.png"
    overlay_mp4    = OUTPUT_DIR  / f"{stem}_trajectory_overlay.mp4"

    print(f"=== pipeline for {stem} ===")

    # 1. Keypoints — interactive; only run if file missing (or --force)
    if _need(keypoints_json, args.force):
        _run([str(VENV_PY), "-m", "backend.pipeline.reconstruct3d.keypoints",
              "--video", str(args.video),
              "--out",   str(keypoints_json),
              "--frame", str(args.frame)])

    # 2. Calibrate
    if _need(pose_json, args.force):
        _run([str(VENV_PY), "-m", "backend.pipeline.reconstruct3d.calibrate",
              "--video",     str(args.video),
              "--keypoints", str(keypoints_json),
              "--out-pose",  str(pose_json),
              "--out-check", str(calib_check)])

    # 3. V8m detection with pose-gated release (system python — has ultralytics + torch-cuda)
    if _need(detections, args.force):
        _run([str(SYSTEM_PY), "-m", "backend.pipeline.reconstruct3d.detect_ball_v8m",
              "--video",     str(args.video),
              "--weights",   str(V8M_WEIGHTS),
              "--keypoints", str(keypoints_json),
              "--out",       str(detections),
              "--conf",      str(V8M_CONF)])

    # 4. Trajectory fit
    fit_cmd = [str(VENV_PY), "-m", "backend.pipeline.reconstruct3d.trajectory_fit",
               "--detections", str(detections),
               "--pose",       str(pose_json),
               "--video",      str(args.video),
               "--out-track",  str(track_json),
               "--out-check",  str(track_check),
               "--release-before", str(args.release_before),
               "--release-after",  str(args.release_after)]
    if args.release_frame is not None:
        fit_cmd += ["--release-frame", str(args.release_frame)]
    if _need(track_json, args.force):
        _run(fit_cmd)

    # 5. Overlay video (always regenerate if track was regenerated)
    if _need(overlay_mp4, args.force) or (track_json.stat().st_mtime > overlay_mp4.stat().st_mtime
                                          if overlay_mp4.exists() else True):
        _run([str(VENV_PY), "-m", "backend.pipeline.reconstruct3d.overlay_video",
              "--video",      str(args.video),
              "--track",      str(track_json),
              "--detections", str(detections),
              "--out",        str(overlay_mp4)])

    # 6. Unity handoff
    if not args.skip_unity:
        UNITY_STREAMING.mkdir(parents=True, exist_ok=True)
        dst = UNITY_STREAMING / "track_3d.json"
        dst.write_bytes(track_json.read_bytes())
        print(f"[unity] copied  {track_json.name}  ->  {dst}")

    print("\n=== done ===")
    print(f"  overlay:  {overlay_mp4}")
    print(f"  unity:    {UNITY_STREAMING / 'track_3d.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
