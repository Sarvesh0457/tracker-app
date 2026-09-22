"""Stage 2 — solve camera pose from 6 clicked keypoints.

Usage:
    python -m backend.pipeline.reconstruct3d.calibrate \
        --video      path/to/clip.mp4 \
        --keypoints  path/to/clip_keypoints.json \
        --out-pose   path/to/clip_camera_pose.json \
        --out-check  path/to/clip_calibration_check.png

Approach:
  All 6 world points lie on the ground plane (Y=0), so the target is planar.
  With an unknown focal length, we search f over a plausible range, run
  solvePnP for each, and pick the f with the lowest mean reprojection error.
  Then we LM-refine pose at the chosen f. Principal point is assumed at the
  image center, skew is zero — standard broadcast-camera assumption.

Verification image:
  Draws each clicked point (red) and its reprojected position (green) with a
  line between them; overlays the pitch rectangle projected into image space.
  You can eyeball whether the calibration is sane in one glance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from .world_points import (
    KEYPOINT_NAMES,
    WORLD_POINTS_M,
    PITCH_LENGTH_M,
    RETURN_HALF_WIDTH_M,
    POPPING_IN_FRONT_M,
)


FOCAL_SEARCH_RANGE = range(500, 15001, 25)  # px; wide range so non-planar points can settle the ambiguity


def _load_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


def _load_keypoints(kp_path: Path):
    data = json.loads(kp_path.read_text(encoding="utf-8"))
    by_name = {p["name"]: p["uv"] for p in data["points"]}
    missing = [n for n in KEYPOINT_NAMES if n not in by_name]
    if missing:
        raise RuntimeError(f"keypoints file missing: {missing}")
    img_pts = np.array([by_name[n] for n in KEYPOINT_NAMES], dtype=np.float64)
    return data, img_pts


def _reprojection_error(world_pts, img_pts, rvec, tvec, K):
    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    per_point = np.linalg.norm(proj - img_pts, axis=1)
    return per_point, proj


def solve_pose(world_pts_np, img_pts_np, image_size):
    W, H = image_size
    cx, cy = W / 2.0, H / 2.0

    best = None   # (mean_err, f, K, rvec, tvec)
    for f in FOCAL_SEARCH_RANGE:
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(
                world_pts_np, img_pts_np, K, None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            continue
        if not ok:
            continue
        per_pt, _ = _reprojection_error(world_pts_np, img_pts_np, rvec, tvec, K)
        mean_err = float(per_pt.mean())
        if best is None or mean_err < best[0]:
            best = (mean_err, f, K, rvec, tvec)

    if best is None:
        raise RuntimeError("solvePnP failed at every focal length")

    _, f, K, rvec, tvec = best
    # LM refinement at chosen f
    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(world_pts_np, img_pts_np, K, None, rvec, tvec)
    per_pt, proj = _reprojection_error(world_pts_np, img_pts_np, rvec_ref, tvec_ref, K)
    return {
        "focal_px": float(f),
        "K": K,
        "rvec": rvec_ref,
        "tvec": tvec_ref,
        "per_point_error_px": per_pt,
        "reprojected_uv": proj,
    }


def pack_pose_json(pose, image_size, keypoints_meta):
    K = pose["K"]
    rvec = pose["rvec"].reshape(3)
    tvec = pose["tvec"].reshape(3)
    R, _ = cv2.Rodrigues(rvec)

    # Camera position in world coordinates: C = -R^T · t
    C_world = (-R.T @ tvec.reshape(3, 1)).reshape(3)

    W, H = image_size
    fx = float(K[0, 0])
    fov_v_deg = float(np.degrees(2.0 * np.arctan(H / (2.0 * fx))))

    per_pt = pose["per_point_error_px"].tolist()

    return {
        "video": keypoints_meta.get("video"),
        "frame_index": keypoints_meta.get("frame_index", 0),
        "image_size": [W, H],
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        },
        "fov_vertical_deg": fov_v_deg,
        "extrinsics_opencv": {
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            "R": R.tolist(),
            "camera_position_world_m": C_world.tolist(),
        },
        "reprojection": {
            "mean_px": float(np.mean(per_pt)),
            "max_px": float(np.max(per_pt)),
            "per_point_px": [
                {"name": KEYPOINT_NAMES[i], "err_px": per_pt[i]} for i in range(len(per_pt))
            ],
        },
    }


def draw_check(frame, img_pts, proj_pts, K, rvec, tvec):
    out = frame.copy()

    # Project pitch rectangle + midline for visual sanity
    z_near = POPPING_IN_FRONT_M
    z_far = PITCH_LENGTH_M - POPPING_IN_FRONT_M
    pitch_outline = np.array([
        [-RETURN_HALF_WIDTH_M, 0.0, z_near],
        [+RETURN_HALF_WIDTH_M, 0.0, z_near],
        [+RETURN_HALF_WIDTH_M, 0.0, z_far],
        [-RETURN_HALF_WIDTH_M, 0.0, z_far],
    ], dtype=np.float64)
    midline = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, PITCH_LENGTH_M],
    ], dtype=np.float64)

    for pts_3d, color in [(pitch_outline, (255, 200, 0)), (midline, (0, 255, 255))]:
        proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, None)
        proj = proj.reshape(-1, 2).astype(int)
        if len(proj) == 4:
            cv2.polylines(out, [proj], True, color, 2)
        else:
            cv2.line(out, tuple(proj[0]), tuple(proj[1]), color, 2)

    for i, (clk, rep) in enumerate(zip(img_pts, proj_pts)):
        clk_i = tuple(int(v) for v in clk)
        rep_i = tuple(int(v) for v in rep)
        cv2.line(out, clk_i, rep_i, (0, 0, 255), 1)
        cv2.circle(out, clk_i, 6, (0, 0, 255), 2)         # click (red ring)
        cv2.circle(out, rep_i, 4, (0, 255, 0), -1)        # reprojection (green dot)
        cv2.putText(out, f"{i}", (clk_i[0] + 8, clk_i[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # legend
    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, "red ring = your click   green dot = reprojection   cyan = pitch midline   orange = crease box",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--keypoints",  required=True, type=Path)
    p.add_argument("--out-pose",   required=True, type=Path)
    p.add_argument("--out-check",  required=True, type=Path)
    args = p.parse_args(argv)

    kp_data, img_pts = _load_keypoints(args.keypoints)
    frame_idx = kp_data.get("frame_index", 0)
    image_size = tuple(kp_data["image_size"])

    frame = _load_frame(args.video, frame_idx)
    if (frame.shape[1], frame.shape[0]) != image_size:
        print(f"warning: frame size {(frame.shape[1], frame.shape[0])} != keypoints image_size {image_size}",
              file=sys.stderr)

    world_pts = np.array(WORLD_POINTS_M, dtype=np.float64)
    pose = solve_pose(world_pts, img_pts, image_size)

    pose_json = pack_pose_json(pose, image_size, kp_data)
    args.out_pose.parent.mkdir(parents=True, exist_ok=True)
    args.out_pose.write_text(json.dumps(pose_json, indent=2), encoding="utf-8")

    check_img = draw_check(frame, img_pts, pose["reprojected_uv"],
                           pose["K"], pose["rvec"], pose["tvec"])
    args.out_check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_check), check_img)

    print(f"wrote pose:  {args.out_pose}")
    print(f"wrote check: {args.out_check}")
    print(f"focal_px:    {pose_json['intrinsics']['fx']:.1f}   fov_v: {pose_json['fov_vertical_deg']:.1f} deg")
    print(f"reprojection mean/max: {pose_json['reprojection']['mean_px']:.2f} / {pose_json['reprojection']['max_px']:.2f} px")
    for entry in pose_json["reprojection"]["per_point_px"]:
        print(f"    {entry['name']:26s}  err = {entry['err_px']:.2f} px")
    cam_pos = pose_json["extrinsics_opencv"]["camera_position_world_m"]
    print(f"camera position (world, metres): ({cam_pos[0]:+.2f}, {cam_pos[1]:+.2f}, {cam_pos[2]:+.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
