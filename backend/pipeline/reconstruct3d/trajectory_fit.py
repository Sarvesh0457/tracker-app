"""Stage 3+4 — back-project 2D detections to 3D rays and fit a physics
trajectory (projectile + one bounce), writing the Unity playback JSON.

Data flow:
    detections.json + camera_pose.json
        │
        ├─ auto-detect release_frame  (first frame of longest coherent run)
        │
        ├─ window = [release-20, release+150] clipped to available detections
        │
        ├─ each detection (u, v) → ray in world 3D via camera intrinsics/pose
        │
        └─ optimize (p0, v0, t_bounce, restitution) such that
               ball(t)  under gravity, with one bounce reflection off Y=0,
               projects back to the observed (u, v) with minimum error
                          │
                          ▼
                    track_3d.json  (Unity)

Usage:
    python -m backend.pipeline.reconstruct3d.trajectory_fit \
        --detections samples/4_027_06_detections_v8m.json \
        --pose       samples/4_027_06_camera_pose.json \
        --video      "D:/intership/Input files/Red_ball/4_027_06.mp4" \
        --out-track  samples/4_027_06_track_3d.json \
        --out-check  samples/4_027_06_track_check.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .world_points import (
    PITCH_LENGTH_M, POPPING_IN_FRONT_M, RETURN_HALF_WIDTH_M, BALL_RADIUS_M,
)


G = np.array([0.0, -9.81, 0.0])   # gravity, world coords (Y is up)


# ────────────────────────────── window selection ─────────────────────────────
def find_release_frame(dets, min_run_len=6, gap=3, min_step_px=15.0):
    """Return the first frame of the earliest detection run that carries a
    BOUNCE SIGNATURE — vertical pixel velocity sign change (ball dropping,
    then rising after hitting the pitch). That's the delivery. Post-impact
    deflections are horizontal-only runs and get skipped.

    Falls back to "longest run with fast motion" if no bounce signature is
    detected anywhere."""
    if not dets:
        return None
    ds = sorted(dets, key=lambda d: d["frame"])
    runs = []
    cur = [ds[0]]
    for d in ds[1:]:
        if d["frame"] - cur[-1]["frame"] <= gap:
            cur.append(d)
        else:
            runs.append(cur); cur = [d]
    runs.append(cur)

    fast_runs = []
    for r in runs:
        if len(r) < min_run_len:
            continue
        uv = np.array([[d["u"], d["v"]] for d in r], dtype=np.float64)
        step_lens = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        if step_lens.mean() < min_step_px:
            continue
        fast_runs.append(r)

    if not fast_runs:
        return None

    # Prefer earliest run whose Y-velocity flips sign (ball down → up).
    for r in fast_runs:
        vy = np.diff([d["v"] for d in r])
        # Look for a clear sign change (not just noise): significant down then up.
        signs = np.sign(vy)
        # Ignore near-zero steps
        signs = signs[np.abs(vy) > 3.0]
        if len(signs) < 4:
            continue
        # Any transition from +1 (going down) to -1 (going up)?
        for i in range(len(signs) - 1):
            if signs[i] > 0 and signs[i + 1] < 0:
                return r[0]["frame"]

    # No bounce signature found → longest fast run.
    return max(fast_runs, key=len)[0]["frame"]


def select_window(dets, release_frame, before=20, after=150, max_gap=6):
    """Grab detections in [release-before, release+after] but drop anything
    separated from the release by a gap larger than max_gap frames — that
    excludes ball-in-hand detections before release and spurious late ones.

    Also deduplicates per frame: when the detector emits multiple candidates
    on one frame (e.g. a stationary false positive on the far stumps + the
    real airborne ball), we keep only the highest-confidence one."""
    lo, hi = release_frame - before, release_frame + after
    band_all = [d for d in dets if lo <= d["frame"] <= hi]
    by_frame: dict[int, dict] = {}
    for d in band_all:
        cur = by_frame.get(d["frame"])
        if cur is None or d.get("conf", 0.0) > cur.get("conf", 0.0):
            by_frame[d["frame"]] = d
    band = sorted(by_frame.values(), key=lambda d: d["frame"])
    if not band:
        return []
    # release_frame itself may not be in the detection list; find nearest
    idx0 = min(range(len(band)), key=lambda i: abs(band[i]["frame"] - release_frame))
    # walk forward while gap ≤ max_gap
    out = [band[idx0]]
    i = idx0 + 1
    while i < len(band) and band[i]["frame"] - out[-1]["frame"] <= max_gap:
        out.append(band[i]); i += 1
    # walk backward
    j = idx0 - 1
    while j >= 0 and out[0]["frame"] - band[j]["frame"] <= max_gap:
        out.insert(0, band[j]); j -= 1

    # Bat-impact cutoff: as soon as consecutive pixel-step vectors reverse by
    # more than 120° OR jump more than 3× in length, cut. That's the bat's
    # signature — the ball changes direction/speed discontinuously on contact.
    if len(out) >= 5:
        uvs = np.array([[d["u"], d["v"]] for d in out], dtype=float)
        steps = np.diff(uvs, axis=0)
        lens = np.linalg.norm(steps, axis=1) + 1e-9
        units = steps / lens[:, None]
        cos_120 = np.cos(np.deg2rad(120))
        for k in range(2, len(steps) - 1):
            cos_ang = float(np.dot(units[k], units[k + 1]))
            ratio   = float(lens[k + 1] / lens[k])
            if cos_ang < cos_120 or ratio > 3.0:
                out = out[:k + 2]   # keep points 0..k+1 (inclusive of the pre-impact step)
                break
    return out


# ─────────────────────────── ray back-projection ─────────────────────────────
def pixel_to_ray(uv, K, R, C):
    """Return (origin, direction) of the world-space ray through pixel uv.

    K (3×3) intrinsics, R (3×3) camera rotation (world→camera), C (3,) camera
    center in world coords. R.T maps camera→world.
    """
    u, v = uv
    Kinv = np.linalg.inv(K)
    d_cam = Kinv @ np.array([u, v, 1.0])
    d_world = R.T @ d_cam
    d_world = d_world / np.linalg.norm(d_world)
    return C, d_world


def project_world(P_world, K, R, t):
    """World point → pixel (u, v). t is camera translation (world→camera)."""
    p_cam = R @ P_world + t
    if p_cam[2] <= 0:
        return np.array([np.nan, np.nan])
    p_img = K @ p_cam
    return p_img[:2] / p_img[2]


# ─────────────────────────── trajectory + bounce ─────────────────────────────
def _solve_bounce_time(p0, v0, ball_r=0.036):
    """Analytical bounce time: first t > 0 where the ball's *centre* reaches
    ball_r above the ground. Solves 0.5*g*t^2 + v0y*t + (p0y - ball_r) = 0.
    Returns +inf if the ball never lands within a reasonable horizon."""
    a = 0.5 * G[1]         # -4.905
    b = v0[1]
    c = p0[1] - ball_r
    disc = b * b - 4 * a * c
    if disc < 0:
        return float("inf")
    sq = np.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    cands = [t for t in (t1, t2) if t > 1e-4]
    return min(cands) if cands else float("inf")


def ball_at_time(t, p0, v0, _tb_ignored, restitution_e, friction_f):
    """Position at time t. Bounce time is derived from p0, v0 and Y=ball_radius,
    which enforces a physical bounce ON the ground (not above or below)."""
    tb = _solve_bounce_time(p0, v0)
    if t <= tb:
        return p0 + v0 * t + 0.5 * G * t**2
    p_b = p0 + v0 * tb + 0.5 * G * tb**2
    v_b = v0 + G * tb
    v_after = np.array([v_b[0] * friction_f, -v_b[1] * restitution_e, v_b[2] * friction_f])
    dt = t - tb
    return p_b + v_after * dt + 0.5 * G * dt**2


def fit_trajectory(times, uvs, K, R, t_vec, C):
    """Least-squares fit: minimise sum of (proj - obs)^2 over all detections.

    Parameters: p0(3), v0(3), t_bounce(1), restitution_e(1), friction_f(1).
    """
    # Initial guess: release near bowler's popping crease at hand height,
    # ball delivered toward striker (Z=0) at ~35 m/s (fast-medium pace),
    # slight downward and slight side-arm angle.
    x0 = np.array([
        0.0,  2.0,  18.9,      # p0  (bowler's popping crease, hand height)
        0.0, -2.0, -35.0,      # v0  (mostly -Z toward striker, slight drop)
        (times[-1] - times[0]) * 0.55,   # t_bounce ≈ 55% of window
        0.50,                             # restitution
        0.70,                             # friction
    ])
    # Physically realistic bounds:
    #   restitution (Y) : real cricket ball 0.30 - 0.55
    #   tangential friction : real cricket ball 0.55 - 0.85 (retains most forward
    #     speed; 0.20 was letting fits collapse to "straight up" post-bounce).
    #   release Z (down-pitch) : bowler releases 0.5-2m in front of the bowler's
    #     crease (which is at Z=18.9); so 15-22m works for most deliveries.
    # v_z upper bound = -20 m/s (72 km/h — slowest realistic bowling delivery);
    # keeps the fit out of scale-collapsed local minima where the ball "moves"
    # at walking pace over 1 m instead of 20 m.
    lo = np.array([-4.0, 0.5, 15.0,   -25, -20, -50, 0.05, 0.30, 0.55])
    hi = np.array([+4.0, 3.5, 25.0,   +25,  +8, -20, 3.00, 0.55, 0.85])

    def residuals(params):
        p0 = params[0:3]; v0 = params[3:6]
        tb = params[6]; e = params[7]; f = params[8]
        r = []
        for t, uv in zip(times, uvs):
            P = ball_at_time(t, p0, v0, tb, e, f)
            pr = project_world(P, K, R, t_vec)
            if np.any(np.isnan(pr)):
                r.extend([1e3, 1e3])
            else:
                r.append(pr[0] - uv[0])
                r.append(pr[1] - uv[1])
        return np.array(r)

    result = least_squares(residuals, x0, bounds=(lo, hi), method="trf", max_nfev=500)
    return result


# ───────────────────────────────── export ────────────────────────────────────
def build_track_json(pose_json, fit_result, dets_window, times, fps, sample_hz=240):
    p0 = fit_result.x[0:3]; v0 = fit_result.x[3:6]
    tb = _solve_bounce_time(p0, v0)   # override — physical value from analytic solver
    e = float(fit_result.x[7]); f = float(fit_result.x[8])
    duration = times[-1] - times[0]
    n_samples = max(2, int(duration * sample_hz) + 1)
    ts = np.linspace(0.0, duration, n_samples)

    traj = []
    for t in ts:
        p = ball_at_time(t, p0, v0, tb, e, f)
        traj.append({"t": round(float(t), 4), "p": [round(float(p[0]), 4),
                                                     round(float(p[1]), 4),
                                                     round(float(p[2]), 4)]})
    bounce_p = ball_at_time(tb, p0, v0, tb, e, f)
    intr = pose_json["intrinsics"]
    W, H = pose_json["image_size"]
    return {
        "video": pose_json.get("video"),
        "fps": fps,
        "render": {
            "resolution": [W, H],
            "aspect_ratio": round(W / H, 6),
            "fov_vertical_deg": pose_json["fov_vertical_deg"],
            "note": "Unity camera MUST use this aspect_ratio (Camera.aspect) and fov_vertical_deg (Camera.fieldOfView). Render target should match `resolution` (or same aspect); background video plane must fill the same viewport 1:1 for the CG ball to line up pixel-for-pixel with the source.",
        },
        "camera": {
            "position_m": pose_json["extrinsics_opencv"]["camera_position_world_m"],
            "rotation_matrix_row_major": [
                float(x) for row in pose_json["extrinsics_opencv"]["R"] for x in row
            ],
            "fov_vertical_deg": pose_json["fov_vertical_deg"],
            "intrinsics_px": intr,
            "image_size": pose_json["image_size"],
        },
        "delivery": {
            "release_frame": int(dets_window[0]["frame"]),
            "last_frame":    int(dets_window[-1]["frame"]),
            "impact_frame":  int(dets_window[-1]["frame"]),   # last physics-consistent detection
            "release_pos_m": [round(float(x), 3) for x in p0],
            "release_vel_ms": [round(float(x), 3) for x in v0],
            "release_speed_ms": round(float(np.linalg.norm(v0)), 3),
            "bounce_time_s": round(tb, 4),
            "bounce_pos_m":  [round(float(x), 3) for x in bounce_p],
            "restitution":   round(e, 3),
            "tangential_friction": round(f, 3),
            "fit_cost_px2":  round(float(fit_result.cost), 3),
        },
        "trajectory": traj,
    }


# ───────────────────────────── check overlay ─────────────────────────────────
def draw_check(video_path, window, times, fit, K, R, t_vec, C, frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read check frame")
    out = frame.copy()

    p0 = fit.x[0:3]; v0 = fit.x[3:6]
    tb = float(fit.x[6]); e = float(fit.x[7]); f = float(fit.x[8])

    # densely sampled fitted trajectory reprojected
    n = 200
    ts_dense = np.linspace(times[0], times[-1], n)
    prev = None
    for t in ts_dense:
        P = ball_at_time(t - times[0], p0, v0, tb, e, f)
        uv = project_world(P, K, R, t_vec)
        if not np.any(np.isnan(uv)):
            pt = (int(uv[0]), int(uv[1]))
            if prev is not None:
                cv2.line(out, prev, pt, (0, 200, 255), 2)   # orange = fitted
            prev = pt

    # bounce point
    P_b = ball_at_time(tb, p0, v0, tb, e, f)
    uv_b = project_world(P_b, K, R, t_vec)
    if not np.any(np.isnan(uv_b)):
        pt_b = (int(uv_b[0]), int(uv_b[1]))
        cv2.circle(out, pt_b, 10, (0, 255, 255), 2)         # yellow ring at bounce
        cv2.putText(out, "bounce", (pt_b[0]+12, pt_b[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # observed detections
    for d in window:
        cv2.circle(out, (int(d["u"]), int(d["v"])), 4, (0, 255, 0), -1)  # green dot

    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out, "green = observed detection   orange = fitted 3D trajectory (reprojected)   "
                     "yellow ring = bounce",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ────────────────────────────────── main ─────────────────────────────────────
def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--pose",       required=True, type=Path)
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--out-track",  required=True, type=Path)
    p.add_argument("--out-check",  required=True, type=Path)
    p.add_argument("--release-before", type=int, default=20)
    p.add_argument("--release-after",  type=int, default=150)
    p.add_argument("--release-frame", type=int, default=None,
                   help="Override auto-detected release frame")
    args = p.parse_args(argv)

    det_data = json.loads(args.detections.read_text(encoding="utf-8"))
    pose = json.loads(args.pose.read_text(encoding="utf-8"))
    fps = float(det_data.get("fps", 25.0))
    all_dets = det_data["detections"]

    if args.release_frame is not None:
        release = args.release_frame
        release_source = "CLI"
    elif det_data.get("release_frame") is not None:
        release = int(det_data["release_frame"])
        release_source = "V8m pose-gated ReleaseDetector"
    else:
        release = find_release_frame(all_dets)
        release_source = "bounce-signature heuristic"
    print(f"release source: {release_source}")
    if release is None:
        print("could not auto-detect release; pass --release-frame N", file=sys.stderr)
        return 1
    window = select_window(all_dets, release, args.release_before, args.release_after)
    if len(window) < 5:
        print(f"window has only {len(window)} detections — too few for a fit", file=sys.stderr)
        return 1

    # Build K, R, t, C
    intr = pose["intrinsics"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1.0]])
    R = np.array(pose["extrinsics_opencv"]["R"], dtype=np.float64)
    tvec = np.array(pose["extrinsics_opencv"]["tvec"], dtype=np.float64)
    C = np.array(pose["extrinsics_opencv"]["camera_position_world_m"], dtype=np.float64)

    # ─── Shrink-until-physics-fits impact discovery ─────────────────────────
    # Try the fit with all detections from release_frame up to a candidate
    # impact. If any detection is more than TOLERANCE_PX from the physics
    # reprojection, drop that candidate and shrink to the previous DETECTED
    # frame (not the previous frame number). Repeat until the fit is
    # physics-consistent or too few points remain.
    TOLERANCE_PX   = 3.0
    MIN_FIT_POINTS = 6

    print(f"release estimate = frame {release}")
    print(f"initial window has {len(window)} detections (frames {window[0]['frame']}..{window[-1]['frame']})")

    fit = None
    per_err = None
    accepted_window = None
    detected_frames_desc = sorted({d["frame"] for d in window}, reverse=True)
    for candidate_impact in detected_frames_desc:
        cand_win = [d for d in window if d["frame"] <= candidate_impact]
        if len(cand_win) < MIN_FIT_POINTS:
            print(f"  too few points ({len(cand_win)}) — stopping shrink loop")
            break
        cand_times = np.array([(d["frame"] - cand_win[0]["frame"]) / fps for d in cand_win])
        cand_uvs   = np.array([[d["u"], d["v"]] for d in cand_win])
        cand_fit = fit_trajectory(cand_times, cand_uvs, K, R, tvec, C)
        cand_err = np.linalg.norm(cand_fit.fun.reshape(-1, 2), axis=1)
        max_err  = float(cand_err.max())
        print(f"  candidate impact = {candidate_impact:3d}  ({len(cand_win)} pts)  "
              f"max_err={max_err:.2f}px  mean_err={cand_err.mean():.2f}px  "
              f"{'ACCEPT' if max_err <= TOLERANCE_PX else 'reject'}")
        if max_err <= TOLERANCE_PX:
            fit = cand_fit
            per_err = cand_err
            accepted_window = cand_win
            break

    if fit is None:
        print(f"no candidate impact produced max_err <= {TOLERANCE_PX} px — no fit accepted",
              file=sys.stderr)
        return 1

    window = accepted_window
    times = np.array([(d["frame"] - window[0]["frame"]) / fps for d in window])
    uvs   = np.array([[d["u"], d["v"]] for d in window])
    print(f"ACCEPTED window: frames {window[0]['frame']}..{window[-1]['frame']} "
          f"({len(window)} pts, {times[-1]:.2f}s)  impact_frame={window[-1]['frame']}")
    print(f"fit converged: {fit.success}, iterations={fit.nfev}, "
          f"reprojection mean={per_err.mean():.2f}px  max={per_err.max():.2f}px")

    track_json = build_track_json(pose, fit, window, times, fps)
    args.out_track.parent.mkdir(parents=True, exist_ok=True)
    args.out_track.write_text(json.dumps(track_json, indent=2), encoding="utf-8")
    print(f"wrote track:  {args.out_track}")

    mid_frame = window[len(window)//2]["frame"]
    check = draw_check(args.video, window, times, fit, K, R, tvec, C, mid_frame)
    args.out_check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_check), check)
    print(f"wrote check:  {args.out_check}")

    d = track_json["delivery"]
    print(f"\nrelease pos:  {d['release_pos_m']} m")
    print(f"release vel:  {d['release_vel_ms']} m/s   (speed {d['release_speed_ms']:.2f} m/s = {d['release_speed_ms']*3.6:.1f} km/h)")
    print(f"bounce at t={d['bounce_time_s']:.3f}s, pos {d['bounce_pos_m']} m")
    print(f"restitution={d['restitution']}, tangential friction={d['tangential_friction']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
