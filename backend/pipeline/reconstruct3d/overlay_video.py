"""Render the fitted 3D trajectory back onto the source video and save mp4.

Draws:
  - the full reprojected trajectory as an orange polyline (all frames)
  - a large filled orange ball at the current time's 3D position (reprojected)
  - green rings around every observed V8m detection in the window
  - a yellow ring at the bounce point

So you can eyeball: does the fitted trajectory PASS THROUGH the observed
ball detections?

Usage:
    python -m backend.pipeline.reconstruct3d.overlay_video \
        --video      "D:/intership/Input files/Red_ball/4_027_06.mp4" \
        --track      samples/4_027_06_track_3d.json \
        --detections samples/4_027_06_detections_v8m.json \
        --out        "D:/intership/OUTPUT/4_027_06_overlay.mp4"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np


def project_world(P, K, R, t):
    p_cam = R @ P + t
    if p_cam[2] <= 0:
        return None
    p_img = K @ p_cam
    return int(round(p_img[0] / p_img[2])), int(round(p_img[1] / p_img[2]))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--track",      required=True, type=Path)
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--out",        required=True, type=Path)
    args = p.parse_args(argv)

    track = json.loads(args.track.read_text(encoding="utf-8"))
    dets  = json.loads(args.detections.read_text(encoding="utf-8"))["detections"]

    # Recover the OpenCV camera model from the track JSON.
    intr = track["camera"]["intrinsics_px"]
    K = np.array([[intr["fx"], 0, intr["cx"]],
                  [0, intr["fy"], intr["cy"]],
                  [0, 0, 1.0]])
    r = track["camera"]["rotation_matrix_row_major"]
    R = np.array(r, dtype=np.float64).reshape(3, 3)
    C = np.array(track["camera"]["position_m"], dtype=np.float64)
    tvec = -R @ C   # world→camera translation

    fps = float(track["fps"])
    release_frame = track["delivery"]["release_frame"]
    last_frame    = track["delivery"]["last_frame"]

    # Trajectory samples: (t, [x,y,z]). t is seconds from release_frame.
    ts    = np.array([pt["t"] for pt in track["trajectory"]])
    Pts3d = np.array([pt["p"] for pt in track["trajectory"]])
    fit_end_idx = len(Pts3d) - 1   # last FITTED sample; anything after is extrapolation

    # Extrapolate from the fit's endpoint under FULL PROJECTILE PHYSICS —
    # gravity + one bounce off Y=0 if the ball would hit the ground before
    # reaching the stump line. Restitution and friction come from the fit.
    # This handles both cases: impact at bounce (immediate flat trajectory
    # forward) and impact mid-air (ball still needs to come down + bounce +
    # continue to stumps).
    STUMP_Z = 0.0
    BALL_R = 0.036
    G = np.array([0.0, -9.81, 0.0])
    restitution = float(track["delivery"].get("restitution", 0.5))
    friction    = float(track["delivery"].get("tangential_friction", 0.7))

    def _time_to_z(p, v, z_target):
        """Solve p.z + v.z * t = z_target for t > 0. Returns +Inf if no solution."""
        if abs(v[2]) < 1e-9: return float("inf")
        t = (z_target - p[2]) / v[2]
        return t if t > 1e-6 else float("inf")

    def _time_to_ground(p, v, y_target=BALL_R):
        """Smallest positive t at which p.y + v.y*t + 0.5*g*t^2 = y_target."""
        a = 0.5 * G[1]                 # -4.905
        b = v[1]
        c = p[1] - y_target
        disc = b * b - 4 * a * c
        if disc < 0: return float("inf")
        sq = np.sqrt(disc)
        cands = [(-b - sq) / (2 * a), (-b + sq) / (2 * a)]
        cands = [t for t in cands if t > 1e-4]
        return min(cands) if cands else float("inf")

    if len(Pts3d) >= 2:
        p_end, p_prev = Pts3d[-1], Pts3d[-2]
        dt = ts[-1] - ts[-2]
        if dt > 0:
            v_end = (p_end - p_prev) / dt
            # Are we already past the stump line?
            direction = np.sign(v_end[2]) if abs(v_end[2]) > 1e-6 else 0.0
            past_line = (direction < 0 and p_end[2] <= STUMP_Z) or \
                        (direction > 0 and p_end[2] >= STUMP_Z) or direction == 0.0
            if not past_line:
                # Walk projectile forward, applying up to 2 bounces if needed
                # (usually 0 or 1 bounces before reaching the stump line).
                cur_p, cur_v, cur_t = p_end.copy(), v_end.copy(), 0.0
                t_base = ts[fit_end_idx]
                for bounce_count in range(3):
                    t_stump = _time_to_z(cur_p, cur_v, STUMP_Z)
                    t_ground = _time_to_ground(cur_p, cur_v) if cur_p[1] > BALL_R + 1e-4 or cur_v[1] > 0 else float("inf")
                    if t_stump <= t_ground:
                        # Reaches stump line before hitting ground → sample & done
                        n_extra = max(4, int(40 * t_stump / max(1e-3, dt * 10)))
                        for k in range(1, n_extra + 1):
                            tau = (k / n_extra) * t_stump
                            p_tau = cur_p + cur_v * tau + 0.5 * G * tau * tau
                            ts    = np.append(ts, t_base + cur_t + tau)
                            Pts3d = np.vstack([Pts3d, p_tau])
                        break
                    else:
                        # Ground bounce before stump line — sample to bounce, reflect, continue
                        n_pre = max(3, int(30 * t_ground / max(1e-3, dt * 10)))
                        for k in range(1, n_pre + 1):
                            tau = (k / n_pre) * t_ground
                            p_tau = cur_p + cur_v * tau + 0.5 * G * tau * tau
                            ts    = np.append(ts, t_base + cur_t + tau)
                            Pts3d = np.vstack([Pts3d, p_tau])
                        # Update state to just after bounce
                        cur_p = cur_p + cur_v * t_ground + 0.5 * G * t_ground * t_ground
                        cur_p[1] = BALL_R
                        v_at_ground = cur_v + G * t_ground
                        cur_v = np.array([v_at_ground[0] * friction,
                                           -v_at_ground[1] * restitution,
                                           v_at_ground[2] * friction])
                        cur_t += t_ground
                        # if velocity essentially dead, stop
                        if abs(cur_v[2]) < 0.1: break

    # Precompute all reprojected trajectory pixel points.
    traj_pix = []
    for P in Pts3d:
        pt = project_world(P, K, R, tvec)
        traj_pix.append(pt)   # None if behind camera

    # Marker at where the fit ends (transition orange → blue).
    impact_pix = traj_pix[fit_end_idx] if 0 <= fit_end_idx < len(traj_pix) else None

    # Bounce point pixel.
    Pb = np.array(track["delivery"]["bounce_pos_m"])
    bounce_pix = project_world(Pb, K, R, tvec)

    # Detections keyed by frame (keep highest-confidence one per frame).
    det_by_frame = {}
    for d in dets:
        cur = det_by_frame.get(d["frame"])
        if cur is None or d.get("conf", 0.0) > cur.get("conf", 0.0):
            det_by_frame[d["frame"]] = d

    # Pure physics: two smooth parabolas (pre-bounce, post-bounce) joined at
    # the bounce point. NO anchoring to detections — detections stay as
    # separate green rings for reference.

    cap = cv2.VideoCapture(str(args.video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (W, H))

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 1a. Fitted portion — RED at 40% opacity (blended via overlay layer).
        overlay = frame.copy()
        fitted = [p for p in traj_pix[:fit_end_idx + 1] if p is not None]
        for a, b in zip(fitted[:-1], fitted[1:]):
            cv2.line(overlay, a, b, (0, 0, 255), 3)   # BGR red
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

        # 1b. Extrapolated portion — solid blue (from Unity-style physics extension).
        extrap = [p for p in traj_pix[fit_end_idx:] if p is not None]
        for a, b in zip(extrap[:-1], extrap[1:]):
            cv2.line(frame, a, b, (255, 120, 0), 2)   # blue

        # 2a. Bounce marker (yellow ring).
        if bounce_pix is not None:
            cv2.circle(frame, bounce_pix, 12, (0, 255, 255), 2)
        # 2b. Impact marker at fit endpoint — where red hands off to blue.
        if impact_pix is not None:
            cv2.circle(frame, impact_pix, 9, (255, 120, 0), -1)
            cv2.circle(frame, impact_pix, 11, (255, 255, 255), 1)
            cv2.putText(frame, "impact", (impact_pix[0] + 12, impact_pix[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)

        # 3. Current-time ball position marker. Rule: at every video frame that
        #    has a detection, the marker sits on the DETECTION (timestamp ↔
        #    location match). Between detections, it falls back to the physics
        #    fit's reprojected position at the corresponding time.
        if release_frame <= f <= last_frame + 30:
            d_now = det_by_frame.get(f)
            pt = None
            if d_now is not None:
                pt = (int(round(d_now["u"])), int(round(d_now["v"])))
            else:
                t_now = (f - release_frame) / fps
                if 0.0 <= t_now <= ts[-1]:
                    idx = int(np.clip(np.searchsorted(ts, t_now), 0, len(ts) - 1))
                    pt = traj_pix[idx]
            if pt is not None:
                cv2.circle(frame, pt, 10, (0, 140, 255), -1)  # solid orange ball
                cv2.circle(frame, pt, 12, (255, 255, 255), 1)

        # 4. Observed detection for this exact frame (green ring).
        d = det_by_frame.get(f)
        if d is not None:
            cv2.circle(frame, (int(d["u"]), int(d["v"])), 6, (0, 255, 0), 2)

        # HUD
        cv2.rectangle(frame, (0, 0), (W, 44), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {f}/{n_total}   orange line = fitted 3D trajectory (all reprojected)   "
                           f"orange ball = current-time model pos   green ring = raw ball detection   "
                           f"yellow ring = bounce",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"wrote {args.out}  ({f} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
