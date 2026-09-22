import cv2
import numpy as np
from config import COLOR_BALL, COLOR_BAT, COLOR_TEXT_FG


def draw_release_to_bounce_parabola(
    frame,
    release_pt,
    bounce_pt,
    alpha: float = 0.40,
    thickness: int = 10,
    arc_ratio: float = 0.25,
    n_samples: int = 80,
    progress: float = 1.0,
    trajectory_points=None,
    frame_range=None,
):
    """Draw a parabolic arc anchored at release_pt and bounce_pt.

    When trajectory_points (list of (frame, x, y)) is provided, the curve is
    a CONSTRAINED quadratic — its endpoints are pinned exactly to release_pt
    (t=0) and bounce_pt (t=1), and its bow (one deflection parameter in each
    of x and y) is fitted via least squares to the interior ball positions.

    frame_range=(f_release, f_bounce) maps trajectory frames onto t ∈ [0, 1].
    If omitted, the min/max frames in trajectory_points are used.

    progress (0..1) animates the sweep: 0 hides the arc, 1 draws it fully.
    """
    if release_pt is None or bounce_pt is None:
        return frame

    progress = max(0.0, min(1.0, float(progress)))
    if progress <= 0.0:
        return frame

    x1, y1 = float(release_pt[0]), float(release_pt[1])
    x2, y2 = float(bounce_pt[0]),  float(bounce_pt[1])

    # ── Constrained quadratic: passes EXACTLY through release & bounce ──
    if trajectory_points and len(trajectory_points) >= 1:
        if frame_range is not None:
            f_lo, f_hi = float(frame_range[0]), float(frame_range[1])
        else:
            fs = [p[0] for p in trajectory_points]
            f_lo, f_hi = float(min(fs)), float(max(fs))

        if f_hi > f_lo:
            ts_p = np.array([(p[0] - f_lo) / (f_hi - f_lo)
                             for p in trajectory_points], dtype=float)
            xs_p = np.array([p[1] for p in trajectory_points], dtype=float)
            ys_p = np.array([p[2] for p in trajectory_points], dtype=float)

            # Model: x(t) = (1-t)*x1 + t*x2 + kx * t*(1-t)
            #        y(t) = (1-t)*y1 + t*y2 + ky * t*(1-t)
            # Solve kx, ky by least squares against the interior points.
            a   = ts_p * (1.0 - ts_p)
            denom = float(np.sum(a * a))
            if denom > 1e-9:
                bx = xs_p - ((1.0 - ts_p) * x1 + ts_p * x2)
                by = ys_p - ((1.0 - ts_p) * y1 + ts_p * y2)
                kx = float(np.sum(a * bx) / denom)
                ky = float(np.sum(a * by) / denom)
            else:
                kx = ky = 0.0

            t = np.linspace(0.0, progress,
                            max(2, int(round(n_samples * progress))))
            a_t = t * (1.0 - t)
            xs = (1.0 - t) * x1 + t * x2 + kx * a_t
            ys = (1.0 - t) * y1 + t * y2 + ky * a_t
            pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

            overlay = frame.copy()
            cv2.polylines(overlay, [pts], isClosed=False,
                          color=(255, 255, 255), thickness=thickness,
                          lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            return frame

    # ── Fallback: synthetic arc with apex above the chord midpoint ──────
    chord = float(np.hypot(x2 - x1, y2 - y1))
    if chord < 2.0:
        return frame
    arc_h = arc_ratio * chord

    t = np.linspace(0.0, progress, max(2, int(round(n_samples * progress))))
    xs = (1.0 - t) * x1 + t * x2
    ys = (1.0 - t) * y1 + t * y2 - 4.0 * arc_h * t * (1.0 - t)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)

    overlay = frame.copy()
    cv2.polylines(overlay, [pts], isClosed=False,
                  color=(255, 255, 255), thickness=thickness,
                  lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame

def draw_detections(frame, detections):
    """Draw a green bounding box + confidence label around each BALL detection."""
    if not detections:
        return frame
    for d in detections:
        if d.get("class_name") != "ball":
            continue
        x1, y1 = int(d["x1"]), int(d["y1"])
        x2, y2 = int(d["x2"]), int(d["y2"])
        conf = float(d.get("conf", 0.0))
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"ball {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(frame, (x1, ty1), (x1 + tw + 6, ty1 + th + 6),
                      color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1 + 3, ty1 + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame
