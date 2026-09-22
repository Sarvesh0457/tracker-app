"""
TRAJECTORY — raw polyline through every real ball detection.

No polyfit, no interpolation, no segmentation. Each consecutive pair of real
ball-center detections is connected with a straight line, producing a trail
that spans from the first ball seen to the last. The 3-pass cylinder
rendering (edge / core / highlight) is preserved for the same visual style
the segmented version had.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

# Interpolator usage disabled — gap-filling is now off everywhere.
# from interpolator import interpolate_gap
# from config import (
#     INTERP_METHOD,
#     INTERP_MAX_GAP_FRAMES,
#     INTERP_ANCHOR_FRAMES,
#     GRAVITY_PX_PER_FRAME2,
#     PITCH_Y,
# )


class Trajectory:
    def __init__(
        self,
        edge_color:      tuple[int, int, int] = (160, 160, 160),  # cylinder shading
        core_color:      tuple[int, int, int] = (255, 255, 255),  # white core
        highlight_color: tuple[int, int, int] = (255, 255, 255),  # bright top streak
        edge_thickness:      int = 14,
        core_thickness:      int = 9,
        highlight_thickness: int = 3,
        highlight_offset_y:  int = -3,
        sample_count: int = 200,
        alpha: float = 0.6,                                       # 40% transparent
    ):
        self._pts: list[tuple[int, int, int]] = []   # (frame, x, y)
        self._edge_color = edge_color
        self._core_color = core_color
        self._highlight_color = highlight_color
        self._edge_t = edge_thickness
        self._core_t = core_thickness
        self._highlight_t = highlight_thickness
        self._hl_dy = highlight_offset_y
        self._n_samples = sample_count
        self._alpha = alpha

        # Direction-change markers: list of (frame, x, y) where the ball's
        # horizontal velocity flipped against the predicted direction.
        self._dir_changes: list[tuple[int, int, int]] = []

    # Minimum velocity magnitude (px/frame) on both predicted and actual
    # sides before we consider deviation meaningful (filters jitter on a
    # near-stationary ball).
    _MIN_SPEED = 1.5
    # Angular deviation threshold (degrees) between predicted and actual
    # velocity vectors. Anything beyond this is flagged as a deviation event.
    _DEVIATION_ANGLE_DEG = 10.0

    def update(
        self,
        detections: list,
        frame_num: int,
        release_frame: Optional[int] = None,
        bounce_frame:  Optional[int] = None,
    ) -> None:
        balls = [d for d in detections if d.get("class_name") == "ball"]
        if not balls:
            return
        best = max(balls, key=lambda d: d["conf"])
        cx = (best["x1"] + best["x2"]) // 2
        cy = (best["y1"] + best["y2"]) // 2
        self._pts.append((frame_num, cx, cy))
        self._check_direction_change(release_frame, bounce_frame)

    def _check_direction_change(
        self,
        release_frame: Optional[int],
        bounce_frame:  Optional[int],
    ) -> None:
        """
        Flag a marker when the predicted horizontal direction (from the
        prior fit) disagrees with the actual motion on the latest frame.
        Gated on release; the bounce frame itself is skipped because vy
        flips there, not vx.
        """
        # Only the first deviation point is recorded.
        if self._dir_changes:
            return

        # Deviation events are only meaningful AFTER bounce — that's when
        # the ball can be deflected by bat/pad/stumps. Pre-bounce path
        # changes are dominated by release wobble and gravity curvature.
        if bounce_frame is None:
            return

        ordered = sorted(self._pts, key=lambda p: p[0])
        post_bounce = [p for p in ordered if p[0] >= bounce_frame]
        if len(post_bounce) < 4:
            return
        post_release = post_bounce

        # Use points strictly before the latest as "prediction context",
        # then compare predicted vx against actual vx into the latest point.
        *prior, latest = post_release
        if len(prior) < 3:
            return

        f_prev, x_prev, y_prev = prior[-1]
        f_now,  x_now,  y_now = latest
        df = max(f_now - f_prev, 1)
        vx_actual = (x_now - x_prev) / df
        vy_actual = (y_now - y_prev) / df

        # Predicted velocity from a linear fit on the last few prior points.
        window = prior[-5:] if len(prior) >= 5 else prior
        ts = np.array([p[0] for p in window], dtype=float)
        xs = np.array([p[1] for p in window], dtype=float)
        ys = np.array([p[2] for p in window], dtype=float)
        if len(np.unique(ts)) < 2:
            return
        slope_x, _ = np.polyfit(ts, xs, 1)
        slope_y, _ = np.polyfit(ts, ys, 1)
        vx_predicted = float(slope_x)
        vy_predicted = float(slope_y)

        # Skip frames immediately after bounce — vy reverses there and the
        # post-bounce fit needs a couple of points to stabilize.
        if abs(f_now - bounce_frame) <= 2:
            return

        speed_pred = (vx_predicted ** 2 + vy_predicted ** 2) ** 0.5
        speed_act  = (vx_actual    ** 2 + vy_actual    ** 2) ** 0.5
        if speed_pred < self._MIN_SPEED or speed_act < self._MIN_SPEED:
            return

        # Angle between predicted and actual velocity vectors.
        cos_theta = (vx_predicted * vx_actual + vy_predicted * vy_actual) \
                    / (speed_pred * speed_act)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = float(np.degrees(np.arccos(cos_theta)))

        if angle_deg >= self._DEVIATION_ANGLE_DEG:
            # Only the FIRST deviation point post-bounce is recorded —
            # subsequent bends (secondary contact, roll, etc.) are ignored.
            if not self._dir_changes:
                self._dir_changes.append((f_prev, x_prev, y_prev))
                print("\n[DEVIATION POINT]")
                print(f"  Frame      : {f_prev}  (confirmed at F{f_now})")
                print(f"  Coordinates: x={x_prev}, y={y_prev}")
                print(f"  Angle      : {angle_deg:.1f} deg")
                print(f"  v_predicted: ({vx_predicted:+.2f}, {vy_predicted:+.2f}) px/frame")
                print(f"  v_actual   : ({vx_actual:+.2f}, {vy_actual:+.2f}) px/frame\n")

    def draw_direction_change_markers(
        self,
        frame: np.ndarray,
        frame_num: int,
        size: Optional[int] = None,
    ) -> np.ndarray:
        """Overlay the cricket ball image at every detected direction-change point."""
        from utils import overlay_ball_image
        from config import BOUNCE_MARKER_SIZE
        if size is None:
            size = BOUNCE_MARKER_SIZE
        for f, x, y in self._dir_changes:
            if frame_num < f:
                continue
            overlay_ball_image(frame, x, y, size)
        return frame

    def _window_points(
        self, lo: int, hi: int
    ) -> list[tuple[int, int, int]]:
        return [(f, x, y) for (f, x, y) in self._pts if lo <= f <= hi]

    # ── DISABLED: interpolator-based dense-window builder ─────────────────
    # def _dense_window_points(
    #     self, lo: int, hi: int
    # ) -> list[tuple[int, int, int]]:
    #     """
    #     Return points covering [lo, hi] with detection gaps filled by
    #     interpolate_gap(). Gaps larger than INTERP_MAX_GAP_FRAMES are left
    #     unfilled (matches tracker.py's policy).
    #     """
    #     anchor_pad = max(INTERP_ANCHOR_FRAMES, 4)
    #     all_pts = sorted(self._pts, key=lambda p: p[0])
    #     if not all_pts:
    #         return []
    #
    #     in_window = [(f, x, y) for (f, x, y) in all_pts if lo <= f <= hi]
    #     if len(in_window) < 2:
    #         return in_window
    #
    #     merged: list[tuple[int, int, int]] = list(in_window)
    #     for i in range(len(in_window) - 1):
    #         f_pre, _, _   = in_window[i]
    #         f_post, _, _  = in_window[i + 1]
    #         gap = list(range(f_pre + 1, f_post))
    #         if not gap or len(gap) > INTERP_MAX_GAP_FRAMES:
    #             continue
    #
    #         pre_idx_global = next(j for j, p in enumerate(all_pts) if p[0] == f_pre)
    #         post_idx_global = next(j for j, p in enumerate(all_pts) if p[0] == f_post)
    #         pre_pts  = all_pts[max(0, pre_idx_global - anchor_pad + 1) : pre_idx_global + 1]
    #         post_pts = all_pts[post_idx_global : post_idx_global + anchor_pad]
    #         if len(pre_pts) < 2 or len(post_pts) < 2:
    #             continue
    #
    #         method = INTERP_METHOD if INTERP_METHOD in {"linear", "parabola", "bezier", "physics"} else "parabola"
    #         synth = interpolate_gap(
    #             pre_pts, post_pts, gap,
    #             method=method,
    #             gravity=GRAVITY_PX_PER_FRAME2,
    #             pitch_y=PITCH_Y,
    #         )
    #         for s in synth:
    #             merged.append((s["frame"], s["x"], s["y"]))
    #
    #     merged.sort(key=lambda p: p[0])
    #     return merged

    def _fit_curve(
        self,
        pts: list[tuple[int, int, int]],
        t_start: float,
        t_end:   float,
    ) -> list[tuple[int, int]]:
        if len(pts) < 2:
            return [(x, y) for (_, x, y) in pts]

        ts = np.array([p[0] for p in pts], dtype=float)
        xs = np.array([p[1] for p in pts], dtype=float)
        ys = np.array([p[2] for p in pts], dtype=float)

        unique_t = len(np.unique(ts))
        deg = 2 if unique_t >= 3 else (1 if unique_t == 2 else 0)
        if deg == 0:
            return [(int(xs[0]), int(ys[0]))]

        coeffs_x = np.polyfit(ts, xs, deg)
        coeffs_y = np.polyfit(ts, ys, deg)

        if t_end <= t_start:
            t_start, t_end = float(ts.min()), float(ts.max())

        t_samples = np.linspace(t_start, t_end, self._n_samples)
        x_samples = np.polyval(coeffs_x, t_samples)
        y_samples = np.polyval(coeffs_y, t_samples)

        return [(int(round(x)), int(round(y)))
                for x, y in zip(x_samples, y_samples)]

    def _draw_polyline(self, frame: np.ndarray, pts: list[tuple[int, int]]) -> None:
        """3-pass cylinder render through the given pixel points (already ordered)."""
        if len(pts) < 2:
            return
        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        # Pass 1 — outer shading (cylinder edge)
        cv2.polylines(frame, [pts_np], isClosed=False,
                      color=self._edge_color, thickness=self._edge_t,
                      lineType=cv2.LINE_AA)
        # Pass 2 — white core (cylinder body)
        cv2.polylines(frame, [pts_np], isClosed=False,
                      color=self._core_color, thickness=self._core_t,
                      lineType=cv2.LINE_AA)
        # Pass 3 — glossy highlight streak offset upward
        hl = [(x, y + self._hl_dy) for (x, y) in pts]
        hl_np = np.array(hl, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [hl_np], isClosed=False,
                      color=self._highlight_color, thickness=self._highlight_t,
                      lineType=cv2.LINE_AA)

    def draw(
        self,
        frame: np.ndarray,
        release_frame: Optional[int] = None,
        bounce_frame:  Optional[int] = None,
        impact_frame:  Optional[int] = None,
    ) -> np.ndarray:
        """
        Raw trajectory: connect every real ball detection in chronological order.
        release_frame / bounce_frame / impact_frame are accepted for API
        compatibility but no longer gate the rendering.
        """
        if len(self._pts) < 2:
            return frame

        ordered = sorted(self._pts, key=lambda p: p[0])
        pixel_pts = [(int(x), int(y)) for (_, x, y) in ordered]

        # Render onto an overlay, then alpha-blend.
        overlay = frame.copy()
        self._draw_polyline(overlay, pixel_pts)
        cv2.addWeighted(overlay, self._alpha, frame, 1.0 - self._alpha, 0, frame)
        return frame

    def draw_direction_arrow(
        self,
        frame: np.ndarray,
        release_frame: Optional[int] = None,
        bounce_frame:  Optional[int] = None,
        predict_frames: int = 12,
        min_fit_points: int = 4,
        color: tuple[int, int, int] = (0, 255, 255),
        thickness: int = 4,
    ) -> np.ndarray:
        """
        Predicted-path arrow using a physics-aware polynomial fit.

        Uses only points from the current phase (after the most recent
        bounce, or after release if no bounce yet) to avoid smearing the
        fit across the bounce discontinuity. Fits x(t) linearly and y(t)
        quadratically (gravity → parabolic vertical motion), then
        extrapolates `predict_frames` frames ahead as a curved polyline
        ending in an arrow head.
        """
        if len(self._pts) < 2:
            return frame

        ordered = sorted(self._pts, key=lambda p: p[0])

        # Phase gating: prefer post-bounce, else post-release, else all.
        phase_start = None
        if bounce_frame is not None:
            phase_start = bounce_frame
        elif release_frame is not None:
            phase_start = release_frame
        if phase_start is not None:
            phase_pts = [p for p in ordered if p[0] >= phase_start]
            if len(phase_pts) >= 2:
                ordered = phase_pts

        # Need enough points for a stable quadratic fit; fall back gracefully.
        pts = ordered[-max(min_fit_points * 2, 10):]
        if len(pts) < 2:
            return frame

        ts = np.array([p[0] for p in pts], dtype=float)
        xs = np.array([p[1] for p in pts], dtype=float)
        ys = np.array([p[2] for p in pts], dtype=float)
        unique_t = len(np.unique(ts))

        # x is near-linear in cricket side view; y is parabolic under gravity.
        deg_x = 1 if unique_t >= 2 else 0
        deg_y = 2 if unique_t >= 3 else (1 if unique_t == 2 else 0)
        if deg_y == 0:
            return frame

        cx = np.polyfit(ts, xs, deg_x)
        cy = np.polyfit(ts, ys, deg_y)

        t_last = float(ts.max())
        t_pred = np.linspace(t_last, t_last + predict_frames, 32)
        x_pred = np.polyval(cx, t_pred)
        y_pred = np.polyval(cy, t_pred)

        curve = [(int(round(x)), int(round(y)))
                 for x, y in zip(x_pred, y_pred)]
        if len(curve) < 2:
            return frame

        # ── Visualization disabled (logic above is still computed and
        #    available for downstream use). Uncomment to re-enable the
        #    predicted-direction arrow overlay.
        # pts_np = np.array(curve, dtype=np.int32).reshape(-1, 1, 2)
        # cv2.polylines(frame, [pts_np], isClosed=False,
        #               color=(0, 0, 0), thickness=thickness + 3,
        #               lineType=cv2.LINE_AA)
        # cv2.polylines(frame, [pts_np], isClosed=False,
        #               color=color, thickness=thickness,
        #               lineType=cv2.LINE_AA)
        #
        # p_end_prev = curve[-2]
        # p_end      = curve[-1]
        # cv2.arrowedLine(frame, p_end_prev, p_end,
        #                 (0, 0, 0), thickness + 3, cv2.LINE_AA, tipLength=1.2)
        # cv2.arrowedLine(frame, p_end_prev, p_end,
        #                 color, thickness, cv2.LINE_AA, tipLength=1.2)
        return frame

    def reset(self) -> None:
        self._pts.clear()
