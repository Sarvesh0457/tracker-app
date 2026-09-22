"""
TRACKER MODULE — Ball Path Visualization

Key features:
  • History stored as list of TrailEntry objects (frame, x, y, interpolated)
    so we always know *which* frame each point belongs to.
  • Occlusion gap filling via the interpolator is DISABLED — only real
    detections populate _history. The `interpolated` field on TrailEntry is
    kept for compatibility but every entry is always real.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from kalman_filter import KalmanBallFilter

from config import (
    TRAIL_LENGTH,
    TRAIL_THICKNESS,
    TRAIL_GRADIENT,
    TRAIL_ALPHA, TRAIL_TAIL_FADE,
    BALL_GLOW_RADIUS, BALL_GLOW_INNER, BALL_GLOW_COLOR, BALL_GLOW_ALPHA,
    INTERP_METHOD, INTERP_MAX_GAP_FRAMES, INTERP_ANCHOR_FRAMES,
    GRAVITY_PX_PER_FRAME2, PITCH_Y,
    BOUNCE_MIN_DESCENT_VY, BOUNCE_MIN_ASCENT_VY, BOUNCE_MARKER_SIZE,
)
# Interpolator usage disabled — gap-filling is now off everywhere.
# from interpolator import interpolate_gap

try:
    from scipy.interpolate import splprep, splev
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrailEntry:
    frame:        int
    x:            int
    y:            int
    interpolated: bool = False   # True → synthesised by interpolator
    y_bottom:     int = 0        # ball bbox bottom y (= y for non-real entries)

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def as_anchor(self) -> tuple[int, int, int]:
        return (self.frame, self.x, self.y)


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers — Hawk-Eye style tube trail
# ─────────────────────────────────────────────────────────────────────────────

def _spline_smooth(pts: list[tuple[int, int]], density: int = 15) -> list[tuple[int, int]]:
    """
    Fit a smooth spline through pts and return a dense list of points.
    density = how many output points per input point.
    Falls back to raw pts if scipy unavailable or < 4 points.
    """
    if not _SCIPY_OK or len(pts) < 4:
        return pts
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    s = max(1.0, len(pts) * 0.8)   # tighter fit — stays close to actual detections
    try:
        tck, _ = splprep([xs, ys], s=s, k=min(3, len(pts) - 1))
        u_fine = np.linspace(0, 1, max(len(pts) * density, 200))
        xf, yf = splev(u_fine, tck)
        return list(zip(xf.astype(int), yf.astype(int)))
    except Exception:
        return pts


def _gradient_color(t: float, stops: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """
    Sample a multi-stop gradient at position t ∈ [0, 1].
    `stops` is a list of BGR tuples evenly spaced from t=0 to t=1.
    """
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    idx = t * n
    lo = int(idx)
    hi = min(lo + 1, n)
    frac = idx - lo
    c0, c1 = stops[lo], stops[hi]
    return (
        int(c0[0] + frac * (c1[0] - c0[0])),
        int(c0[1] + frac * (c1[1] - c0[1])),
        int(c0[2] + frac * (c1[2] - c0[2])),
    )


def _lerp_color(c0, c1, t: float) -> tuple[int, int, int]:
    return (
        int(c0[0] + t * (c1[0] - c0[0])),
        int(c0[1] + t * (c1[1] - c0[1])),
        int(c0[2] + t * (c1[2] - c0[2])),
    )


def _brighten(color: tuple[int, int, int], factor: float = 1.5) -> tuple[int, int, int]:
    """Return a brighter version of color (for tube highlight)."""
    return (
        min(255, int(color[0] * factor)),
        min(255, int(color[1] * factor)),
        min(255, int(color[2] * factor)),
    )


def _darken(color: tuple[int, int, int], factor: float = 0.5) -> tuple[int, int, int]:
    """Return a darker version of color (for tube shadow)."""
    return (
        int(color[0] * factor),
        int(color[1] * factor),
        int(color[2] * factor),
    )


def _draw_hawkeye_tube(
    frame: np.ndarray,
    curve_pts: list[tuple[int, int]],
    gradient_stops: list[tuple[int, int, int]],
    thickness: int,
    trail_alpha: float = TRAIL_ALPHA,
    tail_fade: float = TRAIL_TAIL_FADE,
) -> None:
    """
    Draw a translucent 3D tube trail along curve_pts.

    Per-segment opacity fade: oldest (t=0) uses tail_fade * trail_alpha,
    newest (t=1) uses full trail_alpha — creates a natural fading tail.
    Per-segment thickness taper: thinner at tail, full at head.
    """
    m = len(curve_pts)
    if m < 2:
        return

    # Pass 1: outer glow per segment
    for i in range(1, m):
        t = i / (m - 1) if m > 1 else 1.0
        seg_alpha = trail_alpha * (tail_fade + (1.0 - tail_fade) * t)
        glow_alpha = seg_alpha * 0.35
        color = _gradient_color(t, gradient_stops)
        p0, p1 = curve_pts[i - 1], curve_pts[i]
        overlay = frame.copy()
        cv2.line(overlay, p0, p1, _darken(color, 0.4), thickness + 12, cv2.LINE_AA)
        cv2.addWeighted(overlay, glow_alpha, frame, 1.0 - glow_alpha, 0, frame)

    # Pass 2: shadow + core + highlight per segment
    for i in range(1, m):
        t = i / (m - 1) if m > 1 else 1.0
        seg_alpha = trail_alpha * (tail_fade + (1.0 - tail_fade) * t)
        seg_thickness = thickness
        color = _gradient_color(t, gradient_stops)
        p0, p1 = curve_pts[i - 1], curve_pts[i]
        overlay2 = frame.copy()
        cv2.line(overlay2, p0, p1, _darken(color, 0.6), seg_thickness + 4, cv2.LINE_AA)
        cv2.line(overlay2, p0, p1, color, seg_thickness, cv2.LINE_AA)
        cv2.line(overlay2, p0, p1, _brighten(color, 1.3),
                 max(1, seg_thickness // 3), cv2.LINE_AA)
        cv2.addWeighted(overlay2, seg_alpha, frame, 1.0 - seg_alpha, 0, frame)


def _draw_ball_glow(
    frame: np.ndarray,
    cx: int,
    cy: int,
    frame_num: int,
) -> None:
    """Pulsing glow dot at the current ball position."""
    pulse = 1.0 + 0.25 * np.sin(frame_num * 0.5)
    outer_r = int(BALL_GLOW_RADIUS * pulse)
    overlay = frame.copy()
    cv2.circle(overlay, (cx, cy), outer_r + 4,
               _darken(BALL_GLOW_COLOR, 0.5), -1, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), outer_r,
               BALL_GLOW_COLOR, 2, cv2.LINE_AA)
    cv2.circle(overlay, (cx, cy), BALL_GLOW_INNER,
               (255, 255, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, BALL_GLOW_ALPHA, frame, 1.0 - BALL_GLOW_ALPHA, 0, frame)


# ─────────────────────────────────────────────────────────────────────────────
# BALL TRACKER (trajectory + occlusion interpolation)
# ─────────────────────────────────────────────────────────────────────────────

class BounceDetector:
    """
    Tracks ball position across frames, fills occlusion gaps with the chosen
    interpolation method, and draws the trajectory with visual distinction
    between real and synthesised positions.
    """

    # Maximum history kept (frames); keep more than TRAIL_LENGTH so
    # we have enough anchor points across a long gap.
    _MAX_HISTORY = TRAIL_LENGTH * 4

    def __init__(self, trail_length: int = TRAIL_LENGTH, calib_markers: Optional[dict] = None):
        self._trail_len = trail_length
        self.calib_markers = calib_markers

        # Full ordered history of TrailEntry objects (real + interpolated)
        self._history: list[TrailEntry] = []

        # Kalman-smoothed history — real detections only
        self._smooth_history: list[TrailEntry] = []
        self._kalman = KalmanBallFilter()

        # Frame number of last real detection (for gap tracking)
        self._last_det_frame: Optional[int] = None
        # How many consecutive frames has the ball been missing?
        self._gap_len: int = 0

        # Bounce point: (frame, x, y) once detected, else None.
        self.bounce_point: Optional[tuple[int, int, int]] = None

    # ── Internal history management ──────────────────────────────────────────

    def _append(self, entry: TrailEntry) -> None:
        self._history.append(entry)
        # Trim to avoid unbounded growth
        if len(self._history) > self._MAX_HISTORY:
            self._history = self._history[-self._MAX_HISTORY:]

    def _get_anchors_before(self, n: int) -> list[tuple[int, int, int]]:
        """Return up to n most recent REAL anchor points."""
        real = [e for e in self._history if not e.interpolated]
        return [e.as_anchor() for e in real[-n:]]

    # ── Gap filling ───────────────────────────────────────────────────────────

    def _fill_gap(
        self,
        gap_frames: list[int],
        post_pts: list[tuple[int, int, int]],
    ) -> None:
        """
        Interpolator usage disabled — this is now a no-op. Gaps in the
        ball history are left empty; only real detections populate _history.
        """
        return
        # ── DISABLED: interpolator-based gap fill ─────────────────────────────
        # pre_pts = self._get_anchors_before(INTERP_ANCHOR_FRAMES)
        # if len(pre_pts) < 2 or len(post_pts) < 2:
        #     return
        #
        # synth = interpolate_gap(
        #     pre_pts, post_pts, gap_frames,
        #     method  = INTERP_METHOD,
        #     gravity = GRAVITY_PX_PER_FRAME2,
        #     pitch_y = PITCH_Y,
        # )
        #
        # for pt in synth:
        #     entry = TrailEntry(
        #         frame        = pt["frame"],
        #         x            = pt["x"],
        #         y            = pt["y"],
        #         interpolated = True,
        #     )
        #     self._append(entry)
        #
        # # Re-sort history by frame so draw() always sees chronological data
        # self._history.sort(key=lambda e: e.frame)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list,
        frame_num:  int,
        release_frame: Optional[int] = None,
    ) -> None:
        """Call once per frame to update ball tracking history."""
        balls = [d for d in detections if d["class_name"] == "ball"]

        # Always advance Kalman (even on missed frames)
        self._kalman.predict()

        if balls:
            best = max(balls, key=lambda x: x["conf"])
            cx = (best["x1"] + best["x2"]) // 2
            cy = (best["y1"] + best["y2"]) // 2

            # Kalman update → smoothed position added to smooth_history
            sx, sy, _, _ = self._kalman.update(float(cx), float(cy))
            self._smooth_history.append(
                TrailEntry(frame=frame_num, x=int(round(sx)), y=int(round(sy)))
            )
            if len(self._smooth_history) > self._MAX_HISTORY:
                self._smooth_history = self._smooth_history[-self._MAX_HISTORY:]

            # ── Check if we're reappearing after a gap ────────────────────────
            if self._gap_len > 0 and self._last_det_frame is not None:
                gap_len = self._gap_len
                if 1 <= gap_len <= INTERP_MAX_GAP_FRAMES:
                    gap_frames = list(range(self._last_det_frame + 1, frame_num))
                    post_pts = [(frame_num, cx, cy)]
                    self._pending_fill = {
                        "gap_frames": gap_frames,
                        "post_pts":   post_pts,
                    }
                else:
                    self._pending_fill = None
            else:
                # Execute a pending fill now that we have two post-anchor frames
                if hasattr(self, "_pending_fill") and self._pending_fill is not None:
                    pending = self._pending_fill
                    pending["post_pts"].append((frame_num, cx, cy))
                    if len(pending["post_pts"]) >= 2:
                        self._fill_gap(
                            pending["gap_frames"],
                            pending["post_pts"],
                        )
                        self._pending_fill = None

            # Append real detection AFTER potential gap fill so sort is stable
            self._append(TrailEntry(
                frame=frame_num, x=cx, y=cy,
                interpolated=False, y_bottom=int(best["y2"]),
            ))
            self._last_det_frame = frame_num
            self._gap_len = 0

            # Bounce analysis on the real trajectory — gated on release.
            # Bounce can never occur before the ball has been released.
            self._update_bounce(release_frame=release_frame)

        else:
            # No detection this frame — record gap
            self._gap_len += 1
            self._kalman.handle_gap()

    # ── Bounce detection ──────────────────────────────────────────────────────

    def _update_bounce(self, release_frame: Optional[int] = None) -> None:
        """
        Detect bounce (vy sign flip: descending → ascending) across the FULL
        post-release trajectory. Called once after each new real detection is
        appended.

        Previous version only inspected the last 3 detections, so once the
        ball had been tracked past the bounce for a few frames the sign-flip
        window slid out of view and the bounce was permanently missed. This
        version scans forward from release for the earliest valid sign flip,
        skipping triplets whose per-side frame gap is too wide to give a
        reliable vy (interpolator is disabled, so we don't trust wide gaps).
        """
        if self.bounce_point is not None:
            return

        # Gate: release must be confirmed first
        if release_frame is None:
            return

        # Only consider real detections strictly AFTER release
        real = [e for e in self._history
                if not e.interpolated and e.frame > release_frame]
        if len(real) < 3:
            return

        for i in range(len(real) - 2):
            a, b, c = real[i], real[i + 1], real[i + 2]
            df_pre  = max(1, b.frame - a.frame)
            df_post = max(1, c.frame - b.frame)
            # Skip triplets with wide gaps — a raw vy across a 6-frame gap is
            # dominated by whatever happened in the middle, not by the local
            # motion around b.
            if df_pre > 5 or df_post > 5:
                continue
            vy_pre  = (b.y - a.y) / df_pre   # +ve = descending (image coords)
            vy_post = (c.y - b.y) / df_post

            if vy_pre >= BOUNCE_MIN_DESCENT_VY and vy_post <= -BOUNCE_MIN_ASCENT_VY:
                by = b.y_bottom if b.y_bottom else b.y

                # Constraint: bounce must be between left return crease and right return crease
                if self.calib_markers is not None:
                    bat_l = self.calib_markers.get("bat_L")
                    bowl_l = self.calib_markers.get("bowl_L")
                    bat_r = self.calib_markers.get("bat_R")
                    bowl_r = self.calib_markers.get("bowl_R")
                    
                    if bat_l and bowl_l and bat_r and bowl_r:
                        l_x1, l_y1 = bat_l
                        l_x2, l_y2 = bowl_l
                        r_x1, r_y1 = bat_r
                        r_x2, r_y2 = bowl_r
                        
                        # Constraint: bounce must be on the bowler's side of the batting stumps crease
                        # If bowler is at the bottom (y2 > y1), bounce must be below batting crease (by > y1)
                        # If bowler is at the top (y2 < y1), bounce must be above batting crease (by < y1)
                        if l_y2 > l_y1:
                            if by <= l_y1:
                                continue
                        else:
                            if by >= l_y1:
                                continue

                        # Interpolate left and right return creases at y coordinate of bounce candidate (by)
                        if abs(l_y2 - l_y1) > 1e-5 and abs(r_y2 - r_y1) > 1e-5:
                            left_bound_x = l_x1 + (l_x2 - l_x1) * (by - l_y1) / (l_y2 - l_y1)
                            right_bound_x = r_x1 + (r_x2 - r_x1) * (by - r_y1) / (r_y2 - r_y1)
                            
                            # Ensure left_bound_x is less than right_bound_x
                            if left_bound_x > right_bound_x:
                                left_bound_x, right_bound_x = right_bound_x, left_bound_x
                                
                            if not (left_bound_x <= b.x <= right_bound_x):
                                continue

                self.bounce_point = (b.frame, b.x, by)
                print(f"[BOUNCE] Detected at frame {b.frame} pos=({b.x},{by})")
                return

    # ── Drawing ───────────────────────────────────────────────────────────────

    def get_trail_points(
        self, release_frame: Optional[int] = None
    ) -> list[tuple[int, int]]:
        """Return the drawn trail as an (x, y) polyline for external snapping."""
        if release_frame is None:
            entries = list(self._history)
        else:
            entries = [e for e in self._history if e.frame >= release_frame]
        return [(int(e.x), int(e.y)) for e in entries]

    def draw(
        self,
        frame: np.ndarray,
        frame_num: int,
        release_frame: Optional[int] = None,
    ) -> np.ndarray:
        """
        Trajectory overlay disabled — but the bounce marker is drawn here once
        detected, using the same cricket_ball.png used at the release point.
        """
        from utils import overlay_ball_image
        import cv2

        if self.bounce_point is not None:
            bf, bx, by = self.bounce_point
            if frame_num >= bf:
                # Soft elliptical shadow under the bounce marker — drawn as
                # several stacked translucent ellipses for a feathered look.
                shadow_cy = by + int(BOUNCE_MARKER_SIZE * 0.85) - 8
                base_rx = max(2, int(BOUNCE_MARKER_SIZE * 0.42))
                base_ry = max(2, int(BOUNCE_MARKER_SIZE * 0.168))
                for k, alpha in ((1.6, 0.10), (1.3, 0.18), (1.0, 0.30)):
                    rx = max(2, int(base_rx * k))
                    ry = max(2, int(base_ry * k))
                    overlay = frame.copy()
                    cv2.ellipse(
                        overlay, (int(bx), shadow_cy),
                        (rx, ry), 0, 0, 360,
                        (0, 0, 0), -1, lineType=cv2.LINE_AA,
                    )
                    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

                overlay_ball_image(frame, bx, by, BOUNCE_MARKER_SIZE)

        return frame

    def reset(self) -> None:
        self._history.clear()
        self._smooth_history.clear()
        self._kalman.reset()
        self._last_det_frame = None
        self._gap_len = 0
        self._pending_fill = None
        self.bounce_point = None


# ── BALL PATH TRACKER (backwards-compatible shim) ─────────────────────────────

class BallPathTracker:
    """Simplified tracker kept for backwards compatibility."""

    def __init__(self, trail_length: int = TRAIL_LENGTH):
        self._detector = BounceDetector(trail_length=trail_length)

    def update(self, detections: list):
        # frame_num not tracked here — pass a dummy 0
        self._detector.update(detections, frame_num=0)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        return self._detector.draw(frame, frame_num=0)

    def reset(self):
        self._detector.reset()