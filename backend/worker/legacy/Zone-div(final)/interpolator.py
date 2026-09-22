"""
INTERPOLATOR MODULE — Occlusion-aware ball position synthesis.

Given the last N real detections before an occlusion and the first N after,
this module synthesises plausible ball positions for every missing frame using
one of four methods:

    "linear"       — straight line between entry and exit (fastest, least accurate)
    "parabola"     — unblended one-sided gravity arc (default — preserves sharp
                     bounce corners; no spline rounding)
    "bezier"       — cubic Bézier using velocity-derived control points
    "physics"      — forward + backward physics extrapolation with gravity (blended)

Special case: if the ball bounced inside the gap (descending → ascending),
the gap is split at the estimated bounce point and each half is filled with a
pure one-sided parabola so the two halves meet at a sharp V-shaped corner
rather than a rounded curve.

Each synthesised point is a dict:
    {"frame": int, "x": int, "y": int, "interpolated": True}
"""

from __future__ import annotations
import numpy as np
from typing import Sequence


# ── Type alias ────────────────────────────────────────────────────────────────
# A "real" anchor point: (frame_number, x, y)
AnchorPt = tuple[int, int, int]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def interpolate_gap(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
    method:     str   = "parabola",
    gravity:    float = 0.35,
    pitch_y:    int | None = None,
) -> list[dict]:
    """
    Synthesise ball positions for every frame in gap_frames.

    Parameters
    ----------
    pre_pts     : Anchor points BEFORE the gap, ordered oldest → newest.
    post_pts    : Anchor points AFTER the gap, ordered oldest → newest.
    gap_frames  : Sorted list of frame indices that need to be filled.
    method      : Interpolation method to use.
    gravity     : Downward acceleration in pixels per frame².
    pitch_y     : Y pixel coordinate of the pitch surface (for bounce detection).
                  If None, estimated from the apex of the pre/post trajectory.

    Returns
    -------
    List of dicts, one per gap frame, sorted by frame number.
    """
    if not gap_frames or not pre_pts or not post_pts:
        return []

    # Estimate pitch_y if not provided
    if pitch_y is None:
        pitch_y = _estimate_pitch_y(pre_pts, post_pts)

    # Check for bounce inside the gap
    bounce_in_gap, bounce_frame, bounce_x, bounce_y = _detect_bounce_in_gap(
        pre_pts, post_pts, gap_frames, pitch_y, gravity
    )

    if bounce_in_gap:
        return _interpolate_with_bounce(
            pre_pts, post_pts, gap_frames,
            bounce_frame, bounce_x, bounce_y,
            gravity
        )

    return _dispatch(pre_pts, post_pts, gap_frames, method, gravity)


# ─────────────────────────────────────────────────────────────────────────────
# BOUNCE-IN-GAP DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _velocity(pts: Sequence[AnchorPt], from_end: bool = False) -> tuple[float, float]:
    """
    Estimate velocity (vx, vy) px/frame from the first or last segment of pts.
    from_end=True → use the last two points (entry velocity into gap).
    from_end=False → use the first two points (exit velocity out of gap).
    """
    if len(pts) < 2:
        return 0.0, 0.0
    if from_end:
        f0, x0, y0 = pts[-2]
        f1, x1, y1 = pts[-1]
    else:
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[1]
    dt = f1 - f0
    if dt == 0:
        return 0.0, 0.0
    return (x1 - x0) / dt, (y1 - y0) / dt


def _detect_bounce_in_gap(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
    pitch_y:    int,
    gravity:    float,
) -> tuple[bool, int, int, int]:
    """
    Returns (bounce_detected, bounce_frame, bounce_x, bounce_y).

    Bounce is suspected when:
      - The ball was descending at the end of pre_pts  (vy_pre  > 0, downward in pixel space)
      - The ball was ascending at the start of post_pts (vy_post < 0, upward in pixel space)
    """
    _, vy_pre  = _velocity(pre_pts,  from_end=True)
    _, vy_post = _velocity(post_pts, from_end=False)

    if not (vy_pre > 0 and vy_post < 0):
        return False, 0, 0, 0

    # Estimate time of bounce from entry velocity + gravity
    # Ball enters gap at frame gap_frames[0]-1 with velocity vy_pre (downward)
    # Solve: pitch_y = y_entry + vy_pre*t + 0.5*gravity*t²
    f_entry, x_entry, y_entry = pre_pts[-1]
    vx_pre, _ = _velocity(pre_pts, from_end=True)

    a = 0.5 * gravity
    b = vy_pre
    c = y_entry - pitch_y

    discriminant = b * b - 4 * a * c
    if discriminant < 0 or a == 0:
        # Can't solve quadratic — fall back to midpoint
        bounce_frame = (gap_frames[0] + gap_frames[-1]) // 2
        bounce_x = int(x_entry + vx_pre * (bounce_frame - f_entry))
        bounce_y = pitch_y
        return True, bounce_frame, bounce_x, bounce_y

    # Two roots — take the smaller positive one
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b + sqrt_d) / (2 * a)
    t2 = (-b - sqrt_d) / (2 * a)
    candidates = [t for t in [t1, t2] if t > 0]
    if not candidates:
        bounce_frame = (gap_frames[0] + gap_frames[-1]) // 2
        bounce_x = int(x_entry + vx_pre * (bounce_frame - f_entry))
        bounce_y = pitch_y
        return True, bounce_frame, bounce_x, bounce_y

    t_bounce = min(candidates)
    bounce_frame = int(round(f_entry + t_bounce))
    # Clamp to gap
    bounce_frame = max(gap_frames[0], min(gap_frames[-1], bounce_frame))
    bounce_x = int(round(x_entry + vx_pre * t_bounce))
    bounce_y = pitch_y
    return True, bounce_frame, bounce_x, bounce_y


def _estimate_pitch_y(
    pre_pts:  Sequence[AnchorPt],
    post_pts: Sequence[AnchorPt],
) -> int:
    """
    Auto-estimate pitch Y as the maximum Y observed across all anchor points
    (lowest position in pixel space = closest to ground).
    """
    all_y = [p[2] for p in pre_pts] + [p[2] for p in post_pts]
    return int(max(all_y)) if all_y else 500


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH + PER-METHOD IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

def _dispatch(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
    method:     str,
    gravity:    float,
) -> list[dict]:
    if method == "linear":
        return _linear(pre_pts, post_pts, gap_frames)
    elif method == "bezier":
        return _bezier(pre_pts, post_pts, gap_frames)
    elif method == "physics":
        return _physics(pre_pts, post_pts, gap_frames, gravity)
    else:  # "parabola" is default
        return _parabola(pre_pts, post_pts, gap_frames, gravity)


def _make_result(frame: int, x: float, y: float) -> dict:
    return {"frame": frame, "x": int(round(x)), "y": int(round(y)), "interpolated": True}


# ── 1. Linear ─────────────────────────────────────────────────────────────────

def _linear(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
) -> list[dict]:
    """Straight line between last pre-gap point and first post-gap point."""
    f0, x0, y0 = pre_pts[-1]
    f1, x1, y1 = post_pts[0]
    results = []
    for f in gap_frames:
        t = (f - f0) / (f1 - f0) if f1 != f0 else 0.5
        t = np.clip(t, 0.0, 1.0)
        results.append(_make_result(f, x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return results


# ── 2. Parabola (unblended one-sided gravity arc) ─────────────────────────────

def _parabola(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
    gravity:    float,
) -> list[dict]:
    """
    Single unblended parabola driven by pre-gap entry velocity and gravity.

    Unlike _physics, this does NOT blend with a backward extrapolation from
    post_pts — so when the bounce-split path uses this on each half, the two
    halves meet at the bounce point with opposite vertical velocities,
    producing a sharp V-shaped corner instead of a rounded curve.

    Used by the public API as the default for non-bounce gaps and by
    _interpolate_with_bounce for each half of a split bounce gap.
    """
    f0, x0, y0 = pre_pts[-1]
    vx, vy = _velocity(pre_pts, from_end=True)

    results = []
    for f in gap_frames:
        dt = f - f0
        x = x0 + vx * dt
        y = y0 + vy * dt + 0.5 * gravity * dt ** 2
        results.append(_make_result(f, x, y))
    return results


# ── 3. Bézier ─────────────────────────────────────────────────────────────────

def _bezier(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
) -> list[dict]:
    """
    Cubic Bézier:
      P0 = last pre-gap point
      P3 = first post-gap point
      P1 = P0 + vx_pre * tension  (entry tangent)
      P2 = P3 - vx_post * tension (exit tangent)
    where tension ≈ 1/3 of the gap duration.
    """
    f0, x0, y0 = pre_pts[-1]
    f3, x3, y3 = post_pts[0]
    gap_dur = max(f3 - f0, 1)
    tension = gap_dur / 3.0

    vx_pre,  vy_pre  = _velocity(pre_pts,  from_end=True)
    vx_post, vy_post = _velocity(post_pts, from_end=False)

    # Control points
    p1x, p1y = x0 + vx_pre  * tension, y0 + vy_pre  * tension
    p2x, p2y = x3 - vx_post * tension, y3 - vy_post * tension

    results = []
    for f in gap_frames:
        t = (f - f0) / gap_dur
        t = np.clip(t, 0.0, 1.0)
        # De Casteljau
        bx = (1-t)**3*x0 + 3*(1-t)**2*t*p1x + 3*(1-t)*t**2*p2x + t**3*x3
        by = (1-t)**3*y0 + 3*(1-t)**2*t*p1y + 3*(1-t)*t**2*p2y + t**3*y3
        results.append(_make_result(f, bx, by))
    return results


# ── 4. Physics ────────────────────────────────────────────────────────────────

def _physics(
    pre_pts:    Sequence[AnchorPt],
    post_pts:   Sequence[AnchorPt],
    gap_frames: Sequence[int],
    gravity:    float,
) -> list[dict]:
    """
    Forward-extrapolate from pre_pts entry velocity + gravity.
    Backward-extrapolate from post_pts exit velocity + gravity.
    Blend the two linearly (distance-weighted) so each end is fully trusted
    and the middle is a 50/50 average.
    """
    f0, x0, y0 = pre_pts[-1]
    f3, x3, y3 = post_pts[0]
    vx_pre,  vy_pre  = _velocity(pre_pts,  from_end=True)
    vx_post, vy_post = _velocity(post_pts, from_end=False)

    results = []
    n = len(gap_frames)
    for i, f in enumerate(gap_frames):
        dt_fwd = f  - f0
        dt_bwd = f3 - f

        # Forward prediction
        fwd_x = x0 + vx_pre  * dt_fwd
        fwd_y = y0 + vy_pre  * dt_fwd + 0.5 * gravity * dt_fwd ** 2

        # Backward prediction (reverse time → reverse gravity sign)
        bwd_x = x3 - vx_post * dt_bwd
        bwd_y = y3 - vy_post * dt_bwd + 0.5 * gravity * dt_bwd ** 2

        # Blend: weight forward by how far we are from the end, and vice versa
        w_fwd = dt_bwd / (dt_fwd + dt_bwd + 1e-9)
        w_bwd = dt_fwd / (dt_fwd + dt_bwd + 1e-9)

        bx = w_fwd * fwd_x + w_bwd * bwd_x
        by = w_fwd * fwd_y + w_bwd * bwd_y
        results.append(_make_result(f, bx, by))
    return results


# ── Bounce-in-gap split interpolation ─────────────────────────────────────────

def _interpolate_with_bounce(
    pre_pts:      Sequence[AnchorPt],
    post_pts:     Sequence[AnchorPt],
    gap_frames:   Sequence[int],
    bounce_frame: int,
    bounce_x:     int,
    bounce_y:     int,
    gravity:      float,
) -> list[dict]:
    """
    Split gap at bounce_frame.
    Descent: pre_pts → bounce point  (forward parabola, vy_pre downward)
    Ascent:  bounce point → post_pts (forward parabola, vy reversed via post_pts)

    Each half uses an unblended one-sided parabola, so the two halves meet at
    the bounce with opposite vertical velocities — producing a sharp V corner
    rather than a rounded curve.
    """
    gap_list = list(gap_frames)
    descent_frames = [f for f in gap_list if f <  bounce_frame]
    ascent_frames  = [f for f in gap_list if f >  bounce_frame]

    results = []

    # Mark bounce point itself
    if bounce_frame in gap_list:
        results.append(_make_result(bounce_frame, bounce_x, bounce_y))

    # Descent: forward parabola from pre_pts (its own vy_pre, gravity down)
    if descent_frames:
        results += _parabola(pre_pts, post_pts, descent_frames, gravity)

    # Ascent: forward parabola starting at the bounce point with the ascent
    # velocity inferred from the first post-gap detection. We synthesise a
    # two-point pre_pts so _parabola's _velocity() yields the right vx, vy.
    if ascent_frames:
        f_post, x_post, y_post = post_pts[0]
        df = max(f_post - bounce_frame, 1)
        vx_asc = (x_post - bounce_x) / df
        vy_asc = (y_post - bounce_y) / df - 0.5 * gravity * df  # invert the gravity term
        # Two anchors one frame apart so _velocity picks up (vx_asc, vy_asc)
        synth_prev: AnchorPt = (bounce_frame - 1,
                                int(round(bounce_x - vx_asc)),
                                int(round(bounce_y - vy_asc)))
        synth_anchor: AnchorPt = (bounce_frame, bounce_x, bounce_y)
        results += _parabola([synth_prev, synth_anchor], post_pts, ascent_frames, gravity)

    # Sort by frame
    results.sort(key=lambda d: d["frame"])

    # Tag the bounce point so caller can log it
    for r in results:
        if r["frame"] == bounce_frame:
            r["is_bounce"] = True

    return results
