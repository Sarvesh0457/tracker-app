"""Inverse homography + zone classification for bounce points.

Given the world→pixel homography from zones.build_homography, invert it to map
a bounce pixel back into world meters and classify it onto the ZONES bands.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from zones import HALF_WIDTH, ZONES, build_homography  # noqa: E402


def pixel_to_world(H: np.ndarray, x: float, y: float) -> Optional[Tuple[float, float]]:
    """Map (x, y) px → (X, Y) meters in the pitch world frame, via H⁻¹."""
    if H is None:
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    pts = np.array([[[float(x), float(y)]]], dtype=np.float32)
    world = cv2.perspectiveTransform(pts, H_inv).reshape(-1)
    return float(world[0]), float(world[1])


def classify_bounce(
    markers: dict, bounce_px: Tuple[float, float]
) -> Tuple[Optional[str], Optional[float]]:
    """Return (zone_name | None, length_m | None) for a bounce pixel.

    `markers` is the same 6-key calibration dict used by zones.build_homography.
    Returns (None, None) if the point falls outside the marked pitch area
    (|x| > HALF_WIDTH or y outside [0, BOWL_CREASE_Y)).
    """
    H = build_homography(markers)
    if H is None:
        return None, None
    world = pixel_to_world(H, bounce_px[0], bounce_px[1])
    if world is None:
        return None, None
    wx, wy = world
    if abs(wx) > HALF_WIDTH:
        return None, None
    # ZONES is the spec contract: [(y0, y1, color, name), ...] sorted.
    for y0, y1, _color, name in ZONES:
        if y0 <= wy < y1:
            return name, round(wy, 3)
    return None, round(wy, 3) if 0.0 <= wy else None
