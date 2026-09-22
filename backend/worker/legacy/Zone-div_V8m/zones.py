"""
PITCH ZONE OVERLAY
==================
Builds a world→pixel homography from the 6 calibration markers and renders
the 4 length-bands (yorker / full / good / short) onto a frame at 30 % alpha.

World frame (meters):
    origin  = bat_stump
    +y axis = from batter toward bowler (along pitch length)
    +x axis = perpendicular to pitch (right side of bat_stump)

Known dimensions:
    bat_center  ↔ bowl_center   = 19.68 m  (between popping-crease centers)
    bat_stump   ↔ bat_center    =  1.22 m  (stump sits 1.22 m behind crease)
    return-crease half-width    =  1.32 m  (where bat_L/R, bowl_L/R sit)
"""

import cv2
import numpy as np


# ── World coordinates of the 6 markers, in meters ────────────────────────────
HALF_WIDTH = 1.32       # return-crease distance from middle stump
BAT_CREASE_Y  = 1.22                 # popping crease is 1.22 m in front of stump
BOWL_CREASE_Y = BAT_CREASE_Y + 19.68  # = 20.90 m
BOWL_STUMP_Y  = BOWL_CREASE_Y + 1.22  # = 22.12 m

WORLD_COORDS = {
    "bat_stump":  (0.0,          0.0),
    "bat_L":      (-HALF_WIDTH,  BAT_CREASE_Y),
    "bat_R":      (+HALF_WIDTH,  BAT_CREASE_Y),
    "bowl_L":     (-HALF_WIDTH,  BOWL_CREASE_Y),
    "bowl_R":     (+HALF_WIDTH,  BOWL_CREASE_Y),
    "bowl_stump": (0.0,          BOWL_STUMP_Y),
}

# ── Zone definitions: (y_min_m, y_max_m, BGR_color, label) ──────────────────
ZONES = [
    (0.0,           2.0,           (0,   255, 255), "YORKER"),  # yellow
    (2.0,           6.0,           (230, 200, 100), "FULL"),    # light blue
    (6.0,           8.0,           (0,   255,  50), "GOOD"),    # parrot green
    (8.0,           BOWL_CREASE_Y, (0,     0, 255), "SHORT"),   # red
]


def build_homography(markers: dict) -> np.ndarray | None:
    """
    markers: dict {name -> (px, py)} containing the 6 calibration keys.
    Returns the 3×3 world→pixel homography, or None if not solvable.
    """
    needed = list(WORLD_COORDS.keys())
    if not all(k in markers for k in needed):
        return None

    src_world = np.array([WORLD_COORDS[k] for k in needed], dtype=np.float32)
    dst_pixel = np.array([markers[k]      for k in needed], dtype=np.float32)

    H, _ = cv2.findHomography(src_world, dst_pixel, method=0)
    return H


def world_to_pixel(H: np.ndarray, points_world: np.ndarray) -> np.ndarray:
    """points_world: (N,2) in meters. Returns (N,2) pixel coords."""
    pts = points_world.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def draw_zones(frame: np.ndarray, H: np.ndarray, alpha: float = 0.30,
               label: bool = True) -> np.ndarray:
    """Fill the 4 zone quads on `frame` at the given alpha."""
    if H is None:
        return frame

    overlay = frame.copy()
    for (y0, y1, color, _name) in ZONES:
        corners_w = np.array([
            [-HALF_WIDTH, y0],
            [+HALF_WIDTH, y0],
            [+HALF_WIDTH, y1],
            [-HALF_WIDTH, y1],
        ], dtype=np.float32)
        corners_px = world_to_pixel(H, corners_w).astype(np.int32)
        cv2.fillPoly(overlay, [corners_px], color)

    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    if label:
        for (y0, y1, color, name) in ZONES:
            # Anchor label at 60% of the half-width to the right of centreline
            # so it sits over the right side of each zone strip rather than
            # the middle of the pitch.
            mid_w = np.array(
                [[HALF_WIDTH * 0.60, (y0 + y1) / 2.0]], dtype=np.float32
            )
            mx, my = world_to_pixel(H, mid_w)[0].astype(int)
            cv2.putText(frame, name, (mx - 32, my + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)

    return frame
