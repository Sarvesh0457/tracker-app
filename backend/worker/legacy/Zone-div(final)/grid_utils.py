"""
grid_utils.py — Cricket Pitch Grid Utility Functions
=====================================================
Standalone importable module for cricket pitch grid overlay.

Usage:
    from grid_utils import compute_grid_corners, draw_grid, pixel_to_grid_cell, draw_grid_cell_highlight

    corners = compute_grid_corners(markers)
    frame   = draw_grid(frame, corners, alpha=0.6)
    cell    = pixel_to_grid_cell((x, y), corners)          # → (row, col) or None
    frame   = draw_grid_cell_highlight(frame, corners, row, col)  # highlight that cell

Dependencies: numpy, opencv-python, Python standard library only.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GRID GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

def compute_grid_corners(markers: dict) -> dict:
    """
    Compute the 4 perspective-correct grid corners from 6 calibration markers.

    markers = {
        'bat_L':     (x, y),   # batting crease / left return crease
        'bat_R':     (x, y),   # batting crease / right return crease
        'bowl_L':    (x, y),   # bowling crease / left return crease
        'bowl_R':    (x, y),   # bowling crease / right return crease
        'bat_stump': (x, y),   # base of middle stump, batter's end
        'bowl_stump':(x, y),   # base of middle stump, bowler's end
    }
    Returns {'TL': (x,y), 'TR': (x,y), 'BL': (x,y), 'BR': (x,y)}
    TL/TR = bowler's stump end, BL/BR = batter's stump end.
    """
    bat_L      = np.array(markers['bat_L'],     dtype=float)
    bat_R      = np.array(markers['bat_R'],     dtype=float)
    bowl_L     = np.array(markers['bowl_L'],    dtype=float)
    bowl_R     = np.array(markers['bowl_R'],    dtype=float)
    bat_stump  = np.array(markers['bat_stump'], dtype=float)
    bowl_stump = np.array(markers['bowl_stump'],dtype=float)

    # ── Pitch axis ────────────────────────────────────────────────────────
    # Mid-points of the two crease lines
    bat_mid  = (bat_L  + bat_R)  / 2.0
    bowl_mid = (bowl_L + bowl_R) / 2.0

    # Pitch axis unit vector (batter's end → bowler's end)
    axis = bowl_mid - bat_mid
    D    = float(np.linalg.norm(axis))
    if D < 1e-6:
        raise ValueError("bat_mid and bowl_mid are too close — check marker placement.")
    u = axis / D                             # unit along pitch length

    # ── Perpendicular direction derived from actual marker positions ──────
    # Use the bat_L → bat_R vector so "left" and "right" always match
    # what the user physically clicked, regardless of camera angle.
    lr_bat  = bat_R  - bat_L                # vector from left to right crease (bat end)
    lr_bowl = bowl_R - bowl_L               # vector from left to right crease (bowl end)

    # Average the two end-crease vectors for a stable direction estimate
    lr_avg = lr_bat + lr_bowl
    lr_norm = float(np.linalg.norm(lr_avg))
    if lr_norm < 1e-6:
        # Fallback: 90° CCW of pitch axis
        p = np.array([-u[1], u[0]], dtype=float)
    else:
        p = lr_avg / lr_norm   # unit vector from left → right across pitch

    # Half-widths at each crease (pure distance, sign-independent)
    hw_bat  = float(np.linalg.norm(bat_R  - bat_L))  / 2.0
    hw_bowl = float(np.linalg.norm(bowl_R - bowl_L)) / 2.0

    def get_corners(t):
        """Return (left_pt, right_pt) at parametric position t along pitch."""
        hw = hw_bat + (hw_bowl - hw_bat) * t
        centre = bat_mid + (bowl_mid - bat_mid) * t
        left  = centre - p * hw
        right = centre + p * hw
        return left, right

    # t-values for both stump bases
    # bat_stump is BEHIND the batting crease (t < 0)
    d_bat   = float(np.linalg.norm(bat_stump  - bat_mid))
    # bowl_stump is BEYOND the bowling crease (t > 1)
    d_bowl  = float(np.linalg.norm(bowl_stump - bowl_mid))

    t_bat_stump  = -d_bat  / D
    t_bowl_stump =  1.0 + d_bowl / D

    # Grid corners
    # TL, TR = bowl stump end  (top of grid image = far end)
    # BL, BR = bat  stump end  (bottom of grid image = near end)
    TL, TR = get_corners(t_bowl_stump)
    BL, BR = get_corners(t_bat_stump)

    # Lerp parameters along TL→BL (or TR→BR) where the user's clicked
    # popping creases actually sit. The grid extends from stump base to
    # stump base, so the creases are NOT at fixed 4/85 and 81/85 unless the
    # clicked stump-to-crease distance matches that exact ratio. Derive
    # from the real geometry instead so the dashed lines hug the inputs.
    span = t_bowl_stump - t_bat_stump
    t_lerp_bowl_crease = (t_bowl_stump - 1.0) / span   # near TL (bowler end)
    t_lerp_bat_crease  =  t_bowl_stump        / span   # near BL (batter end)

    return {
        'TL': tuple(TL.astype(int)),
        'TR': tuple(TR.astype(int)),
        'BL': tuple(BL.astype(int)),
        'BR': tuple(BR.astype(int)),
        't_crease_bowl': float(t_lerp_bowl_crease),
        't_crease_bat':  float(t_lerp_bat_crease),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRID RENDERING
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(p1, p2, t):
    """Linear interpolation between two 2-D points."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    return p1 + (p2 - p1) * t


def _draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_len=8, gap_len=6):
    """Draw a dashed line between pt1 and pt2 on img (in-place)."""
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    total = np.linalg.norm(pt2 - pt1)
    if total < 1:
        return
    direction = (pt2 - pt1) / total
    step = dash_len + gap_len
    pos = 0.0
    while pos < total:
        start = pt1 + direction * pos
        end   = pt1 + direction * min(pos + dash_len, total)
        cv2.line(img,
                 (int(round(start[0])), int(round(start[1]))),
                 (int(round(end[0])),   int(round(end[1]))),
                 color, thickness, cv2.LINE_AA)
        pos += step


def draw_grid(frame, corners: dict, alpha: float = 0.6,
              grid_color=(0, 255, 255), highlight: bool = True) -> np.ndarray:
    """
    Draw a perspective-correct 84-row × 18-column pitch grid overlay.

    corners = {'TL': (x,y), 'TR': (x,y), 'BL': (x,y), 'BR': (x,y)}
        TL/TR → bowler's stump end (top of grid)
        BL/BR → batter's stump end (bottom of grid)

    alpha       : overlay opacity (0 = invisible, 1 = fully opaque)
    grid_color  : BGR colour for the regular grid lines
    highlight   : if True, draw special lines (stumps/crease/wide guides)

    Returns a new frame with the composited grid.
    """
    TL = np.array(corners['TL'], dtype=float)
    TR = np.array(corners['TR'], dtype=float)
    BL = np.array(corners['BL'], dtype=float)
    BR = np.array(corners['BR'], dtype=float)

    overlay = frame.copy()

    # ── Regular grid lines ────────────────────────────────────────────────────

    # Horizontal lines: 84 divisions → indices i = 0..85 (85 gaps)
    # i=0  → TL..TR (bowl-stump edge, drawn as stump line below)
    # i=85 → BL..BR (bat-stump edge, drawn as stump line below)
    for i in range(1, 85):          # 84 interior lines
        t = i / 85.0
        p1 = _lerp(TL, BL, t)
        p2 = _lerp(TR, BR, t)
        cv2.line(overlay,
                 (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))),
                 grid_color, 1, cv2.LINE_AA)

    # Vertical lines: 18 divisions → j = 1..18 (17 interior lines)
    for j in range(1, 18):          # 17 interior lines
        t = j / 18.0
        p1 = _lerp(TL, TR, t)
        p2 = _lerp(BL, BR, t)
        cv2.line(overlay,
                 (int(round(p1[0])), int(round(p1[1]))),
                 (int(round(p2[0])), int(round(p2[1]))),
                 grid_color, 1, cv2.LINE_AA)

    # ── Highlighted special lines ─────────────────────────────────────────────
    if highlight:
        # Stump lines — top and bottom edges (bright green, thick)
        stump_color = (0, 255, 0)
        for edge_TL, edge_TR in [(TL, TR), (BL, BR)]:
            cv2.line(overlay,
                     (int(round(edge_TL[0])), int(round(edge_TL[1]))),
                     (int(round(edge_TR[0])), int(round(edge_TR[1]))),
                     stump_color, 2, cv2.LINE_AA)

        # Popping crease lines — use the t-values derived from the user's
        # clicked crease points so the dashed line lands exactly on the
        # input markers, not on a hardcoded 4/85 ratio.
        crease_color = (0, 165, 255)   # orange in BGR
        t_creases = (
            corners.get('t_crease_bowl', 4  / 85.0),
            corners.get('t_crease_bat',  81 / 85.0),
        )
        for t_crease in t_creases:
            p1 = _lerp(TL, BL, t_crease)
            p2 = _lerp(TR, BR, t_crease)
            _draw_dashed_line(overlay,
                              (int(round(p1[0])), int(round(p1[1]))),
                              (int(round(p2[0])), int(round(p2[1]))),
                              crease_color, thickness=2)

        # Wide guide lines — column 3 and 16 (magenta, dashed)
        wide_color = (255, 0, 255)    # magenta in BGR
        for t_wide in (3 / 18.0, 15 / 18.0):
            p1 = _lerp(TL, TR, t_wide)
            p2 = _lerp(BL, BR, t_wide)
            _draw_dashed_line(overlay,
                              (int(round(p1[0])), int(round(p1[1]))),
                              (int(round(p2[0])), int(round(p2[1]))),
                              wide_color, thickness=2)

    # ── Alpha blend ───────────────────────────────────────────────────────────
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


# ─────────────────────────────────────────────────────────────────────────────
# INVERSE MAPPING  (pixel → grid cell)
# ─────────────────────────────────────────────────────────────────────────────

def _inverse_bilinear(p, TL, TR, BL, BR, max_iter=50, tol=1e-6):
    """
    Solve for (u, v) in [0,1]² such that bilinear(u,v) == p,
    where the bilinear patch is defined by corners TL, TR, BL, BR.

    Bilinear mapping:
        Q(u, v) = (1-u)(1-v)*TL + u(1-v)*TR + (1-u)v*BL + u*v*BR
        u=0 → left edge (TL→BL), u=1 → right edge (TR→BR)
        v=0 → top  edge (TL→TR), v=1 → bottom edge (BL→BR)

    Returns (u, v) or None if not converged / outside [0,1]².
    """
    TL = np.array(TL, dtype=float)
    TR = np.array(TR, dtype=float)
    BL = np.array(BL, dtype=float)
    BR = np.array(BR, dtype=float)
    p  = np.array(p,  dtype=float)

    u, v = 0.5, 0.5   # initial guess

    for _ in range(max_iter):
        # Current position
        q = (1-u)*(1-v)*TL + u*(1-v)*TR + (1-u)*v*BL + u*v*BR

        # Jacobian
        dq_du = -(1-v)*TL + (1-v)*TR - v*BL + v*BR
        dq_dv = -(1-u)*TL - u*TR    + (1-u)*BL + u*BR

        r = p - q
        J = np.array([[dq_du[0], dq_dv[0]],
                      [dq_du[1], dq_dv[1]]])
        try:
            delta = np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return None

        u += delta[0]
        v += delta[1]

        if np.linalg.norm(delta) < tol:
            break

    # Check bounds (with small tolerance)
    eps = 1e-4
    if -eps <= u <= 1 + eps and -eps <= v <= 1 + eps:
        u = float(np.clip(u, 0, 1))
        v = float(np.clip(v, 0, 1))
        return u, v
    return None


def pixel_to_grid_cell(pixel: tuple, corners: dict):
    """
    Convert a pixel coordinate to a (row, col) grid cell.

    corners = {'TL': (x,y), 'TR': (x,y), 'BL': (x,y), 'BR': (x,y)}
        TL/TR → bowler's end (top),  BL/BR → batter's end (bottom)

    Grid has 84 rows (0-indexed) and 18 columns (0-indexed).
    Returns (row, col) integers or None if outside the grid.

    Mapping convention:
        u ∈ [0,1] → column direction (left → right, TL side → TR side)
        v ∈ [0,1] → row direction    (top  → bottom, TL/TR → BL/BR)
        row = int(v * 84)  clamped to [0, 83]
        col = int(u * 18)  clamped to [0, 17]
    """
    result = _inverse_bilinear(
        pixel,
        corners['TL'], corners['TR'],
        corners['BL'], corners['BR'],
    )
    if result is None:
        return None
    u, v = result
    row = int(min(v * 84, 83))
    col = int(min(u * 18, 17))
    return row, col


def pixel_to_grid_uv(pixel: tuple, corners: dict):
    """
    Like pixel_to_grid_cell() but returns the raw continuous (u, v) floats
    in [0, 1]² for precise sub-cell positioning on schematics.

    u = 0 → left edge,  u = 1 → right edge
    v = 0 → bowl-stump end (top), v = 1 → bat-stump end (bottom)

    Returns (u, v) or None if outside the grid.
    """
    return _inverse_bilinear(
        pixel,
        corners['TL'], corners['TR'],
        corners['BL'], corners['BR'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CELL HIGHLIGHT
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid_cell_highlight(frame, corners: dict, row: int, col: int,
                             color=(0, 255, 255), alpha: float = 0.45,
                             label: bool = True) -> np.ndarray:
    """
    Shade the (row, col) grid cell with a filled semi-transparent polygon.

    corners = {'TL': (x,y), 'TR': (x,y), 'BL': (x,y), 'BR': (x,y)}
    row  : 0-indexed row    (0 = bowl-stump end, 83 = bat-stump end)
    col  : 0-indexed column (0 = left edge,      17 = right edge)
    color: BGR fill colour for the cell
    alpha: opacity of the cell fill (0 = invisible, 1 = opaque)
    label: if True, print "R{row} C{col}" text near the cell centre

    Returns a new frame with the highlighted cell composited in.
    """
    TL = np.array(corners['TL'], dtype=float)
    TR = np.array(corners['TR'], dtype=float)
    BL = np.array(corners['BL'], dtype=float)
    BR = np.array(corners['BR'], dtype=float)

    # 84 rows (v: 0→1 top→bottom), 18 cols (u: 0→1 left→right)
    # Row band:  v in [row/84, (row+1)/84]   (along TL→BL / TR→BR)
    # Col band:  u in [col/18, (col+1)/18]   (along TL→TR / BL→BR)

    v0, v1 = row       / 84.0, (row + 1) / 84.0
    u0, u1 = col       / 18.0, (col + 1) / 18.0

    def _q(u, v):
        """Bilinear point on the trapezoid patch."""
        pt = (1 - u) * (1 - v) * TL + u * (1 - v) * TR \
           + (1 - u) *       v  * BL + u *       v  * BR
        return (int(round(pt[0])), int(round(pt[1])))

    # Four corners of the cell (in drawing order: TL → TR → BR → BL)
    cell_pts = np.array([
        _q(u0, v0),   # top-left  of cell
        _q(u1, v0),   # top-right of cell
        _q(u1, v1),   # bot-right of cell
        _q(u0, v1),   # bot-left  of cell
    ], dtype=np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [cell_pts], color)
    result = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    if label:
        # Centre of cell
        cx = int(round(sum(p[0] for p in cell_pts) / 4))
        cy = int(round(sum(p[1] for p in cell_pts) / 4))
        txt = f"R{row} C{col}"
        cv2.putText(result, txt,
                    (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (0, 0, 0), 3, cv2.LINE_AA)          # dark outline
        cv2.putText(result, txt,
                    (cx - 20, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 255), 1, cv2.LINE_AA)    # white text

    return result
