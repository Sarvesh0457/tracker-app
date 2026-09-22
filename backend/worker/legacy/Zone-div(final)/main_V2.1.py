
"""
IMPROVED CRICKET DETECTION — ORCHESTRATION SCRIPT
Only 1 BALL and MAX 2 BATS per frame with strict geometric/temporal filtering.

Pitch Grid Integration:
  Before processing begins, the first video frame is shown and the user is
  asked to place 6 calibration markers (or load a previously saved set).
  The resulting perspective-correct 18×84 grid is composited onto every
  output frame.  Each detected ball's grid cell (row, col) is highlighted
  on the frame and stored in the CSV via pixel_to_grid_cell().
"""

import cv2
import json
import time
import datetime
import numpy as np
from pathlib import Path

# Import configuration
from config import VIDEO_PATH, OUTPUT_DIR

# Import pitch-grid utilities
from grid_utils import (compute_grid_corners, draw_grid,
                        pixel_to_grid_cell, draw_grid_cell_highlight)

# Import sub-modulesll
from detector import load_model, detect_objects
from visualizer import draw_detections, draw_release_to_bounce_parabola
from logger import DetectionLogger
from tracker import BounceDetector
from release_detector import ReleaseDetector
from filters import reset_filters
from trajectory import Trajectory
from zones import build_homography, draw_zones
from utils import overlay_ball_image

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION CONFIG
# ─────────────────────────────────────────────────────────────────────────────
CALIB_FILE  = Path(__file__).parent / "calibration.json"
GRID_ALPHA  = 0.55
GRID_COLOR  = (0, 255, 255)          # cyan
FONT        = cv2.FONT_HERSHEY_SIMPLEX

MARKER_KEYS  = ["bat_L", "bat_R", "bowl_L", "bowl_R", "bat_stump", "bowl_stump"]
MARKER_NAMES = {
    "bat_L":      "Bat-L  (batting crease, left return crease)",
    "bat_R":      "Bat-R  (batting crease, right return crease)",
    "bowl_L":     "Bowl-L (bowling crease, left return crease)",
    "bowl_R":     "Bowl-R (bowling crease, right return crease)",
    "bat_stump":  "Bat Stump  (middle stump base, batter's end)",
    "bowl_stump": "Bowl Stump (middle stump base, bowler's end)",
}
MARKER_COLORS = {
    "bat_L":      (0,   200, 255),
    "bat_R":      (0,   200, 255),
    "bowl_L":     (255, 140,   0),
    "bowl_R":     (255, 140,   0),
    "bat_stump":  (0,   255,   0),
    "bowl_stump": (0,   255,   0),
}
KEY_TO_MARKER = {str(i + 1): k for i, k in enumerate(MARKER_KEYS)}

# Create unique output filename with timestamp
timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
base_name    = f"filtered_detection_output_{timestamp}"
counter      = 1
VIDEO_OUTPUT = OUTPUT_DIR / f"{base_name}.mp4"
while VIDEO_OUTPUT.exists():
    VIDEO_OUTPUT = OUTPUT_DIR / f"{base_name}_{counter}.mp4"
    counter += 1


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION STATE
# ─────────────────────────────────────────────────────────────────────────────
class _CalibState:
    def __init__(self):
        self.markers       = {}
        self.active_marker = MARKER_KEYS[0]


def _all_placed(state: _CalibState) -> bool:
    return all(k in state.markers for k in MARKER_KEYS)


def _mouse_callback(event, x, y, flags, state: _CalibState):
    if event != cv2.EVENT_LBUTTONDOWN or state.active_marker is None:
        return
    key = state.active_marker
    state.markers[key] = (x, y)
    remaining = [k for k in MARKER_KEYS if k not in state.markers]
    state.active_marker = remaining[0] if remaining else None


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION UI DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def _draw_calib_ui(frame, state: _CalibState) -> None:
    """Annotate frame in-place with markers, HUD and instructions."""
    h, w = frame.shape[:2]

    # Semi-transparent top panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, "PITCH GRID CALIBRATION — place 6 markers",
                (10, 24), FONT, 0.62, (0, 220, 255), 1, cv2.LINE_AA)

    # Next marker prompt
    if state.active_marker:
        cv2.putText(frame,
                    f"Click: {MARKER_NAMES[state.active_marker]}",
                    (10, 50), FONT, 0.50, (0, 255, 120), 1, cv2.LINE_AA)
    elif _all_placed(state):
        cv2.putText(frame,
                    "All markers placed!  Press ENTER to confirm, R to reset.",
                    (10, 50), FONT, 0.50, (0, 255, 0), 1, cv2.LINE_AA)

    # Marker status grid (2 rows × 3 cols)
    col_w = w // 3
    for i, mk in enumerate(MARKER_KEYS):
        placed = mk in state.markers
        col    = (i % 3) * col_w + 10
        row    = 72 + (i // 3) * 22
        sym    = "[v]" if placed else "[ ]"
        color  = (100, 255, 100) if placed else (100, 100, 255)
        if mk == state.active_marker:
            color = (0, 220, 255)
        cv2.putText(frame, f"{i+1}:{sym} {mk}", (col, row),
                    FONT, 0.40, color, 1, cv2.LINE_AA)

    # Bottom hint bar
    hint = ("Keys: 1-6=reselect  R=reset  S=save  L=load  "
            "ENTER=confirm & start  ESC=quit without grid")
    cv2.putText(frame, hint, (10, h - 8), FONT, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

    # Draw each placed marker
    for mk, (mx, my) in state.markers.items():
        mc = MARKER_COLORS.get(mk, (255, 255, 255))
        cv2.drawMarker(frame, (mx, my), mc,
                       cv2.MARKER_CROSS, markerSize=20, thickness=2,
                       line_type=cv2.LINE_AA)
        cv2.circle(frame, (mx, my), 7, mc, 2, cv2.LINE_AA)
        cv2.putText(frame, mk, (mx + 10, my - 6), FONT, 0.42, mc, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP DIALOG — load saved calibration or start fresh
# ─────────────────────────────────────────────────────────────────────────────
def _startup_dialog() -> str:
    """
    If calibration.json exists, ask the user via an OpenCV window:
      L → load saved markers
      N → place new markers
    Returns 'load' or 'new'.
    """
    if not CALIB_FILE.exists():
        return 'new'

    with open(CALIB_FILE) as f:
        saved = json.load(f)

    import numpy as _np
    dw, dh = 520, 240
    panel  = _np.zeros((dh, dw, 3), dtype='uint8')
    panel[:] = (30, 30, 30)

    lines = [
        ("Saved calibration found!", (0, 220, 255), 0.62),
        (str(CALIB_FILE),            (140, 140, 140), 0.36),
        ("", None, 0.36),
    ]
    for mk in MARKER_KEYS:
        coord = saved.get(mk)
        txt   = f"  {mk}: {tuple(coord)}" if coord else f"  {mk}: MISSING"
        col   = (100, 255, 100) if coord else (80, 80, 220)
        lines.append((txt, col, 0.38))
    lines.append(("", None, 0.36))
    lines.append(("Press  L = load saved    N = new calibration", (0, 200, 255), 0.46))

    y = 26
    for txt, col, sc in lines:
        if txt and col:
            cv2.putText(panel, txt, (14, y), FONT, sc, col, 1, cv2.LINE_AA)
        y += max(int(sc * 38), 16)

    WIN = "Calibration — Load or New?"
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WIN, panel)
    choice = None
    while choice is None:
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('l'), ord('L')):
            choice = 'load'
        elif k in (ord('n'), ord('N')):
            choice = 'new'
        elif k in (27, ord('q')):
            cv2.destroyAllWindows()
            raise SystemExit(0)
    cv2.destroyWindow(WIN)
    return choice


# ─────────────────────────────────────────────────────────────────────────────
# RUN INTERACTIVE CALIBRATION ON FIRST FRAME
# ─────────────────────────────────────────────────────────────────────────────
def run_calibration(first_frame):
    """
    Show first_frame in an interactive OpenCV window and collect 6 markers.

    Returns (corners, markers): corners is the grid corners dict (or None on
    skip) and markers is the raw {'bat_L': (x,y), ..., 'bowl_stump': (x,y)}
    dict from the calibration UI (or None on skip).
    """
    choice = _startup_dialog()

    state = _CalibState()

    if choice == 'load':
        with open(CALIB_FILE) as f:
            data = json.load(f)
        state.markers = {k: tuple(v) for k, v in data.items() if k in MARKER_KEYS}
        state.active_marker = None
        print("[CALIB] Loaded saved calibration markers.")

    WIN = "PITCH CALIBRATION — Place Markers on First Frame"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse_callback, state)

    print("\n[CALIB] Calibration window open.")
    print("  Place the 6 markers in order shown on screen.")
    print("  Keys: 1-6=reselect  R=reset  S=save  L=load  ENTER=confirm  ESC=skip grid\n")

    corners = None
    while True:
        display = first_frame.copy()
        _draw_calib_ui(display, state)

        # Live grid preview once all 6 markers are placed
        if _all_placed(state):
            try:
                preview_corners = compute_grid_corners(state.markers)
                display = draw_grid(display, preview_corners,
                                    alpha=GRID_ALPHA, grid_color=GRID_COLOR,
                                    highlight=True)
            except Exception:
                pass

        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:                   # ESC → skip grid
            print("[CALIB] Skipped — processing without grid overlay.")
            break
        elif key in (13, 10):           # ENTER → confirm
            if _all_placed(state):
                try:
                    corners = compute_grid_corners(state.markers)
                    with open(CALIB_FILE, "w") as f:
                        json.dump({k: list(v) for k, v in state.markers.items()}, f, indent=2)
                    print(f"[CALIB] Confirmed. Calibration saved to {CALIB_FILE}")
                except Exception as e:
                    print(f"[CALIB] Error computing grid: {e}")
                    corners = None
            else:
                print("[CALIB] Not all markers placed yet — place all 6 first.")
                continue
            break
        elif key == ord('r'):
            state.markers       = {}
            state.active_marker = MARKER_KEYS[0]
            print("[CALIB] Markers reset.")
        elif key == ord('s'):
            if _all_placed(state):
                with open(CALIB_FILE, "w") as f:
                    json.dump({k: list(v) for k, v in state.markers.items()}, f, indent=2)
                print(f"[CALIB] Saved to {CALIB_FILE}")
        elif key == ord('l'):
            if CALIB_FILE.exists():
                with open(CALIB_FILE) as f:
                    data = json.load(f)
                state.markers = {k: tuple(v) for k, v in data.items() if k in MARKER_KEYS}
                state.active_marker = None
                print("[CALIB] Loaded saved markers.")
        elif chr(key) in KEY_TO_MARKER:
            mk = KEY_TO_MARKER[chr(key)]
            state.active_marker = mk
            print(f"[CALIB] Ready to reposition: {mk}")

    cv2.destroyWindow(WIN)
    markers = dict(state.markers) if state.markers else None
    return corners, markers


# ─────────────────────────────────────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _draw_progressive_overlays(
    frame,
    frame_count: int,
    detections: list,
    release_detector: ReleaseDetector,
    bounce_detector: BounceDetector,
    trajectory: Trajectory,
    draw_parabolas: bool = False,
    rel_to_bounce_progress: float = 1.0,
    bounce_to_dev_progress: float = 1.0,
    rel_to_bounce_pts=None,
    rel_to_bounce_frame_range=None,
) -> "any":
    """
    Render the overlay stack for a single frame.

    Always: the 3 markers (release / bounce / deviation), each gated by
    frame_count (pass a large frame_count to force all visible).
    Only when draw_parabolas=True: the release→bounce and bounce→deviation
    parabolas — these form the trajectory overlay for segment B. The two
    `*_progress` floats (0.0..1.0) animate each arc; pass 1.0 to draw fully.
    """
    release_frame = (release_detector.release.release_frame
                     if release_detector.release else None)

    if detections:
        frame = draw_detections(frame, detections)

    # Bounce → deviation parabola (segment B only)
    if draw_parabolas and bounce_detector.bounce_point is not None \
            and trajectory._dir_changes and bounce_to_dev_progress > 0.0:
        _bf, _bx, _by = bounce_detector.bounce_point
        _df, _dx, _dy = trajectory._dir_changes[0]
        frame = draw_release_to_bounce_parabola(
            frame, (_bx, _by), (_dx, _dy),
            alpha=0.70, arc_ratio=0.0, thickness=8,
            progress=bounce_to_dev_progress,
        )

    # Deviation marker
    frame = trajectory.draw_direction_change_markers(frame, frame_count)

    # Bounce marker
    frame = bounce_detector.draw(frame, frame_count, release_frame=release_frame)

    # Release → bounce parabola (segment B only) — fitted to actual ball
    # positions between release and bounce when available.
    if draw_parabolas and release_detector.release is not None \
            and bounce_detector.bounce_point is not None \
            and rel_to_bounce_progress > 0.0:
        _bf, _bx, _by = bounce_detector.bounce_point
        frame = draw_release_to_bounce_parabola(
            frame,
            (release_detector.release.x, release_detector.release.y),
            (_bx, _by),
            alpha=0.40,
            progress=rel_to_bounce_progress,
            trajectory_points=rel_to_bounce_pts,
            frame_range=rel_to_bounce_frame_range,
        )

    # Release marker
    frame = release_detector.draw(frame, frame_count)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENT C — frozen-frame trajectory animation helpers
# ─────────────────────────────────────────────────────────────────────────────
def _draw_segment_banner(frame, text: str) -> None:
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.7, 2)
    x1, y1 = w - tw - 24, 10
    x2, y2 = w - 6, 24 + th
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x1 + 8, y2 - 8),
                FONT, 0.7, (0, 220, 255), 2, cv2.LINE_AA)


def _draw_bounce_marker_at(frame, pt):
    if pt is None:
        return frame
    from config import BOUNCE_MARKER_SIZE
    bx, by = pt
    shadow_cy = by + int(BOUNCE_MARKER_SIZE * 0.85) - 8
    base_rx = max(2, int(BOUNCE_MARKER_SIZE * 0.42))
    base_ry = max(2, int(BOUNCE_MARKER_SIZE * 0.168))
    for k, alpha in ((1.6, 0.10), (1.3, 0.18), (1.0, 0.30)):
        rx = max(2, int(base_rx * k))
        ry = max(2, int(base_ry * k))
        overlay = frame.copy()
        cv2.ellipse(overlay, (int(bx), shadow_cy), (rx, ry), 0, 0, 360,
                    (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    overlay_ball_image(frame, bx, by, BOUNCE_MARKER_SIZE)
    return frame


def _fit_constrained_quadratic_k(release_pt, bounce_pt,
                                 trajectory_points, frame_range):
    """
    Same constrained quadratic least-squares fit used inside
    visualizer.draw_release_to_bounce_parabola, but returns (kx, ky)
    so callers can also evaluate the ball position at any t.

    Model: x(t) = (1-t)*x1 + t*x2 + kx * t*(1-t)
           y(t) = (1-t)*y1 + t*y2 + ky * t*(1-t)
    """
    if not trajectory_points or len(trajectory_points) < 1 or frame_range is None:
        return 0.0, 0.0
    f_lo, f_hi = float(frame_range[0]), float(frame_range[1])
    if f_hi <= f_lo:
        return 0.0, 0.0
    x1, y1 = float(release_pt[0]), float(release_pt[1])
    x2, y2 = float(bounce_pt[0]),  float(bounce_pt[1])
    ts_p = np.array([(p[0] - f_lo) / (f_hi - f_lo)
                     for p in trajectory_points], dtype=float)
    xs_p = np.array([p[1] for p in trajectory_points], dtype=float)
    ys_p = np.array([p[2] for p in trajectory_points], dtype=float)
    a = ts_p * (1.0 - ts_p)
    denom = float(np.sum(a * a))
    if denom <= 1e-9:
        return 0.0, 0.0
    bx = xs_p - ((1.0 - ts_p) * x1 + ts_p * x2)
    by = ys_p - ((1.0 - ts_p) * y1 + ts_p * y2)
    return float(np.sum(a * bx) / denom), float(np.sum(a * by) / denom)


def _point_on_constrained_quadratic(p1, p2, kx, ky, t):
    """Position along the fitted release→bounce quadratic at parameter t∈[0,1]."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = t * (1.0 - t)
    x = (1.0 - t) * x1 + t * x2 + kx * a
    y = (1.0 - t) * y1 + t * y2 + ky * a
    return x, y


def _run_segment_c_freeze_animation(out, video_path, fps,
                                    deviation_frame,
                                    release_pt, bounce_pt, deviation_pt,
                                    rel_to_bounce_pts, rel_to_bounce_frame_range,
                                    markers):
    """
    SEGMENT C — frozen frame at deviation_frame with the zone overlay baked in.
    Uses the SAME parabola logic as Segment B:
      • Phase 1: release→bounce constrained-quadratic arc grows via rb_prog
                 (0 → 1). A cricket ball rides the arc.
      • Phase 2: bounce→deviation straight arc grows via bd_prog (0 → 1).
                 Bounce marker pops as soon as the ball reaches the bounce
                 point (start of phase 2). A ball rides the arc.
      • Hold:    both arcs fully drawn + bounce marker only.
    No release or deviation marker is drawn.
    """
    if release_pt is None or bounce_pt is None or deviation_pt is None:
        print("[SEG C] Missing release/bounce/deviation — skipping.")
        return
    if deviation_frame is None:
        print("[SEG C] Missing deviation frame — skipping.")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[SEG C] Could not re-open video — skipping.")
        return
    target_frame = max(0, deviation_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frozen = cap.read()
    cap.release()
    if not ret:
        print(f"[SEG C] Could not read frame {target_frame} — skipping.")
        return

    # ── Bake zones onto the frozen backdrop ─────────────────────────────
    base = frozen.copy()
    H_world = build_homography(markers) if markers else None
    if H_world is not None:
        base = draw_zones(base, H_world, alpha=0.30, label=True)

    # ── Fit constrained quadratic so we can place the ball on the arc ──
    kx, ky = _fit_constrained_quadratic_k(
        release_pt, bounce_pt,
        rel_to_bounce_pts, rel_to_bounce_frame_range,
    )

    # ── Animation duration roughly matches Segment B's 0.5× pacing ─────
    n_phase1 = max(8, int(round(fps * 1.5)))   # release → bounce
    n_phase2 = max(8, int(round(fps * 1.2)))   # bounce  → deviation
    n_hold   = max(8, int(round(fps * 1.5)))   # final settled view

    banner = "SEGMENT C - zones + animated parabolas"

    # ── Phase 1: release → bounce arc grows; ball rides the arc ────────
    for i in range(n_phase1):
        rb_prog = (i + 1) / n_phase1
        frame = base.copy()
        frame = draw_release_to_bounce_parabola(
            frame, release_pt, bounce_pt,
            alpha=0.40, progress=rb_prog,
            trajectory_points=rel_to_bounce_pts,
            frame_range=rel_to_bounce_frame_range,
        )
        bx, by = _point_on_constrained_quadratic(
            release_pt, bounce_pt, kx, ky, rb_prog
        )
        overlay_ball_image(frame, int(bx) + 2, int(by), 14)
        _draw_segment_banner(frame, banner)
        out.write(frame)

    # ── Phase 2: bounce → deviation arc grows; bounce marker appears ───
    for i in range(n_phase2):
        bd_prog = (i + 1) / n_phase2
        frame = base.copy()
        # Full release→bounce arc held behind
        frame = draw_release_to_bounce_parabola(
            frame, release_pt, bounce_pt,
            alpha=0.40, progress=1.0,
            trajectory_points=rel_to_bounce_pts,
            frame_range=rel_to_bounce_frame_range,
        )
        # Bounce marker (only marker drawn in Segment C)
        _draw_bounce_marker_at(frame, bounce_pt)
        # Growing bounce→deviation straight arc
        frame = draw_release_to_bounce_parabola(
            frame, bounce_pt, deviation_pt,
            alpha=0.70, arc_ratio=0.0, thickness=8,
            progress=bd_prog,
        )
        # Ball along the straight bounce→deviation segment
        bx = (1.0 - bd_prog) * bounce_pt[0] + bd_prog * deviation_pt[0]
        by = (1.0 - bd_prog) * bounce_pt[1] + bd_prog * deviation_pt[1]
        overlay_ball_image(frame, int(bx) + 2, int(by), 14)
        _draw_segment_banner(frame, banner)
        out.write(frame)

    # ── Hold: both arcs fully drawn + bounce marker only ───────────────
    final = base.copy()
    final = draw_release_to_bounce_parabola(
        final, release_pt, bounce_pt,
        alpha=0.40, progress=1.0,
        trajectory_points=rel_to_bounce_pts,
        frame_range=rel_to_bounce_frame_range,
    )
    final = draw_release_to_bounce_parabola(
        final, bounce_pt, deviation_pt,
        alpha=0.70, arc_ratio=0.0, thickness=8, progress=1.0,
    )
    _draw_bounce_marker_at(final, bounce_pt)
    _draw_segment_banner(final, banner)
    for _ in range(n_hold):
        out.write(final.copy())

    print(f"[SEG C] Animated {n_phase1 + n_phase2 + n_hold} frames on frozen "
          f"frame {target_frame} (kx={kx:.2f}, ky={ky:.2f}).")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def process_video():
    start_time = time.time()
    reset_filters()
    session = load_model()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps")
    print(f"[INFO] Total frames: {total_frames}\n")

    # ── Extract first frame for calibration ───────────────────────────────
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame for calibration.")

    # ── Interactive pitch-grid calibration ────────────────────────────────
    print("[INFO] Opening calibration window…")
    grid_corners, calib_markers = run_calibration(first_frame)

    if grid_corners is None:
        print("[INFO] No grid calibration — continuing without grid overlay.")
    else:
        print(f"[INFO] Grid corners: {grid_corners}")


    # ──────────────────────────────────────────────────────────────────────
    # PASS 1 — ANALYSIS: detect across the whole video, cache detections,
    # and let the trackers find release / bounce / deviation frames.
    # ──────────────────────────────────────────────────────────────────────
    print("\n[INFO] PASS 1 — analyzing full video to locate key frames…\n")

    reset_filters()
    bounce_detector  = BounceDetector()
    release_detector = ReleaseDetector()
    trajectory       = Trajectory()

    detections_by_frame: dict[int, list] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    f = 0
    ball_count = 0
    bat_count  = 0
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        dets = detect_objects(session, frm, f)
        detections_by_frame[f] = dets

        if any(d["class_name"] == "ball" for d in dets):
            ball_count += 1
        bat_count += sum(1 for d in dets if d["class_name"] == "bat")

        release_detector.update(dets, f)
        rel_f = (release_detector.release.release_frame
                 if release_detector.release else None)
        bounce_detector.update(dets, f, release_frame=rel_f)
        bf = (bounce_detector.bounce_point[0]
              if bounce_detector.bounce_point else None)
        trajectory.update(dets, f, release_frame=rel_f, bounce_frame=bf)

        if f % 100 == 0 and total_frames > 0:
            print(f"  [analysis] {(f/total_frames)*100:.1f}%")
        f += 1

    total = f

    if release_detector.release is None:
        raise RuntimeError("Release point not detected — cannot trim video.")
    if bounce_detector.bounce_point is None:
        raise RuntimeError("Bounce point not detected — cannot trim video.")
    if not trajectory._dir_changes:
        raise RuntimeError("Deviation point not detected — cannot trim video.")

    release_f = release_detector.release.release_frame
    bounce_f  = bounce_detector.bounce_point[0]
    dev_f     = trajectory._dir_changes[0][0]

    seg_a_start = 0
    seg_a_end   = min(dev_f + 10, total - 1)
    seg_b_start = max(0, release_f - 7)
    seg_b_end   = seg_a_end

    print(f"\n[INFO] Release : F{release_f}")
    print(f"[INFO] Bounce  : F{bounce_f}")
    print(f"[INFO] Deviate : F{dev_f}")
    print(f"[INFO] SEG A  -> [{seg_a_start} .. {seg_a_end}]  (normal speed)")
    print(f"[INFO] SEG B  -> [{seg_b_start} .. {seg_b_end}]  (0.5x, with trajectory)\n")

    # Snapshot the fully-analyzed state for pre-seeding segment B.
    full_release       = release_detector.release
    full_bounce        = bounce_detector.bounce_point
    full_dev_changes   = list(trajectory._dir_changes)
    full_traj_pts      = [p for p in trajectory._pts if p[0] <= seg_b_end]

    # ──────────────────────────────────────────────────────────────────────
    # PASS 2 — RENDER SEGMENT A (live, progressive, no trajectory line)
    # ──────────────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(str(VIDEO_OUTPUT), fourcc, fps, (width, height))
    logger = DetectionLogger(output_dir=OUTPUT_DIR, video_fps=fps)

    print("[INFO] PASS 2 — writing segment A (release → deviation+10)…")
    reset_filters()
    bounce_detector  = BounceDetector()
    release_detector = ReleaseDetector()
    trajectory       = Trajectory()

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for fnum in range(0, seg_a_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections_by_frame.get(fnum, [])

        # Evolve tracker state in order so markers appear progressively.
        release_detector.update(dets, fnum)
        rel_f = (release_detector.release.release_frame
                 if release_detector.release else None)
        bounce_detector.update(dets, fnum, release_frame=rel_f)
        bf = (bounce_detector.bounce_point[0]
              if bounce_detector.bounce_point else None)
        trajectory.update(dets, fnum, release_frame=rel_f, bounce_frame=bf)

        if fnum < seg_a_start:
            continue

        logger.log(fnum, dets, grid_corners=grid_corners)

        frame = _draw_progressive_overlays(
            frame, fnum, dets,
            release_detector, bounce_detector, trajectory,
            draw_parabolas=False,
        )

        rel_str = f"Release: F{rel_f}" if rel_f is not None else "Release: --"
        balls_in_frame = sum(1 for d in dets if d["class_name"] == "ball")
        bats_in_frame  = sum(1 for d in dets if d["class_name"] == "bat")
        status = (f"SEG A | Frame: {fnum} | Ball: {balls_in_frame} "
                  f"| Bat: {bats_in_frame} | {rel_str}")
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (255, 255, 255), 2)

        out.write(frame)

    # ──────────────────────────────────────────────────────────────────────
    # PASS 3 — RENDER SEGMENT B (0.5x replay, trajectory + 3 markers from frame 1)
    # ──────────────────────────────────────────────────────────────────────
    print("[INFO] PASS 3 — writing segment B (replay, 0.5x with trajectory)…")
    bounce_detector  = BounceDetector()
    release_detector = ReleaseDetector()
    trajectory       = Trajectory()

    # Pre-seed analyzed state so all overlays render from frame 1 of seg B.
    release_detector.release       = full_release
    bounce_detector.bounce_point   = full_bounce
    trajectory._dir_changes        = list(full_dev_changes)
    trajectory._pts                = list(full_traj_pts)

    # Sentinel frame number that lies above every marker frame, so every
    # gated overlay (bounce, deviation, release→bounce parabola, etc.) draws.
    big_fnum = seg_b_end + 10_000

    # Each arc animates over its own [start, end] frame span. Two output
    # frames are written per source frame (0.5x), so progress advances in
    # half-frame steps for a smoother sweep.
    rb_span = max(1, bounce_f - release_f)
    bd_span = max(1, dev_f    - bounce_f)

    # Ball detections strictly between release and bounce — used to fit a
    # quadratic that hugs the actual flight path for the first parabola.
    rel_to_bounce_pts = [
        (f, x, y) for (f, x, y) in full_traj_pts
        if release_f <= f <= bounce_f
    ]
    print(f"[INFO] Release→bounce parabola fitted to {len(rel_to_bounce_pts)} ball points")

    cap.set(cv2.CAP_PROP_POS_FRAMES, seg_b_start)
    for fnum in range(seg_b_start, seg_b_end + 1):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detections_by_frame.get(fnum, [])

        for sub in (0, 1):
            f_eff = fnum + 0.5 * sub
            rb_prog = max(0.0, min(1.0, (f_eff - release_f) / rb_span))
            bd_prog = max(0.0, min(1.0, (f_eff - bounce_f)  / bd_span))

            frame_out = _draw_progressive_overlays(
                frame.copy(), big_fnum, dets,
                release_detector, bounce_detector, trajectory,
                draw_parabolas=True,
                rel_to_bounce_progress=rb_prog,
                bounce_to_dev_progress=bd_prog,
                rel_to_bounce_pts=rel_to_bounce_pts,
                rel_to_bounce_frame_range=(release_f, bounce_f),
            )

            status = f"SEG B (0.5x) | Frame: {fnum}"
            cv2.putText(frame_out, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, (0, 200, 255), 2)
            out.write(frame_out)

    cap.release()

    # ──────────────────────────────────────────────────────────────────────
    # PASS 4 — RENDER SEGMENT C (zones + parabola animation on frozen frame)
    # ──────────────────────────────────────────────────────────────────────
    print("[INFO] PASS 4 — writing segment C (zones + animated parabolas)…")
    release_pt   = (full_release.x, full_release.y) if full_release else None
    bounce_pt    = (full_bounce[1], full_bounce[2]) if full_bounce  else None
    deviation_pt = ((full_dev_changes[0][1], full_dev_changes[0][2])
                    if full_dev_changes else None)
    _run_segment_c_freeze_animation(
        out, VIDEO_PATH, fps,
        deviation_frame=dev_f,
        release_pt=release_pt,
        bounce_pt=bounce_pt,
        deviation_pt=deviation_pt,
        rel_to_bounce_pts=rel_to_bounce_pts,
        rel_to_bounce_frame_range=(release_f, bounce_f),
        markers=calib_markers,
    )

    out.release()
    logger.close()
    frame_count = total

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\n[INFO] Processing complete!")
    print(f"[INFO] Total execution time      : {elapsed_time:.2f} seconds")
    print(f"[INFO] Frames with ball detected : {ball_count}/{frame_count}")
    print(f"[INFO] Total bat detections      : {bat_count}")
    rel = release_detector.release
    if rel:
        print(f"[INFO] Release point detected    : frame {rel.release_frame} "
              f"at ({rel.x},{rel.y})")
    else:
        print(f"[INFO] Release point             : not detected")

    bp = bounce_detector.bounce_point
    if bp:
        print(f"[INFO] Bounce point detected     : frame {bp[0]} at ({bp[1]},{bp[2]})")
    else:
        print(f"[INFO] Bounce point              : not detected")

    if trajectory._dir_changes:
        df, dx, dy = trajectory._dir_changes[0]
        print(f"[INFO] Deviation point detected  : frame {df} at ({dx},{dy})")
    else:
        print(f"[INFO] Deviation point           : not detected")

    print(f"[INFO] Output video              : {VIDEO_OUTPUT}")
    print(f"[INFO] Detection CSV             : {logger.csv_path}")



if __name__ == "__main__":
    process_video()