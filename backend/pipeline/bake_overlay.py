"""Bake the frontend SVG overlay into a downloadable MP4.

Mirrors the per-segment overlay logic in
`tracker_app/frontend/src/features/tracking/PitchOverlay.tsx` so the downloaded
video matches what the user sees in the Review screen. Reads `output.mp4` +
`results.json` from a job directory and writes `output_full.mp4` next to them.

This module is intentionally self-contained — the alpha-keyed cricket-ball PNG
is read from `tracker_app/frontend/public/`, and the stump/parabola math is
re-implemented here rather than imported from `process.py` so the bake doesn't
depend on the legacy tracker bootstrap.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Optional

import cv2
import numpy as np

def _find_ffmpeg() -> Optional[str]:
    """Locate an ffmpeg binary: PATH first, then the imageio-ffmpeg bundle.
    Inlined here so this module has no dependency on `.process` (which pulls
    in the legacy Zone-div tracker modules that the API container doesn't
    ship)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants — mirror PitchOverlay.tsx
# ─────────────────────────────────────────────────────────────────────────────
_STUMP_DIAM_M      = 0.03493
_STUMP_GAP_M       = 0.054
_STUMP_PITCH_M     = _STUMP_DIAM_M + _STUMP_GAP_M
_STUMP_SET_WIDTH_M = 3 * _STUMP_DIAM_M + 2 * _STUMP_GAP_M
_STUMP_HEIGHT_M    = 0.65
_RETURN_CREASE_M   = 2.64
_STUMP_OPACITY     = 0.70

# BGR (OpenCV) versions of PitchOverlay.tsx STUMP_LIGHT/DARK/OUTLINE.
_STUMP_LIGHT   = (240, 245, 248)
_STUMP_DARK    = (210, 204, 200)
_STUMP_OUTLINE = (130, 120, 110)

# Ball PNG marker — same size as the SVG overlay (PitchOverlay BALL_MARKER_SIZE_PX).
_BALL_MARKER_PX = 21
_N_PARABOLA_SAMPLES = 80


# ─────────────────────────────────────────────────────────────────────────────
# Ball-PNG cache (alpha-keyed from luminance, matching SVG feColorMatrix)
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_FRONTEND_PUBLIC = Path(_os.environ.get(
    "TRACKER_BALL_ASSETS_DIR",
    str(Path(__file__).resolve().parents[2] / "frontend" / "public"),
))
_BALL_PATHS = {
    "white": _FRONTEND_PUBLIC / "cricket_ball.png",
    "red":   _FRONTEND_PUBLIC / "cricket_ball_red.png",
    "pink":  _FRONTEND_PUBLIC / "cricket_ball_pink.png",
}

# Trajectory-line colors per ball type (BGR — OpenCV convention).
# Each is a LIGHTER shade of the corresponding ball, so the trail reads as
# "same ball, but the traveled path" instead of an unrelated overlay.
_TRAJECTORY_COLOR_BGR = {
    "white": (244, 236, 230),  # #e6ecf4  soft cool off-white/silver
    "red":   (112, 138, 255),  # #ff8a70  light coral (lighter red)
    "pink":  (220, 201, 255),  # #ffc9dc  very light pink
}
_DEFAULT_TRAJECTORY_BGR = _TRAJECTORY_COLOR_BGR["white"]
_BALL_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _load_ball(ball_type: str, size: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Returns (bgr, mask) cropped+resized to a `size`×`size` square."""
    key = (ball_type, size)
    if key in _BALL_CACHE:
        return _BALL_CACHE[key]
    path = _BALL_PATHS.get(ball_type, _BALL_PATHS["white"])
    src = cv2.imread(str(path))
    if src is None:
        return None
    # Mirror the SVG <image preserveAspectRatio="xMidYMid meet"> behavior:
    # render the whole PNG centered into a `size × size` box without
    # stretching. We pad the source to a square first, then resize. This
    # keeps the ball's apparent diameter matching the on-screen overlay,
    # which is smaller than `size` when the PNG has transparent padding.
    sh, sw = src.shape[:2]
    side_src = max(sh, sw)
    pad_t = (side_src - sh) // 2
    pad_b = side_src - sh - pad_t
    pad_l = (side_src - sw) // 2
    pad_r = side_src - sw - pad_l
    src = cv2.copyMakeBorder(src, pad_t, pad_b, pad_l, pad_r,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bgr = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    mask_full = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(mask_full, 20, 255, cv2.THRESH_BINARY)
    _BALL_CACHE[key] = (bgr, mask)
    return bgr, mask


def _draw_ball_marker(frame: np.ndarray, x: float, y: float,
                      ball_type: str, size: int = _BALL_MARKER_PX) -> None:
    res = _load_ball(ball_type, size)
    if res is None:
        return
    bgr, mask = res
    h, w = frame.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    half = size // 2
    x0 = cx - half
    y0 = cy - half
    x1 = x0 + size
    y1 = y0 + size
    # Clip to frame bounds.
    sx0 = max(0, -x0); sy0 = max(0, -y0)
    sx1 = size - max(0, x1 - w); sy1 = size - max(0, y1 - h)
    dx0 = max(0, x0); dy0 = max(0, y0)
    dx1 = min(w, x1); dy1 = min(h, y1)
    if dx1 <= dx0 or dy1 <= dy0:
        return
    roi = frame[dy0:dy1, dx0:dx1]
    src = bgr[sy0:sy1, sx0:sx1]
    m = mask[sy0:sy1, sx0:sx1]
    if roi.shape[:2] != src.shape[:2]:
        return
    np.copyto(roi, src, where=(m[..., None] > 0))


# ─────────────────────────────────────────────────────────────────────────────
# Stumps + pitching line — port of PitchOverlay StumpSet / PitchingLine
# ─────────────────────────────────────────────────────────────────────────────
def _draw_stump_set(frame: np.ndarray, base: tuple[float, float], crease_w: float,
                    alpha: float = _STUMP_OPACITY) -> None:
    if crease_w <= 0:
        return
    px_per_m = crease_w / _RETURN_CREASE_M
    half_w   = _STUMP_PITCH_M * px_per_m
    height   = _STUMP_HEIGHT_M * px_per_m
    diam     = max(2.0, _STUMP_DIAM_M * px_per_m)
    thick    = max(2, int(round(diam)))
    cx, cy = float(base[0]), float(base[1])
    top_y  = cy - height
    overlay = frame.copy()
    xs = [cx - half_w, cx, cx + half_w]
    for x in xs:
        cv2.line(overlay, (int(x), int(cy)), (int(x), int(top_y)),
                 _STUMP_OUTLINE, thick + 2, lineType=cv2.LINE_AA)
        cv2.line(overlay, (int(x), int(cy)), (int(x), int(top_y)),
                 _STUMP_LIGHT, thick, lineType=cv2.LINE_AA)
        sh = max(1, int(thick * 0.32))
        ox = int(diam * 0.22)
        cv2.line(overlay, (int(x + ox), int(cy)), (int(x + ox), int(top_y)),
                 _STUMP_DARK, sh, lineType=cv2.LINE_AA)
    bail_thick = max(2, int(diam * 0.7))
    for a_x, b_x in ((xs[0], xs[1]), (xs[1], xs[2])):
        cv2.line(overlay, (int(a_x), int(top_y)), (int(b_x), int(top_y)),
                 _STUMP_OUTLINE, bail_thick + 1, lineType=cv2.LINE_AA)
        cv2.line(overlay, (int(a_x), int(top_y)), (int(b_x), int(top_y)),
                 _STUMP_DARK, bail_thick, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_pitching_line(frame: np.ndarray, cal: dict) -> None:
    bowl_stump = cal["bowl_stump"]; bat_stump = cal["bat_stump"]
    bowl_w = _dist(cal["bowl_L"], cal["bowl_R"])
    bat_w  = _dist(cal["bat_L"],  cal["bat_R"])
    if bowl_w <= 0 or bat_w <= 0:
        return
    dx = bat_stump[0] - bowl_stump[0]
    dy = bat_stump[1] - bowl_stump[1]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return
    ux, uy = dx / length, dy / length
    px_dir, py_dir = -uy, ux
    half_bowl = (_STUMP_SET_WIDTH_M / 2) * (bowl_w / _RETURN_CREASE_M)
    half_bat  = (_STUMP_SET_WIDTH_M / 2) * (bat_w  / _RETURN_CREASE_M)
    bL = (bowl_stump[0] - px_dir * half_bowl, bowl_stump[1] - py_dir * half_bowl)
    aL = (bat_stump[0]  - px_dir * half_bat,  bat_stump[1]  - py_dir * half_bat)
    aR = (bat_stump[0]  + px_dir * half_bat,  bat_stump[1]  + py_dir * half_bat)
    bR = (bowl_stump[0] + px_dir * half_bowl, bowl_stump[1] + py_dir * half_bowl)
    pts = np.array(
        [[int(p[0]), int(p[1])] for p in (bL, aL, aR, bR)], dtype=np.int32,
    )
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ─────────────────────────────────────────────────────────────────────────────
# Parabola — constrained quadratic fit, port of PitchOverlay
# ─────────────────────────────────────────────────────────────────────────────
def _fit_kxky(rb_pts, x1, y1, x2, y2, f_lo, f_hi) -> tuple[float, float]:
    if not rb_pts or f_hi <= f_lo:
        return 0.0, 0.0
    sumAA = sumABx = sumABy = 0.0
    span = f_hi - f_lo
    for p in rb_pts:
        t = (p["frame"] - f_lo) / span
        a = t * (1 - t)
        sumAA  += a * a
        sumABx += a * (p["x"] - ((1 - t) * x1 + t * x2))
        sumABy += a * (p["y"] - ((1 - t) * y1 + t * y2))
    if sumAA < 1e-9:
        return 0.0, 0.0
    return sumABx / sumAA, sumABy / sumAA


def _point_on_quad(x1, y1, x2, y2, kx, ky, t) -> tuple[float, float]:
    a = t * (1 - t)
    return ((1 - t) * x1 + t * x2 + kx * a,
            (1 - t) * y1 + t * y2 + ky * a)


def _draw_parabola_segment(frame, pts: list[tuple[float, float]],
                            stroke: tuple[int, int, int], thickness: int,
                            alpha: float) -> None:
    if len(pts) < 2:
        return
    overlay = frame.copy()
    arr = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
    cv2.polylines(overlay, [arr], isClosed=False, color=stroke,
                  thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ─────────────────────────────────────────────────────────────────────────────
# Phase routing — exact port of PitchOverlay.currentPhase / parabolaProgress
# ─────────────────────────────────────────────────────────────────────────────
def _phase(t: float, segs: dict) -> str:
    if t < segs["seg_a_start_sec"]:    return "before"
    if t < segs["seg_a_end_sec"]:      return "segA"
    if t < segs["seg_b_end_sec"]:      return "segB"
    end_c = segs.get("seg_c_end_sec")
    if end_c is not None and t < end_c: return "segC"
    return "after"


def _progress(t: float, segs: dict, phase: str) -> tuple[float, float]:
    """(rb, bd) — fraction of release→bounce and bounce→deviation drawn."""
    if phase in ("before", "segA"):
        return 0.0, 0.0
    if phase == "segB":
        r = segs.get("seg_b_release_sec")
        b = segs.get("seg_b_bounce_sec")
        d = segs.get("seg_b_deviation_sec")
        if r is None or b is None:
            return 0.0, 0.0
        if t < r:  return 0.0, 0.0
        if t < b:  return (t - r) / max(1e-6, b - r), 0.0
        if d is None: return 1.0, 0.0
        if t < d:  return 1.0, (t - b) / max(1e-6, d - b)
        return 1.0, 1.0
    if phase == "segC":
        s    = segs.get("seg_c_start_sec")
        rEnd = segs.get("seg_c_rb_end_sec")
        bEnd = segs.get("seg_c_bd_end_sec")
        if s is None or rEnd is None or bEnd is None:
            return 1.0, 1.0
        if t < rEnd: return (t - s) / max(1e-6, rEnd - s), 0.0
        if t < bEnd: return 1.0, (t - rEnd) / max(1e-6, bEnd - rEnd)
        return 1.0, 1.0
    return 1.0, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Frame composer
# ─────────────────────────────────────────────────────────────────────────────
def _compose_frame(frame: np.ndarray, t: float, segs: dict, results: dict,
                   ball_type: str) -> None:
    """Mutate `frame` in place with the overlays appropriate to time `t`."""
    cal = results["calibration"]
    ev = results["events"]
    release = ev.get("release")
    bounce  = ev.get("bounce")
    dev     = ev.get("deviation")

    phase = _phase(t, segs)
    if phase in ("before", "after"):
        return

    bowl_w = _dist(cal["bowl_L"], cal["bowl_R"])
    bat_w  = _dist(cal["bat_L"],  cal["bat_R"])

    if phase == "segA":
        # Progressive ball markers only.
        if release and segs.get("seg_a_release_sec") is not None and \
                t >= segs["seg_a_release_sec"]:
            _draw_ball_marker(frame, release["x"], release["y"], ball_type)
        if bounce and segs.get("seg_a_bounce_sec") is not None and \
                t >= segs["seg_a_bounce_sec"]:
            _draw_ball_marker(frame, bounce["x"], bounce["y"], ball_type)
        if dev and segs.get("seg_a_deviation_sec") is not None and \
                t >= segs["seg_a_deviation_sec"]:
            _draw_ball_marker(frame, dev["x"], dev["y"], ball_type)
        return

    # SEG B / SEG C share the stump + pitching-line stack.
    # Layer order from PitchOverlay (bottom → top):
    #   1. Batter-end stumps   2. Parabola+ball   3. Pitching line
    #   4. Bowler-end stumps   5. Event markers
    _draw_stump_set(frame, cal["bat_stump"], bat_w)

    # Parabola + traveling ball.
    if release and bounce:
        rb, bd = _progress(t, segs, phase)
        x1, y1 = release["x"], release["y"]
        x2, y2 = bounce["x"],  bounce["y"]
        f_lo, f_hi = release["frame"], bounce["frame"]
        rb_pts = [p for p in results["trajectory"]
                  if f_lo <= p["frame"] <= f_hi]
        kx, ky = _fit_kxky(rb_pts, x1, y1, x2, y2, f_lo, f_hi)
        trail_bgr = _TRAJECTORY_COLOR_BGR.get(ball_type, _DEFAULT_TRAJECTORY_BGR)

        if rb > 0:
            n = max(2, round(_N_PARABOLA_SAMPLES * rb))
            pts = [_point_on_quad(x1, y1, x2, y2, kx, ky, (i / (n - 1)) * rb)
                   for i in range(n)]
            _draw_parabola_segment(frame, pts, trail_bgr, 16, 0.40)

        if dev and bd > 0:
            ex = bounce["x"] + bd * (dev["x"] - bounce["x"])
            ey = bounce["y"] + bd * (dev["y"] - bounce["y"])
            _draw_parabola_segment(frame,
                                   [(bounce["x"], bounce["y"]), (ex, ey)],
                                   trail_bgr, 13, 0.70)

        # SEG C traveling ball leads the sweep.
        if phase == "segC":
            ball_pos = None
            if 0 < rb < 1:
                ball_pos = _point_on_quad(x1, y1, x2, y2, kx, ky, rb)
            elif dev and rb >= 1 and 0 < bd < 1:
                ball_pos = (bounce["x"] + bd * (dev["x"] - bounce["x"]),
                            bounce["y"] + bd * (dev["y"] - bounce["y"]))
            if ball_pos is not None:
                _draw_ball_marker(frame, ball_pos[0], ball_pos[1], ball_type)

    _draw_pitching_line(frame, cal)
    _draw_stump_set(frame, cal["bowl_stump"], bowl_w)

    # Event markers.
    if phase == "segB":
        if release: _draw_ball_marker(frame, release["x"], release["y"], ball_type)
        if bounce:  _draw_ball_marker(frame, bounce["x"],  bounce["y"],  ball_type)
        if dev:     _draw_ball_marker(frame, dev["x"],     dev["y"],     ball_type)
    elif phase == "segC":
        bEnd = segs.get("seg_c_bd_end_sec")
        if bEnd is not None and t >= bEnd and bounce:
            _draw_ball_marker(frame, bounce["x"], bounce["y"], ball_type)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry — lazy per-job bake with a process-local lock
# ─────────────────────────────────────────────────────────────────────────────
_JOB_LOCKS: dict[str, Lock] = {}
_LOCKS_MASTER = Lock()


def _job_lock(job_id: str) -> Lock:
    with _LOCKS_MASTER:
        lk = _JOB_LOCKS.get(job_id)
        if lk is None:
            lk = _JOB_LOCKS[job_id] = Lock()
        return lk


def _reencode_h264(src: Path, dst: Path) -> bool:
    """Mirror of process._reencode_h264 — kept local to avoid coupling."""
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return False
    return result.returncode == 0


def bake_full_overlay(job_id: str, job_dir: Path, ball_type: str) -> Path:
    """Returns the path to output_full.mp4, generating it if absent.

    Idempotent and serialized per job_id: concurrent callers see the same
    finished file.
    """
    out_final = job_dir / "output_full.mp4"
    with _job_lock(job_id):
        if out_final.exists():
            return out_final
        src = job_dir / "output.mp4"
        results_path = job_dir / "results.json"
        if not src.exists() or not results_path.exists():
            raise FileNotFoundError(
                f"job {job_id}: missing output.mp4 or results.json")

        results = json.loads(results_path.read_text())
        segs = results["segments"]
        fps = float(results["video"]["fps"])
        width = int(results["video"]["width"])
        height = int(results["video"]["height"])

        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"job {job_id}: cannot open {src}")

        temp = job_dir / "_temp_full.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp), fourcc, fps, (width, height))
        try:
            f_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = f_idx / fps
                _compose_frame(frame, t, segs, results, ball_type)
                writer.write(frame)
                f_idx += 1
        finally:
            cap.release()
            writer.release()

        if _reencode_h264(temp, out_final):
            try: temp.unlink()
            except OSError: pass
        else:
            shutil.move(str(temp), str(out_final))
        return out_final
