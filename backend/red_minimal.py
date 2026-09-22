"""
Minimal Kalman-first red-ball pipeline.

Deliberately excludes:
  - Bat class, size/aspect filters, MAX_POSITION_JUMP, zone classification,
    impact detector, trail/glow/stumps/pitching-line rendering, TRACKER_DEBUG
    per-frame chatter.
Kept:
  - YOLOv8 detection (dynamic-axes ONNX)
  - 2-state confidence: PRIMARY floor for seeding & anchor detections,
    RECOVERY floor for on-Kalman-path recovery in gap frames
  - Constant-velocity + gravity Kalman filter with Mahalanobis chi-square gate
  - Sliding-window (3/6) release detection
  - Bounce = lowest-y point on the Kalman-smoothed trajectory (post-release)
  - Deviation = first significant post-bounce residual break
  - Parabola fit through Kalman-smoothed release->bounce points
  - Overlay: parabola drawn progressively, ball markers at release/bounce/dev,
    straight line bounce->deviation

Usage:
    cd D:\\intership\\tracker_app\\backend
    python red_minimal.py `
        --video "D:\\intership\\Input files\\Red_ball\\4_025_05.mp4" `
        --weights "worker\\models\\red.onnx" `
        --out "D:\\intership\\OUTPUT\\4_025_05_min.mp4"

Optional:
    --imgsz 1280           letterbox size (default 1280)
    --primary 0.60         primary conf floor (default 0.60)
    --recovery 0.30        recovery conf floor (default 0.30)
    --seed 0.70            seed confidence (must exceed this to initialize KF)
    --gate 12.0            Mahalanobis chi2 gate for recovery (2 DoF, ~99.7%)
    --debug                print per-frame decisions to stdout
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort


# ─────────────────────────────────────────────────────────────────────────────
# Kalman filter — 4-state (x, y, vx, vy), constant velocity + gravity on vy.
# ─────────────────────────────────────────────────────────────────────────────
class KalmanBallFilter:
    """Image-space ball tracker.

    Process model: constant velocity, plus a constant vy acceleration (gravity)
    applied at each predict step. Measurement is (cx, cy) center in original
    frame coords.
    """

    def __init__(self, gravity: float = 0.35, process_noise: float = 1.0,
                 meas_noise: float = 5.0):
        self.gravity = gravity
        self._Q = np.eye(4) * process_noise
        self._R = np.eye(2) * meas_noise
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)
        self.reset()

    def reset(self) -> None:
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 100.0
        self.is_initialized = False

    def initialize(self, cx: float, cy: float,
                   vx: float = 0.0, vy: float = 0.0) -> None:
        self.x = np.array([[cx], [cy], [vx], [vy]], dtype=float)
        self.P = np.diag([10.0, 10.0, 25.0, 25.0])
        self.is_initialized = True

    def predict(self) -> None:
        self.x = self._F @ self.x
        self.x[3, 0] += self.gravity
        self.P = self._F @ self.P @ self._F.T + self._Q

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        z = np.array([[cx], [cy]], dtype=float)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self._H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    def mahalanobis(self, cx: float, cy: float) -> float:
        """Chi-square distance of (cx, cy) to current prediction. 2 DoF."""
        z = np.array([[cx], [cy]], dtype=float)
        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self._R
        return float((y.T @ np.linalg.inv(S) @ y)[0, 0])

    def handle_gap(self) -> None:
        """Inflate uncertainty during a missed frame — grows Mahalanobis gate
        for the next detection so a ball that reappears farther away is still
        admitted."""
        self.P = self.P + np.eye(4) * 4.0

    @property
    def pos(self) -> tuple[float, float]:
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def vel(self) -> tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])


# ─────────────────────────────────────────────────────────────────────────────
# Detection — YOLOv8 single-class, dynamic input.
# ─────────────────────────────────────────────────────────────────────────────
def _letterbox(img: np.ndarray, sz: int) -> tuple[np.ndarray, float, float, float]:
    h, w = img.shape[:2]
    r = min(sz / h, sz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (sz - nw) / 2, (sz - nh) / 2
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img_r, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, r, dw, dh


def _xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_thr]
    return keep


@dataclass
class RawDet:
    cx: float
    cy: float
    w: float
    h: float
    score: float


def detect_raw(session: ort.InferenceSession, frame: np.ndarray, imgsz: int,
               floor: float, iou_thr: float = 0.45) -> list[RawDet]:
    """Run one inference, apply floor + NMS, return dets in ORIGINAL frame coords."""
    padded, ratio, dw, dh = _letterbox(frame, imgsz)
    blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]

    input_name = session.get_inputs()[0].name
    raw = session.run(None, {input_name: blob})[0][0]  # [C, N] for YOLOv8

    # Normalize to [N, 4+C]
    if raw.shape[0] < raw.shape[1]:
        raw = raw.T
    # Single-class YOLOv8: cols are [cx, cy, w, h, class0]
    scores = raw[:, 4]
    mask = scores >= floor
    if not mask.any():
        return []

    kept = raw[mask]
    kept_scores = scores[mask]
    kept_boxes_xyxy = _xywh2xyxy(kept[:, :4])
    keep_idx = _nms(kept_boxes_xyxy, kept_scores, iou_thr)

    dets = []
    for i in keep_idx:
        b = kept[i]
        cx = (b[0] - dw) / ratio
        cy = (b[1] - dh) / ratio
        w = b[2] / ratio
        h = b[3] / ratio
        dets.append(RawDet(cx=float(cx), cy=float(cy),
                           w=float(w), h=float(h), score=float(kept_scores[i])))
    return dets


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory record
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrajPoint:
    frame: int
    cx: float             # raw detection center (or predicted, if from_pred)
    cy: float
    sx: float             # Kalman-smoothed center
    sy: float
    score: float
    from_pred: bool       # True if this frame had no accepted detection
    recovery: bool        # True if accepted below primary (via Kalman gate)
    w: float = 0.0        # raw detection box width  (0 when from_pred)
    h: float = 0.0        # raw detection box height (0 when from_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — collect raw detections at RECOVERY floor for every frame.
# ─────────────────────────────────────────────────────────────────────────────
def collect_raw(session, cap, imgsz: int, recovery_floor: float,
                debug: bool) -> tuple[dict[int, list[RawDet]], int, tuple[int, int, float]]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame: dict[int, list[RawDet]] = {}
    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detect_raw(session, frame, imgsz, floor=recovery_floor)
        per_frame[f] = dets
        if debug and dets:
            top = max(dets, key=lambda d: d.score)
            print(f"[RAW] f={f} n={len(dets)} top={top.score:.3f} "
                  f"xy=({int(top.cx)},{int(top.cy)}) wh=({int(top.w)}x{int(top.h)})")
        f += 1
    return per_frame, f, (w, h, fps)


# ─────────────────────────────────────────────────────────────────────────────
# Seed selection — pick the first frame with a strong (>= seed_conf) detection
# that is followed by another strong detection within `look_ahead` frames whose
# position is a plausible ball motion from the seed. This double-check kills
# static false-positive seeds (a shadow scoring 0.75 at the same xy in 5 frames
# in a row would fail the motion test).
# ─────────────────────────────────────────────────────────────────────────────
def pick_seed(per_frame: dict[int, list[RawDet]], seed_conf: float,
              look_ahead: int = 8, min_motion: float = 4.0,
              max_motion_per_frame: float = 150.0) -> Optional[tuple[int, RawDet, int, RawDet]]:
    frames = sorted(per_frame.keys())
    for i, f in enumerate(frames):
        strong = [d for d in per_frame[f] if d.score >= seed_conf]
        if not strong:
            continue
        seed = max(strong, key=lambda d: d.score)
        for g in frames[i + 1:i + 1 + look_ahead]:
            gap = g - f
            if gap <= 0 or gap > look_ahead:
                continue
            for cand in per_frame[g]:
                if cand.score < seed_conf:
                    continue
                d = ((cand.cx - seed.cx) ** 2 + (cand.cy - seed.cy) ** 2) ** 0.5
                per_frame_motion = d / gap
                if min_motion <= d and per_frame_motion <= max_motion_per_frame:
                    return f, seed, g, cand
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Forward Kalman pass from the seed onward, with recovery via the chi2 gate.
# ─────────────────────────────────────────────────────────────────────────────
def kalman_forward(per_frame: dict[int, list[RawDet]], seed_frame: int,
                   seed_det: RawDet, second_frame: int, second_det: RawDet,
                   total_frames: int, primary: float, recovery: float,
                   gate: float, debug: bool) -> list[TrajPoint]:
    kf = KalmanBallFilter(gravity=0.35)
    # Initialize with velocity implied by seed->second
    dt = max(1, second_frame - seed_frame)
    vx = (second_det.cx - seed_det.cx) / dt
    vy = (second_det.cy - seed_det.cy) / dt
    kf.initialize(seed_det.cx, seed_det.cy, vx=vx, vy=vy)

    traj: list[TrajPoint] = []
    traj.append(TrajPoint(frame=seed_frame, cx=seed_det.cx, cy=seed_det.cy,
                          sx=seed_det.cx, sy=seed_det.cy,
                          score=seed_det.score, from_pred=False, recovery=False,
                          w=seed_det.w, h=seed_det.h))

    for f in range(seed_frame + 1, total_frames):
        kf.predict()
        cands = per_frame.get(f, [])

        # Two acceptance rules:
        #  a) score >= primary AND passes gate (relaxed if kalman uncertain)
        #  b) score >= recovery AND passes gate (recovery: only work if Kalman
        #     agrees with the candidate)
        best: Optional[RawDet] = None
        best_d2 = float("inf")
        best_recovery = False
        for c in cands:
            d2 = kf.mahalanobis(c.cx, c.cy)
            if c.score >= primary and d2 <= gate:
                if d2 < best_d2:
                    best, best_d2, best_recovery = c, d2, False
            elif c.score >= recovery and d2 <= gate:
                # Recovery candidate — only consider if nothing better found
                if best is None and d2 < best_d2:
                    best, best_d2, best_recovery = c, d2, True

        if best is not None:
            sx, sy = kf.update(best.cx, best.cy)
            traj.append(TrajPoint(frame=f, cx=best.cx, cy=best.cy,
                                  sx=sx, sy=sy, score=best.score,
                                  from_pred=False, recovery=best_recovery,
                                  w=best.w, h=best.h))
            if debug:
                tag = "REC" if best_recovery else "PRI"
                print(f"[KF] f={f} {tag} conf={best.score:.2f} d2={best_d2:.1f}")
        else:
            kf.handle_gap()
            px, py = kf.pos
            traj.append(TrajPoint(frame=f, cx=px, cy=py, sx=px, sy=py,
                                  score=0.0, from_pred=True, recovery=False))
            if debug:
                print(f"[KF] f={f} GAP pred=({int(px)},{int(py)})")
    return traj


# ─────────────────────────────────────────────────────────────────────────────
# Release detection — first frame in the 3-of-6 sliding window of REAL
# (non-predicted) trajectory points. Scans from the seed onward.
# ─────────────────────────────────────────────────────────────────────────────
def find_release(traj: list[TrajPoint], window: int = 6, min_hits: int = 3) -> Optional[TrajPoint]:
    for i in range(len(traj) - window + 1):
        w = traj[i:i + window]
        reals = [p for p in w if not p.from_pred]
        if len(reals) >= min_hits:
            return reals[0]
    if traj and not traj[0].from_pred:
        return traj[0]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Direction-change detection — bounce and deviation are BOTH points where the
# ball's direction changes sustainedly. Bounce is the FIRST such change after
# release (ball hits the pitch); deviation is the SECOND (ball hits bat/pad
# after the bounce). Both use the same angular test — reference direction from
# the two frames right after the previous anchor, then walk forward and flag
# the first frame with `consecutive` frames whose per-step velocity vector is
# more than `angle_deg_thresh` away from the reference.
# ─────────────────────────────────────────────────────────────────────────────
def _first_direction_change(post: list[TrajPoint], angle_deg_thresh: float,
                            consecutive: int) -> Optional[TrajPoint]:
    if len(post) < 4:
        return None
    a, b = post[0], post[1]
    ref = np.array([b.sx - a.sx, b.sy - a.sy])
    ref_n = np.linalg.norm(ref)
    if ref_n < 1e-3:
        return None
    ref /= ref_n
    thresh = np.cos(np.deg2rad(angle_deg_thresh))
    streak = 0
    first_off = None
    for i in range(2, len(post)):
        cur = np.array([post[i].sx - post[i - 1].sx,
                        post[i].sy - post[i - 1].sy])
        n = np.linalg.norm(cur)
        if n < 1e-3:
            continue
        cur /= n
        cos = float(ref @ cur)
        if cos < thresh:
            streak += 1
            if first_off is None:
                first_off = post[i]
            if streak >= consecutive:
                return first_off
        else:
            streak = 0
            first_off = None
    return None


MIN_RELEASE_TO_BOUNCE_FRAMES = 4  # bounce must be at least this many frames
                                  # after release (physical delivery guard)


def find_bounce(traj: list[TrajPoint], release: TrajPoint,
                angle_deg_thresh: float = 20.0,
                consecutive: int = 2) -> Optional[TrajPoint]:
    """First sustained direction change after release, subject to a minimum
    frame gap so we don't flag early Kalman jitter as the bounce."""
    post = [p for p in traj if p.frame > release.frame and not p.from_pred]
    cand = _first_direction_change(post, angle_deg_thresh, consecutive)
    while cand is not None and (cand.frame - release.frame) < MIN_RELEASE_TO_BOUNCE_FRAMES:
        # Reject and scan again starting after this candidate.
        post = [p for p in post if p.frame > cand.frame]
        cand = _first_direction_change(post, angle_deg_thresh, consecutive)
    return cand


def find_deviation(traj: list[TrajPoint], bounce: TrajPoint,
                   angle_deg_thresh: float = 20.0,
                   consecutive: int = 2) -> Optional[TrajPoint]:
    """First sustained direction change after bounce (second overall)."""
    post = [p for p in traj if p.frame > bounce.frame and not p.from_pred]
    return _first_direction_change(post, angle_deg_thresh, consecutive)


# ─────────────────────────────────────────────────────────────────────────────
# Parabola — fit y = a*x^2 + b*x + c through Kalman-smoothed release->bounce.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ParabolaFit:
    x1: float; y1: float          # release
    x2: float; y2: float          # bounce
    kx: float; ky: float          # perpendicular control offset
    f_lo: int; f_hi: int          # frame endpoints


def fit_parabola(traj: list[TrajPoint], release: TrajPoint,
                 bounce: TrajPoint) -> Optional[ParabolaFit]:
    """Parametric quadratic anchored at release and bounce, with a control
    offset (kx, ky) fit least-squares to the intermediate Kalman-smoothed
    detections. Ported from bake_overlay._fit_kxky so the shape matches the
    rest of the app. Handles any orientation of the release->bounce vector
    (a purely-vertical trajectory would explode a horizontal y=ax^2+bx+c fit;
    this parametric form doesn't)."""
    x1, y1 = release.sx, release.sy
    x2, y2 = bounce.sx, bounce.sy
    f_lo, f_hi = release.frame, bounce.frame
    if f_hi <= f_lo:
        return None
    mid = [p for p in traj
           if f_lo < p.frame < f_hi and not p.from_pred]
    sumAA = sumABx = sumABy = 0.0
    span = f_hi - f_lo
    for p in mid:
        t = (p.frame - f_lo) / span
        a = t * (1 - t)
        sumAA  += a * a
        sumABx += a * (p.sx - ((1 - t) * x1 + t * x2))
        sumABy += a * (p.sy - ((1 - t) * y1 + t * y2))
    if sumAA < 1e-9:
        kx, ky = 0.0, 0.0
    else:
        kx = sumABx / sumAA
        ky = sumABy / sumAA
    return ParabolaFit(x1=x1, y1=y1, x2=x2, y2=y2, kx=kx, ky=ky,
                       f_lo=f_lo, f_hi=f_hi)


def _point_on_quad(fit: ParabolaFit, t: float) -> tuple[float, float]:
    a = t * (1 - t)
    return ((1 - t) * fit.x1 + t * fit.x2 + fit.kx * a,
            (1 - t) * fit.y1 + t * fit.y2 + fit.ky * a)


# ─────────────────────────────────────────────────────────────────────────────
# Baking — second pass over the video, drawing progressive parabola + markers.
# Style ported from bake_overlay.py so this matches the rest of the app.
# ─────────────────────────────────────────────────────────────────────────────
_BALL_SPRITE_PX = 34
_BALL_SPRITE_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _load_ball_sprite(ball_type: str, size: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Load the cricket_ball_{red,white}.png sprite, alpha-keyed from luminance.
    Returns (bgr, mask) at `size` × `size`."""
    key = (ball_type, size)
    if key in _BALL_SPRITE_CACHE:
        return _BALL_SPRITE_CACHE[key]
    name = "cricket_ball_red.png" if ball_type == "red" else "cricket_ball.png"
    p = Path(__file__).resolve().parents[1] / "frontend" / "public" / name
    src = cv2.imread(str(p))
    if src is None:
        return None
    sh, sw = src.shape[:2]
    side = max(sh, sw)
    pt = (side - sh) // 2; pb = side - sh - pt
    pl = (side - sw) // 2; pr = side - sw - pl
    src = cv2.copyMakeBorder(src, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    bgr = cv2.resize(src, (size, size), interpolation=cv2.INTER_AREA)
    mask_full = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    _, mask = cv2.threshold(mask_full, 20, 255, cv2.THRESH_BINARY)
    _BALL_SPRITE_CACHE[key] = (bgr, mask)
    return bgr, mask


def _draw_ball_marker(frame: np.ndarray, x: float, y: float,
                      ball_type: str = "red",
                      size: int = _BALL_SPRITE_PX) -> None:
    res = _load_ball_sprite(ball_type, size)
    if res is None:
        # Fallback: solid circle if sprite is missing
        cv2.circle(frame, (int(x), int(y)), size // 2, (0, 0, 200), -1, cv2.LINE_AA)
        return
    bgr, mask = res
    h, w = frame.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    half = size // 2
    x0, y0 = cx - half, cy - half
    x1, y1 = x0 + size, y0 + size
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


def _sample_parabola_pts(fit: "ParabolaFit", progress: float,
                         n: int = 80) -> list[tuple[float, float]]:
    """Sample n points along the parametric quadratic between t=0 and
    t=progress (progress clamped to [0,1])."""
    prog = max(0.0, min(1.0, progress))
    if prog <= 0.0:
        return []
    ts = np.linspace(0.0, prog, n)
    return [_point_on_quad(fit, float(t)) for t in ts]


def _draw_parabola_segment(frame: np.ndarray, pts: list[tuple[float, float]],
                           stroke: tuple[int, int, int], thickness: int,
                           alpha: float) -> None:
    """Ported from pipeline/bake_overlay.py — soft alpha-blended polyline."""
    if len(pts) < 2:
        return
    overlay = frame.copy()
    arr = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
    cv2.polylines(overlay, [arr], isClosed=False, color=stroke,
                  thickness=thickness, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def _draw_bbox_with_conf(frame: np.ndarray, cx: float, cy: float,
                         w: float, h: float, conf: float,
                         recovery: bool = False) -> None:
    """Per-frame ball detection box + confidence label. Green = primary,
    yellow = recovery (accepted via Kalman gate at lower confidence)."""
    color = (0, 255, 255) if recovery else (0, 255, 0)
    x1 = int(round(cx - w / 2)); y1 = int(round(cy - h / 2))
    x2 = int(round(cx + w / 2)); y2 = int(round(cy + h / 2))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"ball {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 6)),
                  (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def bake(video_path: Path, out_path: Path, traj: list[TrajPoint],
         release: Optional[TrajPoint], bounce: Optional[TrajPoint],
         deviation: Optional[TrajPoint], parabola: Optional[ParabolaFit],
         parabola2: Optional[ParabolaFit] = None,
         last_real: Optional[TrajPoint] = None,
         ball_type: str = "red") -> None:
    """Second video pass. Draws per frame:
       1. per-frame ball bbox + confidence label (green=primary, yellow=recovery)
       2. progressive parabola (release -> bounce), then straight line
          bounce -> deviation. Styled as soft alpha-blended polylines matching
          pipeline/bake_overlay.py — wide dim outer + tight bright core.
       3. cricket-ball sprite markers at release/bounce/deviation, persistent
          once each event frame is reached
       4. HUD line summarizing state
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    traj_by_frame = {p.frame: p for p in traj}

    # Warm-gold soft-glow polyline stroke (matches bake_overlay.py convention).
    PARA_OUTER = (255, 255, 255)
    PARA_CORE  = (0, 200, 255)
    # Parabola 2 (release -> last detected) — cool cyan so it visibly differs
    # from parabola 1 (release -> bounce) when both are drawn simultaneously.
    PARA2_OUTER = (255, 255, 255)
    PARA2_CORE  = (255, 200, 0)

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # (1) per-frame ball bbox
        p = traj_by_frame.get(f)
        if p is not None and not p.from_pred and p.w > 0 and p.h > 0:
            _draw_bbox_with_conf(frame, p.cx, p.cy, p.w, p.h, p.score, p.recovery)

        # (2a) progressive parabola release -> bounce (parametric quadratic).
        if (parabola is not None and release is not None and bounce is not None
                and f >= release.frame):
            span = max(1, bounce.frame - release.frame)
            prog = (f - release.frame) / span
            pts = _sample_parabola_pts(parabola, prog)
            _draw_parabola_segment(frame, pts, PARA_OUTER, 10, 0.40)
            _draw_parabola_segment(frame, pts, PARA_CORE,   4, 0.90)

        # (2a2) parabola 2 — release -> last detected ball, drawn together
        # with parabola 1 so the difference between "detected bounce" and
        # "actual last-seen ball" is visible.
        if (parabola2 is not None and release is not None and last_real is not None
                and f >= release.frame):
            span2 = max(1, last_real.frame - release.frame)
            prog2 = (f - release.frame) / span2
            pts2 = _sample_parabola_pts(parabola2, prog2)
            _draw_parabola_segment(frame, pts2, PARA2_OUTER, 10, 0.35)
            _draw_parabola_segment(frame, pts2, PARA2_CORE,   4, 0.85)

        # (2b) straight line bounce -> deviation
        if bounce is not None and deviation is not None and f >= bounce.frame:
            span = max(1, deviation.frame - bounce.frame)
            prog = min(1.0, (f - bounce.frame) / span)
            ex = bounce.sx + (deviation.sx - bounce.sx) * prog
            ey = bounce.sy + (deviation.sy - bounce.sy) * prog
            pts = [(bounce.sx, bounce.sy), (ex, ey)]
            _draw_parabola_segment(frame, pts, PARA_OUTER, 10, 0.40)
            _draw_parabola_segment(frame, pts, PARA_CORE,   4, 0.90)

        # (3) sprite markers
        if release is not None and f >= release.frame:
            _draw_ball_marker(frame, release.sx, release.sy, ball_type)
        if bounce is not None and f >= bounce.frame:
            _draw_ball_marker(frame, bounce.sx, bounce.sy, ball_type)
        if deviation is not None and f >= deviation.frame:
            _draw_ball_marker(frame, deviation.sx, deviation.sy, ball_type)

        # (4) HUD
        hud = f"f={f}/{total}"
        if release is not None:
            hud += f"  R@{release.frame}"
        if bounce is not None:
            hud += f"  B@{bounce.frame}"
        if deviation is not None:
            hud += f"  D@{deviation.frame}"
        cv2.putText(frame, hud, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    print(f"[BAKE] wrote {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--primary", type=float, default=0.60)
    ap.add_argument("--recovery", type=float, default=0.30)
    ap.add_argument("--seed", type=float, default=0.70)
    ap.add_argument("--gate", type=float, default=12.0)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    print(f"[MODEL] loading {args.weights} providers={providers}")
    session = ort.InferenceSession(str(args.weights), providers=providers)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"can't open video: {args.video}")

    print("[PASS1] collecting raw detections at recovery floor "
          f"{args.recovery} across the whole video")
    per_frame, total, (w, h, fps) = collect_raw(
        session, cap, args.imgsz, args.recovery, args.debug)
    cap.release()
    print(f"[PASS1] {sum(1 for f in per_frame if per_frame[f])}/{total} "
          f"frames have >=1 raw candidate above {args.recovery}")

    seed = pick_seed(per_frame, args.seed)
    if seed is None:
        print(f"[SEED] no confident-and-moving anchor at >= {args.seed} — "
              "cannot proceed. Try lowering --seed.")
        return 1
    sf, sd, gf, gd = seed
    print(f"[SEED] anchor at f={sf} conf={sd.score:.2f} -> f={gf} conf={gd.score:.2f}")

    traj = kalman_forward(per_frame, sf, sd, gf, gd, total,
                          args.primary, args.recovery, args.gate, args.debug)
    kept_real = sum(1 for p in traj if not p.from_pred)
    print(f"[KALMAN] {kept_real}/{len(traj)} post-seed frames accepted a detection")

    release = find_release(traj)
    print(f"[RELEASE] {'f=' + str(release.frame) if release else 'NOT FOUND'}")

    bounce = find_bounce(traj, release) if release else None
    print(f"[BOUNCE]  {'f=' + str(bounce.frame) if bounce else 'NOT FOUND'}")

    deviation = find_deviation(traj, bounce) if bounce else None
    print(f"[DEV]     {'f=' + str(deviation.frame) if deviation else 'NOT FOUND'}")

    parabola = fit_parabola(traj, release, bounce) if (release and bounce) else None
    print(f"[PARABOLA] {'fit' if parabola is not None else 'NOT FIT (need >=3 points)'}")

    # Parabola 2 — release -> last detected ball (extends the trajectory arc
    # all the way to wherever we last saw the ball, regardless of bounce).
    parabola2 = None
    last_real = None
    if release is not None:
        reals = [p for p in traj if not p.from_pred]
        if reals:
            last_real = reals[-1]
            if last_real.frame > release.frame:
                parabola2 = fit_parabola(traj, release, last_real)
    print(f"[PARABOLA2] {'fit release->f=' + str(last_real.frame) if parabola2 else 'NOT FIT'}")

    bake(args.video, args.out, traj, release, bounce, deviation, parabola,
         parabola2=parabola2, last_real=last_real)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
