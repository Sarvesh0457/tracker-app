"""
RELEASE DETECTOR (sliding-window)
==================================
Rule: confirm release once the ball has been detected in at least
MIN_DETECTIONS_IN_WINDOW frames within a sliding window of WINDOW_FRAMES.
The release frame is the *first* detection inside that window.

A sliding window tolerates intermittent detection (1–2 dropped frames in a
row) far better than a "consecutive-run" counter, which resets on any miss.
That matters for wide-angle / small-ball footage where the detector genuinely
skips frames even when it's finding the ball reliably overall.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

from config import RELEASE_DISPLAY_FRAMES, RELEASE_COLOR, RELEASE_MARKER_RADIUS
from pose_detector import bowl_stump_top_y

# ── Tuneable constants ───────────────────────────────────────────────────────
WINDOW_FRAMES              = 6   # sliding-window length in frames
MIN_DETECTIONS_IN_WINDOW   = 3   # minimum ball hits within the window to confirm

# Shoulder-anchored release-search box (image px). Ball centre must sit inside:
#   |cx - shoulder_x| <= BALL_BOX_HALF_W
#   shoulder_y - BALL_BOX_ABOVE <= cy <= shoulder_y + BALL_BOX_BELOW
# Symmetric in x so it works for both right- and left-arm bowlers.
BALL_BOX_HALF_W  = 60   # ±60 px horizontally → 120 px wide
BALL_BOX_ABOVE   = 120  # px above shoulder (upper edge)
BALL_BOX_BELOW   = 20   # px below shoulder (lower edge)


@dataclass
class ReleasePoint:
    release_frame: int   # index of the first ball-detection frame in the window
    x: int               # ball centre-x at that frame
    y: int               # ball centre-y at that frame


class ReleaseDetector:
    """
    Watches the ball detection stream frame-by-frame.
    Maintains a sliding window of the last WINDOW_FRAMES frames; when the
    number of ball-present frames inside that window first reaches
    MIN_DETECTIONS_IN_WINDOW, it locks in the earliest hit as the release.
    """

    def __init__(
        self,
        window_frames: int = WINDOW_FRAMES,
        min_detections: int = MIN_DETECTIONS_IN_WINDOW,
        calib_markers: Optional[dict] = None,
    ):
        self.window_frames = window_frames
        self.min_detections = min_detections
        self.calib_markers = calib_markers

        # Each entry: (frame_num, (cx, cy)) if ball present that frame, else None
        self._window: deque = deque(maxlen=window_frames)

        self.release: Optional[ReleasePoint] = None     # locked once confirmed
        self.last_wrist_xy: Optional[tuple[float, float]] = None
        self.last_wrist_frame: int = -999
        self.last_shoulder_xy: Optional[tuple[float, float]] = None
        self.last_shoulder_frame: int = -999

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        detections: list,
        frame_num: int,
        pose_confirmed: bool = True,
        wrist_xy: Optional[tuple[float, float]] = None,
        shoulder_xy: Optional[tuple[float, float]] = None,
        hand_raise_frame: Optional[int] = None,
    ) -> Optional[ReleasePoint]:
        """
        Call once per frame with the list of detections for that frame.
        Returns the ReleasePoint the moment it is confirmed (only fires once),
        or None every other frame.

        `pose_confirmed` gates release: the sliding window is not filled while
        it is False, so no release can be locked in before the bowler pose has
        been detected.
        """
        if self.release is not None:
            return None
        if not pose_confirmed:
            return None

        if hand_raise_frame is not None and frame_num < hand_raise_frame:
            return None

        # Keep track of bowler's raised hand wrist coordinates
        if wrist_xy is not None:
            self.last_wrist_xy = wrist_xy
            self.last_wrist_frame = frame_num

        if shoulder_xy is not None:
            self.last_shoulder_xy = shoulder_xy
            self.last_shoulder_frame = frame_num

        ball = self._get_ball(detections)
        if ball is not None:
            cx = (ball["x1"] + ball["x2"]) // 2
            cy = (ball["y1"] + ball["y2"]) // 2

            # Constraint: release must be generally on the bowler's side AND above the TOP of the bowling stumps
            if self.calib_markers is not None:
                bowl_l = self.calib_markers.get("bowl_L")
                bat_l = self.calib_markers.get("bat_L")
                bowl_r = self.calib_markers.get("bowl_R")
                bat_r = self.calib_markers.get("bat_R")

                # Stump-top gate — physical top of the bowling stumps, estimated
                # from pitch geometry. Falls back to the crease/base line if
                # stump markers are missing.
                stump_top_y = bowl_stump_top_y(self.calib_markers)
                if stump_top_y is not None and cy >= stump_top_y:
                    self._window.append(None)
                    return None

                if bowl_l and bat_l:
                    bowl_y = bowl_l[1]

                    # 1. Fallback stump gate when bowl/bat stumps missing — use crease base.
                    if stump_top_y is None and cy >= bowl_y:
                        self._window.append(None)
                        return None

                # 2. (Pitch-half cutoff removed — the shoulder-anchored box below
                #    already localises the release moment, and the pitch-half
                #    check wrongly rejects airborne balls near the bowler.)

                # 3. Must be between the left and right return creases (interpolated at cy)
                if all([bowl_l, bat_l, bowl_r, bat_r]):
                    l_x1, l_y1 = bowl_l
                    l_x2, l_y2 = bat_l
                    r_x1, r_y1 = bowl_r
                    r_x2, r_y2 = bat_r

                    # Interpolate left and right boundaries at the y coordinate of the release candidate (cy)
                    # Note: Since the ball is in the air (cy < bowl_y), this extrapolates the pitch lines 
                    # accurately matching the camera's perspective.
                    left_bound_x = l_x1 + (l_x2 - l_x1) * (cy - l_y1) / (l_y2 - l_y1)
                    right_bound_x = r_x1 + (r_x2 - r_x1) * (cy - r_y1) / (r_y2 - r_y1)

                    if left_bound_x > right_bound_x:
                        left_bound_x, right_bound_x = right_bound_x, left_bound_x

                    # Allow a generous margin (e.g. 50px) because the bowler's arm can extend outside the return crease
                    margin = 50
                    if not (left_bound_x - margin <= cx <= right_bound_x + margin):
                        self._window.append(None)
                        return None

            # Constraint: ball must sit inside a box anchored on the bowler's
            # shoulder — ±BALL_BOX_HALF_W px in x, and no more than
            # BALL_BOX_BELOW px below the shoulder in y (unbounded above).
            # Uses the current-frame shoulder when available, else the most
            # recent one within 5 frames.
            active_shoulder = None
            if shoulder_xy is not None:
                active_shoulder = shoulder_xy
            elif self.last_shoulder_xy is not None and (frame_num - self.last_shoulder_frame) <= 5:
                active_shoulder = self.last_shoulder_xy

            if active_shoulder is not None:
                sx, sy = float(active_shoulder[0]), float(active_shoulder[1])
                if abs(cx - sx) > BALL_BOX_HALF_W:
                    self._window.append(None)
                    return None
                if cy > sy + BALL_BOX_BELOW or cy < sy - BALL_BOX_ABOVE:
                    self._window.append(None)
                    return None

            self._window.append((frame_num, (cx, cy)))
        else:
            self._window.append(None)

        hits = [e for e in self._window if e is not None]
        if len(hits) >= self.min_detections:
            first_frame, (first_x, first_y) = hits[0]
            self.release = ReleasePoint(
                release_frame=first_frame,
                x=first_x,
                y=first_y,
            )
            print(
                f"[RELEASE] Confirmed at frame {self.release.release_frame} "
                f"pos=({self.release.x},{self.release.y}) "
                f"({len(hits)}/{self.window_frames} in sliding window)"
            )
            return self.release

        return None

    def draw(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        """
        Draw the release marker on the frame for RELEASE_DISPLAY_FRAMES frames
        after the release point was confirmed.
        """
        if self.release is None:
            return frame

        from utils import overlay_ball_image
        cx, cy = self.release.x, self.release.y
        overlay_ball_image(frame, cx + 2, cy, 17)
        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_ball(detections: list) -> Optional[dict]:
        """Return the first ball detection, or None."""
        for d in detections:
            if d.get("class_name") == "ball":
                return d
        return None
