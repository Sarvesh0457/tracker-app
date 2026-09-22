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

# ── Tuneable constants ───────────────────────────────────────────────────────
WINDOW_FRAMES              = 6   # sliding-window length in frames
MIN_DETECTIONS_IN_WINDOW   = 3   # minimum ball hits within the window to confirm


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
    ):
        self.window_frames = window_frames
        self.min_detections = min_detections

        # Each entry: (frame_num, (cx, cy)) if ball present that frame, else None
        self._window: deque = deque(maxlen=window_frames)

        self.release: Optional[ReleasePoint] = None     # locked once confirmed

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, detections: list, frame_num: int) -> Optional[ReleasePoint]:
        """
        Call once per frame with the list of detections for that frame.
        Returns the ReleasePoint the moment it is confirmed (only fires once),
        or None every other frame.
        """
        if self.release is not None:
            return None

        ball = self._get_ball(detections)
        if ball is not None:
            cx = (ball["x1"] + ball["x2"]) // 2
            cy = (ball["y1"] + ball["y2"]) // 2
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
