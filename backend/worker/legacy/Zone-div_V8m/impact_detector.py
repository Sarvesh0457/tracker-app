"""
IMPACT POINT DETECTOR — DISABLED
================================
All impact-detection logic has been removed. This stub keeps the class
and its public methods so the rest of the pipeline (main_V2.1.py,
visualizer, trajectory, logger) continues to import and call into it
without changes. No impact is ever locked; `self.impact` stays None
for the entire run.

Reinstate detection by restoring the previous parabola-deviation /
direction-change / vanish-fallback implementation from git history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ImpactPoint:
    frame: int
    x:     int
    y:     int
    reason: str


class ImpactDetector:
    """No-op impact detector. Always reports no impact."""

    def __init__(self, *args, **kwargs):
        self.impact: Optional[ImpactPoint] = None

    # ── Public API (signatures preserved) ──────────────────────────────────────

    def set_grid_corners(self, corners: Optional[dict]) -> None:
        return None

    def update(
        self,
        detections: list,
        frame_num: int,
        bounce_point: Optional[tuple[int, int, int]],
    ) -> Optional[ImpactPoint]:
        return None

    def finalize(self) -> Optional[ImpactPoint]:
        return None

    def last_known_fallback(self) -> Optional[ImpactPoint]:
        return None

    def draw(self, frame: np.ndarray, frame_num: int) -> np.ndarray:
        return frame
