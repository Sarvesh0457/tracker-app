"""
KALMAN FILTER MODULE
====================
Provides a mathematically optimal estimator for the cricket ball's true position
and velocity. By incorporating a Constant Velocity (CV) physical transition model
with gravity as a control input, it reduces frame-to-frame pixel-level detection noise
and provides a clean velocity/jerk signal for downstream detectors.

Includes Mahalanobis distance gating to reject erratic detections.
"""

import numpy as np
from typing import Optional, Tuple

class KalmanBallFilter:
    """
    A 2D Kalman Filter tracking the state vector:
        x = [x, y, vx, vy]^T
    Measurements:
        z = [x, y]^T
    Control input:
        u = [g] (downward gravity acceleration)
    """

    def __init__(
        self,
        gravity: float = 0.35,
        process_noise_std: float = 8.0,    # Increased: ball moves fast, so model uncertainty must be wider
        measurement_noise_std: float = 8.0, # Increased: detector is noisy on a fast ball
        gate_threshold: float = 50.0,       # Permissive gate; chi-sq 99% (9.21) was rejecting real balls
        warmup_frames: int = 5,             # Skip gating for first N updates after initialization
    ):
        self.g = gravity
        self.gate_threshold = gate_threshold
        self.warmup_frames = warmup_frames
        self._update_count = 0  # counts successful updates since last init

        # Time step (dt = 1 frame)
        self.dt = 1.0

        # State transition matrix (F)
        self.F = np.array([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

        # Control input matrix (B)
        # y = y_prev + vy * dt + 0.5 * g * dt^2
        # vy = vy_prev + g * dt
        self.B = np.array([
            [0.0],
            [0.5 * (self.dt ** 2)],
            [0.0],
            [self.dt]
        ], dtype=float)

        # Measurement matrix (H) - we only measure x and y coordinates
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=float)

        # Measurement noise covariance (R)
        r_var = measurement_noise_std ** 2
        self.R = np.eye(2, dtype=float) * r_var

        # Process noise covariance (Q)
        # We assume standard white noise acceleration model for x and y separately
        q_var = process_noise_std ** 2
        dt3 = (self.dt ** 3) / 3.0
        dt2 = (self.dt ** 2) / 2.0
        dt1 = self.dt
        self.Q = np.array([
            [dt3 * q_var, 0.0, dt2 * q_var, 0.0],
            [0.0, dt3 * q_var, 0.0, dt2 * q_var],
            [dt2 * q_var, 0.0, dt1 * q_var, 0.0],
            [0.0, dt2 * q_var, 0.0, dt1 * q_var]
        ], dtype=float)

        # Filter state
        self.x = np.zeros((4, 1), dtype=float)  # State estimate [x, y, vx, vy]^T
        self.P = np.eye(4, dtype=float) * 1000.0  # State covariance (high initial uncertainty)
        
        self.is_initialized = False
        self.consecutive_gaps = 0
        self._update_count = 0

    def initialize(self, cx: float, cy: float):
        """Initialize filter state on the first detection."""
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=float)
        # Initial covariance: moderate uncertainty for position, higher for velocity
        self.P = np.diag([50.0, 50.0, 500.0, 500.0]).astype(float)
        self.is_initialized = True
        self.consecutive_gaps = 0
        self._update_count = 0  # reset warm-up counter on re-init

    def predict(self) -> Tuple[float, float]:
        """
        Predict step: project the state and covariance ahead.
        Returns the predicted (x, y) coordinates.
        """
        if not self.is_initialized:
            return 0.0, 0.0

        # Control input: gravity (downward)
        u = np.array([[self.g]], dtype=float)

        # x = F*x + B*u
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)

        # P = F*P*F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def validate_detection(self, cx: float, cy: float) -> Tuple[bool, float]:
        """
        Check if detection (cx, cy) is a valid candidate using Mahalanobis Gating.
        Returns (is_valid, mahalanobis_distance_squared).
        """
        if not self.is_initialized:
            return True, 0.0

        # During warm-up, always accept detections so the filter can build
        # a reliable velocity estimate before gating kicks in.
        if self._update_count < self.warmup_frames:
            return True, 0.0

        z = np.array([[cx], [cy]], dtype=float)
        
        # Innovation: y = z - H*x
        y_innov = z - np.dot(self.H, self.x)
        
        # Innovation covariance: S = H*P*H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Mahalanobis distance squared: d2 = y_innov^T * S^-1 * y_innov
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(np.dot(np.dot(y_innov.T, S_inv), y_innov).item())
        except np.linalg.LinAlgError:
            # Fallback to standard Euclidean gating if inversion fails
            dist = np.hypot(y_innov[0, 0], y_innov[1, 0])
            d2 = (dist / 10.0) ** 2  # arbitrary scaling

        is_valid = d2 <= self.gate_threshold
        return is_valid, d2

    def update(self, cx: float, cy: float) -> Tuple[float, float, float, float]:
        """
        Update state with a new measurement (cx, cy).
        Returns the smoothed (x, y, vx, vy).
        """
        if not self.is_initialized:
            self.initialize(cx, cy)
            return cx, cy, 0.0, 0.0

        z = np.array([[cx], [cy]], dtype=float)
        
        # Innovation: y = z - H*x
        y_innov = z - np.dot(self.H, self.x)
        
        # Innovation covariance: S = H*P*H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        
        # Kalman Gain: K = P * H^T * S^-1
        try:
            S_inv = np.linalg.inv(S)
            K = np.dot(np.dot(self.P, self.H.T), S_inv)
        except np.linalg.LinAlgError:
            # Standard fallback (e.g. constant small gain)
            K = np.zeros((4, 2), dtype=float)
            K[0, 0] = 0.5
            K[1, 1] = 0.5

        # Update state: x = x + K*y
        self.x = self.x + np.dot(K, y_innov)
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(4, dtype=float)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        
        self.consecutive_gaps = 0
        self._update_count += 1

        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        )

    def handle_gap(self):
        """Increment occlusion counter if the ball is not detected in this frame."""
        self.consecutive_gaps += 1
        # If tracking is lost for too long, reset so the filter can re-initialize cleanly.
        # Raised from 25 → 40 frames to be more tolerant of occlusion.
        if self.consecutive_gaps > 40:
            self.reset()

    def reset(self):
        """Reset filter states entirely."""
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float) * 1000.0
        self.is_initialized = False
        self.consecutive_gaps = 0
        self._update_count = 0

    @property
    def current_state(self) -> Tuple[float, float, float, float]:
        """Return the current smoothed (x, y, vx, vy) of the filter."""
        return (
            float(self.x[0, 0]),
            float(self.x[1, 0]),
            float(self.x[2, 0]),
            float(self.x[3, 0])
        )
