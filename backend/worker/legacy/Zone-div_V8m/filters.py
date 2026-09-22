import numpy as np
from kalman_filter import KalmanBallFilter
from config import (
    MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MIN_BOX_AREA, MAX_BOX_AREA,
    MAX_BOX_WIDTH_BALL, MAX_BOX_HEIGHT_BALL, MIN_ASPECT_RATIO_BALL, MAX_ASPECT_RATIO_BALL,
    MAX_BOX_WIDTH_BAT, MAX_BOX_HEIGHT_BAT, MIN_ASPECT_RATIO_BAT, MAX_ASPECT_RATIO_BAT,
    MAX_POSITION_JUMP, GRAVITY_PX_PER_FRAME2
)

# Shared tracking states for temporal consistency
tracking_state = {
    "last_ball_pos": None,
    "last_bat_positions": [],
    "ball_misses": 0,  # consecutive frames without an accepted ball
    "kalman_filter": KalmanBallFilter(gravity=GRAVITY_PX_PER_FRAME2)
}

def reset_filters():
    """Resets temporal consistency state."""
    tracking_state["last_ball_pos"] = None
    tracking_state["last_bat_positions"] = []
    tracking_state["ball_misses"] = 0
    if tracking_state.get("kalman_filter") is not None:
        tracking_state["kalman_filter"].reset()

def is_valid_detection(det, class_name):
    """Apply geometric and confidence filters"""
    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
    w = x2 - x1
    h = y2 - y1
    area = w * h
    
    # Basic size checks
    if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
        return False, f"too small ({w}x{h})"
    
    if area < MIN_BOX_AREA:
        return False, f"area too small ({area}px²)"
    
    if area > MAX_BOX_AREA:
        return False, f"area too large ({area}px²)"
    
    # Calculate aspect ratio
    aspect_ratio = w / h if h > 0 else 0
    
    # Class-specific filters
    if class_name == "ball":
        if w > MAX_BOX_WIDTH_BALL or h > MAX_BOX_HEIGHT_BALL:
            return False, f"ball too large ({w}x{h})"
        
        if aspect_ratio < MIN_ASPECT_RATIO_BALL or aspect_ratio > MAX_ASPECT_RATIO_BALL:
            return False, f"ball aspect ratio {aspect_ratio:.2f} out of range"
    
    elif class_name == "bat":
        if w > MAX_BOX_WIDTH_BAT or h > MAX_BOX_HEIGHT_BAT:
            return False, f"bat too large ({w}x{h})"
        
        if aspect_ratio < MIN_ASPECT_RATIO_BAT or aspect_ratio > MAX_ASPECT_RATIO_BAT:
            return False, f"bat aspect ratio {aspect_ratio:.2f} out of range"
    
    return True, "valid"

def filter_by_temporal_consistency(det, class_name):
    """Filter based on previous frame position"""
    cx = (det["x1"] + det["x2"]) // 2
    cy = (det["y1"] + det["y2"]) // 2
    
    if class_name == "ball":
        kf = tracking_state.get("kalman_filter")
        if kf is not None and kf.is_initialized:
            is_valid, d2 = kf.validate_detection(cx, cy)
            if not is_valid:
                return False, f"Kalman gate rejected: Mahalanobis d² = {d2:.2f} (max {kf.gate_threshold})"
        
        if tracking_state["last_ball_pos"] is not None:
            dist = np.sqrt((cx - tracking_state["last_ball_pos"][0])**2 + (cy - tracking_state["last_ball_pos"][1])**2)
            if dist > MAX_POSITION_JUMP:
                return False, f"jumped {dist:.0f}px (max {MAX_POSITION_JUMP})"
        return True, "valid"
    
    elif class_name == "bat":
        # For bat, check it's not too far from any previous bat
        if tracking_state["last_bat_positions"]:
            min_dist = min([np.sqrt((cx - bx)**2 + (cy - by)**2) 
                           for bx, by in tracking_state["last_bat_positions"]])
            if min_dist > MAX_POSITION_JUMP * 1.5:  # Bats can move more
                return False, f"jumped {min_dist:.0f}px"
        return True, "valid"
