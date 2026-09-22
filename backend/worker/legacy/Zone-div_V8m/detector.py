# FIX NOTE (2026-06-12) — ball lost on wide-angle / far-camera videos
# Cause: execution bug, not the model. The YOLOv5 ONNX export already applies
# sigmoid in its Detect head, but this file applied sigmoid AGAIN, squashing
# every score into ~[0.36, 0.48]. Zoomed videos (true ball score 0.6-0.75)
# scraped past the 0.45 threshold; wide videos (true score 0.45-0.7, smaller
# ball) fell just below it and were rejected.
# Fix: sigmoid is now applied only when the output is raw logits (values
# outside [0,1]); measured true scores: ball in flight 0.45-0.75, background
# junk < 0.30 — so CONF_THRESH_BALL is set to 0.40 in config.py.
import numpy as np
import onnxruntime as ort
import cv2
from config import (
    WEIGHTS_PATH, CONF_THRESH_BALL_PRIMARY, CONF_THRESH_BALL_RECOVERY,
    IOU_THRESH, IMGSZ, BALL_REACQUIRE_FRAMES,
)
from utils import letterbox, sigmoid, xywh2xyxy, nms
from filters import is_valid_detection, filter_by_temporal_consistency, tracking_state

def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Model not found: {WEIGHTS_PATH}")
    
    print(f"[INFO] Loading model: {WEIGHTS_PATH.name}")
    
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    
    session = ort.InferenceSession(str(WEIGHTS_PATH), providers=providers)
    print(f"[INFO] Model loaded - providers: {providers}\n")
    
    return session

def detect_objects(session, frame, frame_num):
    """Run detection with strict filtering"""
    orig_h, orig_w = frame.shape[:2]
    
    # Kalman filter prediction
    kf = tracking_state.get("kalman_filter")
    if kf is not None and kf.is_initialized:
        kf.predict()
    
    # Preprocess
    padded_img, ratio, dw, dh = letterbox(frame, IMGSZ)
    blob = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]
    
    # Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw_output = session.run([output_name], {input_name: blob})[0][0]
    
    # YOLOv8 exports channels-first as [4+num_classes, num_dets] with no
    # separate objectness — transpose to [num_dets, 4+num_classes] and treat
    # the class score directly as the confidence. YOLOv5 exports were
    # [num_dets, 5+num_classes] (with objectness in col 4); keep that path.
    if raw_output.shape[0] < raw_output.shape[1]:
        raw_output = raw_output.T
        conf_cols = raw_output[:, 4:]
        if conf_cols.min() < 0.0 or conf_cols.max() > 1.0:
            raw_output[:, 4:] = sigmoid(conf_cols)
        class_probs = raw_output[:, 4:]
        class_ids = class_probs.argmax(axis=1)
        scores = class_probs.max(axis=1)
    else:
        conf_cols = raw_output[:, 4:]
        if conf_cols.min() < 0.0 or conf_cols.max() > 1.0:
            raw_output[:, 4:] = sigmoid(conf_cols)
        objectness = raw_output[:, 4]
        class_probs = raw_output[:, 5:] if raw_output.shape[1] > 5 else np.ones((len(raw_output), 1))
        class_ids = class_probs.argmax(axis=1)
        class_scores = class_probs.max(axis=1)
        scores = objectness * class_scores
    
    # ── Two-stage ball keep (BALL ONLY — bat class is ignored entirely) ──────
    # PRIMARY floor gates unconditionally. RECOVERY floor gates only when
    # Kalman tracking is initialized: the Mahalanobis gate + MAX_POSITION_JUMP
    # then act as the "is this candidate on the ball's path?" filter, so
    # low-confidence detections on-trajectory are kept, off-trajectory ones
    # are rejected.
    kf = tracking_state.get("kalman_filter")
    _kalman_ready = kf is not None and kf.is_initialized
    _floor = CONF_THRESH_BALL_RECOVERY if _kalman_ready else CONF_THRESH_BALL_PRIMARY
    ball_mask = (class_ids == 0) & (scores >= _floor)

    # Optional per-frame diagnostic — enable with TRACKER_DEBUG=1 in worker env.
    # Prints the raw candidate distribution so we can tell whether the ball is
    # missing because the model doesn't see it (top_conf < 0.15) vs. because
    # the threshold rejects a real detection (top_conf ~ primary floor).
    import os as _os_debug
    if _os_debug.environ.get("TRACKER_DEBUG") == "1":
        _ball_scores_all = scores[class_ids == 0]
        _top = float(_ball_scores_all.max()) if _ball_scores_all.size else 0.0
        _kept = int(ball_mask.sum())
        _wh = ""
        _xy = ""
        if _kept > 0:
            _keep_scores = scores[ball_mask]
            _keep_boxes = xywh2xyxy(raw_output[ball_mask][:, :4])
            _best = _keep_scores.argmax()
            _b = _keep_boxes[_best]
            # Convert best-kept box (currently in letterboxed coords) back to
            # original-frame coords so trajectory Y is directly interpretable.
            _bx1 = (_b[0] - dw) / ratio
            _by1 = (_b[1] - dh) / ratio
            _bx2 = (_b[2] - dw) / ratio
            _by2 = (_b[3] - dh) / ratio
            _cx = int((_bx1 + _bx2) / 2)
            _cy = int((_by1 + _by2) / 2)
            _wh = f" wh=({int(_bx2-_bx1)}x{int(_by2-_by1)})"
            _xy = f" xy=({_cx},{_cy})"
        _tag = "REC" if _kalman_ready else "PRI"
        print(f"[DBG] f={frame_num} floor={_tag}({_floor:.2f}) "
              f"ball_raw_top={_top:.3f} kept={_kept}{_wh}{_xy}", flush=True)

    all_detections = []

    if ball_mask.any():
        ball_output = raw_output[ball_mask]
        ball_scores = scores[ball_mask]
        ball_boxes = xywh2xyxy(ball_output[:, :4])

        # NMS for balls
        ball_keep = nms(ball_boxes, ball_scores, IOU_THRESH)
        ball_boxes = ball_boxes[ball_keep]
        ball_scores = ball_scores[ball_keep]

        # Unpad and scale
        ball_boxes[:, [0, 2]] -= dw
        ball_boxes[:, [1, 3]] -= dh
        ball_boxes /= ratio
        ball_boxes[:, [0, 2]] = ball_boxes[:, [0, 2]].clip(0, orig_w)
        ball_boxes[:, [1, 3]] = ball_boxes[:, [1, 3]].clip(0, orig_h)

        for box, score in zip(ball_boxes, ball_scores):
            det = {
                "x1": int(box[0]), "y1": int(box[1]),
                "x2": int(box[2]), "y2": int(box[3]),
                "conf": float(score),
                "class_id": 0,
                "class_name": "ball",
            }
            valid, _ = is_valid_detection(det, "ball")
            if not valid:
                continue
            valid, _ = filter_by_temporal_consistency(det, "ball")
            if valid:
                all_detections.append(det)

    # SELECT BEST BALL (bat class dropped — never emitted)
    final_detections = []
    balls = all_detections
    
    if balls:
        if kf is not None and kf.is_initialized:
            # Pick the ball closest to the Kalman filter predicted position to avoid false positives
            pred_x, pred_y = kf.x[0, 0], kf.x[1, 0]
            best_ball = min(balls, key=lambda b: (
                ((b["x1"] + b["x2"]) / 2.0 - pred_x) ** 2 +
                ((b["y1"] + b["y2"]) / 2.0 - pred_y) ** 2
            ))
        elif tracking_state.get("last_ball_pos") is not None:
            # Fallback to closest to last known position
            last_x, last_y = tracking_state["last_ball_pos"]
            best_ball = min(balls, key=lambda b: (
                ((b["x1"] + b["x2"]) / 2.0 - last_x) ** 2 +
                ((b["y1"] + b["y2"]) / 2.0 - last_y) ** 2
            ))
        else:
            # Fallback to highest confidence if there is no path/tracking state yet
            best_ball = max(balls, key=lambda x: x["conf"])
        cx = (best_ball["x1"] + best_ball["x2"]) // 2
        cy = (best_ball["y1"] + best_ball["y2"]) // 2
        
        if kf is not None:
            if not kf.is_initialized:
                kf.initialize(cx, cy)
                smoothed_x, smoothed_y = cx, cy
            else:
                smoothed_x, smoothed_y, _, _ = kf.update(cx, cy)
            
            # Apply smoothed center to original detection bounding box
            w = best_ball["x2"] - best_ball["x1"]
            h = best_ball["y2"] - best_ball["y1"]
            best_ball["x1"] = int(round(smoothed_x - w / 2))
            best_ball["y1"] = int(round(smoothed_y - h / 2))
            best_ball["x2"] = int(round(smoothed_x + w / 2))
            best_ball["y2"] = int(round(smoothed_y + h / 2))
            
            cx = (best_ball["x1"] + best_ball["x2"]) // 2
            cy = (best_ball["y1"] + best_ball["y2"]) // 2
            
        final_detections.append(best_ball)
        tracking_state["last_ball_pos"] = (cx, cy)
        tracking_state["ball_misses"] = 0
    else:
        # Handle occlusion gap
        if kf is not None and kf.is_initialized:
            kf.handle_gap()
        # Re-acquisition: after a long run of missed frames the stored
        # last_ball_pos is stale (it may even be a false positive), and the
        # MAX_POSITION_JUMP check would reject the real ball indefinitely.
        # Drop the stale state so the next confident ball is accepted fresh.
        tracking_state["ball_misses"] += 1
        if tracking_state["ball_misses"] > BALL_REACQUIRE_FRAMES:
            tracking_state["last_ball_pos"] = None
            if kf is not None:
                kf.reset()
    
    return final_detections
