import os
import cv2
import numpy as np

# ── Cricket ball image overlay ────────────────────────────────────────────────
_BALL_IMG_CACHE: dict = {}
_BALL_IMG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cricket_ball.png")


def overlay_ball_image(frame: np.ndarray, cx: int, cy: int, size: int = 16) -> np.ndarray:
    """Overlay the cricket ball image centred at (cx, cy) with given pixel diameter."""
    key = size
    if key not in _BALL_IMG_CACHE:
        ball = cv2.imread(_BALL_IMG_PATH)
        if ball is None:
            return frame

        # Crop the source image to a tight square around the ball (no stretching)
        gray_src = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_src, 20, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh)
        if coords is not None:
            rx, ry, rw, rh = cv2.boundingRect(coords)
            side = max(rw, rh)
            bcx, bcy = rx + rw // 2, ry + rh // 2
            # Pad into a centred square so the ball stays in the middle even
            # when the content sits near the edge of the source image.
            out = np.zeros((side, side, ball.shape[2]), dtype=ball.dtype)
            sx0, sy0 = bcx - side // 2, bcy - side // 2
            ix0, iy0 = max(0, sx0), max(0, sy0)
            ix1, iy1 = min(ball.shape[1], sx0 + side), min(ball.shape[0], sy0 + side)
            dx0, dy0 = ix0 - sx0, iy0 - sy0
            out[dy0:dy0 + (iy1 - iy0), dx0:dx0 + (ix1 - ix0)] = ball[iy0:iy1, ix0:ix1]
            ball = out

        ball = cv2.resize(ball, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ball, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        _BALL_IMG_CACHE[key] = (ball, mask)

    ball_img, mask = _BALL_IMG_CACHE[key]

    fh, fw = frame.shape[:2]
    x1, y1 = cx - size // 2, cy - size // 2
    x2, y2 = x1 + size, y1 + size

    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x2, fw), min(y2, fh)
    if fx1 >= fx2 or fy1 >= fy2:
        return frame

    bx1, by1 = fx1 - x1, fy1 - y1
    bx2, by2 = bx1 + (fx2 - fx1), by1 + (fy2 - fy1)

    roi        = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    ball_roi   = ball_img[by1:by2, bx1:bx2].astype(np.float32)
    mask_f     = mask[by1:by2, bx1:bx2].astype(np.float32)[:, :, np.newaxis] / 255.0

    frame[fy1:fy2, fx1:fx2] = np.clip(ball_roi * mask_f + roi * (1 - mask_f), 0, 255).astype(np.uint8)
    return frame


def letterbox(img, sz=960):
    h, w = img.shape[:2]
    r = min(sz/h, sz/w)
    nw, nh = int(round(w*r)), int(round(h*r))
    dw, dh = (sz-nw)/2, (sz-nh)/2
    
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-.1)), int(round(dh+.1))
    left, right = int(round(dw-.1)), int(round(dw+.1))
    
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, 
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, r, dw, dh

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

def xywh2xyxy(boxes):
    output = np.zeros_like(boxes)
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return output

def nms(boxes, scores, iou_threshold):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        
        order = order[1:][iou <= iou_threshold]
    
    return keep
