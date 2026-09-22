"""Manual 8-point stump keypoint picker.

Usage:
    python -m backend.pipeline.reconstruct3d.keypoints \
        --video path/to/clip.mp4 \
        --out   path/to/keypoints.json \
        [--frame 0] [--extend]

Opens an OpenCV window on the chosen frame. Click the 8 points in the on-screen
order. Left-click adds a point, right-click / 'u' undoes the last, 'r' resets,
Enter saves and exits, Esc quits without saving.

--extend: if --out already exists with 6 points from a previous session, load
those and only prompt for the 2 new points (stump tops).

Output JSON:
{
  "video": "clip.mp4",
  "frame_index": 0,
  "image_size": [width, height],
  "points": [ {"name": "popping_striker_leg", "uv": [u, v]}, ... ]  # 8 entries
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .world_points import KEYPOINT_NAMES


PROMPT_HELP = [
    "STRIKER end - popping crease LEFT corner   (as seen on screen)",
    "STRIKER end - popping crease RIGHT corner  (as seen on screen)",
    "NON-STRIKER end - popping crease LEFT corner",
    "NON-STRIKER end - popping crease RIGHT corner",
    "STRIKER end - MIDDLE STUMP BASE",
    "NON-STRIKER end - MIDDLE STUMP BASE",
]
N_POINTS = len(PROMPT_HELP)

WINDOW_NAME = "stump keypoints — L-click add, R-click undo, 'r' reset, Enter save, Esc quit"


def _extract_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= total:
        cap.release()
        raise RuntimeError(f"frame_index {frame_index} out of range [0, {total})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame {frame_index}")
    return frame


WINDOW_NAME2 = "stump keypoints — L-click add, R-click undo, 'r' reset all, 1..8 re-do that point, Enter save, Esc quit"


def _draw(base_img, points, redo_idx):
    img = base_img.copy()
    h, w = img.shape[:2]
    banner_h = 60
    cv2.rectangle(img, (0, 0), (w, banner_h), (0, 0, 0), -1)

    if redo_idx is not None:
        msg = f"[REDO {redo_idx + 1}/{N_POINTS}] {PROMPT_HELP[redo_idx]}"
        color = (0, 200, 255)
    else:
        n_placed = sum(1 for p in points if p is not None)
        if n_placed < N_POINTS:
            # next empty slot in order
            next_i = next(i for i, p in enumerate(points) if p is None)
            msg = f"[{next_i + 1}/{N_POINTS}] {PROMPT_HELP[next_i]}"
            color = (0, 255, 255)
        else:
            msg = f"All {N_POINTS} marked. Enter=save   1..8=redo one   r=reset all   Esc=quit"
            color = (0, 255, 0)
    cv2.putText(img, msg, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    for i, uv in enumerate(points):
        if uv is None:
            continue
        u, v = uv
        is_redo = (redo_idx == i)
        ring_col = (0, 200, 255) if is_redo else (255, 255, 255)
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.circle(img, (u, v), 8, ring_col, 2 if is_redo else 1)
        cv2.putText(img, f"{i+1}:{KEYPOINT_NAMES[i]}", (u + 10, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def pick_keypoints(video_path: Path, frame_index: int = 0, seed_points=None):
    frame = _extract_frame(video_path, frame_index)
    h, w = frame.shape[:2]
    # Slot-based storage: fixed list of N_POINTS entries, each either (u,v) or None.
    points: list = [None] * N_POINTS
    if seed_points:
        for i, uv in enumerate(seed_points[:N_POINTS]):
            points[i] = (int(uv[0]), int(uv[1]))
    redo_slot: list = [None]     # boxed so callback can mutate

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if redo_slot[0] is not None:
                points[redo_slot[0]] = (int(x), int(y))
                redo_slot[0] = None
            else:
                # first empty slot
                for i in range(N_POINTS):
                    if points[i] is None:
                        points[i] = (int(x), int(y))
                        return
        elif event == cv2.EVENT_RBUTTONDOWN:
            # undo the most recently filled slot
            for i in range(N_POINTS - 1, -1, -1):
                if points[i] is not None:
                    points[i] = None
                    return

    cv2.namedWindow(WINDOW_NAME2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME2, on_mouse)

    saved = False
    while True:
        cv2.imshow(WINDOW_NAME2, _draw(frame, points, redo_slot[0]))
        key = cv2.waitKey(20) & 0xFFFF
        if key == 27:                                            # Esc
            break
        if key in (ord('r'), ord('R')):                          # reset all
            for i in range(N_POINTS): points[i] = None
            redo_slot[0] = None
        elif key in (ord('u'), ord('U')):                        # undo last filled
            for i in range(N_POINTS - 1, -1, -1):
                if points[i] is not None:
                    points[i] = None
                    break
        elif ord('1') <= key <= ord('0') + N_POINTS:             # redo point 1..N
            redo_slot[0] = key - ord('1')
        elif key in (13, 10):                                    # Enter
            if all(p is not None for p in points) and redo_slot[0] is None:
                saved = True
                break

    cv2.destroyWindow(WINDOW_NAME2)

    if not saved:
        return None

    return {
        "video": Path(video_path).name,
        "frame_index": frame_index,
        "image_size": [w, h],
        "points": [
            {"name": KEYPOINT_NAMES[i], "uv": [points[i][0], points[i][1]]}
            for i in range(N_POINTS)
        ],
    }


def _load_seed(out_path: Path):
    """Load existing 6- or 8-point JSON, return list[(u,v)] in order matching
    KEYPOINT_NAMES prefix. Silently returns None if the file is missing or malformed."""
    if not out_path.exists():
        return None
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
        by_name = {p["name"]: tuple(p["uv"]) for p in data["points"]}
        seed = []
        for name in KEYPOINT_NAMES:
            if name not in by_name:
                break
            seed.append(by_name[name])
        return seed if seed else None
    except Exception:
        return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--extend", action="store_true",
                   help="preload the existing 6 points from --out and prompt only for the new ones")
    args = p.parse_args(argv)

    if not args.video.exists():
        print(f"video not found: {args.video}", file=sys.stderr)
        return 2

    seed = _load_seed(args.out) if args.extend else None
    if args.extend and seed:
        print(f"loaded {len(seed)} existing keypoints from {args.out} — click the remaining {N_POINTS - len(seed)}")

    result = pick_keypoints(args.video, args.frame, seed_points=seed)
    if result is None:
        print("cancelled — no output written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
