"""Isolate the delivery window from a noisy detections.json.

A real delivery has three signatures in 2D:
  - length: ~10-30 frames at 25-30 fps (release to slightly-past bounce)
  - directional motion: consecutive pixel velocities point in similar direction
  - large travel: total pixel distance from first to last point is a big fraction
    of the sum of frame-to-frame steps (i.e., not a wobble around one spot)

Score each candidate run and pick the best. Emits:
  - <video>_delivery_window.json: {first_frame, last_frame, detection_indices, score}
  - <video>_delivery_check.png: composite image drawing all detections and the
    picked delivery in a different color, so the pick can be eyeballed.

Usage:
    python -m backend.pipeline.reconstruct3d.isolate_delivery \
        --video       path/to/clip.mp4 \
        --detections  path/to/detections.json \
        --out-window  path/to/delivery_window.json \
        --out-check   path/to/delivery_check.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


MIN_WIN_LEN = 8        # frames — smallest plausible delivery window
MAX_WIN_LEN = 30       # frames — largest plausible delivery window
MIN_AVG_STEP_PX = 6.0  # frames slower than this per step are ball-at-rest


def _score_window(win):
    """Higher is better. Zero if disqualified."""
    n = len(win)
    if n < MIN_WIN_LEN:
        return 0.0, {}

    uvs = np.array([[d["u"], d["v"]] for d in win], dtype=np.float64)
    steps = np.diff(uvs, axis=0)
    step_lens = np.linalg.norm(steps, axis=1)
    total_step = float(step_lens.sum())
    endpoint_dist = float(np.linalg.norm(uvs[-1] - uvs[0]))
    avg_step = float(step_lens.mean())

    if total_step < 1e-6 or avg_step < MIN_AVG_STEP_PX:
        return 0.0, {}

    straightness = endpoint_dist / total_step

    if len(steps) >= 2:
        norms = np.linalg.norm(steps, axis=1, keepdims=True) + 1e-9
        unit = steps / norms
        cos_sims = np.sum(unit[:-1] * unit[1:], axis=1)
        coherence = float(np.clip(cos_sims.mean(), -1, 1))
    else:
        coherence = 0.0

    mean_conf = float(np.mean([d["conf"] for d in win]))

    score = (
        straightness * 2.0
        + max(coherence, 0.0) * 1.5
        + mean_conf
        + math.log1p(avg_step) * 0.5
    )
    meta = {
        "n": n,
        "endpoint_dist_px": round(endpoint_dist, 1),
        "total_step_px": round(total_step, 1),
        "straightness": round(straightness, 3),
        "coherence": round(coherence, 3),
        "mean_conf": round(mean_conf, 3),
        "avg_step_px": round(avg_step, 1),
    }
    return score, meta


def _enumerate_windows(dets):
    """Sliding contiguous-index windows of length MIN..MAX over the sorted
    detections list. We slide by DETECTION INDEX, not frame number, so a big
    frame gap inside a window is fine as long as the window contains enough
    consecutive detections."""
    windows = []
    for start in range(len(dets)):
        for length in range(MIN_WIN_LEN, MAX_WIN_LEN + 1):
            end = start + length
            if end > len(dets):
                break
            windows.append(dets[start:end])
    return windows


def _draw_check(video_path, all_dets, chosen_run, chosen_frame_idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("failed to read reference frame for check image")

    out = frame.copy()
    # all detections in dim red
    for d in all_dets:
        cv2.circle(out, (int(d["u"]), int(d["v"])), 3, (60, 60, 200), 1)

    # chosen delivery in green with numbers
    prev = None
    for i, d in enumerate(chosen_run):
        pt = (int(d["u"]), int(d["v"]))
        cv2.circle(out, pt, 5, (0, 255, 0), -1)
        if i % 3 == 0:
            cv2.putText(out, str(d["frame"]), (pt[0] + 6, pt[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        if prev is not None:
            cv2.line(out, prev, pt, (0, 255, 0), 1)
        prev = pt

    cv2.rectangle(out, (0, 0), (out.shape[1], 60), (0, 0, 0), -1)
    cv2.putText(out,
                f"green = delivery ({chosen_run[0]['frame']}-{chosen_run[-1]['frame']}, "
                f"{len(chosen_run)} pts)   dim red = all other detections",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video",      required=True, type=Path)
    p.add_argument("--detections", required=True, type=Path)
    p.add_argument("--out-window", required=True, type=Path)
    p.add_argument("--out-check",  required=True, type=Path)
    args = p.parse_args(argv)

    data = json.loads(args.detections.read_text(encoding="utf-8"))
    dets = sorted(data["detections"], key=lambda d: d["frame"])

    windows = _enumerate_windows(dets)
    scored = [(w, *_score_window(w)) for w in windows]
    scored = [(w, s, m) for (w, s, m) in scored if s > 0.0]
    if not scored:
        print("no plausible delivery window found — all windows too short/slow/degenerate", file=sys.stderr)
        return 1
    scored.sort(key=lambda x: -x[1])

    print("Top 5 candidate windows (score, frames, meta):")
    for w, s, m in scored[:5]:
        print(f"  score={s:5.2f}  frames={w[0]['frame']:3d}..{w[-1]['frame']:3d} "
              f"(n={m['n']:2d})  straight={m['straightness']:.2f}  "
              f"coh={m['coherence']:+.2f}  conf={m['mean_conf']:.2f}  "
              f"step={m['avg_step_px']:.1f}px")

    best_run, best_score, best_meta = scored[0]
    window = {
        "video": data.get("video"),
        "first_frame": best_run[0]["frame"],
        "last_frame": best_run[-1]["frame"],
        "num_detections": len(best_run),
        "score": best_score,
        "metrics": best_meta,
        "detections": best_run,
    }
    args.out_window.parent.mkdir(parents=True, exist_ok=True)
    args.out_window.write_text(json.dumps(window, indent=2), encoding="utf-8")

    check_frame_idx = best_run[len(best_run) // 2]["frame"]
    check_img = _draw_check(args.video, dets, best_run, check_frame_idx)
    args.out_check.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_check), check_img)

    print(f"\npicked: frames {window['first_frame']}-{window['last_frame']} ({window['num_detections']} points)")
    print(f"wrote  window: {args.out_window}")
    print(f"wrote  check:  {args.out_check}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
