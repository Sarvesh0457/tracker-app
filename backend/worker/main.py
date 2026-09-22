"""Worker service — the 'pipeline runner'.

GPU Cloud Run service. Wakes up when Cloud Tasks POSTs to /process with a
job_id. Pulls the job's input.mp4 from GCS, runs the pipeline, uploads
artifacts, updates Firestore.

The legacy tracker modules at TRACKER_LEGACY_DIR_{WHITE,RED} hold
module-level state, so per-job process isolation matters. On Cloud Run we
get that automatically: each container instance handles concurrency=1, and
once it returns the response Cloud Run can recycle the instance. We don't
need the subprocess hop the original jobs.py used.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

# Mount the existing backend modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.errors import PipelineError  # noqa: E402
from storage import make_storage  # noqa: E402

from api import firestore_jobs  # noqa: E402 — share the Firestore schema

# NOTE: `pipeline.process_video` is imported LAZILY inside _run_pipeline below.
# pipeline/__init__.py binds TRACKER_LEGACY_DIR onto sys.path at import time,
# and once the legacy detector/filters/tracker modules are loaded they cache
# in sys.modules — so an early top-level import would lock the worker to
# whichever folder was on disk when the container started (always white via
# the fallback). Per-job we evict and re-import to honor ball_type.


# ─────────────────────────────────────────────────────────────────────────────
# Per-ball-type model + legacy folder. Container paths, baked in at build.
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))
LEGACY_ROOT = Path(os.environ.get("LEGACY_ROOT", "/app/legacy"))

BALL_PIPELINES = {
    "white": {
        "weights": Path(os.environ.get(
            "TRACKER_WEIGHTS_WHITE", MODELS_DIR / "white.onnx")),
        "legacy_dir": Path(os.environ.get(
            "TRACKER_LEGACY_DIR_WHITE", LEGACY_ROOT / "Zone-div(final)")),
    },
    "red": {
        "weights": Path(os.environ.get(
            "TRACKER_WEIGHTS_RED", MODELS_DIR / "red.onnx")),
        "legacy_dir": Path(os.environ.get(
            "TRACKER_LEGACY_DIR_RED", LEGACY_ROOT / "Zone-div_V8m")),
    },
}

WORK_DIR = Path(os.environ.get("TRACKER_WORK_DIR", "/tmp/_tracker_work"))
storage = make_storage(WORK_DIR)

app = FastAPI(title="Cricket Ball Tracker — Worker")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "worker",
        "pipelines": {
            name: {
                "weights_present": cfg["weights"].exists(),
                "legacy_dir_present": cfg["legacy_dir"].exists(),
            }
            for name, cfg in BALL_PIPELINES.items()
        },
    }


@app.post("/process")
async def process(request: Request):
    """Process one job. Cloud Tasks delivers `{"job_id": "..."}`."""
    body = await request.json()
    job_id = body.get("job_id")
    if not job_id:
        raise HTTPException(status_code=400, detail="missing job_id")

    rec = firestore_jobs.get_job(job_id)
    if rec is None:
        # Avoid Cloud Tasks retrying forever on a job that was deleted.
        return {"ok": False, "reason": "unknown job_id"}

    if rec.get("cancel_requested"):
        firestore_jobs.update_job(job_id, status="cancelled",
                                  error={"code": "CANCELLED", "message": "Cancelled before start."})
        return {"ok": True, "status": "cancelled"}

    ball_type = rec.get("ball_type", "white")
    if ball_type not in BALL_PIPELINES:
        firestore_jobs.update_job(job_id, status="failed",
                                  error={"code": "INVALID_BALL_TYPE",
                                         "message": f"unknown ball_type {ball_type}"})
        return {"ok": False, "reason": "invalid ball_type"}
    cfg = BALL_PIPELINES[ball_type]

    # Signal the ball type to the legacy config.py so it picks the correct
    # per-ball-type CONF_THRESH_BALL. Must be set BEFORE _bind_legacy /
    # _import_process_video, because those imports read the env var at
    # module load time.
    os.environ["TRACKER_BALL_TYPE"] = ball_type

    # Bind the legacy folder for this ball type, evict any cached legacy /
    # pipeline modules from a previous job, and force a fresh import of
    # pipeline.process_video below.
    _bind_legacy(cfg["legacy_dir"])
    process_video = _import_process_video()

    job_dir = storage.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    video_path = job_dir / "input.mp4"

    # Pull the input video from GCS into the worker's local temp.
    # GCSStorage's job_dir() is local; we need to download input.mp4 explicitly.
    _download_input(job_id, video_path)

    firestore_jobs.update_job(
        job_id, status="processing",
        progress={"frame": 0, "total": 0, "percent": 0.0},
    )

    # Firestore-backed cancel poll, throttled to avoid hammering the DB on
    # every rendered frame. process_video calls cancel_check() at least once
    # per frame in every pass (PASS 1..4); we sample Firestore roughly once
    # per second and cache the answer between samples.
    import time as _time
    _cancel_cache: dict[str, float | bool] = {"ts": 0.0, "flag": False}
    _CANCEL_POLL_SEC = 1.0

    def _cancel_check() -> bool:
        now = _time.time()
        if now - float(_cancel_cache["ts"]) < _CANCEL_POLL_SEC:
            return bool(_cancel_cache["flag"])
        rec_now = firestore_jobs.get_job(job_id)
        flag = bool(rec_now and rec_now.get("cancel_requested"))
        _cancel_cache["ts"] = now
        _cancel_cache["flag"] = flag
        return flag

    def _progress(frame: int, total: int) -> None:
        # Progress-only update; cancel is handled by _cancel_check above.
        pct = (frame / total * 100.0) if total > 0 else 0.0
        firestore_jobs.update_job(
            job_id,
            progress={"frame": frame, "total": total, "percent": round(pct, 1)},
        )

    try:
        process_video(
            video_path=video_path,
            output_dir=job_dir,
            calibration=rec["calibration"],
            weights_path=cfg["weights"],
            interp_method=rec.get("interp_method", "parabola"),
            progress_callback=_progress,
            cancel_check=_cancel_check,
            job_id=job_id,
            video_filename=rec.get("filename"),
        )
        result = json.loads((job_dir / "results.json").read_text(encoding="utf-8"))
        storage.upload_job_artifacts(job_id, job_dir)
        firestore_jobs.update_job(job_id, status="done", result=result)
        return {"ok": True, "status": "done"}
    except Exception as e:  # noqa: BLE001
        # Duck-type on `.code` instead of isinstance(PipelineError): the class
        # object captured by the top-level import is stale after _bind_legacy()
        # evicts + re-imports the pipeline package per job, so the error class
        # raised by process_video is a different class object than the one this
        # module imported at startup. `hasattr(e, "code")` survives that.
        code = getattr(e, "code", None) if hasattr(e, "code") else None
        if code == "CANCELLED":
            firestore_jobs.update_job(
                job_id, status="cancelled",
                error={"code": "CANCELLED", "message": str(e) or "Cancelled by user."},
            )
            return {"ok": True, "status": "cancelled"}
        if code is not None:
            firestore_jobs.update_job(
                job_id, status="failed",
                error={"code": code, "message": str(e)},
            )
            return {"ok": False, "status": "failed", "code": code}
        traceback.print_exc()
        firestore_jobs.update_job(
            job_id, status="failed",
            error={"code": "INTERNAL", "message": str(e)},
        )
        return {"ok": False, "status": "failed", "code": "INTERNAL"}
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-job legacy-folder rebinding
# ─────────────────────────────────────────────────────────────────────────────
# Module names from the legacy Zone-div folders that must be evicted between
# jobs so a re-import picks up the right ball-type config + detector/filters.
_LEGACY_MODULE_NAMES = {
    "config", "detector", "filters", "grid_utils", "impact_detector",
    "interpolator", "kalman_filter", "logger", "release_detector",
    "tracker", "trajectory", "utils", "visualizer", "zones",
}


def _bind_legacy(target: Path) -> None:
    """Make `target` the active legacy folder: remove any other legacy folder
    already on sys.path, prepend `target`, evict cached legacy + pipeline
    modules so the next import re-resolves against `target`."""
    target_str = str(target)
    other_legacies = {str(c["legacy_dir"]) for c in BALL_PIPELINES.values()
                      if str(c["legacy_dir"]) != target_str}
    sys.path[:] = [p for p in sys.path if p not in other_legacies]
    if target_str in sys.path:
        sys.path.remove(target_str)
    sys.path.insert(0, target_str)
    os.environ["TRACKER_LEGACY_DIR"] = target_str
    for name in list(sys.modules):
        if name in _LEGACY_MODULE_NAMES or name == "pipeline" \
                or name.startswith("pipeline."):
            del sys.modules[name]


def _import_process_video():
    from pipeline import process_video  # re-resolved against the bound folder
    return process_video


def _download_input(job_id: str, dst: Path) -> None:
    """Fetch jobs/{job_id}/input.mp4 from GCS into dst. No-op if already present
    (local dev)."""
    if dst.exists():
        return
    bucket_name = os.environ.get("GCS_BUCKET")
    if not bucket_name:
        raise HTTPException(status_code=500, detail="GCS_BUCKET not set and input missing")
    from google.cloud import storage as gcs
    client = gcs.Client()
    blob = client.bucket(bucket_name).blob(f"jobs/{job_id}/input.mp4")
    if not blob.exists():
        raise HTTPException(status_code=404, detail="input.mp4 not found in bucket")
    dst.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dst))
