"""API service — the 'receptionist'.

Lives on a small CPU Cloud Run instance. Receives uploads, extracts first
frame, suggests calibration, then enqueues a Cloud Tasks message for the
worker service to actually run the pipeline. Worker progress is observed
via Firestore.

Routes (all under /api/track):
  POST /first-frame             upload video → first frame URL + upload_id
  GET  /uploads/{id}/suggest    auto-place calibration markers
  POST /analyze                 create Firestore job + enqueue Cloud Task
  GET  /jobs                    recent jobs
  GET  /jobs/{id}               poll status
  POST /jobs/{id}/cancel        request cancellation
  GET  /jobs/{id}/result        full Section-4 JSON + fresh signed URLs
  POST /jobs/{id}/feedback      submit feedback
  GET  /jobs/{id}/feedback      retrieve feedback
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Imports from the existing backend root (one level up).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.errors import InvalidVideoError  # noqa: E402
from storage import LocalStorage, make_storage  # noqa: E402

from . import firestore_jobs, tasks  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
MAX_CLIP_SECONDS = float(os.environ.get("MAX_CLIP_SECONDS", "30"))
WORK_DIR = Path(os.environ.get("TRACKER_WORK_DIR", "./_tracker_work"))

ACCEPTED_MIME = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/avi",
}

# ball_type → weights path (used by the worker; api just validates the type).
SUPPORTED_BALL_TYPES = {"white", "red"}


def _client_id(x_client_id: Optional[str] = Header(None)) -> str:
    """Per-browser identity. Required for any endpoint that creates or scopes jobs."""
    if not x_client_id:
        raise HTTPException(status_code=401, detail={"code": "MISSING_CLIENT_ID",
                                                     "message": "X-Client-Id header required"})
    return x_client_id


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Cricket Ball Tracker — API")
_cors_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = make_storage(WORK_DIR)
_uploads_root: Path = (
    storage.upload_dir() if isinstance(storage, LocalStorage)
    else WORK_DIR / "uploads"
)
_uploads_root.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _save_upload_streaming(src: UploadFile, dst: Path, max_bytes: int) -> int:
    written = 0
    chunk = 1 << 20
    with open(dst, "wb") as f:
        while True:
            data = src.file.read(chunk)
            if not data:
                break
            written += len(data)
            if written > max_bytes:
                f.close()
                dst.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail={
                        "code": "VIDEO_TOO_LARGE",
                        "message": f"max {max_bytes // (1<<20)} MiB",
                    },
                )
            f.write(data)
    return written


def _extract_first_frame(video_path: Path, out_jpg: Path) -> tuple[int, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InvalidVideoError("cannot open")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise InvalidVideoError("no first frame")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cv2.imwrite(str(out_jpg), frame)
    return w, h, (frames / fps if fps > 0 else 0.0)


def _result_with_urls(rec: dict) -> dict:
    if not rec.get("result"):
        return {}
    out = json.loads(json.dumps(rec["result"]))
    job_id = rec["job_id"]
    out["outputs"] = {
        "annotated_video_url": storage.signed_url(job_id, "output.mp4"),
        "csv_url": storage.signed_url(
            job_id, rec["result"]["outputs"].get("csv_url", "detections.csv")
        ),
        "first_frame_url": storage.signed_url(job_id, "first_frame.jpg"),
    }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/track/first-frame")
def first_frame(request: Request, video: UploadFile = File(...)):
    if video.content_type not in ACCEPTED_MIME:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_VIDEO",
                    "message": f"unsupported content-type {video.content_type}"},
        )
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = _uploads_root / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    src = upload_dir / "input.mp4"
    _save_upload_streaming(video, src, MAX_UPLOAD_MB * (1 << 20))

    try:
        w, h, duration = _extract_first_frame(src, upload_dir / "first_frame.jpg")
    except InvalidVideoError as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_VIDEO", "message": str(e)},
        )

    if duration > MAX_CLIP_SECONDS + 0.5:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=422,
            detail={"code": "VIDEO_TOO_LONG",
                    "message": f"clip is {duration:.1f} s — max {MAX_CLIP_SECONDS:.0f} s"},
        )

    # Persist the upload to GCS so the analyze step (potentially on a different
    # container instance) can find it.
    storage.put_blob(f"uploads/{upload_id}/input.mp4", src)
    storage.put_blob(f"uploads/{upload_id}/first_frame.jpg", upload_dir / "first_frame.jpg")
    # We can clear the local copy now — analyze() pulls from GCS.
    shutil.rmtree(upload_dir, ignore_errors=True)

    return {
        "upload_id": upload_id,
        "filename": video.filename or "upload.mp4",
        "width": w,
        "height": h,
        "duration_sec": round(duration, 2),
        "first_frame_url": storage.signed_url_for(f"uploads/{upload_id}/first_frame.jpg"),
    }


@app.get("/api/track/uploads/{upload_id}/suggest")
def suggest_calibration(upload_id: str):
    p = _uploads_root / upload_id / "first_frame.jpg"
    if not p.exists():
        # Pull from GCS — the API container that ran /first-frame may have recycled.
        if not storage.get_blob(f"uploads/{upload_id}/first_frame.jpg", p):
            raise HTTPException(status_code=404, detail="upload not found")
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(status_code=422, detail="cannot read first frame")
    h, w = img.shape[:2]
    cx = w / 2
    bat_y = int(h * 0.30)
    bowl_y = int(h * 0.82)
    bat_half = w * 0.10
    bowl_half = w * 0.22
    stump_off = (bowl_y - bat_y) * (1.22 / 15.24)
    return {
        "calibration": {
            "bat_L":  [int(cx - bat_half),  bat_y],
            "bat_R":  [int(cx + bat_half),  bat_y],
            "bowl_L": [int(cx - bowl_half), bowl_y],
            "bowl_R": [int(cx + bowl_half), bowl_y],
            "bat_stump":  [int(cx), int(bat_y - stump_off)],
            "bowl_stump": [int(cx), int(bowl_y + stump_off)],
        },
        "video": {"width": w, "height": h},
    }


@app.get("/api/track/uploads/{upload_id}/first_frame.jpg")
def get_upload_first_frame(upload_id: str):
    p = _uploads_root / upload_id / "first_frame.jpg"
    if not p.exists():
        if not storage.get_blob(f"uploads/{upload_id}/first_frame.jpg", p):
            raise HTTPException(status_code=404, detail="not found")
    return FileResponse(p, media_type="image/jpeg")


@app.post("/api/track/analyze")
def analyze(
    upload_id: str = Form(...),
    calibration: str = Form(...),
    interp_method: str = Form("parabola"),
    ball_type: str = Form("white"),
    filename: Optional[str] = Form(None),
    user_id: str = Depends(_client_id),
):
    """Create a Firestore job and enqueue a Cloud Task for the worker."""
    # Verify the upload exists in GCS (resilient to container recycles).
    if hasattr(storage, "_bucket"):
        upload_blob = storage._bucket.blob(f"uploads/{upload_id}/input.mp4")
        if not upload_blob.exists():
            raise HTTPException(
                status_code=404,
                detail={"code": "INVALID_VIDEO", "message": "upload_id not found"},
            )
    if ball_type not in SUPPORTED_BALL_TYPES:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_BALL_TYPE",
                    "message": f"ball_type must be one of {sorted(SUPPORTED_BALL_TYPES)}"},
        )
    try:
        calib_data = json.loads(calibration)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVALID_CALIBRATION", "message": str(e)},
        )

    job_id = firestore_jobs.create_job(
        upload_id=upload_id,
        calibration=calib_data,
        filename=filename or "video.mp4",
        interp_method=interp_method,
        ball_type=ball_type,
        user_id=user_id,
    )

    # Server-side copy from uploads/{upload_id}/ to jobs/{job_id}/ in GCS.
    # No local filesystem involved — fast and stateless.
    storage.copy_blob(f"uploads/{upload_id}/input.mp4", f"jobs/{job_id}/input.mp4")
    storage.copy_blob(f"uploads/{upload_id}/first_frame.jpg", f"jobs/{job_id}/first_frame.jpg")

    # Drop the note in the mailbox.
    tasks.enqueue_job(job_id)

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )


@app.get("/api/track/jobs")
def list_jobs(limit: int = 20, user_id: str = Depends(_client_id)):
    items = []
    for rec in firestore_jobs.list_jobs(user_id=user_id, limit=limit):
        bounce = (rec.get("result") or {}).get("events", {}).get("bounce") if rec.get("result") else None
        items.append({
            "job_id": rec["job_id"],
            "filename": rec.get("filename"),
            "status": rec["status"],
            "ball_type": rec.get("ball_type"),
            "created_at": rec.get("created_at"),
            "zone": bounce.get("zone") if bounce else None,
            "first_frame_url": (
                storage.signed_url(rec["job_id"], "first_frame.jpg")
                if rec["status"] == "done" else None
            ),
        })
    return {"jobs": items}


def _owned_job_or_404(job_id: str, user_id: str) -> dict:
    rec = firestore_jobs.get_job(job_id)
    if rec is None or rec.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return rec


@app.get("/api/track/jobs/{job_id}")
def get_job(job_id: str, user_id: str = Depends(_client_id)):
    return firestore_jobs.to_public(_owned_job_or_404(job_id, user_id))


@app.post("/api/track/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user_id: str = Depends(_client_id)):
    rec = _owned_job_or_404(job_id, user_id)
    if not firestore_jobs.request_cancel(job_id):
        raise HTTPException(status_code=409, detail="cannot cancel")
    # Fast-path: if the worker hasn't started yet (job is still queued behind
    # other single-concurrency jobs), flip straight to cancelled. Otherwise
    # the user waits minutes for Cloud Tasks to dispatch and the worker's
    # boot-time check to honour the flag. Once status is "processing" we let
    # the worker's Firestore poll pick up cancel_requested mid-flight.
    if rec.get("status") == "queued":
        firestore_jobs.update_job(
            job_id,
            status="cancelled",
            error={"code": "CANCELLED", "message": "Cancelled by user."},
        )
        return {"job_id": job_id, "status": "cancelled"}
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/api/track/jobs/{job_id}/result")
def get_result(job_id: str, user_id: str = Depends(_client_id)):
    rec = _owned_job_or_404(job_id, user_id)
    if rec["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail={"status": rec["status"], "message": "job not done"},
        )
    return _result_with_urls(rec)


@app.get("/api/track/jobs/{job_id}/download_full")
def download_full(job_id: str, user_id: str = Depends(_client_id)):
    """Lazy-bake and serve `output_full.mp4` — annotated video with all
    SVG overlays (stumps, pitching line, parabolas, event ball markers)
    baked in. First call per job pays the render cost (~5–15s for a 30s
    clip); subsequent calls return the cached file from GCS."""
    rec = _owned_job_or_404(job_id, user_id)
    if rec["status"] != "done":
        raise HTTPException(
            status_code=409,
            detail={"status": rec["status"], "message": "job not done"},
        )

    bucket = os.environ.get("GCS_BUCKET")
    full_key = f"jobs/{job_id}/output_full.mp4"
    ball_type = rec.get("ball_type", "white")

    # Cache-tmp lives outside the per-request TemporaryDirectory so subsequent
    # requests can hit it without re-baking (the GCS round-trip below also caches).
    cache_root = Path(os.environ.get("TRACKER_WORK_DIR", "/tmp/_tracker_work"))
    cache_dir = cache_root / "downloads" / job_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    baked_local = cache_dir / "output_full.mp4"

    # Fast path 1: bucket cache.
    if bucket and hasattr(storage, "get_blob") and not baked_local.exists():
        storage.get_blob(full_key, baked_local)  # no-op if absent

    # Slow path: bake.
    if not baked_local.exists():
        from pipeline.bake_overlay import bake_full_overlay
        src_mp4 = cache_dir / "output.mp4"
        src_json = cache_dir / "results.json"
        if hasattr(storage, "get_blob"):
            ok1 = storage.get_blob(f"jobs/{job_id}/output.mp4", src_mp4)
            ok2 = storage.get_blob(f"jobs/{job_id}/results.json", src_json)
        else:
            src_dir = storage.job_dir(job_id)
            shutil.copy(src_dir / "output.mp4", src_mp4)
            shutil.copy(src_dir / "results.json", src_json)
            ok1 = ok2 = True
        if not (ok1 and ok2):
            raise HTTPException(status_code=404, detail="job artifacts missing")
        try:
            baked = bake_full_overlay(job_id, cache_dir, ball_type)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"bake failed: {e}")
        if baked != baked_local:
            shutil.move(str(baked), str(baked_local))
        if bucket and hasattr(storage, "put_blob"):
            try:
                storage.put_blob(full_key, baked_local)
            except Exception:
                pass  # caching is best-effort; download still works

    return FileResponse(
        baked_local, media_type="video/mp4", filename="annotated.mp4",
    )


@app.post("/api/track/jobs/{job_id}/feedback")
async def submit_feedback(job_id: str, request: Request):
    rec = firestore_jobs.get_job(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid JSON")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="expected object")
    payload["job_id"] = job_id
    payload["submitted_at"] = datetime.now(timezone.utc).isoformat()
    path = storage.job_dir(job_id) / "feedback.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    storage.upload_job_artifacts(job_id, path.parent)
    return {"ok": True}


@app.get("/api/track/jobs/{job_id}/feedback")
def get_feedback(job_id: str):
    path = storage.job_dir(job_id) / "feedback.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no feedback")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/api/track/health")
def health():
    return {"ok": True, "service": "api"}
