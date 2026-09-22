"""Firestore-backed JobStore — replacement for the in-memory store in jobs.py.

Schema for collection `jobs`:
  job_id           (doc id, string)
  status           "queued" | "processing" | "done" | "failed" | "cancelled"
  progress         { frame, total, percent } | None
  error            { code, message } | None
  result           dict | None
  created_at       float (unix epoch)
  upload_id        string | None
  calibration      dict | None
  filename         string | None
  interp_method    string
  ball_type        string
  cancel_requested bool
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from google.cloud import firestore

_COLLECTION = "jobs"


def _client() -> firestore.Client:
    return firestore.Client()


def create_job(**fields: Any) -> str:
    job_id = uuid.uuid4().hex[:8]
    doc = {
        "job_id": job_id,
        "status": "queued",
        "progress": None,
        "error": None,
        "result": None,
        "created_at": time.time(),
        "cancel_requested": False,
        **fields,
    }
    _client().collection(_COLLECTION).document(job_id).set(doc)
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    snap = _client().collection(_COLLECTION).document(job_id).get()
    return snap.to_dict() if snap.exists else None


def update_job(job_id: str, **patch: Any) -> None:
    _client().collection(_COLLECTION).document(job_id).update(patch)


def list_jobs(user_id: Optional[str] = None, limit: int = 20) -> list[dict]:
    q = _client().collection(_COLLECTION)
    if user_id:
        q = q.where("user_id", "==", user_id)
    q = q.order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit)
    return [d.to_dict() for d in q.stream()]


def request_cancel(job_id: str) -> bool:
    doc_ref = _client().collection(_COLLECTION).document(job_id)
    snap = doc_ref.get()
    if not snap.exists:
        return False
    rec = snap.to_dict()
    if rec.get("status") in ("done", "failed", "cancelled"):
        return False
    doc_ref.update({"cancel_requested": True})
    return True


def to_public(rec: dict) -> dict:
    return {
        "job_id": rec["job_id"],
        "status": rec["status"],
        "progress": rec.get("progress"),
        "error": rec.get("error"),
    }
