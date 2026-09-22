"""Pluggable storage backend.

Two implementations sharing one interface:
  - LocalStorage  : writes under TRACKER_WORK_DIR; "signed_url" is just a
                    relative path served by the FastAPI app's /files route.
                    Default for development.
  - GCSStorage    : uploads to gs://$GCS_BUCKET, mints v4-signed URLs with
                    SIGNED_URL_TTL_SECONDS. Selected when GCS_BUCKET is set.
"""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import Optional


class Storage(ABC):
    @abstractmethod
    def job_dir(self, job_id: str) -> Path:
        """Local working directory for job artifacts (always local)."""

    @abstractmethod
    def upload_job_artifacts(self, job_id: str, local_dir: Path) -> None:
        """Push everything in `local_dir` to remote storage (no-op locally)."""

    @abstractmethod
    def signed_url(self, job_id: str, filename: str) -> str:
        """Return a URL the browser can fetch for the given artifact."""


# ─────────────────────────────────────────────────────────────────────────────
# Local-filesystem backend
# ─────────────────────────────────────────────────────────────────────────────
class LocalStorage(Storage):
    def __init__(self, work_dir: Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        d = self.work_dir / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def upload_dir(self) -> Path:
        d = self.work_dir / "uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def upload_job_artifacts(self, job_id: str, local_dir: Path) -> None:
        target = self.job_dir(job_id)
        if Path(local_dir).resolve() == target.resolve():
            return
        for item in Path(local_dir).iterdir():
            shutil.copy2(item, target / item.name)

    def signed_url(self, job_id: str, filename: str) -> str:
        # Served by the FastAPI app (see app.py /api/track/files/{job}/{name}).
        return f"/api/track/files/{job_id}/{filename}"

    # ── Generic blob helpers (parallel to GCSStorage). For local dev, the
    # "gcs_path" is just a relative path under work_dir. ──
    def put_blob(self, gcs_path: str, local_file: Path) -> None:
        dst = self.work_dir / gcs_path
        if Path(local_file).resolve() == dst.resolve():
            return  # already there — happens when upload_dir() is the same root
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_file, dst)

    def get_blob(self, gcs_path: str, local_file: Path) -> bool:
        src = self.work_dir / gcs_path
        if not src.exists():
            return False
        local_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_file)
        return True

    def signed_url_for(self, gcs_path: str) -> str:
        return f"/api/track/files/{gcs_path}"

    def copy_blob(self, src_path: str, dst_path: str) -> None:
        src = self.work_dir / src_path
        dst = self.work_dir / dst_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ─────────────────────────────────────────────────────────────────────────────
# GCS backend
# ─────────────────────────────────────────────────────────────────────────────
class GCSStorage(Storage):
    def __init__(self, bucket: str, work_dir: Path, ttl_seconds: int = 3600):
        from google.cloud import storage as gcs  # lazy import

        self.bucket_name = bucket
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self._client = gcs.Client()
        self._bucket = self._client.bucket(bucket)

    def _signed(self, path: str) -> str:
        # On Cloud Run the ADC is a compute-engine token with no private key.
        # generate_signed_url routes signing through the IAM Credentials API
        # when given (service_account_email, access_token) — that needs the
        # caller SA to hold roles/iam.serviceAccountTokenCreator on itself.
        from google.auth import default
        from google.auth.transport.requests import Request
        creds, _ = default()
        creds.refresh(Request())
        sa_email = (
            os.environ.get("SERVICE_ACCOUNT_EMAIL")
            or getattr(creds, "service_account_email", None)
        )
        if not sa_email or sa_email == "default":
            raise RuntimeError(
                "Cannot determine service account email for signed URL. "
                "Set SERVICE_ACCOUNT_EMAIL env var."
            )
        return self._bucket.blob(path).generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=self.ttl),
            method="GET",
            service_account_email=sa_email,
            access_token=creds.token,
        )

    def job_dir(self, job_id: str) -> Path:
        d = self.work_dir / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def upload_job_artifacts(self, job_id: str, local_dir: Path) -> None:
        prefix = f"jobs/{job_id}/"
        for item in Path(local_dir).iterdir():
            if not item.is_file():
                continue
            blob = self._bucket.blob(prefix + item.name)
            blob.upload_from_filename(str(item))

    def signed_url(self, job_id: str, filename: str) -> str:
        return self._signed(f"jobs/{job_id}/{filename}")

    # ── Generic blob helpers (used by the API for upload persistence). ──
    def put_blob(self, gcs_path: str, local_file: Path) -> None:
        self._bucket.blob(gcs_path).upload_from_filename(str(local_file))

    def get_blob(self, gcs_path: str, local_file: Path) -> bool:
        blob = self._bucket.blob(gcs_path)
        if not blob.exists():
            return False
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        return True

    def signed_url_for(self, gcs_path: str) -> str:
        return self._signed(gcs_path)

    def copy_blob(self, src_path: str, dst_path: str) -> None:
        src = self._bucket.blob(src_path)
        self._bucket.copy_blob(src, self._bucket, dst_path)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_storage(work_dir: Optional[Path] = None) -> Storage:
    work_dir = Path(
        work_dir or os.environ.get("TRACKER_WORK_DIR", "./_tracker_work")
    )
    bucket = os.environ.get("GCS_BUCKET")
    if bucket:
        ttl = int(os.environ.get("SIGNED_URL_TTL_SECONDS", "3600"))
        return GCSStorage(bucket, work_dir, ttl_seconds=ttl)
    return LocalStorage(work_dir)
