"""In-process job store + subprocess job execution.

Phase 1 per spec §5.4: in-memory dict, single threading.Lock so only one
video processes at a time (CPU/GPU-bound — parallel jobs thrash). Replace
with Redis/Firestore + worker queue when we outgrow this.

Each job runs in its own subprocess (pipeline/worker.py) so the legacy
tracker modules — which are chosen per ball type via TRACKER_LEGACY_DIR and
hold module-level state — never collide between jobs or with the server.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

BACKEND_DIR = Path(__file__).resolve().parent


JobStatus = str  # "queued" | "processing" | "done" | "failed" | "cancelled"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus = "queued"
    progress: dict | None = None  # {frame, total, percent}
    error: dict | None = None     # {code, message}
    result: dict | None = None
    created_at: float = field(default_factory=time.time)
    upload_id: Optional[str] = None
    calibration: Optional[dict] = None
    filename: Optional[str] = None
    interp_method: str = "parabola"
    ball_type: str = "white"
    cancel_requested: bool = False


class JobStore:
    def __init__(self):
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        # Serializes pipeline execution (spec §5.4 phase 1).
        self.exec_lock = threading.Lock()

    def create(self, **kwargs) -> JobRecord:
        with self._lock:
            jid = uuid.uuid4().hex[:8]
            while jid in self._jobs:
                jid = uuid.uuid4().hex[:8]
            rec = JobRecord(job_id=jid, **kwargs)
            self._jobs[jid] = rec
            return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._jobs.get(job_id)

    def list(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return jobs[:limit]

    def update(self, job_id: str, **patch: Any) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None:
                return
            for k, v in patch.items():
                setattr(rec, k, v)

    def request_cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec is None or rec.status in ("done", "failed", "cancelled"):
                return False
            rec.cancel_requested = True
            return True

    def to_public(self, rec: JobRecord) -> dict:
        return {
            "job_id": rec.job_id,
            "status": rec.status,
            "progress": rec.progress,
            "error": rec.error,
        }


def run_job(
    store: JobStore,
    job_id: str,
    video_path: Path,
    output_dir: Path,
    calibration: dict,
    weights_path: Path,
    interp_method: str,
    legacy_dir: Path,
    on_done: Optional[Callable[[str], None]] = None,
) -> None:
    """Top-level worker. Runs pipeline/worker.py in a subprocess bound to
    `legacy_dir` (white or red pipeline), relaying progress and mapping
    failures to typed error codes."""
    rec = store.get(job_id)
    if rec is None:
        return

    def _cancel() -> bool:
        cur = store.get(job_id)
        return bool(cur and cur.cancel_requested)

    with store.exec_lock:
        if _cancel():
            store.update(job_id, status="cancelled")
            if on_done is not None:
                on_done(job_id)
            return
        store.update(job_id, status="processing",
                     progress={"frame": 0, "total": 0, "percent": 0.0})

        args_file = output_dir / "job_args.json"
        args_file.write_text(json.dumps({
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "calibration": calibration,
            "weights_path": str(weights_path),
            "interp_method": interp_method,
            "job_id": job_id,
            "video_filename": rec.filename,
        }), encoding="utf-8")

        # Point the legacy pipeline at (a) the folder containing coloured ball
        # PNGs and (b) which colour to use for this job. utils.overlay_ball_image
        # reads both env vars; without them the SEG-A baked markers fall back to
        # the legacy cricket_ball.png (white) regardless of ball type.
        _frontend_assets = BACKEND_DIR.parent / "frontend" / "public"
        env = {
            **os.environ,
            "TRACKER_LEGACY_DIR": str(legacy_dir),
            "TRACKER_BALL_TYPE": rec.ball_type,
            "TRACKER_BALL_ASSETS_DIR": os.environ.get(
                "TRACKER_BALL_ASSETS_DIR", str(_frontend_assets)),
        }
        log_tail: list[str] = []
        cancelled = False
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "pipeline.worker", str(args_file)],
                cwd=str(BACKEND_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            def _pump_stdout() -> None:
                for line in proc.stdout:  # type: ignore[union-attr]
                    line = line.rstrip()
                    if line.startswith("PROGRESS "):
                        try:
                            _, frame_s, total_s = line.split()
                            frame, total = int(frame_s), int(total_s)
                        except ValueError:
                            continue
                        pct = (frame / total * 100.0) if total > 0 else 0.0
                        store.update(job_id, progress={
                            "frame": frame, "total": total,
                            "percent": round(pct, 1),
                        })
                    else:
                        log_tail.append(line)
                        del log_tail[:-30]

            reader = threading.Thread(target=_pump_stdout, daemon=True)
            reader.start()

            while proc.poll() is None:
                if _cancel():
                    cancelled = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(0.5)
            proc.wait()
            reader.join(timeout=5)

            if cancelled:
                store.update(job_id, status="cancelled",
                             error={"code": "CANCELLED", "message": "Cancelled."})
            elif proc.returncode == 0:
                result = json.loads(
                    (output_dir / "results.json").read_text(encoding="utf-8")
                )
                store.update(job_id, status="done", result=result)
            else:
                error_file = output_dir / "error.json"
                if error_file.exists():
                    error = json.loads(error_file.read_text(encoding="utf-8"))
                else:
                    error = {
                        "code": "INTERNAL",
                        "message": "\n".join(log_tail[-5:])
                                   or f"worker exited with {proc.returncode}",
                    }
                store.update(job_id, status="failed", error=error)
        except Exception as e:  # noqa: BLE001 — failure launching/reading worker
            store.update(
                job_id,
                status="failed",
                error={"code": "INTERNAL", "message": str(e)},
            )
        finally:
            args_file.unlink(missing_ok=True)
            if on_done is not None:
                on_done(job_id)
