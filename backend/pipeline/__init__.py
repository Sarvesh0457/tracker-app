"""
Headless, server-safe cricket-ball tracker pipeline.

This package wraps the existing Zone-div(final) tracker logic into a single
callable `process_video()` with no module-level state. The public surface is:

    from pipeline import process_video, PipelineError
    from pipeline.errors import (
        InvalidVideoError, NoBallDetectedError, HomographyFailedError,
        EncodingFailedError, ModelLoadFailedError, PipelineCancelled,
    )

See pipeline/process.py for the entry point.
"""
import os
import sys
from pathlib import Path

# Make the legacy tracker modules importable. The folder stays untouched and
# is used as-is; TRACKER_LEGACY_DIR selects white (Zone-div(final)) vs red
# (Zone-div RED) and must be set before this package is imported — each job
# runs in its own worker subprocess so the binding is per-job.
_env_legacy = os.environ.get("TRACKER_LEGACY_DIR")
if _env_legacy:
    _LEGACY_DIR = Path(_env_legacy)
else:
    # Local-dev fallback: ../../June/working perfection/Zone-div(final) from this file.
    # In the deployed worker container, TRACKER_LEGACY_DIR is always set so this branch
    # never runs.
    _resolved = Path(__file__).resolve()
    if len(_resolved.parents) > 3:
        _LEGACY_DIR = _resolved.parents[3] / "June" / "working perfection" / "Zone-div(final)"
    else:
        _LEGACY_DIR = Path("/app/legacy/Zone-div(final)")
if str(_LEGACY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGACY_DIR))

from .errors import (  # noqa: E402
    PipelineError,
    InvalidVideoError,
    NoBallDetectedError,
    HomographyFailedError,
    EncodingFailedError,
    ModelLoadFailedError,
    PipelineCancelled,
    InvalidCalibrationError,
)


def __getattr__(name):
    # Lazy-load process_video so the API container (which doesn't ship the
    # legacy modules) can import pipeline.errors without triggering process.py's
    # top-level legacy imports.
    if name == "process_video":
        from .process import process_video
        return process_video
    raise AttributeError(f"module 'pipeline' has no attribute {name!r}")


__all__ = [
    "process_video",
    "PipelineError",
    "InvalidVideoError",
    "NoBallDetectedError",
    "HomographyFailedError",
    "EncodingFailedError",
    "ModelLoadFailedError",
    "PipelineCancelled",
    "InvalidCalibrationError",
]
