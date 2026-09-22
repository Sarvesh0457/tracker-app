"""Double-run isolation test (spec §3.6).

Calls process_video() twice in one process on the same clip and asserts the
results.json outputs are equal. Proves no module-level state leaks between
runs. Skipped when the legacy weights/test video aren't available.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Make pipeline package importable without an install step
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import process_video  # noqa: E402


WEIGHTS = Path(
    os.environ.get(
        "TRACKER_WEIGHTS_PATH",
        r"D:\intership\may\version 2\model_weights\best.onnx",
    )
)
VIDEO = Path(
    os.environ.get(
        "TRACKER_TEST_VIDEO",
        r"D:\intership\Input files\white ball\2.mp4",
    )
)
CALIB_JSON = Path(
    os.environ.get(
        "TRACKER_TEST_CALIB",
        Path(__file__).resolve().parents[3]
        / "June" / "working perfection" / "Zone-div(final)" / "calibration.json",
    )
)


def _have_fixtures() -> bool:
    return WEIGHTS.exists() and VIDEO.exists() and CALIB_JSON.exists()


@pytest.mark.skipif(not _have_fixtures(), reason="test fixtures not present")
def test_double_run_isolated(tmp_path):
    with open(CALIB_JSON) as f:
        calib = json.load(f)

    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    out1.mkdir()
    out2.mkdir()

    r1 = process_video(
        VIDEO, out1, calib, WEIGHTS, interp_method="parabola",
        job_id="run1", video_filename="test.mp4",
    )
    r2 = process_video(
        VIDEO, out2, calib, WEIGHTS, interp_method="parabola",
        job_id="run2", video_filename="test.mp4",
    )

    # Strip per-run fields that legitimately differ
    for r in (r1, r2):
        r["job_id"] = "X"
        r["stats"]["processing_time_sec"] = 0
        r["outputs"]["csv_url"] = "csv"  # name contains timestamp

    assert r1 == r2, "Two runs in one process produced different results"
