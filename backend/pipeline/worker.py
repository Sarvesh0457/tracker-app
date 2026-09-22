"""Subprocess entry point — runs exactly one process_video() job.

Launched by jobs.run_job as `python -m pipeline.worker <args.json>` with
TRACKER_LEGACY_DIR pointing at the white-ball (Zone-div(final)) or red-ball
(Zone-div RED) folder. Because the legacy modules are bound at import time
and hold module-level state, per-job process isolation is what lets one
server serve both pipelines.

Protocol with the parent:
  - progress  → `PROGRESS <frame> <total>` lines on stdout
  - success   → exit 0, results.json in the output dir
  - failure   → exit non-zero, error.json {code, message} in the output dir
  - cancel    → the parent terminates this process
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> int:
    args = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    from pipeline import process_video
    from pipeline.errors import PipelineError

    def _progress(frame: int, total: int) -> None:
        print(f"PROGRESS {frame} {total}", flush=True)

    try:
        process_video(
            video_path=Path(args["video_path"]),
            output_dir=output_dir,
            calibration=args["calibration"],
            weights_path=Path(args["weights_path"]),
            interp_method=args.get("interp_method", "parabola"),
            progress_callback=_progress,
            job_id=args.get("job_id"),
            video_filename=args.get("video_filename"),
        )
        return 0
    except PipelineError as e:
        (output_dir / "error.json").write_text(
            json.dumps({"code": e.code, "message": str(e)}), encoding="utf-8"
        )
        return 2
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        (output_dir / "error.json").write_text(
            json.dumps({"code": "INTERNAL", "message": str(e)}), encoding="utf-8"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
