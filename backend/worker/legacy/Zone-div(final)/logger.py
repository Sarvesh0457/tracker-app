import csv
from pathlib import Path
from datetime import datetime

class DetectionLogger:
    """
    Writes per-frame detection data to:
      - CSV  : one file containing two tables (ball and bat)
    """

    def __init__(self, output_dir: Path, video_fps: float):
        self.output_dir = Path(output_dir)
        self.fps        = video_fps

        # CSV setup
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.output_dir / f"detections_{ts}.csv"
        
        self.header = [
            "frame", "timestamp_sec",
            "event_type",
            "class", "conf",
            "x1", "y1", "x2", "y2",
            "cx", "cy", "width", "height",
            "grid_row", "grid_col"
        ]
        
        self.ball_rows = []
        self.bat_rows = []

        print(f"[LOGGER] CSV -> {self.csv_path}")

    def log(self, frame_num: int, detections: list,
             grid_corners: dict = None):
        """Call once per frame with the final detection list.

        grid_corners: optional {'TL':…,'TR':…,'BL':…,'BR':…} used to
                      compute and record (grid_row, grid_col) for each ball.
        """
        # Import here to avoid circular deps; grid_utils has no dep on logger
        if grid_corners is not None:
            from grid_utils import pixel_to_grid_cell as _p2g
        else:
            _p2g = None

        t = round(frame_num / self.fps, 4)

        for d in detections:
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2
            w  = d["x2"] - d["x1"]
            h  = d["y2"] - d["y1"]

            grow, gcol = "", ""
            if _p2g is not None and d["class_name"] == "ball":
                cell = _p2g((cx, cy), grid_corners)
                if cell is not None:
                    grow, gcol = cell

            row = [
                frame_num, t,
                "detection",
                d["class_name"], round(d["conf"], 4),
                d["x1"], d["y1"], d["x2"], d["y2"],
                cx, cy, w, h,
                grow, gcol
            ]
            if d["class_name"] == "ball":
                self.ball_rows.append(row)
            elif d["class_name"] == "bat":
                self.bat_rows.append(row)

    def close(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            
            # Write Ball Table
            writer.writerow(["--- BALL DETECTIONS ---"])
            writer.writerow(self.header)
            writer.writerows(self.ball_rows)
            
            writer.writerow([]) # Empty row for separation
            
            # Write Bat Table
            writer.writerow(["--- BAT DETECTIONS ---"])
            writer.writerow(self.header)
            writer.writerows(self.bat_rows)
            
        print(f"[LOGGER] Saved {len(self.ball_rows)} ball rows and {len(self.bat_rows)} bat rows to CSV.")
