import { useEffect, useMemo, useState } from "react";
import { suggestCalibration } from "./api";
import { CalibrationCanvas } from "./CalibrationCanvas";
import { loadCalibration, saveCalibration } from "./calibrationStorage";
import { Calibration, FirstFrameResponse, MARKER_ORDER } from "./types";

interface Props {
  upload: FirstFrameResponse;
  onConfirm: (cal: Calibration) => void;
  onBack: () => void;
}

export function CalibrationStep({ upload, onConfirm, onBack }: Props) {
  const [partial, setPartial] = useState<Partial<Calibration>>({});
  const [seed, setSeed] = useState<Partial<Calibration> | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Prefer the user's last confirmed calibration when the source video has
    // the same dimensions (pixel coords are resolution-specific). Fall back to
    // the backend's auto-placement otherwise.
    const saved = loadCalibration(upload.width, upload.height);
    if (saved) {
      setSeed(saved);
      return () => { cancelled = true; };
    }
    suggestCalibration(upload.upload_id)
      .then((r) => { if (!cancelled) setSeed(r.calibration); })
      .catch(() => { if (!cancelled) setSeed({}); });
    return () => { cancelled = true; };
  }, [upload.upload_id, upload.width, upload.height]);

  const validation = useMemo(() => validate(partial, upload.width, upload.height), [partial, upload]);

  if (seed === null) {
    return <div className="panel">Preparing calibration…</div>;
  }

  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>Calibrate the pitch</h2>
      <p style={{ marginTop: 0, marginBottom: 8 }}>
        Place the four red corners at the intersections of the popping and return crease.
        Mark the base/bottom of the middle stump at each end.
      </p>
      <div className="muted" style={{ marginTop: 0, marginBottom: 12, fontSize: 13 }}>
        Optimal tracking will be available if:
        <ol style={{ margin: "4px 0 0 18px", padding: 0 }}>
          <li>The camera display is fixed <b>straight on with the pitch</b>.</li>
          <li>The <b>ball stays visible to the camera through the entire delivery</b> — from release to pitching to reaching the batter.</li>
          <li>The camera is <b>static</b> (no pan/zoom) and the <b>first frame is clear</b>.</li>
        </ol>
      </div>
      <CalibrationCanvas
        firstFrameUrl={upload.first_frame_url}
        videoWidth={upload.width}
        videoHeight={upload.height}
        initial={seed}
        onChange={setPartial}
      />
      {validation.error && <div className="error" style={{ marginTop: 12 }}>{validation.error}</div>}
      <div className="row" style={{ marginTop: 16, justifyContent: "space-between" }}>
        <button className="ghost" onClick={onBack}>Back</button>
        <button
          disabled={!validation.complete || !!validation.error}
          onClick={() => {
            const cal = partial as Calibration;
            saveCalibration(cal, upload.width, upload.height);
            onConfirm(cal);
          }}
        >
          Continue
        </button>
      </div>
    </div>
  );
}

function validate(partial: Partial<Calibration>, w: number, h: number) {
  const placed = MARKER_ORDER.filter((k) => partial[k]);
  if (placed.length < MARKER_ORDER.length) {
    return { complete: false, error: null as string | null };
  }
  const c = partial as Calibration;
  for (const k of MARKER_ORDER) {
    const [x, y] = c[k];
    if (x < 0 || y < 0 || x >= w || y >= h) {
      return { complete: true, error: `${k} is outside the frame` };
    }
  }
  if (c.bat_L[0] >= c.bat_R[0]) {
    return { complete: true, error: "Batter's left corner must be left of right corner" };
  }
  if (c.bowl_L[0] >= c.bowl_R[0]) {
    return { complete: true, error: "Bowler's left corner must be left of right corner" };
  }
  const batMeanY = (c.bat_L[1] + c.bat_R[1]) / 2;
  const bowlMeanY = (c.bowl_L[1] + c.bowl_R[1]) / 2;
  if (Math.abs(batMeanY - bowlMeanY) < 50) {
    return { complete: true, error: "Batter-end and bowler-end of the polygon are not vertically separated" };
  }
  // Sanity-check the stump bases sit roughly between the crease corners.
  const inXRange = (x: number, a: number, b: number) =>
    x >= Math.min(a, b) - 20 && x <= Math.max(a, b) + 20;
  if (!inXRange(c.bat_stump[0], c.bat_L[0], c.bat_R[0])) {
    return { complete: true, error: "Batter stump base should sit between the batter-end crease corners" };
  }
  if (!inXRange(c.bowl_stump[0], c.bowl_L[0], c.bowl_R[0])) {
    return { complete: true, error: "Bowler stump base should sit between the bowler-end crease corners" };
  }
  return { complete: true, error: null };
}
