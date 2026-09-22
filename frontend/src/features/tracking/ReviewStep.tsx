import { useState } from "react";
import { analyze } from "./api";
import { BallType, Calibration, ERROR_MESSAGES, FirstFrameResponse } from "./types";
import { useNavigate } from "react-router-dom";

interface Props {
  upload: FirstFrameResponse;
  calibration: Calibration;
  ballType: BallType;
  onBack: () => void;
}

export function ReviewStep({ upload, calibration, ballType, onBack }: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  async function submit() {
    setError(null);
    setBusy(true);
    try {
      const { job_id } = await analyze(
        upload.upload_id,
        calibration,
        ballType,
        upload.filename,
      );
      sessionStorage.setItem("activeJobId", job_id);
      navigate("/delivery-analysis", { state: { jobId: job_id } });
    } catch (e: any) {
      setError(ERROR_MESSAGES[e.code] || e.message || ERROR_MESSAGES.INTERNAL);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>Confirm and analyze</h2>
      <div className="row" style={{ alignItems: "flex-start" }}>
        <img src={upload.first_frame_url} alt="thumb" style={{ maxWidth: 320, borderRadius: 6 }} />
        <div className="stack" style={{ flex: 1 }}>
          <div><span className="muted">File</span><br />{upload.filename}</div>
          <div><span className="muted">Resolution</span><br />{upload.width} × {upload.height}</div>
          <div><span className="muted">Duration</span><br />{upload.duration_sec.toFixed(1)} s</div>
          <div>
            <span className="muted">Ball</span>
            <br />
            <span className={`ball-swatch inline ${ballType}`} aria-hidden />
            {" "}{ballType === "white" ? "White ball" : ballType === "red" ? "Red ball" : "Pink ball"}
          </div>
        </div>
      </div>
      {error && <div className="error" style={{ marginTop: 12 }}>{error}</div>}
      <div className="row" style={{ marginTop: 16, justifyContent: "space-between" }}>
        <button className="ghost" onClick={onBack} disabled={busy}>Back</button>
        <button onClick={submit} disabled={busy}>
          {busy ? "Submitting…" : "Analyze delivery"}
        </button>
      </div>
    </div>
  );
}
