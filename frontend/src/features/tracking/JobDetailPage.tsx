import { useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { useJobPolling } from "./useJobPolling";
import { ResultsView } from "./ResultsView";
import { ERROR_MESSAGES } from "./types";
import { cancelJob } from "./api";
import { ThemeToggle } from "../../ThemeToggle";

export function JobDetailPage() {
  const location = useLocation();
  const jobId =
    (location.state as { jobId?: string } | null)?.jobId ||
    sessionStorage.getItem("activeJobId") ||
    undefined;
  const { job, result, error } = useJobPolling(jobId);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  async function onCancel() {
    if (!jobId || cancelling) return;
    setCancelling(true);
    setCancelError(null);
    try {
      await cancelJob(jobId);
      // useJobPolling will pick up status=cancelled on the next tick.
    } catch (e) {
      const err = e as Error & { code?: string };
      setCancelError(err.code === "HTTP_404"
        ? "This job doesn't belong to your device — can't cancel it here."
        : err.message || "Cancel failed. Try again in a moment.");
      setCancelling(false);
    }
  }

  return (
    <div className="page">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <Link to="/tracking">← Back</Link>
      </div>
      <div className="page-header">
        <h1>Delivery analysis</h1>
        <ThemeToggle />
      </div>
      {!job && !error && <div className="panel">Loading…</div>}
      {error && <div className="panel error">{error}</div>}
      {job && job.status === "queued" && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Queued</h3>
          <div className="muted">Waiting — another video may be processing.</div>
          <button
            className="ghost"
            style={{ marginTop: 12 }}
            onClick={onCancel}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
          {cancelError && (
            <div className="error" style={{ marginTop: 8 }}>{cancelError}</div>
          )}
        </div>
      )}
      {job && job.status === "processing" && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Processing</h3>
          <div className="progress-bar"><div style={{ width: `${job.progress?.percent || 0}%` }} /></div>
          <div className="muted" style={{ marginTop: 8 }}>
            Frame {job.progress?.frame ?? 0} of {job.progress?.total ?? "?"} ·
            {" "}{(job.progress?.percent ?? 0).toFixed(1)}%
          </div>
          <button
            className="ghost"
            style={{ marginTop: 12 }}
            onClick={onCancel}
            disabled={cancelling}
          >
            {cancelling ? "Cancelling…" : "Cancel"}
          </button>
          {cancelError && (
            <div className="error" style={{ marginTop: 8 }}>{cancelError}</div>
          )}
        </div>
      )}
      {job && job.status === "failed" && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Analysis failed</h3>
          <div className="error">
            {ERROR_MESSAGES[job.error?.code || "INTERNAL"] || job.error?.message}
          </div>
          <div className="muted" style={{ marginTop: 8 }}>Code: {job.error?.code}</div>
          <Link to="/tracking">
            <button style={{ marginTop: 12 }}>Try again</button>
          </Link>
        </div>
      )}
      {job && job.status === "cancelled" && (
        <div className="panel">
          <h3 style={{ marginTop: 0 }}>Cancelled</h3>
          <Link to="/tracking"><button>Start over</button></Link>
        </div>
      )}
      {result && <ResultsView result={result} />}
    </div>
  );
}
