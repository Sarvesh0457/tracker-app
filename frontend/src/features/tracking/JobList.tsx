import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listJobs, getFeedback } from "./api";
import { JobListItem, ZONE_COLORS, DeliveryFeedback } from "./types";
import { FeedbackModal } from "./FeedbackModal";

export function JobList() {
  const [items, setItems] = useState<JobListItem[] | null>(null);
  const [feedbackJobId, setFeedbackJobId] = useState<string | null>(null);
  const [feedbackInitial, setFeedbackInitial] = useState<DeliveryFeedback | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    let alive = true;
    listJobs(20).then((r) => { if (alive) setItems(r.jobs); }).catch(() => setItems([]));
    return () => { alive = false; };
  }, []);

  async function openEdit(jobId: string) {
    try {
      const existing = await getFeedback(jobId);
      setFeedbackInitial(existing);
    } catch {
      setFeedbackInitial(null);
    }
    setFeedbackJobId(jobId);
  }

  function openNew(jobId: string) {
    setFeedbackInitial(null);
    setFeedbackJobId(jobId);
  }

  function markGiven(jobId: string) {
    localStorage.setItem(`tracker_feedback_done:${jobId}`, "1");
    setItems((prev) =>
      prev ? prev.map((j) => j.job_id === jobId ? { ...j, has_feedback: true } : j) : prev,
    );
    setFeedbackJobId(null);
  }

  if (!items) return null;
  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>Recent Deliveries</h2>
      {items.length === 0 ? (
        <div className="muted">No deliveries analyzed yet.</div>
      ) : (
        <table className="job-table">
          <thead>
            <tr>
              <th></th>
              <th>File</th>
              <th>Created</th>
              <th>Status</th>
              <th>Bounce</th>
              <th>Feedback</th>
            </tr>
          </thead>
          <tbody>
            {items.map((j) => (
              <tr
                key={j.job_id}
                className="job-row"
                onClick={() => {
                  sessionStorage.setItem("activeJobId", j.job_id);
                  navigate("/delivery-analysis", { state: { jobId: j.job_id } });
                }}
                tabIndex={0}
                role="link"
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    sessionStorage.setItem("activeJobId", j.job_id);
                    navigate("/delivery-analysis", { state: { jobId: j.job_id } });
                  }
                }}
              >
                <td>
                  {j.first_frame_url ? (
                    <img src={j.first_frame_url} alt="" style={{ width: 64, borderRadius: 4 }} />
                  ) : <span className="muted">—</span>}
                </td>
                <td>{j.filename || "Untitled delivery"}</td>
                <td className="muted">{new Date(j.created_at * 1000).toLocaleString()}</td>
                <td><span className={`pill ${j.status}`}>{j.status}</span></td>
                <td>
                  {j.zone ? (
                    <span
                      className={`zone-badge ${j.zone}`}
                      style={{ background: ZONE_COLORS[j.zone] }}
                    >
                      {j.zone}
                    </span>
                  ) : <span className="muted">—</span>}
                </td>
                <td>
                  {j.status !== "done" ? (
                    <span className="muted">—</span>
                  ) : j.has_feedback ? (
                    <button
                      className="fb-done-badge"
                      style={{ border: "none", cursor: "pointer" }}
                      onClick={(e) => { e.stopPropagation(); openEdit(j.job_id); }}
                      title="Edit feedback"
                    >Given · Edit</button>
                  ) : (
                    <button
                      className="fb-pending-badge"
                      onClick={(e) => { e.stopPropagation(); openNew(j.job_id); }}
                    >Give feedback</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {feedbackJobId && (
        <FeedbackModal
          jobId={feedbackJobId}
          initial={feedbackInitial}
          onClose={() => setFeedbackJobId(null)}
          onSubmitted={() => markGiven(feedbackJobId)}
        />
      )}
    </div>
  );
}
