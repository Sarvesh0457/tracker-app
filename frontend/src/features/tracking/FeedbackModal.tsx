import { useEffect, useState } from "react";
import {
  DeliveryFeedback, EventVerdict, OverlayAlignment,
  TrackingQuality, WhatWentWrong,
} from "./types";
import { submitFeedback } from "./api";

interface Props {
  jobId: string;
  initial?: DeliveryFeedback | null;
  onClose: () => void;
  onSubmitted: () => void;
}

const EMPTY: DeliveryFeedback = {
  per_event: { release: null, bounce: null, deviation: null },
  tracking_quality: null,
  overlay_alignment: null,
  what_went_wrong: [],
  comment: "",
  overall_stars: 0,
};

const WRONG_CHIPS: { value: WhatWentWrong; label: string }[] = [
  { value: "wrong_frame", label: "Wrong frame" },
  { value: "wrong_zone_color", label: "Wrong zone color" },
  { value: "wrong_position", label: "Wrong position" },
  { value: "not_the_ball", label: "Not the ball" },
  { value: "missed_entirely", label: "Missed entirely" },
];

export function FeedbackModal({ jobId, initial, onClose, onSubmitted }: Props) {
  const [fb, setFb] = useState<DeliveryFeedback>(initial ?? EMPTY);
  const [submitting, setSubmitting] = useState(false);
  const [showThanks, setShowThanks] = useState(false);
  const isEdit = !!initial;

  const anyWrong =
    fb.per_event.release === "wrong" ||
    fb.per_event.bounce === "wrong" ||
    fb.per_event.deviation === "wrong";

  useEffect(() => {
    if (!showThanks) return;
    const t = setTimeout(() => {
      onSubmitted();
    }, 3000);
    return () => clearTimeout(t);
  }, [showThanks, onSubmitted]);

  function setVerdict(key: "release" | "bounce" | "deviation", v: EventVerdict) {
    setFb((f) => ({ ...f, per_event: { ...f.per_event, [key]: v } }));
  }

  function toggleChip(c: WhatWentWrong) {
    setFb((f) => ({
      ...f,
      what_went_wrong: f.what_went_wrong.includes(c)
        ? f.what_went_wrong.filter((x) => x !== c)
        : [...f.what_went_wrong, c],
    }));
  }

  async function handleSubmit() {
    setSubmitting(true);
    try {
      await submitFeedback(jobId, fb);
      setShowThanks(true);
    } catch {
      setSubmitting(false);
    }
  }

  if (showThanks) {
    return (
      <div className="fb-backdrop">
        <div className="fb-modal fb-thanks" onClick={(e) => e.stopPropagation()}>
          <div style={{ position: "absolute", top: 12, right: 16, fontSize: 28, color: "var(--accent, #2ecc71)" }} aria-label="Submitted">✓</div>
          <h3 style={{ margin: "8px 0 0" }}>Thanks for your feedback!</h3>
        </div>
      </div>
    );
  }

  return (
    <div className="fb-backdrop" onClick={onClose}>
      <div className="fb-modal" onClick={(e) => e.stopPropagation()}>
        <div className="fb-header">
          <h3 style={{ margin: 0 }}>{isEdit ? "Update your feedback" : "How did the tracking look?"}</h3>
          <button className="fb-close" onClick={onClose} aria-label="Close">×</button>
        </div>

        <FieldLabel n={1}>Per-event correctness</FieldLabel>
        <div className="fb-event-grid">
          {(["release", "bounce", "deviation"] as const).map((k) => (
            <EventRow
              key={k}
              label={k === "bounce" ? "Bounce + zone" : k[0].toUpperCase() + k.slice(1)}
              value={fb.per_event[k]}
              onChange={(v) => setVerdict(k, v)}
            />
          ))}
        </div>

        <FieldLabel n={2}>Ball tracking quality</FieldLabel>
        <ChoiceRow<TrackingQuality>
          options={[
            { value: "smooth", label: "Smooth" },
            { value: "jumpy", label: "Jumpy" },
            { value: "lost", label: "Lost" },
          ]}
          value={fb.tracking_quality}
          onChange={(v) => setFb((f) => ({ ...f, tracking_quality: v }))}
        />

        <FieldLabel n={3}>Overlay alignment</FieldLabel>
        <ChoiceRow<OverlayAlignment>
          options={[
            { value: "good", label: "Good" },
            { value: "slightly_off", label: "Slightly off" },
            { value: "way_off", label: "Way off" },
          ]}
          value={fb.overlay_alignment}
          onChange={(v) => setFb((f) => ({ ...f, overlay_alignment: v }))}
        />

        {anyWrong && (
          <div className="fb-chip-panel">
            <FieldLabel n={4}>What went wrong?</FieldLabel>
            <div className="fb-chips">
              {WRONG_CHIPS.map((c) => (
                <button
                  key={c.value}
                  className={`fb-chip ${fb.what_went_wrong.includes(c.value) ? "on" : ""}`}
                  onClick={() => toggleChip(c.value)}
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
        )}

        <FieldLabel n={anyWrong ? 5 : 4}>Anything else? <span className="muted">(optional)</span></FieldLabel>
        <input
          type="text"
          className="fb-input"
          placeholder="One quick note…"
          value={fb.comment}
          onChange={(e) => setFb((f) => ({ ...f, comment: e.target.value }))}
        />

        <FieldLabel n={anyWrong ? 6 : 5}>Overall accuracy</FieldLabel>
        <div className="fb-stars">
          {[1, 2, 3, 4, 5].map((n) => (
            <button
              key={n}
              className={`fb-star ${n <= fb.overall_stars ? "on" : ""}`}
              onClick={() => setFb((f) => ({ ...f, overall_stars: n }))}
              aria-label={`${n} star${n > 1 ? "s" : ""}`}
            >★</button>
          ))}
        </div>

        <div className="fb-actions">
          <button className="ghost" onClick={onClose} disabled={submitting}>{isEdit ? "Cancel" : "Skip for now"}</button>
          <button
            onClick={handleSubmit}
            disabled={submitting || fb.overall_stars === 0}
          >{submitting ? "Saving…" : isEdit ? "Save changes" : "Submit"}</button>
        </div>
      </div>
    </div>
  );
}

function FieldLabel({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <div className="fb-field-label">{n}. {children}</div>
  );
}

function EventRow({
  label, value, onChange,
}: { label: string; value: EventVerdict | null; onChange: (v: EventVerdict) => void }) {
  return (
    <>
      <span className="muted">{label}</span>
      <button className={`fb-vbtn ok ${value === "correct" ? "on" : ""}`} onClick={() => onChange("correct")}>✓</button>
      <button className={`fb-vbtn bad ${value === "wrong" ? "on" : ""}`} onClick={() => onChange("wrong")}>✕</button>
    </>
  );
}

function ChoiceRow<T extends string>({
  options, value, onChange,
}: { options: { value: T; label: string }[]; value: T | null; onChange: (v: T) => void }) {
  return (
    <div className="fb-choices">
      {options.map((o) => (
        <button
          key={o.value}
          className={`fb-choice ${value === o.value ? "on" : ""}`}
          onClick={() => onChange(o.value)}
        >{o.label}</button>
      ))}
    </div>
  );
}
