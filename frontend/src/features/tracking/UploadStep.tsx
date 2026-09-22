import { useRef, useState } from "react";
import { uploadFirstFrame, FirstFrameProgress } from "./api";
import { ERROR_MESSAGES, FirstFrameResponse } from "./types";

const MAX_MB = 500;
const MAX_SECONDS = 30;
const ACCEPTED = ["video/mp4", "video/quicktime", "video/x-msvideo"];

function probeDuration(f: File): Promise<number> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata";
    v.src = url;
    v.onloadedmetadata = () => {
      const d = v.duration;
      URL.revokeObjectURL(url);
      resolve(isFinite(d) ? d : 0);
    };
    v.onerror = () => {
      URL.revokeObjectURL(url);
      resolve(0); // let the backend decide
    };
  });
}

export function UploadStep({ onUploaded }: { onUploaded: (r: FirstFrameResponse) => void }) {
  const [over, setOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<FirstFrameProgress | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  function pick() { inputRef.current?.click(); }

  async function handleFile(f: File) {
    setError(null);
    if (!ACCEPTED.includes(f.type) && !/\.(mp4|mov|avi)$/i.test(f.name)) {
      setError("Only MP4, MOV and AVI videos are supported.");
      return;
    }
    if (f.size > MAX_MB * (1 << 20)) {
      setError(ERROR_MESSAGES.VIDEO_TOO_LARGE);
      return;
    }
    const dur = await probeDuration(f);
    if (dur > MAX_SECONDS + 0.5) {
      setError(ERROR_MESSAGES.VIDEO_TOO_LONG);
      return;
    }
    abortRef.current = new AbortController();
    try {
      const r = await uploadFirstFrame(f, setProgress, abortRef.current.signal);
      onUploaded(r);
    } catch (e: any) {
      setError(ERROR_MESSAGES[e.code] || e.message || ERROR_MESSAGES.INTERNAL);
    } finally {
      setProgress(null);
    }
  }

  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>Upload a delivery video</h2>
      <div
        className={`dropzone ${over ? "over" : ""}`}
        onClick={pick}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handleFile(f);
        }}
      >
        <div>
          {progress
            ? `Uploading… ${Math.round((progress.loaded / progress.total) * 100)}%`
            : "Drop a delivery video to analyze. One delivery per video works best."}
        </div>
        <div className="muted" style={{ marginTop: 6 }}>
          MP4 / MOV / AVI, up to {MAX_MB} MB and {MAX_SECONDS} seconds
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo"
          style={{ display: "none" }}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) handleFile(f);
          }}
        />
        {progress && (
          <div className="progress-bar" style={{ marginTop: 12 }}>
            <div style={{ width: `${(progress.loaded / progress.total) * 100}%` }} />
          </div>
        )}
      </div>
      {error && <div className="error" style={{ marginTop: 12 }}>{error}</div>}
    </div>
  );
}
