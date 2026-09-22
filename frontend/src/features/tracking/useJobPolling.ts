import { useEffect, useState } from "react";
import { getJob, getResult } from "./api";
import { JobRecord, TrackingResult } from "./types";

const FAST_MS = 2000;
const SLOW_MS = 5000;
const SLOW_AFTER_MS = 120_000;

export function useJobPolling(jobId: string | undefined) {
  const [job, setJob] = useState<JobRecord | null>(null);
  const [result, setResult] = useState<TrackingResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    const abort = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const start = Date.now();
    let resultFetched = false;

    async function tick() {
      try {
        const r = await getJob(jobId!, abort.signal);
        if (abort.signal.aborted) return;
        setJob(r);
        if (r.status === "done" && !resultFetched) {
          resultFetched = true;
          const res = await getResult(jobId!, abort.signal);
          if (!abort.signal.aborted) setResult(res);
          return; // stop polling
        }
        if (r.status === "failed" || r.status === "cancelled") {
          return;
        }
        const ms = Date.now() - start > SLOW_AFTER_MS ? SLOW_MS : FAST_MS;
        timer = setTimeout(tick, ms);
      } catch (e: any) {
        if (abort.signal.aborted) return;
        setError(e.message || "polling failed");
        timer = setTimeout(tick, SLOW_MS);
      }
    }
    tick();
    return () => {
      abort.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  return { job, result, error };
}
