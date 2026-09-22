import { Calibration } from "./types";

// Persist the last confirmed calibration so the next upload starts from it
// rather than the backend's auto-placement. Coordinates are pixel-space, so
// we gate restoration on matching source video dimensions — a calibration
// confirmed for 1920×1080 footage would not project correctly onto a 1280×720
// frame.

const KEY = "tracker.lastCalibration.v1";

interface Stored {
  width: number;
  height: number;
  calibration: Calibration;
  savedAt: number;
}

export function saveCalibration(
  cal: Calibration, width: number, height: number,
): void {
  try {
    const payload: Stored = { width, height, calibration: cal, savedAt: Date.now() };
    localStorage.setItem(KEY, JSON.stringify(payload));
  } catch {
    // Quota / privacy mode — silently skip; restoration just won't happen.
  }
}

export function loadCalibration(
  width: number, height: number,
): Calibration | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Stored;
    if (parsed.width !== width || parsed.height !== height) return null;
    return parsed.calibration;
  } catch {
    return null;
  }
}

export function clearCalibration(): void {
  try { localStorage.removeItem(KEY); } catch { /* ignore */ }
}
