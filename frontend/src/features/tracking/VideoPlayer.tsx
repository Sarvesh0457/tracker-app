import {
  ReactNode,
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";

export interface PlayerMarker {
  label: string;
  time: number;
  color: string;
}

export interface VideoPlayerHandle {
  seek: (t: number) => void;
}

interface Props {
  src: string;
  poster?: string;
  fps?: number;
  markers?: PlayerMarker[];
  /** Absolute-positioned overlay rendered on top of the video frame. */
  overlay?: ReactNode;
  /** Called whenever the video's playback time changes. */
  onTime?: (t: number) => void;
  /** Called when playback reaches the end of the video. */
  onEnded?: () => void;
}

const SPEEDS = [0.25, 0.5, 1, 1.5, 2];

function fmt(t: number): string {
  if (!isFinite(t)) return "0:00";
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, Props>(
  function VideoPlayer({ src, poster, fps = 30, markers = [], overlay, onTime, onEnded }, ref) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const wrapRef = useRef<HTMLDivElement>(null);
    const trackRef = useRef<HTMLDivElement>(null);
    const [playing, setPlaying] = useState(false);
    const [muted, setMuted] = useState(false);
    const [time, setTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [speed, setSpeed] = useState(1);
    const [fullscreen, setFullscreen] = useState(false);
    const [blobSrc, setBlobSrc] = useState<string>("");

    // Fetch the video as a blob and play from a blob: URL so the real
    // backend URL doesn't appear in DevTools' Media tab. The request
    // still shows up under Fetch/XHR, but without the media resource entry.
    useEffect(() => {
      if (!src) {
        setBlobSrc("");
        return;
      }
      let cancelled = false;
      let createdUrl = "";
      fetch(src, { credentials: "include" })
        .then((r) => r.blob())
        .then((b) => {
          if (cancelled) return;
          createdUrl = URL.createObjectURL(b);
          setBlobSrc(createdUrl);
        })
        .catch(() => {
          if (!cancelled) setBlobSrc(src);
        });
      return () => {
        cancelled = true;
        if (createdUrl) URL.revokeObjectURL(createdUrl);
      };
    }, [src]);

    useImperativeHandle(ref, () => ({
      seek(t: number) {
        const v = videoRef.current;
        if (!v) return;
        v.currentTime = t;
        v.play().catch(() => {});
      },
    }));

    useEffect(() => {
      const onFsChange = () =>
        setFullscreen(document.fullscreenElement === wrapRef.current);
      document.addEventListener("fullscreenchange", onFsChange);
      return () => document.removeEventListener("fullscreenchange", onFsChange);
    }, []);

    // Smooth currentTime updates: HTMLMediaElement's `timeupdate` event only
    // fires ~4–15 Hz, which makes any time-driven overlay animation visibly
    // step. While playing, drive onTime + time state off requestAnimationFrame
    // (60 Hz) reading video.currentTime directly so SVG overlays animate
    // smoothly. The native timeupdate handler still covers paused/seek cases.
    useEffect(() => {
      if (!playing) return;
      let raf = 0;
      const tick = () => {
        const v = videoRef.current;
        if (v) {
          const t = v.currentTime;
          setTime(t);
          onTime?.(t);
        }
        raf = requestAnimationFrame(tick);
      };
      raf = requestAnimationFrame(tick);
      return () => cancelAnimationFrame(raf);
    }, [playing, onTime]);

    function togglePlay() {
      const v = videoRef.current;
      if (!v) return;
      if (v.paused) v.play().catch(() => {});
      else v.pause();
    }

    function stepFrame(dir: 1 | -1) {
      const v = videoRef.current;
      if (!v) return;
      v.pause();
      v.currentTime = Math.max(
        0,
        Math.min(v.duration || 0, v.currentTime + dir / fps),
      );
    }

    function cycleSpeed() {
      const v = videoRef.current;
      const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
      setSpeed(next);
      if (v) v.playbackRate = next;
    }

    function toggleFullscreen() {
      const w = wrapRef.current;
      if (!w) return;
      if (document.fullscreenElement === w) document.exitFullscreen();
      else w.requestFullscreen().catch(() => {});
    }

    function exitFullscreenIfActive() {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    }

    async function rotateToLandscape() {
      const w = wrapRef.current;
      if (!w) return;
      try {
        if (document.fullscreenElement !== w) await w.requestFullscreen();
        // Screen Orientation API — works on Android Chrome/Edge; Safari iOS no-op.
        const so = (screen.orientation as ScreenOrientation & {
          lock?: (o: string) => Promise<void>;
        });
        if (so?.lock) await so.lock("landscape");
      } catch { /* iOS / unsupported — fullscreen still helps */ }
    }

    function seekFromPointer(clientX: number) {
      const track = trackRef.current;
      const v = videoRef.current;
      if (!track || !v || !duration) return;
      const rect = track.getBoundingClientRect();
      const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
      v.currentTime = frac * duration;
      setTime(frac * duration);
    }

    function onTrackPointerDown(e: React.PointerEvent<HTMLDivElement>) {
      e.preventDefault();
      trackRef.current?.setPointerCapture(e.pointerId);
      seekFromPointer(e.clientX);
    }

    function onTrackPointerMove(e: React.PointerEvent<HTMLDivElement>) {
      if (trackRef.current?.hasPointerCapture(e.pointerId)) {
        seekFromPointer(e.clientX);
      }
    }

    function onKeyDown(e: React.KeyboardEvent) {
      const v = videoRef.current;
      if (!v) return;
      if (e.key === " " || e.key === "k") {
        e.preventDefault();
        togglePlay();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        v.currentTime = Math.min(v.duration || 0, v.currentTime + 5);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        v.currentTime = Math.max(0, v.currentTime - 5);
      } else if (e.key === ".") {
        stepFrame(1);
      } else if (e.key === ",") {
        stepFrame(-1);
      } else if (e.key === "f") {
        toggleFullscreen();
      } else if (e.key === "m") {
        setMuted((m) => {
          v.muted = !m;
          return !m;
        });
      }
    }

    const progressPct = duration ? (time / duration) * 100 : 0;

    return (
      <div
        ref={wrapRef}
        className={`player ${fullscreen ? "fullscreen" : ""}`}
        tabIndex={0}
        onKeyDown={onKeyDown}
      >
        <div className="player-video-wrap" style={{ position: "relative" }}>
          <video
            ref={videoRef}
            src={blobSrc || undefined}
            poster={poster}
            playsInline
            onClick={togglePlay}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onTimeUpdate={(e) => {
              setTime(e.currentTarget.currentTime);
              onTime?.(e.currentTarget.currentTime);
            }}
            onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
            onDurationChange={(e) => setDuration(e.currentTarget.duration)}
            onEnded={() => onEnded?.()}
          />
          {overlay && (
            <div className="player-overlay" style={{
              position: "absolute", inset: 0, pointerEvents: "none",
            }}>
              {overlay}
            </div>
          )}
          <button
            className="player-mobile-close"
            onClick={exitFullscreenIfActive}
            aria-label="Close fullscreen"
            title="Close"
          >×</button>
        </div>

        <div className="player-controls">
          <div
            ref={trackRef}
            className="player-track"
            onPointerDown={onTrackPointerDown}
            onPointerMove={onTrackPointerMove}
            role="slider"
            aria-label="Seek"
            aria-valuemin={0}
            aria-valuemax={duration}
            aria-valuenow={time}
          >
            <div className="player-track-fill" style={{ width: `${progressPct}%` }} />
            {duration > 0 &&
              markers.map((m) => (
                <div
                  key={m.label}
                  className="player-marker"
                  style={{
                    left: `${(m.time / duration) * 100}%`,
                    background: m.color,
                  }}
                  title={`${m.label} — ${m.time.toFixed(2)} s`}
                  onPointerDown={(e) => {
                    e.stopPropagation();
                    const v = videoRef.current;
                    if (v) v.currentTime = m.time;
                  }}
                />
              ))}
          </div>

          <div className="player-buttons">
            <button className="player-btn" onClick={togglePlay} title="Play/pause (space)">
              {playing ? "❚❚" : "▶"}
            </button>
            <button className="player-btn" onClick={() => stepFrame(-1)} title="Previous frame (,)">
              ⏮
            </button>
            <button className="player-btn" onClick={() => stepFrame(1)} title="Next frame (.)">
              ⏭
            </button>
            <span className="player-time">
              {fmt(time)} / {fmt(duration)}
            </span>
            <span className="player-spacer" />
            <button className="player-btn" onClick={cycleSpeed} title="Playback speed">
              {speed}×
            </button>
            <button
              className="player-btn"
              onClick={() => {
                const v = videoRef.current;
                if (v) v.muted = !muted;
                setMuted(!muted);
              }}
              title="Mute (m)"
            >
              {muted ? "🔇" : "🔊"}
            </button>
            <button
              className="player-btn player-mobile-rotate"
              onClick={rotateToLandscape}
              title="Rotate to landscape"
              aria-label="Rotate to landscape"
            >⤾</button>
            <button className="player-btn" onClick={toggleFullscreen} title="Fullscreen (f)">
              ⛶
            </button>
          </div>
        </div>
      </div>
    );
  },
);
