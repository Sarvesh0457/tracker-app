import { useEffect, useLayoutEffect, useRef, useState } from "react";
import {
  Calibration, MARKER_LABELS, MARKER_ORDER, MARKER_SHORT_LABELS, MarkerName,
  POLYGON_ORDER, STUMP_MARKERS,
} from "./types";

interface Props {
  firstFrameUrl: string;
  videoWidth: number;
  videoHeight: number;
  initial?: Partial<Calibration>;
  onChange: (cal: Partial<Calibration>) => void;
}

interface ImgPoint { x: number; y: number; }

export function CalibrationCanvas({
  firstFrameUrl, videoWidth, videoHeight, initial, onChange,
}: Props) {
  const [markers, setMarkers] = useState<Partial<Calibration>>(initial || {});
  const [activeKey, setActiveKey] = useState<MarkerName | null>(() => {
    if (!initial) return MARKER_ORDER[0];
    const pending = MARKER_ORDER.find((k) => !initial[k]);
    return pending || null;
  });
  const [dragging, setDragging] = useState<MarkerName | null>(null);
  const [hover, setHover] = useState<ImgPoint | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [past, setPast] = useState<Partial<Calibration>[]>([]);
  const [future, setFuture] = useState<Partial<Calibration>[]>([]);
  const dragSnapshot = useRef<Partial<Calibration> | null>(null);

  function pushHistory(prev: Partial<Calibration>) {
    setPast((p) => [...p, prev]);
    setFuture([]);
  }

  function undo() {
    setPast((p) => {
      if (p.length === 0) return p;
      const prev = p[p.length - 1];
      setFuture((f) => [markers, ...f]);
      setMarkers(prev);
      const next = MARKER_ORDER.find((k) => !prev[k]);
      setActiveKey(next || null);
      return p.slice(0, -1);
    });
  }

  function redo() {
    setFuture((f) => {
      if (f.length === 0) return f;
      const next = f[0];
      setPast((p) => [...p, markers]);
      setMarkers(next);
      const pending = MARKER_ORDER.find((k) => !next[k]);
      setActiveKey(pending || null);
      return f.slice(1);
    });
  }

  useEffect(() => onChange(markers), [markers, onChange]);

  useLayoutEffect(() => {
    function recalc() {
      const w = imgRef.current?.clientWidth || 1;
      setScale(w / videoWidth);
    }
    recalc();
    window.addEventListener("resize", recalc);
    return () => window.removeEventListener("resize", recalc);
  }, [videoWidth]);

  function toVideo(clientX: number, clientY: number): ImgPoint | null {
    const wrap = wrapRef.current;
    if (!wrap) return null;
    const r = wrap.getBoundingClientRect();
    const x = (clientX - r.left) / scale;
    const y = (clientY - r.top) / scale;
    if (x < 0 || y < 0 || x >= videoWidth || y >= videoHeight) return null;
    return { x: Math.round(x), y: Math.round(y) };
  }

  function place(key: MarkerName, p: ImgPoint) {
    pushHistory(markers);
    setMarkers((m) => ({ ...m, [key]: [p.x, p.y] }));
    const next = MARKER_ORDER.find((k) => k !== key && !markers[k]);
    setActiveKey(next || null);
  }

  function handleCanvasPointerDown(e: React.PointerEvent) {
    if (!activeKey) return;
    const p = toVideo(e.clientX, e.clientY);
    if (!p) return;
    place(activeKey, p);
  }

  function handleMarkerPointerDown(e: React.PointerEvent, key: MarkerName) {
    e.stopPropagation();
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    dragSnapshot.current = markers;
    setDragging(key);
    setActiveKey(key);
  }

  function handlePointerMove(e: React.PointerEvent) {
    const p = toVideo(e.clientX, e.clientY);
    setHover(p);
    if (dragging && p) {
      setMarkers((m) => ({ ...m, [dragging]: [p.x, p.y] }));
    }
  }

  function handlePointerUp() {
    if (dragging && dragSnapshot.current) {
      const snap = dragSnapshot.current;
      const cur = markers[dragging];
      const prev = snap[dragging];
      if (!prev || !cur || prev[0] !== cur[0] || prev[1] !== cur[1]) {
        pushHistory(snap);
      }
    }
    dragSnapshot.current = null;
    setDragging(null);
  }

  function reset() {
    pushHistory(markers);
    setMarkers({});
    setActiveKey(MARKER_ORDER[0]);
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const mod = e.ctrlKey || e.metaKey;
      if (!mod) return;
      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey) { e.preventDefault(); undo(); }
      else if ((key === "z" && e.shiftKey) || key === "y") { e.preventDefault(); redo(); }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function selectMarker(k: MarkerName) {
    setActiveKey(k);
  }

  const allPlaced = MARKER_ORDER.every((k) => markers[k]);
  const polygonReady = POLYGON_ORDER.every((k) => markers[k]);
  const polygonPoints = polygonReady
    ? POLYGON_ORDER.map((k) => {
        const [x, y] = markers[k]!;
        return `${x * scale},${y * scale}`;
      }).join(" ")
    : "";

  return (
    <div>
      {activeKey && (
        <div style={{ marginBottom: 8, color: "var(--accent)" }}>
          {MARKER_LABELS[activeKey]}
        </div>
      )}
      {!activeKey && allPlaced && (
        <div style={{ marginBottom: 8, color: "var(--good)" }}>
          All markers placed. Drag any vertex or stump base to fine-tune.
        </div>
      )}
      <div className="row" style={{ alignItems: "flex-start" }}>
        <div
          ref={wrapRef}
          className="canvas-wrap"
          onPointerDown={handleCanvasPointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          style={{ cursor: activeKey ? "crosshair" : "default", flex: 1 }}
        >
          <img
            ref={imgRef}
            src={firstFrameUrl}
            alt="first frame"
            draggable={false}
            onLoad={(e) => {
              setScale((e.target as HTMLImageElement).clientWidth / videoWidth);
            }}
          />
          {polygonReady && (
            <svg
              className="calib-svg"
              style={{
                position: "absolute",
                left: 0, top: 0,
                width: "100%", height: "100%",
                pointerEvents: "none",
              }}
            >
              <polygon
                points={polygonPoints}
                fill="var(--accent)"
                fillOpacity={0.20}
                stroke="var(--accent)"
                strokeOpacity={0.70}
                strokeWidth={2}
              />
              {POLYGON_ORDER.map((k) => {
                const [x, y] = markers[k]!;
                const isActive = activeKey === k || dragging === k;
                return (
                  <g key={k}>
                    {isActive && (
                      <circle
                        cx={x * scale}
                        cy={y * scale}
                        r={13}
                        fill="none"
                        stroke="var(--accent)"
                        strokeOpacity={0.45}
                        strokeWidth={3}
                        style={{ pointerEvents: "none" }}
                      />
                    )}
                    <circle
                      cx={x * scale}
                      cy={y * scale}
                      r={isActive ? 8 : 7}
                      fill="var(--accent)"
                      stroke="#fff"
                      strokeWidth={2}
                      style={{
                        pointerEvents: "all",
                        cursor: dragging === k ? "grabbing" : "grab",
                      }}
                      onPointerDown={(e) => handleMarkerPointerDown(e, k)}
                    />
                  </g>
                );
              })}
            </svg>
          )}
          {/* Crease corners not yet placed: render simple markers so the user
              can still click-place them through the activeKey workflow. */}
          {POLYGON_ORDER.filter((k) => !markers[k]).map(() => null)}
          {/* Stump-base markers — distinct from polygon vertices. */}
          {STUMP_MARKERS.map((k) => {
            const m = markers[k];
            if (!m) return null;
            return (
              <div
                key={k}
                className={`marker stump ${dragging === k ? "dragging" : ""} ${activeKey === k ? "active" : ""}`}
                style={{ left: m[0] * scale, top: m[1] * scale }}
                onPointerDown={(e) => handleMarkerPointerDown(e, k)}
              >
                <div className="crosshair-h" />
                <div className="crosshair-v" />
                <div className="label">{MARKER_SHORT_LABELS[k]}</div>
              </div>
            );
          })}
          {dragging && hover && (
            <Loupe x={hover.x} y={hover.y} scale={scale}
                   srcUrl={firstFrameUrl}
                   videoWidth={videoWidth} videoHeight={videoHeight} />
          )}
        </div>
        <div className="calib-sidebar">
          <ul className="checklist">
            {MARKER_ORDER.map((k) => (
              <li
                key={k}
                className={`${markers[k] ? "done" : ""} ${activeKey === k ? "active" : ""}`}
                onClick={() => selectMarker(k)}
              >
                <span>{MARKER_SHORT_LABELS[k]}</span>
                <span>{markers[k] ? "✓" : "·"}</span>
              </li>
            ))}
          </ul>
          <div className="row" style={{ marginTop: 12, gap: 8 }}>
            <button
              className="ghost"
              style={{ flex: 1 }}
              onClick={undo}
              disabled={past.length === 0}
              title="Undo (Ctrl+Z)"
            >
              ↶ Undo
            </button>
            <button
              className="ghost"
              style={{ flex: 1 }}
              onClick={redo}
              disabled={future.length === 0}
              title="Redo (Ctrl+Shift+Z)"
            >
              ↷ Redo
            </button>
          </div>
          <button className="ghost" style={{ marginTop: 8, width: "100%" }} onClick={reset}>
            Reset all
          </button>
        </div>
      </div>
    </div>
  );
}

function Loupe({
  x, y, scale, srcUrl, videoWidth, videoHeight,
}: {
  x: number; y: number; scale: number;
  srcUrl: string; videoWidth: number; videoHeight: number;
}) {
  const size = 120;
  const zoom = 3;
  const dispX = x * scale;
  const dispY = y * scale;
  const offX = dispX > 200 ? -160 : 40;
  const offY = -size / 2;
  return (
    <div
      className="loupe"
      style={{ left: dispX + offX, top: dispY + offY }}
    >
      <img
        src={srcUrl}
        alt=""
        style={{
          width: videoWidth * scale * zoom,
          height: videoHeight * scale * zoom,
          marginLeft: -(x * scale * zoom) + size / 2,
          marginTop: -(y * scale * zoom) + size / 2,
          maxWidth: "none",
        }}
      />
    </div>
  );
}
