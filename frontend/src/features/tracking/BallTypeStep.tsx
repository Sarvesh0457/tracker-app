import { BallType } from "./types";

interface Props {
  selected: BallType | null;
  onSelect: (b: BallType) => void;
  onBack: () => void;
}

export function BallTypeStep({ selected, onSelect, onBack }: Props) {
  return (
    <div className="panel">
      <h2 style={{ marginTop: 0 }}>Which ball is in this delivery?</h2>
      <p className="muted" style={{ marginTop: 0 }}>
        The white-ball and red-ball detectors are tuned differently — pick the
        one that matches the footage.
      </p>
      <div className="ball-grid">
        <button
          type="button"
          className={`ball-card ${selected === "white" ? "selected" : ""}`}
          onClick={() => onSelect("white")}
        >
          <span className="ball-swatch white" aria-hidden />
          <span className="ball-card-title">White ball</span>
          <span className="muted">Limited-overs / day-night / nets footage</span>
        </button>
        <button
          type="button"
          className={`ball-card ${selected === "red" ? "selected" : ""}`}
          onClick={() => onSelect("red")}
        >
          <span className="ball-swatch red" aria-hidden />
          <span className="ball-card-title">Red ball</span>
          <span className="muted">Test / first-class / red-leather practice</span>
        </button>
        <button
          type="button"
          className={`ball-card ${selected === "pink" ? "selected" : ""}`}
          onClick={() => onSelect("pink")}
        >
          <span className="ball-swatch pink" aria-hidden />
          <span className="ball-card-title">Pink ball</span>
          <span className="muted">Day-night Tests / pink-leather practice</span>
        </button>
      </div>
      <div className="row" style={{ marginTop: 16, justifyContent: "space-between" }}>
        <button className="ghost" onClick={onBack}>Back</button>
      </div>
    </div>
  );
}
