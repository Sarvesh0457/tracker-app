import { useState } from "react";
import { BallTypeStep } from "./BallTypeStep";
import { CalibrationStep } from "./CalibrationStep";
import { ReviewStep } from "./ReviewStep";
import { UploadStep } from "./UploadStep";
import { JobList } from "./JobList";
import { BallType, Calibration, FirstFrameResponse } from "./types";
import { ThemeToggle } from "../../ThemeToggle";

type Step = "upload" | "calibrate" | "ball" | "review";

export function TrackingPage() {
  const [step, setStep] = useState<Step>("upload");
  const [upload, setUpload] = useState<FirstFrameResponse | null>(null);
  const [calibration, setCalibration] = useState<Calibration | null>(null);
  const [ballType, setBallType] = useState<BallType | null>(null);

  return (
    <div className="page">
      <div className="page-header">
        <h1>Ball Tracking</h1>
        <ThemeToggle />
      </div>
      <Stepper step={step} />
      {step === "upload" && (
        <UploadStep
          onUploaded={(r) => {
            setUpload(r);
            setStep("calibrate");
          }}
        />
      )}
      {step === "calibrate" && upload && (
        <CalibrationStep
          upload={upload}
          onConfirm={(c) => {
            setCalibration(c);
            setStep("ball");
          }}
          onBack={() => setStep("upload")}
        />
      )}
      {step === "ball" && (
        <BallTypeStep
          selected={ballType}
          onSelect={(b) => {
            setBallType(b);
            setStep("review");
          }}
          onBack={() => setStep("calibrate")}
        />
      )}
      {step === "review" && upload && calibration && ballType && (
        <ReviewStep
          upload={upload}
          calibration={calibration}
          ballType={ballType}
          onBack={() => setStep("ball")}
        />
      )}
      <JobList />
    </div>
  );
}

const STEP_LABELS: Record<Step, string> = {
  upload: "Upload",
  calibrate: "Calibrate",
  ball: "Ball",
  review: "Review",
};

function Stepper({ step }: { step: Step }) {
  const order: Step[] = ["upload", "calibrate", "ball", "review"];
  const idx = order.indexOf(step);
  return (
    <div className="stepper">
      {order.map((s, i) => (
        <div
          key={s}
          className={`step ${i === idx ? "active" : ""} ${i < idx ? "done" : ""}`}
        >
          {i + 1}. {STEP_LABELS[s]}
        </div>
      ))}
    </div>
  );
}
