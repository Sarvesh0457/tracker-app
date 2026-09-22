"""Typed errors raised by process_video(). Each maps to a spec §10 error_code."""


class PipelineError(Exception):
    code = "INTERNAL"

    def __init__(self, message: str = ""):
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__


class InvalidVideoError(PipelineError):
    code = "INVALID_VIDEO"


class InvalidCalibrationError(PipelineError):
    code = "INVALID_CALIBRATION"


class ModelLoadFailedError(PipelineError):
    code = "MODEL_LOAD_FAILED"


class NoBallDetectedError(PipelineError):
    code = "NO_BALL_DETECTED"


class HomographyFailedError(PipelineError):
    code = "HOMOGRAPHY_FAILED"


class EncodingFailedError(PipelineError):
    code = "ENCODING_FAILED"


class PipelineCancelled(PipelineError):
    code = "CANCELLED"
