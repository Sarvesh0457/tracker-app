from pathlib import Path

# ── WEIGHTS AND INPUTS ────────────────────────────────────────────────────────
WEIGHTS_PATH = Path("D:\\intership\\may\model\\Weights\\red\\red20.onnx")
VIDEO_PATH = Path("C:\\Users\\Mukund\\Downloads\\4_013_06.mp4")
OUTPUT_DIR = Path('D:\\intership\\OUTPUT')

# ── MODEL THRESHOLDS & PARAMS ──────────────────────────────────────────────────
# Two-stage ball-only detection:
#   PRIMARY: high-confidence anchor detections. Used to establish the track.
#   RECOVERY: lower floor for on-trajectory recovery. Once Kalman tracking is
#     initialized (i.e. we have a confirmed ball path), the detector accepts
#     candidates down to this floor because the Kalman Mahalanobis gate and
#     MAX_POSITION_JUMP filter will reject anything off the ball's path.
# This gives us the "keep the low-confidence detection IF it's on the ball's
# path" behavior in one clean rule, without a second full pass over the video.
CONF_THRESH_BALL_PRIMARY  = 0.55
CONF_THRESH_BALL_RECOVERY = 0.30

# Back-compat alias — some legacy call sites (release_detector, tests) still
# reference CONF_THRESH_BALL; keep it pointing at the primary floor.
CONF_THRESH_BALL = CONF_THRESH_BALL_PRIMARY

IOU_THRESH = 0.45        # Slightly higher IOU for NMS
IMGSZ = 1280   # Default detection scale — 960 was tight for wide-angle / small-
               # ball footage; 1280 preserves ~1.33× more pixel on the ball.

# ── SIZE FILTERS ──────────────────────────────────────────────────────────────
# Ball on far-camera / Test-match footage measures ~5–10 px across (verified
# on 4_025_05.mp4: f=208 conf 0.69, wh=6x6). The old 10×10 / area-100 floor
# rejected every real ball as "too small" on those clips. The model's own
# confidence already gates noise; geometry just needs to admit small balls.
MIN_BOX_AREA = 30        # was 100. 5×5=25 admits FP kit blobs; 6×6=36 keeps
                         # genuine small-ball hits and drops 4×3/3×3 spurious ones.
MAX_BOX_AREA = 50000
MIN_BOX_WIDTH = 6        # was 10
MIN_BOX_HEIGHT = 6       # was 10
MAX_BOX_WIDTH_BALL = 150 # Ball shouldn't be too large
MAX_BOX_HEIGHT_BALL = 150
MAX_BOX_WIDTH_BAT = 600  # Bat can be longer
MAX_BOX_HEIGHT_BAT = 400

# ── ASPECT RATIO FILTERS ──────────────────────────────────────────────────────
# Widened from 0.5/2.0 for the YOLOv8m export — boxes hug tighter and
# motion-blurred balls elongate more, so the old bounds were false-rejecting
# real detections and breaking the release detector's consecutive-frame streak.
MIN_ASPECT_RATIO_BALL = 0.4
MAX_ASPECT_RATIO_BALL = 2.5
MIN_ASPECT_RATIO_BAT = 0.3   # Bat can be elongated
MAX_ASPECT_RATIO_BAT = 5.0

# ── TEMPORAL CONSISTENCY ──────────────────────────────────────────────────────
# Raised from 300 for the YOLOv8m export — the new model misses a frame here
# and there, and after a 1-frame gap at 30 fps the ball has moved ~200-350 px.
# Kalman gate still catches wild jumps independently.
MAX_POSITION_JUMP = 500  # Max pixels ball/bat can move between frames

# If no ball is accepted for this many consecutive frames, the tracker state
# (Kalman filter + last_ball_pos) is reset so the ball can be re-acquired
# anywhere in the frame. Without this, one early false positive poisoned
# last_ball_pos and the MAX_POSITION_JUMP check rejected the real ball for
# the rest of the video (the wide-angle failure mode).
BALL_REACQUIRE_FRAMES = 12

# ── CLASS MAP & COLORS ────────────────────────────────────────────────────────
CLASS_NAMES = {0: "ball", 1: "bat"}
COLOR_BALL = (255, 255, 255)  # White (BGR)
COLOR_BAT = (0, 255, 0)       # Green (BGR)
COLOR_TEXT_FG = (255, 255, 255)

# ── TRAIL CONFIG ──────────────────────────────────────────────────────────────
TRAIL_LENGTH      = 120      # frames of history to keep visible

TRAIL_GRADIENT = [
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
    (255, 255, 255),
]

# Tube rendering
TRAIL_THICKNESS   = 6        # core tube width
TRAIL_ALPHA       = 0.92     # overall trail opacity

# Opacity falloff — oldest point gets this fraction of trail_alpha
TRAIL_TAIL_FADE   = 0.15

# Live ball glow dot at current position
BALL_GLOW_RADIUS      = 9
BALL_GLOW_INNER       = 5
BALL_GLOW_COLOR       = (0, 255, 255)   # bright cyan-white (BGR)
BALL_GLOW_ALPHA       = 0.85

# ── OCCLUSION INTERPOLATION ───────────────────────────────────────────────────
# Method used to fill frames where the ball was not detected.
# Options: "linear" | "cubic_spline" | "bezier" | "physics"
INTERP_METHOD          = "cubic_spline"

# Maximum consecutive missing frames we'll attempt to fill (skip if longer)
INTERP_MAX_GAP_FRAMES  = 20

# Number of real detections on each side of the gap used as anchors
INTERP_ANCHOR_FRAMES   = 4

# Gravity in pixel-space (px per frame²). Tune to your video resolution & fps.
# At 30 fps, 1080p: ~0.35 is a reasonable starting point.
GRAVITY_PX_PER_FRAME2  = 0.35

# Y pixel row of the pitch surface for bounce-inside-gap estimation.
# Set to an int if you know it, or leave None for auto-estimation.
PITCH_Y                = None

# Colour and style for synthesised (interpolated) trail segments
INTERP_COLOR           = (200, 180, 0)   # warm gold (BGR)
INTERP_DASH_GAP        = 6              # pixels between dashes on the synth trail


# ── RELEASE POINT DETECTION ───────────────────────────────────────────────────
# Minimum ball speed (px/frame) before release detection is even attempted.
# Keeps false triggers while the ball is stationary in the hand.
RELEASE_MIN_SPEED         = 6.0

# Speed must jump by at least this factor compared to the recent average.
# e.g. 2.0 = ball must suddenly be moving 2× faster than the preceding frames.
RELEASE_JERK_FACTOR       = 2.0

# Number of frames in the "before" window used to compute the average speed
# and direction prior to the candidate release frame.
RELEASE_WINDOW_PRE        = 5

# After a candidate release frame, the velocity direction must be consistent
# over this many frames (low angular variance = ballistic flight).
RELEASE_WINDOW_POST       = 4

# Maximum allowed angular spread (radians) of velocity direction in the
# post-release window for the detection to be confirmed.
RELEASE_DIR_MAX_STD       = 0.45   # ~26°

# How many frames to display the release marker on-screen.
RELEASE_DISPLAY_FRAMES    = 120

# Visual style
RELEASE_COLOR             = (0, 255, 180)   # bright green-cyan (BGR)
RELEASE_MARKER_RADIUS     = 10


# ── BOUNCE POINT DETECTION ────────────────────────────────────────────────────
# Bounce = vertical velocity sign flip on real detections (descending → ascending).

# Minimum |vy| (px/frame) before the bounce to qualify (filters apex jitter).
# On oblique/wide cameras the ball's motion projects mostly onto x — actual
# per-frame vy around the bounce can be as low as 0.5-1.0 px. Sign flip is
# still required, which is what actually rules out apex jitter (a random
# noisy detection is unlikely to produce a coherent flip across a triplet).
BOUNCE_MIN_DESCENT_VY     = 0.5
BOUNCE_MIN_ASCENT_VY      = 0.3

# Marker size (cricket_ball.png overlay) — matches the release marker size.
BOUNCE_MARKER_SIZE        = 17


# ── IMPACT POINT DETECTION ────────────────────────────────────────────────────
# Impact = ball drifts off the predicted post-bounce parabolic path for
# IMPACT_CONSECUTIVE_FRAMES consecutive real detections. The impact point is
# marked at the FIRST off-path frame in that streak.
#
# Fallback: if the ball goes missing for IMPACT_DISAPPEAR_FRAMES consecutive
# frames post-bounce while it was last seen ON the predicted path, the
# last seen position is taken as the impact point.

# Pixel deviation from the predicted parabola that counts as "off path".
IMPACT_DEVIATION_PX       = 25

# Number of consecutive off-path real detections required to confirm impact.
# Setting this to 1 made any single jittery detection lock impact prematurely;
# 3 is the smallest streak that filters out detection-box wobble without
# missing a real deflection.
IMPACT_CONSECUTIVE_FRAMES = 3

# Number of consecutive missing-ball frames (post-bounce, last seen on path)
# that trigger the vanish fallback.
IMPACT_DISAPPEAR_FRAMES   = 5

# Marker size (cricket_ball.png overlay) — matches release/bounce marker size.
IMPACT_MARKER_SIZE        = 16

# ── IMPACT SECOND-PASS (DIRECTION-CHANGE FALLBACK) ────────────────────────────
# If the first-pass parabola-deviation rule fails to find an impact, finalize()
# replays the post-bounce ball history and compares the ACTUAL per-frame
# velocity vector to the PREDICTED (gravity-corrected) one. A sustained angle
# divergence indicates a direction-change impact (edge / glance / pad).

# Minimum angle (degrees) between actual and predicted velocity vectors that
# counts as a direction change.
IMPACT_DIRECTION_ANGLE_DEG = 25

# Number of consecutive direction-mismatch frames required to confirm impact.
IMPACT_DIRECTION_CONSECUTIVE = 2

