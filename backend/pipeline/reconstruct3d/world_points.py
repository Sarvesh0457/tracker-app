"""Known 3D positions of the 6 calibration points, in metres.

Coordinate frame (right-handed; flipped to Unity left-handed at export time):
  origin = middle stump base at STRIKER'S end
  +X     = one side of pitch (mapped from LEFT-in-frame convention at click time)
  +Y     = up
  +Z     = toward the bowler / non-striker's end (down the pitch)

Cricket regulation dimensions used:
  pitch length, stump to stump ........... 20.12 m  (22 yards)
  popping crease, in front of stumps ..... 1.22 m   (4 ft)
  return crease, half-width from middle .. 1.32 m   (4 ft 4 in)

The 6 calibration points — all on the ground plane (Y=0):
  0  popping_striker_leg      — striker end popping × leg return crease
  1  popping_striker_off      — striker end popping × off return crease
  2  popping_nonstriker_leg   — non-striker end popping × leg return crease
  3  popping_nonstriker_off   — non-striker end popping × off return crease
  4  stump_base_striker       — middle stump base, striker's end (origin)
  5  stump_base_nonstriker    — middle stump base, non-striker's end

NOTE: All 6 points are coplanar (Y=0). This is a degenerate configuration for
single-view calibration — focal length and camera distance are ambiguous.
The trajectory fit compensates with tight physical bounds (v_z ≤ -20 m/s
for real deliveries, bounce Y=0 constraint) that keep the reconstructed 3D
trajectory physical despite the calibration scale ambiguity.
"""

PITCH_LENGTH_M = 20.12
POPPING_IN_FRONT_M = 1.22
RETURN_HALF_WIDTH_M = 1.32
STUMP_HEIGHT_M = 0.711
BALL_DIAMETER_M = 0.072
BALL_RADIUS_M = BALL_DIAMETER_M / 2.0

KEYPOINT_NAMES = [
    "popping_striker_leg",
    "popping_striker_off",
    "popping_nonstriker_leg",
    "popping_nonstriker_off",
    "stump_base_striker",
    "stump_base_nonstriker",
]

WORLD_POINTS_M = [
    (-RETURN_HALF_WIDTH_M, 0.0, +POPPING_IN_FRONT_M),                       # 0
    (+RETURN_HALF_WIDTH_M, 0.0, +POPPING_IN_FRONT_M),                       # 1
    (-RETURN_HALF_WIDTH_M, 0.0, PITCH_LENGTH_M - POPPING_IN_FRONT_M),       # 2
    (+RETURN_HALF_WIDTH_M, 0.0, PITCH_LENGTH_M - POPPING_IN_FRONT_M),       # 3
    ( 0.0,                 0.0, 0.0),                                       # 4
    ( 0.0,                 0.0, PITCH_LENGTH_M),                            # 5
]
