"""Central settings. Change numbers here instead of hunting through the code."""

# Audio is resampled to this rate before analysis.
# 22,050 Hz keeps sounds up to ~11 kHz, which covers saw motor and blade noise.
# (MIMII public data is 16,000 Hz; pass sr=16000 when using it.)
SAMPLE_RATE = 22050

# Each "window" is a short slice of audio that gets one score.
WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.5  # windows overlap by half, so a new score every 0.5 s

# Feature settings
N_FFT = 2048
FRAME_HOP = 512
N_MELS = 64

# Phases: the meter only judges while the motor is running.
# The motor-on loudness threshold itself is learned from the normal recordings.
MOTOR_ON_WINDOWS = 3      # loudness must stay above the threshold this many windows to count as ON
MOTOR_OFF_WINDOWS = 4     # ... and below it this many windows (2 s) to count as OFF
MAX_PULSE_FOR_ON = 0.35   # a door slam / table bump (big pulse_depth) can't switch the motor "on"
SPINUP_S = 10.0           # first seconds after ON are "transition" (spin-up / blade lowering), not judged
SPINDOWN_WINDOWS = 2      # last windows before OFF are "transition" (spin-down)
MIN_CUT_S = 30.0          # OFF after at least this long ON counts as "cut finished"

# Detector: which measurements go into the "distance from normal".
# dominant_hz and pulse_hz jump around on real recordings, so they're shown but not scored.
DISTANCE_EXCLUDE = ("dominant_hz", "pulse_hz")
N_CLUSTERS = 3            # kinds of normal (e.g. early cut, steady cut)
SMOOTH_WINDOWS = 3        # scores are averaged over this many windows (1.5 s)

# Alert thresholds, as percentiles of normal scores (set during training)
YELLOW_PERCENTILE = 99.0
RED_PERCENTILE = 99.9
RED_AFTER_YELLOW_SECONDS = 3.0
RED_MIN_RATIO = 1.25      # red is never closer than 25% above yellow (matters with few recordings)

# Fraction of normal runs held back for the final honest test
HOLDOUT_FRACTION = 0.30
RANDOM_SEED = 42

# --- blade break-through indicator (src/breakthrough.py) ---
BT_SMOOTH_WINDOWS = 20      # load number is averaged over the last 10 s
BT_LOADED_WINDOWS = 120     # the blade must read "in the steel" for 60 s before "through" can fire
BT_LOADED_LEVEL = -0.3      # ... meaning the load number stayed below this
BT_HOLD_WINDOWS = 20        # "through" fires after 10 s above BT_FIRE_LEVEL
BT_FIRE_LEVEL = 0.0
BT_TAIL_WINDOWS = 30        # without a user mark: the last 15 s before motor-off count as "through"
