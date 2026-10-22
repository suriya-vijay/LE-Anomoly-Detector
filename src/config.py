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
MIN_CUT_S = 30.0
MOTOR_OFF_BELOW_LOUD_DB = 6.5  # phone cuts: motor counts as OFF this far under the loud centre (through-phase is ~4 dB under, off >= 8)          # OFF after at least this long ON counts as "cut finished"

# Detector: which measurements go into the "distance from normal".
# dominant_hz and pulse_hz jump around on real recordings, so they're shown but not scored.
DISTANCE_EXCLUDE = ("dominant_hz", "pulse_hz")
N_CLUSTERS = 3            # kinds of normal (e.g. early cut, steady cut)
SMOOTH_WINDOWS = 3        # scores are averaged over this many windows (1.5 s)

# Alert thresholds, as percentiles of normal scores (set during training)
YELLOW_PERCENTILE = 99.0
RED_PERCENTILE = 99.9
RED_AFTER_YELLOW_SECONDS = 3.0
STOP_CONFIRM_SECONDS = 6.0  # a STOP shows only once its reason has stood this long with the motor still running (yellow until then):
                            # the motor winding down after switch-off read as STOP for 1-5 s on cuts 13/20/23/24; the real fault (cut 11) lasted 15 s
RED_MIN_RATIO = 1.25      # red is never closer than 25% above yellow (matters with few recordings)
LINE_SPREAD_SDS = 1.0     # yellow = median of the per-cut 99th percentiles + this many robust (MAD) SDs of them;
                          # a pooled percentile let a few quiet cuts drag the line down (2026-10-15: 6 of 14 good cuts STOP unseen)
LOUD_RISE_DB = 6.0        # plain 'too loud' STOP: 10 s loudness this far above the cut's own first-minute median
                          # (cut 11, the real should-stop event: +8.4 dB for 16 s; loudest good cut +4.0 dB)
LOUD_RISE_HOLD_SECONDS = 2.0
LOUD_BASE_SECONDS = 60.0  # the cut's own loudness reference = median of its first minute of running
LOUD_SMOOTH_SECONDS = 10.0

# Fraction of normal runs held back for the final honest test
HOLDOUT_FRACTION = 0.30
RANDOM_SEED = 42

# --- blade break-through indicator (src/breakthrough.py) ---
BT_SMOOTH_WINDOWS = 20      # load number is averaged over the last 10 s
BT_MIN_RUN_WINDOWS = 120    # no guess in the first 60 s of running (was 120 s: cut 13 ended at 1:32 with no guess)
BT_LOOKBACK_WINDOWS = 360   # the cut's own "low" = BT_LOW_PERCENTILE of the last 180 s of load numbers
BT_LOW_PERCENTILE = 10
BT_DELTA = 1.0              # "through" fires when the load is this far above the cut's own low ... (0.8 until 2026-10-15: guesses ran 20 s early)
BT_HOLD_WINDOWS = 30        # ... for 15 s in a row (was 10 s)
BT_RETRACT_DELTA = 0.5      # the guess is taken back if the load falls under low + this for 10 s
BT_QUIET_DB = 0.5           # ... and the sound must also be this much quieter (10 s mean of rms_db) than ...
BT_QUIET_REF_WINDOWS = 120  # ... its median over the 60 s before that (cut 15: load crept up for 2 min with no through)
BT_STAGE2_HOLD_WINDOWS = 10  # banner ("coming through"): the heads-up must stand AND stay quieter for 5 s more
BT_BRIGHT_PCT = 0.0         # optional banner gate: 10 s brightness this fraction above the previous minute (0 = off;
                            # 3% fixed cut 15 but lost cuts 12 and 14 on 2026-10-15, 2% changed nothing)
FINISH_WINDOW_S = (-60, 5)   # scorecard target (fixed 2026-10-22 before cuts 28+): 'Finishing' shows 0-60 s before the user's
                            # 'through' tap, never more than 5 s after. The sound slides over the last 20-60 s, so an exact
                            # 15 s target was out of reach (rule 7/21, learned model ~9/19); this one: 15 of 21 unseen
BT_PROGRESS_SPAN = 1.5      # the page's progress bar is full when the load is this far above the cut's own low
BT_TAIL_WINDOWS = 30        # without a user mark: the last 15 s before motor-off count as "through"
