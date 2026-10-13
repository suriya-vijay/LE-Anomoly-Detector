"""Step 4a: is the motor running? Which windows should the meter judge?

Plain language: door slams, the table moving and people walking about all happen
while the motor is OFF. We don't want alarms for those, so each window is first
labelled:

  "off"         motor not running  -> the meter shows grey, "not judging"
  "transition"  spinning up or down -> also grey
  "running"     motor on and steady -> judged (green / yellow / red)

The rule is simple: the motor is ON when the loudness stays above a threshold for
a few windows in a row. The threshold is learned from the normal recordings
(halfway between the quiet "off" level and the steady "on" level).
"""
from __future__ import annotations

import numpy as np

from . import config
from .features import feature_names

OFF, TRANSITION, RUNNING = "off", "transition", "running"

_RMS = feature_names().index("rms_db")
_PULSE = feature_names().index("pulse_depth")


MIN_ON_OFF_GAP_DB = 4.0


def motor_on_threshold(rms_db: np.ndarray) -> float:
    """Loudness (dB) halfway between the quiet and the loud group of windows.

    A tiny 2-means split: start from the quietest windows (1st percentile) and the
    loud ones (90th) and move each centre to the mean of the windows nearest to it
    until nothing changes. Starting from the very bottom matters: a cut saved by the
    phone page has only a few seconds of motor-off, and starting at the 10th
    percentile would split the loud part in two and miss half the cut."""
    x = np.asarray(rms_db, dtype=float)
    lo, hi = np.percentile(x, 1), np.percentile(x, 90)
    for _ in range(50):
        near_lo = np.abs(x - lo) <= np.abs(x - hi)
        new_lo, new_hi = x[near_lo].mean(), x[~near_lo].mean() if (~near_lo).any() else hi
        if np.isclose(new_lo, lo) and np.isclose(new_hi, hi):
            break
        lo, hi = new_lo, new_hi
    if hi - lo < MIN_ON_OFF_GAP_DB:          # no real quiet part: assume everything is "on"
        return float(hi - MIN_ON_OFF_GAP_DB)
    return float((lo + hi) / 2)


def motor_state(features: np.ndarray, threshold: float,
                on_windows: int = config.MOTOR_ON_WINDOWS,
                off_windows: int = config.MOTOR_OFF_WINDOWS,
                max_pulse: float = config.MAX_PULSE_FOR_ON) -> np.ndarray:
    """True for every window where the motor is ON (with hysteresis so short
    bangs don't flip it). Works window by window, so the phone can do the same live."""
    loud = features[:, _RMS] > threshold
    steady = features[:, _PULSE] < max_pulse
    on = np.zeros(len(features), dtype=bool)
    state, run, on_since = False, 0, 0
    for i in range(len(features)):
        if not state:
            run = run + 1 if (loud[i] and steady[i]) else 0
            if run >= on_windows:
                state, run, on_since = True, 0, i - on_windows + 1
                on[on_since:i + 1] = True            # the windows that convinced us count too
            continue
        on[i] = True
        # OFF when most of the recent stretch is quiet. Brief bangs while handling the
        # sample after the motor stops don't keep it "on"; a loud thumping cut stays "on"
        # so that it gets judged (and goes red) instead of silently switching to grey.
        lo = max(on_since, i - 2 * off_windows + 1)
        recent_quiet = ~loud[lo:i + 1]
        if recent_quiet.sum() >= off_windows:
            state = False
            first_quiet = lo + int(np.argmax(recent_quiet))
            on[first_quiet:i + 1] = False
    return on


def label_phases(features: np.ndarray, threshold: float,
                 hop_s: float = config.HOP_SECONDS) -> np.ndarray:
    """Per-window label: "off", "transition" or "running"."""
    on = motor_state(features, threshold)
    labels = np.where(on, RUNNING, OFF).astype(object)
    spinup = int(round(config.SPINUP_S / hop_s))
    starts = np.flatnonzero(np.diff(on.astype(int), prepend=0) == 1)
    stops = np.flatnonzero(np.diff(on.astype(int), append=0) == -1)   # last ON window of each stretch
    for s in starts:
        labels[s:s + spinup] = np.where(on[s:s + spinup], TRANSITION, OFF)
    for e in stops:
        lo = max(0, e - config.SPINDOWN_WINDOWS + 1)
        labels[lo:e + 1] = np.where(on[lo:e + 1], TRANSITION, OFF)
    return labels


def on_off_times(labels: np.ndarray, hop_s: float = config.HOP_SECONDS) -> list[tuple[float, float]]:
    """[(on_time_s, off_time_s), ...] for each stretch the motor was on."""
    on = labels != OFF
    starts = np.flatnonzero(np.diff(on.astype(int), prepend=0) == 1)
    stops = np.flatnonzero(np.diff(on.astype(int), append=0) == -1)
    return [(float(s * hop_s), float((e + 1) * hop_s + config.WINDOW_SECONDS - hop_s)) for s, e in zip(starts, stops)]


def cut_done_events(labels: np.ndarray, hop_s: float = config.HOP_SECONDS) -> list[float]:
    """Times (s) at which a cut finished: the motor went OFF after >= MIN_CUT_S on."""
    return [off for on, off in on_off_times(labels, hop_s) if off - on >= config.MIN_CUT_S]


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
