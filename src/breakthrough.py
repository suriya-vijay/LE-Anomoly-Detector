"""Blade break-through: "is the blade still in the steel, or through it?"

Plain language:
  While the blade is in the steel the sound is dull and heavy; the moment it comes
  through, the brightness comes back (loudness hardly changes). We calculate ONE direction
  in the measurement space that separates "in the steel" from "through", using every
  good phone cut: the middle of each cut is "in the steel", the last seconds before the
  motor stops are "through" (or, better, the moment the user tapped "Blade is through").

  Every cut is first centred on its OWN "in the steel" sound before the direction is
  calculated, because the phone's position and the phone itself shift all the numbers
  from one cut to the next (cut 4 on 2026-10-14 sat on the "through" side of an absolute
  rule for its whole length and was missed).

  Live, each half-second gets a load number (about -1 in the steel, +1 through, give or
  take the cut's own offset). The indicator does not compare that number with a fixed
  line; it compares it with the cut's OWN low of the last few minutes and fires when the
  number has risen clearly above that low for 10 s. If the number then falls back, the
  guess is taken back (the page hides its banner) and a later rise can fire again.

The same maths runs on the phone (web/detector.js, class Breakthrough).
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from . import config
from .phases import OFF, RUNNING


class BreakthroughModel:
    def __init__(self):
        self.w: np.ndarray | None = None      # direction in standardised feature space
        self.mid: np.ndarray | None = None    # point where the load number reads -1 for a typical cut

    # ----- calculating the direction -----
    def fit(self, loaded: list[np.ndarray], unloaded: list[np.ndarray]) -> "BreakthroughModel":
        """loaded / unloaded: standardised feature rows, one array per cut, for the two states.
        Each cut is centred on its own 'in the steel' median first (phone offsets cancel)."""
        bases = [np.median(L, axis=0) for L in loaded]
        Lc = np.vstack([L - b for L, b in zip(loaded, bases)]).astype(float)
        Uc = np.vstack([U - b for U, b in zip(unloaded, bases)]).astype(float)
        pooled = LedoitWolf().fit(np.vstack([Lc - Lc.mean(0), Uc - Uc.mean(0)]))
        w = pooled.precision_ @ (Uc.mean(0) - Lc.mean(0))
        scale = ((Uc @ w).mean() - (Lc @ w).mean()) / 2          # loaded ~ -1, through ~ +1
        w = w / (scale if scale > 0 else 1.0)
        mid = np.mean(bases, axis=0)
        offset = -1.0 - (Lc @ w).mean()                          # fold "-1" into mid: (Z - mid').w = (Z - mid).w + offset
        self.w, self.mid = w, mid - offset * w / float(w @ w)
        return self

    # ----- scoring -----
    def load(self, Z: np.ndarray) -> np.ndarray:
        """Raw per-window load number for standardised features Z (n_windows, n_features)."""
        return (Z - self.mid) @ self.w

    def track(self, raw: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, list[list[int | None]]]:
        """Causal state machine, identical to the phone. Returns (smoothed load, events), where
        events is a list of [fired_index, taken_back_index or None], in order."""
        out = np.full(len(raw), np.nan)
        hist: list[float] = []
        recent: list[float] = []
        run = hold = 0
        fired = False
        low_at_fire = 0.0
        events: list[list[int | None]] = []
        for i, (r, p) in enumerate(zip(raw, phases)):
            if p != RUNNING:
                hist, recent, run, hold, fired = [], [], 0, 0, False
                continue
            hist.append(float(r)); hist = hist[-config.BT_SMOOTH_WINDOWS:]
            sm = float(np.mean(hist)); out[i] = sm
            run += 1
            recent.append(sm); recent = recent[-config.BT_LOOKBACK_WINDOWS:]
            low = float(np.percentile(recent, config.BT_LOW_PERCENTILE))
            if not fired:
                hold = hold + 1 if (run >= config.BT_MIN_RUN_WINDOWS and sm > low + config.BT_DELTA) else 0
                if hold >= config.BT_HOLD_WINDOWS:
                    fired, low_at_fire, hold = True, low, 0
                    events.append([i, None])
            else:
                hold = hold + 1 if sm < low_at_fire + config.BT_RETRACT_DELTA else 0
                if hold >= config.BT_HOLD_WINDOWS:
                    fired, hold = False, 0
                    events[-1][1] = i
        return out, events

    # ----- saving -----
    def to_dict(self) -> dict:
        return {"w": self.w.round(8).tolist(), "mid": self.mid.round(8).tolist(),
                "smooth_windows": config.BT_SMOOTH_WINDOWS, "hold_windows": config.BT_HOLD_WINDOWS,
                "min_run_windows": config.BT_MIN_RUN_WINDOWS, "lookback_windows": config.BT_LOOKBACK_WINDOWS,
                "low_percentile": config.BT_LOW_PERCENTILE, "delta": config.BT_DELTA,
                "retract_delta": config.BT_RETRACT_DELTA}

    @classmethod
    def from_dict(cls, d: dict) -> "BreakthroughModel":
        m = cls(); m.w, m.mid = np.array(d["w"]), np.array(d["mid"]); return m


def fired_index(events: list[list[int | None]]) -> int | None:
    """The guess that stood at the end: the last fire that was not taken back, else None."""
    standing = [f for f, back in events if back is None]
    return standing[-1] if standing else None


def taken_back(events: list[list[int | None]]) -> int:
    """How many guesses were taken back (false alarms the page would have shown briefly)."""
    return sum(back is not None for _, back in events)


def segments(Z: np.ndarray, phases: np.ndarray, through_at_s: float | None = None) -> tuple[np.ndarray, np.ndarray] | None:
    """Pick the 'in the steel' and 'through' rows of one cut.

    With a "Blade is through" mark: in-the-steel = 20 s after the cut starts up to 10 s before
    the mark; through = 5 s after the mark to the end. Without one (weak labels): the middle
    40% of the cut vs the last BT_TAIL_WINDOWS before the motor stops. None if too short."""
    run = np.flatnonzero(phases == RUNNING)
    if len(run) < 4 * config.BT_TAIL_WINDOWS:
        return None
    hop = config.HOP_SECONDS
    if through_at_s is not None:
        on = int(np.flatnonzero(phases != OFF)[0])            # the page counts from the first non-off window
        k = on + int(round(through_at_s / hop))
        loaded = run[(run >= run[0] + int(20 / hop)) & (run <= k - int(10 / hop))]
        unloaded = run[run >= k + int(5 / hop)]
        if len(loaded) >= 20 and len(unloaded) >= 10:
            return Z[loaded], Z[unloaded]
    n = len(run)
    return Z[run[int(0.3 * n):int(0.7 * n)]], Z[run[-config.BT_TAIL_WINDOWS:]]
