"""Blade break-through: "is the blade still in the steel, or through it?"

Plain language:
  While the blade is in the steel the sound is dull and heavy; the moment it comes
  through, the brightness comes back (loudness hardly changes). We learn ONE direction
  in the measurement space that separates "in the steel" from "through", using every
  good phone cut: the middle of each cut is "in the steel", the last seconds before the
  motor stops are "through" (or, better, the moment the user tapped "Blade is through").

  Live, each half-second gets a load number: about -1 while cutting, about +1 once
  through. The indicator only fires after the blade has clearly been IN the steel for a
  while, so the idle motor at the start (which also sounds "through") can't trigger it.

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
        self.mid: np.ndarray | None = None    # halfway point between the two sounds

    # ----- learning -----
    def fit(self, loaded: list[np.ndarray], unloaded: list[np.ndarray]) -> "BreakthroughModel":
        """loaded / unloaded: standardised feature rows (one array per cut) for the two states."""
        L, U = np.vstack(loaded).astype(float), np.vstack(unloaded).astype(float)
        pooled = LedoitWolf().fit(np.vstack([L - L.mean(0), U - U.mean(0)]))
        w = pooled.precision_ @ (U.mean(0) - L.mean(0))
        mid = (U.mean(0) + L.mean(0)) / 2
        scale = (((U - mid) @ w).mean() - ((L - mid) @ w).mean()) / 2    # loaded ~ -1, through ~ +1
        self.w, self.mid = w / (scale if scale > 0 else 1.0), mid
        return self

    # ----- scoring -----
    def load(self, Z: np.ndarray) -> np.ndarray:
        """Raw per-window load number for standardised features Z (n_windows, n_features)."""
        return (Z - self.mid) @ self.w

    def track(self, raw: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, int | None]:
        """Causal state machine, identical to the phone. Returns (smoothed load, index of the
        window where "blade is through" fired, or None)."""
        out = np.full(len(raw), np.nan)
        hist: list[float] = []
        loaded_run = hold = 0
        in_steel, fired = False, None
        for i, (r, p) in enumerate(zip(raw, phases)):
            if p != RUNNING:
                hist, loaded_run, hold, in_steel = [], 0, 0, False
                continue
            hist.append(float(r)); hist = hist[-config.BT_SMOOTH_WINDOWS:]
            sm = float(np.mean(hist)); out[i] = sm
            if not in_steel:
                loaded_run = loaded_run + 1 if sm < config.BT_LOADED_LEVEL else 0
                if loaded_run >= config.BT_LOADED_WINDOWS:
                    in_steel = True
            else:
                hold = hold + 1 if sm > config.BT_FIRE_LEVEL else 0
                if hold >= config.BT_HOLD_WINDOWS and fired is None:
                    fired = i
        return out, fired

    # ----- saving -----
    def to_dict(self) -> dict:
        return {"w": self.w.round(8).tolist(), "mid": self.mid.round(8).tolist(),
                "smooth_windows": config.BT_SMOOTH_WINDOWS, "hold_windows": config.BT_HOLD_WINDOWS,
                "loaded_windows": config.BT_LOADED_WINDOWS, "loaded_level": config.BT_LOADED_LEVEL,
                "fire_level": config.BT_FIRE_LEVEL}

    @classmethod
    def from_dict(cls, d: dict) -> "BreakthroughModel":
        m = cls(); m.w, m.mid = np.array(d["w"]), np.array(d["mid"]); return m


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
