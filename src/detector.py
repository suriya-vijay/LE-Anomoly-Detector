"""The detector learns a few "kinds of normal" from my good cuts and scores every half second by how far it is from the closest one, so a big score means the saw sounds weird (just statistics, no neural network)."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

from . import config
from .features import (ENVELOPE_FRAME_S, PULSE_FFT, PULSE_MAX_HZ, PULSE_MIN_HZ, ROLLOFF_FRACTION,
                       feature_names, mel_filterbank)
from .phases import OFF, RUNNING

GOOD, ODD, STOP = "good", "odd", "stop"
MIN_CLUSTER_WINDOWS = 20


class SawDetector:
    def __init__(self, n_clusters: int = config.N_CLUSTERS, seed: int = config.RANDOM_SEED):
        self.n_clusters = n_clusters
        self.seed = seed
        names = feature_names()
        self.feature_names = names
        self.distance_index = [i for i, n in enumerate(names) if n not in config.DISTANCE_EXCLUDE]
        self.mean = self.std = None
        self.clusters: list[dict] = []
        self.yellow = self.red = None
        self.thresholds_by_source: dict[str, tuple[float, float]] = {}
        self.breakthrough = None              # BreakthroughModel or None (needs >= 2 phone cuts)
        self.motor_threshold_db = None
        self.trained_on: list[str] = []
        self.sr = config.SAMPLE_RATE

    # ----- learning -----
    def fit(self, runs_features: list[np.ndarray], groups: list[str] | None = None) -> "SawDetector":
        """runs_features: one array (n_windows, n_features) of RUNNING windows per normal run.
        groups: optional capture source per run (e.g. "file" / "phone"). Each source gets its
        own kinds of normal, so a louder phone-on-the-housing cut can't crowd out the others."""
        groups = groups or ["all"] * len(runs_features)
        X = np.vstack(runs_features)[:, self.distance_index].astype(float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0) + 1e-6
        self.clusters = []
        for g in dict.fromkeys(groups):                      # unique, in order
            Xg = np.vstack([f for f, s in zip(runs_features, groups) if s == g])[:, self.distance_index].astype(float)
            Z = (Xg - self.mean) / self.std
            k = max(1, min(self.n_clusters, len(Z) // MIN_CLUSTER_WINDOWS))
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.seed).fit_predict(Z) if k > 1 else np.zeros(len(Z), int)
            for c in range(k):
                Zc = Z[labels == c]
                if len(Zc) < 2:
                    continue
                lw = LedoitWolf().fit(Zc)
                self.clusters.append({"mean": lw.location_, "precision": lw.precision_, "n_windows": int(len(Zc)), "source": g})
        return self

    # ----- scoring -----
    def raw_scores(self, features: np.ndarray) -> np.ndarray:
        """Per-window distance to the nearest kind of normal (no smoothing)."""
        Z = (features[:, self.distance_index] - self.mean) / self.std
        d2 = np.full(len(Z), np.inf)
        for c in self.clusters:
            diff = Z - c["mean"]
            d2 = np.minimum(d2, np.einsum("ij,jk,ik->i", diff, c["precision"], diff))
        return np.sqrt(np.maximum(d2, 0) / len(self.distance_index))

    def score(self, features: np.ndarray) -> np.ndarray:
        """Smoothed score: trailing average over SMOOTH_WINDOWS (same as the phone does live)."""
        return smooth(self.raw_scores(features))

    def verdicts(self, scores: np.ndarray, phases: np.ndarray, rms: np.ndarray | None = None) -> np.ndarray:
        """Per window: "off" (not judged), "good", "odd" or "stop". STOP when the score crosses red or (when rms
        is given) when the cut gets LOUD_RISE_DB louder than its own first minute for LOUD_RISE_HOLD_SECONDS: the
        plain 'too loud' rule. A long stretch over yellow but under red stays ODD. A STOP reason must stand for
        STOP_CONFIRM_SECONDS before STOP shows (ODD until then), so the few seconds of the motor winding down
        after switch-off never flash STOP."""
        confirm = stop_confirm_windows()
        loud_hold, base_n, sm_n = loud_windows()
        out = np.full(len(scores), OFF, dtype=object)
        loud_streak = stop_streak = 0
        base: list[float] = []; sm: list[float] = []
        for i, (s, p) in enumerate(zip(scores, phases)):
            if p != RUNNING:
                loud_streak = stop_streak = 0; base = []; sm = []
                continue
            loud = False
            if rms is not None and config.LOUD_RISE_DB > 0:
                if len(base) < base_n:
                    base.append(float(rms[i]))
                sm.append(float(rms[i])); sm = sm[-sm_n:]
                over = len(base) >= base_n and len(sm) >= sm_n and \
                    float(np.mean(sm)) > float(np.median(base)) + config.LOUD_RISE_DB
                loud_streak = loud_streak + 1 if over else 0
                loud = loud_streak >= loud_hold
            stop = s > self.red or loud
            stop_streak = stop_streak + 1 if stop else 0
            out[i] = STOP if stop_streak > confirm else ODD if (stop or s > self.yellow) else GOOD
        return out

    # ----- saving / loading (one JSON format for Python and the phone) -----
    def to_dict(self) -> dict:
        mel = mel_filterbank(self.sr)
        sparse = []
        for row in mel:
            nz = np.flatnonzero(row)
            start, end = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
            sparse.append({"start": start, "weights": row[start:end].round(8).tolist()})
        return {
            "version": 1,
            "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
            "trained_on": self.trained_on,
            "audio": {"sr": self.sr, "n_fft": config.N_FFT, "frame_hop": config.FRAME_HOP, "n_mels": config.N_MELS,
                      "window_s": config.WINDOW_SECONDS, "hop_s": config.HOP_SECONDS,
                      "rolloff_fraction": ROLLOFF_FRACTION,
                      "pulse": {"frame_s": ENVELOPE_FRAME_S, "min_hz": PULSE_MIN_HZ, "max_hz": PULSE_MAX_HZ, "fft": PULSE_FFT}},
            "feature_names": self.feature_names,
            "distance_index": self.distance_index,
            "scaler": {"mean": self.mean.tolist(), "std": self.std.tolist()},
            "clusters": [{"mean": c["mean"].tolist(), "precision": c["precision"].tolist(), "n_windows": c["n_windows"],
                          "source": c.get("source", "all")} for c in self.clusters],
            "thresholds": {"yellow": self.yellow, "red": self.red,
                           "stop_confirm_windows": stop_confirm_windows(),
                           "loud_rise_db": config.LOUD_RISE_DB, "loud_hold_windows": loud_windows()[0],
                           "loud_base_windows": loud_windows()[1], "loud_smooth_windows": loud_windows()[2],
                           "by_source": {k: {"yellow": y, "red": r} for k, (y, r) in self.thresholds_by_source.items()}},
            "smooth_windows": config.SMOOTH_WINDOWS,
            "motor": {"threshold_db": self.motor_threshold_db, "on_windows": config.MOTOR_ON_WINDOWS,
                      "off_windows": config.MOTOR_OFF_WINDOWS, "max_pulse": config.MAX_PULSE_FOR_ON,
                      "spinup_windows": int(round(config.SPINUP_S / config.HOP_SECONDS)),
                      "spindown_windows": config.SPINDOWN_WINDOWS, "min_cut_s": config.MIN_CUT_S},
            "mel": sparse,
            "breakthrough": self.breakthrough.to_dict() if self.breakthrough is not None else None,
            "scorecard": getattr(self, "scorecard", None),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SawDetector":
        det = cls(n_clusters=max(1, len(d["clusters"])))
        det.feature_names = d["feature_names"]
        det.distance_index = d["distance_index"]
        det.mean = np.array(d["scaler"]["mean"])
        det.std = np.array(d["scaler"]["std"])
        det.clusters = [{"mean": np.array(c["mean"]), "precision": np.array(c["precision"]), "n_windows": c["n_windows"],
                         "source": c.get("source", "all")} for c in d["clusters"]]
        det.yellow, det.red = d["thresholds"]["yellow"], d["thresholds"]["red"]
        det.thresholds_by_source = {k: (v["yellow"], v["red"]) for k, v in d["thresholds"].get("by_source", {}).items()}
        det.motor_threshold_db = d["motor"]["threshold_db"]
        det.trained_on = d.get("trained_on", [])
        if d.get("breakthrough"):
            from .breakthrough import BreakthroughModel
            det.breakthrough = BreakthroughModel.from_dict(d["breakthrough"])
        det.sr = d["audio"]["sr"]
        return det

    def save(self, *paths: str | Path) -> None:
        text = json.dumps(self.to_dict())
        for p in paths:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text(text)

    @classmethod
    def load(cls, path: str | Path) -> "SawDetector":
        return cls.from_dict(json.loads(Path(path).read_text()))


def stop_confirm_windows() -> int:
    return int(round(config.STOP_CONFIRM_SECONDS / config.HOP_SECONDS))


def loud_windows() -> tuple[int, int, int]:
    """(hold, baseline, smoothing) window counts for the 'too loud' rule."""
    h = config.HOP_SECONDS
    return (int(round(config.LOUD_RISE_HOLD_SECONDS / h)), int(round(config.LOUD_BASE_SECONDS / h)),
            int(round(config.LOUD_SMOOTH_SECONDS / h)))


def smooth(x: np.ndarray, k: int = config.SMOOTH_WINDOWS) -> np.ndarray:
    """Trailing (causal) moving average, so offline and live scores match."""
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = x[max(0, i - k + 1):i + 1].mean()
    return out
