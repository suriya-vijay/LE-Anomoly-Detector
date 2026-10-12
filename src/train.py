"""Train the saw detector from the good cuts in data/runs.csv.

    python -m src.train

Plain language:
  1. Read the logbook, compute (or load cached) features for every recording.
  2. Learn the motor-on loudness threshold and label each window off/transition/running.
  3. Set the yellow/red lines honestly: each good cut is scored by a model that
     never heard it ("leave one run out"), and the lines are the 99th / 99.9th
     percentile of those scores.
  4. Learn the final model from ALL good cuts and save it for Python
     (models/saw.json) and for the phone page (web/model.json).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .detector import SawDetector
from .phases import RUNNING, fmt_time, label_phases, motor_on_threshold, on_off_times
from .runlog import Run, features_for, load_runs, usable_mask

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "saw.json"
WEB_MODEL_PATH = ROOT / "web" / "model.json"


@dataclass
class Prepared:
    runs: list[Run]
    feats: dict[str, np.ndarray]          # filename -> (n_windows, n_features)
    times: dict[str, np.ndarray]          # filename -> window start times
    phases: dict[str, np.ndarray]         # filename -> off/transition/running per window
    usable: dict[str, np.ndarray]         # filename -> True where inside use_from/use_until
    motor_threshold_db: float
    train_runs: list[Run] = field(default_factory=list)

    def running(self, run: Run) -> np.ndarray:
        """Feature rows that are judged (motor running and inside the usable range)."""
        m = (self.phases[run.filename] == RUNNING) & self.usable[run.filename]
        return self.feats[run.filename][m]


def prepare(root: Path = ROOT, verbose: bool = True) -> Prepared:
    runs = load_runs(root)
    if verbose:
        print(f"Found {len(runs)} recordings in the logbook.")
    feats, times = {}, {}
    for r in runs:
        feats[r.filename], times[r.filename] = features_for(r, root)
    train_runs = [r for r in runs if r.label == "normal" and r.train]
    if len(train_runs) < 2:
        sys.exit("Need at least 2 normal recordings with train=yes in data/runs.csv.")
    usable = {r.filename: usable_mask(r, times[r.filename]) for r in runs}
    thr = motor_on_threshold(np.concatenate([feats[r.filename][usable[r.filename], 0] for r in train_runs]))
    phases = {r.filename: label_phases(feats[r.filename], thr) for r in runs}
    p = Prepared(runs, feats, times, phases, usable, thr, train_runs)
    if verbose:
        print(f"Motor counts as ON above {thr:.1f} dB.")
        for r in runs:
            spans = ", ".join(f"{fmt_time(a)}-{fmt_time(b)}" for a, b in on_off_times(phases[r.filename]))
            used = f" (using {fmt_time(r.use_from_s)}-{fmt_time(r.use_until_s)})" if r.use_until_s else ""
            print(f"  {r.filename:30s} [{r.label}{', train' if r in train_runs else ''}] motor on {spans}{used}; "
                  f"{len(p.running(r))} judged windows")
    return p


def leave_one_out_models(p: Prepared) -> dict[str, SawDetector]:
    """For each training run, a model fitted on all the OTHER training runs."""
    return {r.filename: SawDetector().fit([p.running(o) for o in p.train_runs if o is not r])
            for r in p.train_runs}


def leave_one_out_scores(p: Prepared, models: dict[str, SawDetector] | None = None) -> dict[str, np.ndarray]:
    """Score each training run's judged windows with a model that never heard it. Honest, not flattering."""
    models = models or leave_one_out_models(p)
    return {r.filename: models[r.filename].score(p.running(r)) for r in p.train_runs}


def thresholds_from(loo: dict[str, np.ndarray]) -> tuple[float, float]:
    allscores = np.concatenate(list(loo.values()))
    yellow = float(np.percentile(allscores, config.YELLOW_PERCENTILE))
    red = max(float(np.percentile(allscores, config.RED_PERCENTILE)), yellow * config.RED_MIN_RATIO)
    return yellow, red


def train(root: Path = ROOT, verbose: bool = True) -> tuple[SawDetector, Prepared, dict[str, np.ndarray]]:
    p = prepare(root, verbose)
    loo = leave_one_out_scores(p)
    yellow, red = thresholds_from(loo)
    det = SawDetector().fit([p.running(r) for r in p.train_runs])
    det.yellow, det.red = yellow, red
    det.motor_threshold_db = p.motor_threshold_db
    det.trained_on = [r.filename for r in p.train_runs]
    det.save(MODEL_PATH, WEB_MODEL_PATH)
    if verbose:
        n = len(p.train_runs)
        typical = float(np.median(np.concatenate(list(loo.values()))))
        print(f"\nLearned {len(det.clusters)} kinds of normal from {n} good cuts "
              f"({sum(len(p.running(r)) for r in p.train_runs)} judged windows).")
        print(f"Typical score for a good cut: {typical:.2f}.  Yellow (odd) above {yellow:.2f}.  Red (stop) above {red:.2f}.")
        if n < 10:
            print(f"CAUTION: only {n} good cuts. The yellow/red lines are provisional until there are ~10-20.")
        print(f"Saved {MODEL_PATH.relative_to(root)} and {WEB_MODEL_PATH.relative_to(root)}.")
    return det, p, loo


if __name__ == "__main__":
    train()
