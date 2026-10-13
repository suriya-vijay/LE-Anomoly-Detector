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
    motor_threshold_db: dict[str, float]  # per capture source ("file" / "phone")
    train_runs: list[Run] = field(default_factory=list)

    def threshold_for(self, run: Run) -> float:
        return self.motor_threshold_db.get(source_of(run), self.motor_threshold_db["file"])

    def groups(self, runs: list[Run]) -> list[str]:
        return [source_of(r) for r in runs]

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
    # The phone on the saw housing hears the motor louder than a recording made elsewhere,
    # so each capture source learns its own "motor on" loudness level.
    thr: dict[str, float] = {}
    for src in ("file", "phone"):
        rs = [r for r in train_runs if source_of(r) == src]
        if rs:
            thr[src] = motor_on_threshold(np.concatenate([feats[r.filename][usable[r.filename], 0] for r in rs]))
    if "file" not in thr:
        thr["file"] = thr["phone"]
    p = Prepared(runs, feats, times, {}, usable, thr, train_runs)
    p.phases = {r.filename: label_phases(feats[r.filename], p.threshold_for(r)) for r in runs}
    if verbose:
        print("Motor counts as ON above " + ", ".join(f"{v:.1f} dB ({k} recordings)" for k, v in thr.items()) + ".")
        for r in runs:
            spans = ", ".join(f"{fmt_time(a)}-{fmt_time(b)}" for a, b in on_off_times(p.phases[r.filename]))
            used = f" (using {fmt_time(r.use_from_s)}-{fmt_time(r.use_until_s)})" if r.use_until_s else ""
            print(f"  {r.filename:30s} [{r.label}{', train' if r in train_runs else ''}] motor on {spans}{used}; "
                  f"{len(p.running(r))} judged windows")
    return p


def leave_one_out_models(p: Prepared) -> dict[str, SawDetector]:
    """For each training run, a model fitted on all the OTHER training runs."""
    out = {}
    for r in p.train_runs:
        others = [o for o in p.train_runs if o is not r]
        out[r.filename] = SawDetector().fit([p.running(o) for o in others], p.groups(others))
    return out


def leave_one_out_scores(p: Prepared, models: dict[str, SawDetector] | None = None) -> dict[str, np.ndarray]:
    """Score each training run's judged windows with a model that never heard it. Honest, not flattering."""
    models = models or leave_one_out_models(p)
    return {r.filename: models[r.filename].score(p.running(r)) for r in p.train_runs}


def source_of(run: Run) -> str:
    """'phone' for cuts judged live on the phone page, 'file' for audio recordings."""
    return "phone" if run.features is not None else "file"


def threshold_runs(p: Prepared) -> list[Run]:
    """Training runs that may set the yellow/red lines: a run only counts if at least one
    OTHER training run was captured the same way (phone page vs recording file). Otherwise
    the leave-one-out model has never heard that kind of sound and would score it as weird,
    which would push the lines up and make the meter blind."""
    counts = {}
    for r in p.train_runs:
        counts[source_of(r)] = counts.get(source_of(r), 0) + 1
    return [r for r in p.train_runs if counts[source_of(r)] >= 2]


def _lines(scores: np.ndarray) -> tuple[float, float]:
    yellow = float(np.percentile(scores, config.YELLOW_PERCENTILE))
    red = max(float(np.percentile(scores, config.RED_PERCENTILE)), yellow * config.RED_MIN_RATIO)
    return yellow, red


def thresholds_by_source(loo: dict[str, np.ndarray], p: Prepared) -> dict[str, tuple[float, float]]:
    """Yellow/red lines per capture source. Phone cuts vary from one to the next more than
    recordings do (phone position, device), so each source gets lines from its own honest scores;
    one shared pair would be too strict for the phone or too loose for the files."""
    eligible = threshold_runs(p)
    return {src: _lines(np.concatenate([loo[r.filename] for r in eligible if source_of(r) == src]))
            for src in dict.fromkeys(p.groups(eligible))}


def web_source(by_source: dict) -> str:
    """The phone page hears what the phone cuts hear, so it gets the phone's lines when they exist."""
    return "phone" if "phone" in by_source else "file"


def thresholds_from(loo: dict[str, np.ndarray], p: Prepared | None = None) -> tuple[float, float]:
    """The lines the phone page will use (the phone's own once there are >= 2 phone cuts, else the recordings')."""
    if p is None:
        return _lines(np.concatenate(list(loo.values())))
    by = thresholds_by_source(loo, p)
    return by[web_source(by)]


def train(root: Path = ROOT, verbose: bool = True) -> tuple[SawDetector, Prepared, dict[str, np.ndarray]]:
    p = prepare(root, verbose)
    loo = leave_one_out_scores(p)
    by_source = thresholds_by_source(loo, p)
    yellow, red = by_source[web_source(by_source)]
    skipped = [r for r in p.train_runs if r not in threshold_runs(p)]
    if verbose and skipped:
        print(f"\nNOTE: {len(skipped)} cut(s) joined the model but could not help set the yellow/red lines yet "
              f"(first of their kind; need a second cut captured the same way): " + ", ".join(r.filename for r in skipped))
    det = SawDetector().fit([p.running(r) for r in p.train_runs], p.groups(p.train_runs))
    det.yellow, det.red = yellow, red
    det.thresholds_by_source = by_source
    # the phone page captures sound like the phone cuts do, so it gets the phone's motor level
    det.motor_threshold_db = p.motor_threshold_db.get("phone", p.motor_threshold_db["file"])
    det.trained_on = [r.filename for r in p.train_runs]
    det.save(MODEL_PATH, WEB_MODEL_PATH)
    if verbose:
        n = len(p.train_runs)
        typical = float(np.median(np.concatenate(list(loo.values()))))
        by_src = {s: sum(1 for c in det.clusters if c.get("source") == s) for s in dict.fromkeys(p.groups(p.train_runs))}
        print(f"\nLearned {len(det.clusters)} kinds of normal from {n} good cuts "
              f"({sum(len(p.running(r)) for r in p.train_runs)} judged windows): "
              + ", ".join(f"{k} from {s} recordings" for s, k in by_src.items()) + ".")
        for src, (y, rd) in by_source.items():
            print(f"Lines for {src} cuts: yellow (odd) above {y:.2f}, red (stop) above {rd:.2f}"
                  + ("  <- the phone page uses these" if src == web_source(by_source) else ""))
        print(f"Typical honest score for a good cut: {typical:.2f}.")
        if n < 10:
            print(f"CAUTION: only {n} good cuts. The yellow/red lines are provisional until there are ~10-20.")
        print(f"Saved {MODEL_PATH.relative_to(root)} and {WEB_MODEL_PATH.relative_to(root)}.")
    return det, p, loo


if __name__ == "__main__":
    train()
