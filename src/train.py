"""Train the saw detector from the good cuts in data/runs.csv.

    python -m src.train

Plain language:
  1. Read the logbook, compute (or load cached) features for every recording.
  2. Learn the motor-on loudness threshold and label each window off/transition/running.
  3. Set the yellow/red lines honestly: each good cut is scored by a model that
     never heard it ("leave one run out"), and the lines sit one robust spread above
     the typical cut's 99th / 99.9th percentile of those scores.
  4. Learn the final model from ALL good cuts and save it for Python
     (models/saw.json) and for the phone page (web/model.json).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .breakthrough import BreakthroughModel, segments, fired_index, heads_up_index, taken_back
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


def _lines(per_cut: list[np.ndarray]) -> tuple[float, float]:
    """Yellow = median of the cuts' own 99th percentiles plus LINE_SPREAD_SDS robust SDs of those percentiles
    (so one odd cut, or a few very quiet ones, cannot move the line much); red = the same on the 99.9th
    percentile, never under RED_MIN_RATIO x yellow."""
    p_y = np.array([np.percentile(s, config.YELLOW_PERCENTILE) for s in per_cut])
    p_r = np.array([np.percentile(s, config.RED_PERCENTILE) for s in per_cut])
    spread = 1.4826 * float(np.median(np.abs(p_y - np.median(p_y)))) * config.LINE_SPREAD_SDS
    yellow = float(np.median(p_y)) + spread
    red = max(float(np.median(p_r)) + spread, yellow * config.RED_MIN_RATIO)
    return yellow, red


def thresholds_by_source(loo: dict[str, np.ndarray], p: Prepared) -> dict[str, tuple[float, float]]:
    """Yellow/red lines per capture source. Phone cuts vary from one to the next more than
    recordings do (phone position, device), so each source gets lines from its own honest scores;
    one shared pair would be too strict for the phone or too loose for the files."""
    eligible = threshold_runs(p)
    return {src: _lines([loo[r.filename] for r in eligible if source_of(r) == src])
            for src in dict.fromkeys(p.groups(eligible))}


def web_source(by_source: dict) -> str:
    """The phone page hears what the phone cuts hear, so it gets the phone's lines when they exist."""
    return "phone" if "phone" in by_source else "file"


def thresholds_from(loo: dict[str, np.ndarray], p: Prepared | None = None) -> tuple[float, float]:
    """The lines the phone page will use (the phone's own once there are >= 2 phone cuts, else the recordings')."""
    if p is None:
        return _lines(list(loo.values()))
    by = thresholds_by_source(loo, p)
    return by[web_source(by)]


def breakthrough_segments(p: Prepared, det: SawDetector, runs: list[Run]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """'In the steel' / 'through' rows per phone cut, standardised like the detector does."""
    out = {}
    for r in runs:
        Z = (p.feats[r.filename][:, det.distance_index] - det.mean) / det.std
        seg = segments(Z, p.phases[r.filename], r.through_at_s)
        if seg is not None:
            out[r.filename] = seg
    return out


def fit_breakthrough(segs: dict[str, tuple[np.ndarray, np.ndarray]], skip: str | None = None) -> BreakthroughModel | None:
    use = [v for k, v in segs.items() if k != skip]
    if len(use) < 2:
        return None
    return BreakthroughModel().fit([u[0] for u in use], [u[1] for u in use])


def scorecard(guesses: dict, p: Prepared) -> dict:
    """'Finishing' (the heads-up, stage 1) vs the user's tap, each cut judged unseen. Window: config.FINISH_WINDOW_S
    (60 s before to 5 s after the tap). 'false_early' = shown before 60% of the cut's running time had passed."""
    lo, hi = config.FINISH_WINDOW_S
    tapped = within = late = missed = false_early = 0
    for name, (_banner, finishing, truth, _back) in guesses.items():
        if truth is None:
            continue
        tapped += 1
        if finishing is None:
            missed += 1
            continue
        lead = finishing - truth
        if lo <= lead <= hi:
            within += 1
        elif lead > hi:
            late += 1
        run_s = float(np.sum(p.phases[name] != "off")) * config.HOP_SECONDS
        if finishing < 0.6 * run_s:
            false_early += 1
    return {"tapped": tapped, "within_window": within, "late": late, "missed": missed, "false_early": false_early,
            "window_s": [lo, hi], "provisional": tapped < 20,
            "pass": tapped >= 20 and within >= 0.75 * tapped and late == 0}


def scorecard_text(sc: dict) -> str:
    if not sc or not sc["tapped"]:
        return "SCORECARD: no tapped cuts yet."
    verdict = "provisional until 20 tapped cuts" if sc["provisional"] else ("PASS" if sc["pass"] else "FAIL")
    lo, hi = sc["window_s"]
    return (f"SCORECARD ('Finishing' vs your tap, each cut judged unseen): {sc['within_window']} of {sc['tapped']} within "
            f"{-lo} s before to {hi} s after the tap, {sc['late']} late, {sc['missed']} missed, "
            f"{sc['false_early']} shown in the first 60% of the cut. {verdict}.")


def breakthrough_guesses(p: Prepared, det: SawDetector) -> dict[str, tuple[float | None, float | None, float | None, int]]:
    """For each phone cut, judged by a model that never saw it: (banner seconds after motor-on or None,
    heads-up seconds or None, the user's own mark or None, how many heads-ups were taken back)."""
    phone = [r for r in p.train_runs if source_of(r) == "phone"]
    segs = breakthrough_segments(p, det, phone)
    # ODD cuts never train the meter, but a 'Blade is through' tap on one is still a fair test of the banner
    # (the through model never saw them, so no skip is needed)
    odd_tapped = [r for r in p.runs if source_of(r) == "phone" and r not in p.train_runs and r.through_at_s is not None]
    out = {}
    for r in phone + odd_tapped:
        m = fit_breakthrough(segs, skip=r.filename)
        if m is None:
            continue
        F = p.feats[r.filename]
        Z = (F[:, det.distance_index] - det.mean) / det.std
        ph = p.phases[r.filename]
        on = int(np.flatnonzero(ph != "off")[0]) if np.any(ph != "off") else 0
        start = None if r.started_at_s is None else on + int(round(r.started_at_s / config.HOP_SECONDS))
        _, events = m.track(m.load(Z), ph, F[:, 0], F[:, 2], start)
        secs = lambda k: None if k is None else (k - on) * config.HOP_SECONDS
        out[r.filename] = (secs(fired_index(events)), secs(heads_up_index(events)), r.through_at_s, taken_back(events))
    return out


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
    phone = [r for r in p.train_runs if source_of(r) == "phone"]
    det.breakthrough = fit_breakthrough(breakthrough_segments(p, det, phone))
    guesses = breakthrough_guesses(p, det) if det.breakthrough is not None else {}
    det.scorecard = scorecard(guesses, p)
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
        if det.breakthrough is None:
            print("Blade-through indicator: not learned yet (needs at least 2 phone cuts).")
        else:
            marked = sum(r.through_at_s is not None for r in phone)
            print(f"Blade-through indicator learned from {len(phone)} phone cuts ({marked} with a 'Blade is through' mark).")
            for name, (guess, heads, truth, back) in guesses.items():
                h = "never" if heads is None else fmt_time(heads)
                t = "" if truth is None else f"  (you marked {fmt_time(truth)}: {'%+.0f s' % (heads - truth) if heads is not None else 'missed'})"
                b = f"  [{back} taken back]" if back else ""
                print(f"  {name:28s} unseen: 'Finishing' at {h}{t}{b}")
            print(scorecard_text(det.scorecard))
        print(f"Saved {MODEL_PATH.relative_to(root)} and {WEB_MODEL_PATH.relative_to(root)}.")
    return det, p, loo


if __name__ == "__main__":
    train()
