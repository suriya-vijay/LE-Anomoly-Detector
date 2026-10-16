"""Report how the detector does on every recording in the logbook.

    python -m src.evaluate

Plain language: every recording gets scored second by second and turned into the
verdict the phone meter would have shown (GOOD / ODD / STOP). Good cuts used for
training are scored by a model that never heard them, so the numbers are honest.
Prints a table and a short summary you can show a supervisor, and saves a
timeline picture to reports/weirdness.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from . import config
from .detector import GOOD, ODD, STOP, SawDetector
from .phases import OFF, RUNNING, fmt_time
from .train import (ROOT, breakthrough_guesses, leave_one_out_models, leave_one_out_scores, source_of, threshold_runs,
from .train import scorecard, scorecard_text
                    thresholds_by_source, train, web_source)

REPORT_PATH = ROOT / "reports" / "weirdness.png"
COLOURS = {OFF: "#9e9e9e", GOOD: "#2e7d32", ODD: "#f9a825", STOP: "#c62828"}


def evaluate(root: Path = ROOT, verbose: bool = True, plot_path: Path = REPORT_PATH) -> list[dict]:
    det, p, _ = train(root, verbose=False)
    loo_models = leave_one_out_models(p)
    by_source = thresholds_by_source(leave_one_out_scores(p, loo_models), p)
    web = by_source[web_source(by_source)]

    rows = []
    for r in p.runs:
        model = loo_models.get(r.filename, det)
        # each recording is judged by the lines of its own capture source (phone page vs file)
        model.yellow, model.red = yellow, red = by_source.get(source_of(r), web)
        feats, times = p.feats[r.filename], p.times[r.filename]
        phases = p.phases[r.filename].copy()
        phases[~p.usable[r.filename]] = OFF                     # trimmed parts are not judged
        scores = model.score(feats)
        verdicts = model.verdicts(scores, phases)
        judged = phases == RUNNING
        js = scores[judged]
        first = {v: (times[verdicts == v][0] if np.any(verdicts == v) else None) for v in (ODD, STOP)}
        rows.append(dict(
            run=r, times=times, scores=scores, verdicts=verdicts, phases=phases,
            unseen=r.filename in loo_models,
            median=float(np.median(js)) if len(js) else np.nan,
            max=float(js.max()) if len(js) else np.nan,
            t_max=float(times[judged][np.argmax(js)]) if len(js) else np.nan,
            pct_odd=float(np.mean(np.isin(verdicts[judged], [ODD, STOP])) * 100) if len(js) else np.nan,
            pct_stop=float(np.mean(verdicts[judged] == STOP) * 100) if len(js) else np.nan,
            first_odd=first[ODD], first_stop=first[STOP],
            judged_s=float(judged.sum() * config.HOP_SECONDS), yellow=yellow, red=red,
        ))

    if verbose:
        _print_table(rows, by_source)
        _print_summary(rows, p, by_source)
        _print_breakthrough(p, det)
    _plot(rows, plot_path)
    if verbose:
        print(f"\nTimeline picture saved to {plot_path.relative_to(root)}")
    return rows


def _verdict_text(row: dict) -> str:
    if row["first_stop"] is not None:
        return f"STOP at {fmt_time(row['first_stop'])}"
    if row["first_odd"] is not None:
        return f"ODD at {fmt_time(row['first_odd'])} ({row['pct_odd']:.1f}% of the time)"
    return "GOOD throughout"


def _print_table(rows: list[dict], by_source: dict) -> None:
    print("Lines: " + "; ".join(f"{s} cuts odd above {y:.2f}, stop above {r:.2f}" for s, (y, r) in by_source.items())
          + f". The phone page uses the {web_source(by_source)} lines. Scores: ~1-2 is typical for a file, ~3 for the phone.\n")
    print(f"{'recording':30s} {'label':9s} {'judged':>7s} {'typical':>8s} {'worst':>6s} {'when':>6s}  meter would have shown")
    for row in rows:
        r = row["run"]
        tag = r.label + ("*" if row["unseen"] else "") + ("+" if r.features is not None else "")
        if row["judged_s"] == 0:
            print(f"{r.filename[:30]:30s} {tag:9s} {'0:00':>7s}  (motor never detected as running)")
            continue
        print(f"{r.filename[:30]:30s} {tag:9s} {fmt_time(row['judged_s']):>7s} {row['median']:8.2f} "
              f"{row['max']:6.2f} {fmt_time(row['t_max']):>6s}  {_verdict_text(row)}")
    print("* scored by a model that never heard this recording   + judged live on the phone page")


def _print_summary(rows: list[dict], p, by_source: dict) -> None:
    eligible = {r.filename for r in threshold_runs(p)}
    normals = [r for r in rows if r["unseen"] and r["run"].filename in eligible]
    firsts = [r for r in rows if r["unseen"] and r["run"].filename not in eligible]
    others = [r for r in rows if not r["unseen"]]
    n_stop = sum(r["first_stop"] is not None for r in normals)
    n_odd = sum(r["first_stop"] is None and r["first_odd"] is not None for r in normals)
    print("\nSUMMARY (plain language)")
    print(f"- The detector learned what a good cut sounds like from {len(p.train_runs)} recordings "
          f"({fmt_time(sum(r['judged_s'] for r in normals))} of judged cutting time).")
    print(f"- Tested honestly on those good cuts (each scored by a model that never heard it): "
          f"{n_stop} would have shown STOP, {n_odd} would have shown ODD only, "
          f"{len(normals) - n_stop - n_odd} stayed GOOD throughout.")
    if normals:
        fa = np.mean([r["pct_odd"] for r in normals])
        print(f"- False alarms: on average {fa:.1f}% of good-cut seconds showed ODD or STOP "
              f"(about {fa * 36:.0f} seconds per hour of cutting).")
    for r in firsts:
        print(f"- {r['run'].filename} is the first good cut captured its way (phone page vs recording file). "
              f"It joined the model, but it can't be tested honestly until a second one exists "
              f"(a model that never heard it shows {_verdict_text(r)}, as expected).")
    for r in others:
        print(f"- {r['run'].filename} ({r['run'].label}): {_verdict_text(r)}.")
    if len(p.train_runs) < 10:
        print(f"- CAUTION: only {len(p.train_runs)} good cuts so far. The yellow/red lines are provisional; "
              f"expect them to settle after ~10-20 good cuts.")


def _print_breakthrough(p, det) -> None:
    guesses = breakthrough_guesses(p, det)
    if not guesses:
        return
    print("\nBLADE-THROUGH INDICATOR (each phone cut judged by a model that never saw it)")
    for name, (guess, heads, truth, back) in guesses.items():
        h = "no heads-up" if heads is None else f"heads-up at {fmt_time(heads)}"
        g = "banner never shown" if guess is None else f"banner at {fmt_time(guess)} after motor-on"
        t = ("  <- you marked " + fmt_time(truth) + (f", so the banner was {guess - truth:+.0f} s" if guess is not None else ", so it missed")) \
            if truth is not None else "  (no mark from you to check against)"
        b = f"  [{back} heads-up(s) taken back]" if back else ""
        print(f"- {name}: {h}, {g}{t}{b}")
    print(scorecard_text(scorecard(guesses, p)))


def _plot(rows: list[dict], path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 2.4 * len(rows)), squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        t, s, v = row["times"] + config.WINDOW_SECONDS / 2, row["scores"], row["verdicts"]
        ax.plot(t, s, color="#444", lw=0.6, zorder=1)
        for verdict, colour in COLOURS.items():
            m = v == verdict
            if m.any():
                ax.scatter(t[m], s[m], s=8, color=colour, zorder=2, label=verdict)
        ax.axhline(row["yellow"], color=COLOURS[ODD], ls="--", lw=1)
        ax.axhline(row["red"], color=COLOURS[STOP], ls="--", lw=1)
        title = f"{row['run'].filename} [{row['run'].label}]" + ("  (model never heard this one)" if row["unseen"] else "")
        ax.set_title(title + f"  ->  {_verdict_text(row)}", fontsize=10, loc="left")
        ax.set_ylabel("weirdness"); ax.set_ylim(0, max(6, np.nanmax(s) * 1.05))
        ax.set_xlim(0, t[-1]); ax.grid(alpha=0.3)
        step = 30 if t[-1] > 150 else 10
        ax.set_xticks(np.arange(0, t[-1] + step, step))
        ax.set_xticklabels([fmt_time(x) for x in np.arange(0, t[-1] + step, step)], fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=4)
    axes[-1, 0].set_xlabel("time in recording (grey = motor off, not judged)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    evaluate()
