"""Read data/runs.csv and turn each recording into cached features.

Plain language: runs.csv is your logbook. Each row says which file it is, whether
the cut was normal or abnormal, and (optionally) which part of the recording to
use. Features for each file are computed once and cached, so retraining takes
seconds instead of minutes.

Optional columns:
  train         yes/no  - use this run to learn "normal" (default: yes for normal runs)
  use_from_s    seconds - ignore everything before this time (default: start)
  use_until_s   seconds - ignore everything after this time (e.g. people talking)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .audio_io import list_audio, load_audio, split_windows, window_times
from .features import extract_all

DATA_FOLDERS = ("normal", "abnormal", "test")


@dataclass
class Run:
    filename: str
    label: str            # normal / abnormal / unknown
    path: Path | None     # None for cuts saved by the phone page (features only, no audio)
    train: bool
    use_from_s: float
    use_until_s: float | None
    notes: str = ""
    features: np.ndarray | None = None   # pre-computed features (phone cuts)
    through_at_s: float | None = None    # phone cuts: when the user tapped "Blade is through" (s after motor-on)
    started_at_s: float | None = None    # phone cuts: when the user tapped "Blade touched steel" (s after motor-on)

    def __str__(self) -> str:
        return self.filename


def _truthy(value, default: bool) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "y", "yes", "true")


def _float_or(value, default):
    try:
        v = float(value)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def load_runs(root: str | Path) -> list[Run]:
    """All runs in runs.csv whose audio file exists. Warns about files not in the log."""
    root = Path(root)
    df = pd.read_csv(root / "data" / "runs.csv", dtype=str, keep_default_na=False)
    on_disk = {p.name: p for folder in DATA_FOLDERS for p in list_audio(root / "data" / folder)}
    runs = []
    for row in df.to_dict("records"):
        name = row.get("filename", "").strip()
        if name.startswith("EXAMPLE") or not name:
            continue
        if name not in on_disk:
            print(f"  (skipping '{name}': not found in data/normal, data/abnormal or data/test)")
            continue
        label = row.get("label", "unknown").strip().lower() or "unknown"
        runs.append(Run(
            filename=name, label=label, path=on_disk.pop(name),
            train=_truthy(row.get("train"), default=(label == "normal")),
            use_from_s=_float_or(row.get("use_from_s"), 0.0),
            use_until_s=_float_or(row.get("use_until_s"), None),
            notes=row.get("notes", ""),
        ))
    for name in on_disk:
        print(f"  (warning: '{name}' is in the data folder but has no row in runs.csv, so it is ignored)")
    # cuts judged on the phone page (data/saved_cuts/*.json) join in as runs without audio
    from .saved_cuts import load_saved_cuts
    for c in load_saved_cuts(root):
        runs.append(Run(filename=c.name, label=c.label, path=None, train=(c.label == "normal" and c.train), through_at_s=c.through_at_s, started_at_s=c.started_at_s,
                        use_from_s=c.use_from_s, use_until_s=c.use_until_s,
                        notes=" - ".join(x for x in (c.material, c.notes) if x), features=c.features))
    return runs


def features_for(run: Run, root: str | Path, sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """(features, window_start_times) for a run, cached in data/cache/."""
    if run.features is not None:                      # phone cut: features were computed on the phone
        return run.features, window_times(len(run.features))
    root = Path(root)
    cache_dir = root / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = run.path.stat()
    cache = cache_dir / f"{run.path.stem}.npz"
    if cache.exists():
        z = np.load(cache)
        if int(z["mtime"]) == int(stat.st_mtime) and int(z["size"]) == stat.st_size and z["feats"].shape[1] == len(_names()):
            return z["feats"], z["times"]
    audio = load_audio(run.path, sr)
    feats = extract_all(split_windows(audio, sr), sr)
    times = window_times(len(feats))
    np.savez(cache, feats=feats, times=times, mtime=int(stat.st_mtime), size=stat.st_size)
    return feats, times


def usable_mask(run: Run, times: np.ndarray) -> np.ndarray:
    """True for windows inside the run's use_from_s / use_until_s range."""
    mask = times >= run.use_from_s
    if run.use_until_s is not None:
        mask &= (times + config.WINDOW_SECONDS) <= run.use_until_s
    return mask


def _names():
    from .features import feature_names
    return feature_names()
