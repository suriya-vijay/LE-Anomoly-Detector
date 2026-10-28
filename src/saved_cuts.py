"""Read the files downloaded from the phone page ("That cut was GOOD / ODD").

    python -m src.saved_cuts        # list what's in data/saved_cuts/

Plain language: the phone saves each judged cut as a list of per-window
measurements (the same 70 numbers Python computes), plus your label, material
and notes. No audio. These cuts take part in training exactly like recordings:
GOOD ones teach "normal", ODD ones are reported on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .features import feature_names


@dataclass
class SavedCut:
    name: str              # "<file>#<n>" so each cut has a unique id
    label: str             # normal / abnormal
    material: str
    notes: str
    saved_at: str
    duration_s: float
    through_at_s: float | None   # seconds after motor-on when the user tapped "Blade is through" (None if not tapped)
    started_at_s: float | None   # seconds after motor-on when the user tapped "Blade touched steel" (None if not tapped)
    features: np.ndarray   # (n_windows, n_features)
    train: bool = True     # False keeps a GOOD cut out of learning "normal" (e.g. loud shop noises in it)
    use_from_s: float = 0.0          # like runs.csv: ignore the recording before / after these times
    use_until_s: float | None = None # (e.g. the phone was still sliding about for the first 40 s)


def load_saved_cuts(root: str | Path, verbose: bool = False) -> list[SavedCut]:
    folder = Path(root) / "data" / "saved_cuts"
    if not folder.exists():
        return []
    cuts, names = [], feature_names()
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(f"  (skipping {path.name}: {e})")
            continue
        for i, c in enumerate(data.get("cuts", [])):
            if c.get("feature_names") != names:
                print(f"  (skipping cut {i} in {path.name}: it was saved by a different version of the page)")
                continue
            feats = np.asarray(c.get("features", []), dtype=np.float32)
            if feats.ndim != 2 or len(feats) < 20:
                print(f"  (skipping cut {i} in {path.name}: too short)")
                continue
            cuts.append(SavedCut(
                name=f"{path.stem}#{i}", label=c.get("label", "unknown"), material=c.get("material", ""),
                notes=c.get("notes", ""), saved_at=c.get("at", ""),
                duration_s=float(c.get("duration_s", len(feats) * config.HOP_SECONDS)),
                through_at_s=None if c.get("through_at_s") is None else float(c["through_at_s"]),
                started_at_s=None if c.get("started_at_s") is None else float(c["started_at_s"]), features=feats,
                train=bool(c.get("train", True)), use_from_s=float(c.get("use_from_s", 0.0)),
                use_until_s=None if c.get("use_until_s") is None else float(c["use_until_s"])))
    if verbose:
        print(f"{len(cuts)} saved cut(s) from the phone in {folder}:")
        for c in cuts:
            thr = f"through at {c.through_at_s:.0f}s" if c.through_at_s is not None else "no through-mark"
            print(f"  {c.name:28s} {c.label:9s} {c.duration_s:5.0f}s  {thr:18s} {c.saved_at[:16]:16s} {c.material}  {c.notes}")
    return cuts


if __name__ == "__main__":
    load_saved_cuts(Path(__file__).resolve().parents[1], verbose=True)
