"""A file downloaded from the phone page must load and take part in training like a recording."""
import json

import numpy as np

from src.audio_io import split_windows
from src.features import extract_all, feature_names
from src.runlog import features_for, load_runs
from src.synth import make_saw_run

SR = 22050


def _phone_file(label, seed, fault=None):
    audio, _ = make_saw_run(seed=seed, sr=SR, fault=fault)
    feats = extract_all(split_windows(audio, SR), SR)
    return {"label": label, "material": "synthetic steel", "notes": "test", "at": "2026-10-13T10:00:00",
            "duration_s": len(feats) * 0.5, "feature_names": feature_names(),
            "features": np.round(feats, 3).tolist()}          # the phone rounds to 3 decimals


def test_saved_cuts_become_runs(tmp_path):
    (tmp_path / "data" / "saved_cuts").mkdir(parents=True)
    (tmp_path / "data" / "normal").mkdir(); (tmp_path / "data" / "abnormal").mkdir(); (tmp_path / "data" / "test").mkdir()
    (tmp_path / "data" / "runs.csv").write_text("filename,label\n")
    cuts = [_phone_file("normal", 50), _phone_file("abnormal", 51, "thump"),
            {**_phone_file("normal", 52), "feature_names": ["old", "version"]},   # must be skipped
            {**_phone_file("normal", 53), "train": False}]                        # good, but kept out of learning
    (tmp_path / "data" / "saved_cuts" / "saved-cuts-2026-10-13.json").write_text(json.dumps({"cuts": cuts}))

    runs = load_runs(tmp_path)
    assert [r.label for r in runs] == ["normal", "abnormal", "normal"]
    assert runs[0].train and not runs[1].train and not runs[2].train
    assert runs[0].filename == "saved-cuts-2026-10-13#0" and "synthetic steel" in runs[0].notes
    feats, times = features_for(runs[0], tmp_path)
    assert feats.shape[1] == len(feature_names()) and len(times) == len(feats)
    np.testing.assert_allclose(times[:3], [0, 0.5, 1.0])
