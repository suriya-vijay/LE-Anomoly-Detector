"""Python and the phone's JavaScript must compute the SAME features and scores."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from src.audio_io import split_windows
from src.breakthrough import BreakthroughModel, fired_index
from src.detector import SawDetector
from src.features import extract_all
from src.phases import label_phases, motor_on_threshold
from src.synth import make_saw_run

SR = 22050
ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js not installed; JS parity not checked")
def test_js_matches_python(tmp_path):
    normal = [extract_all(split_windows(make_saw_run(seed=s, sr=SR)[0], SR), SR) for s in (30, 31, 32)]
    thr = motor_on_threshold(np.concatenate([f[:, 0] for f in normal]))
    det = SawDetector().fit([f[label_phases(f, thr) == "running"] for f in normal])
    det.yellow, det.red, det.motor_threshold_db = 3.0, 4.0, thr
    rng = np.random.default_rng(0)                       # any direction will do: we only check Python == JS
    bt = BreakthroughModel()
    bt.w, bt.mid = rng.normal(size=len(det.distance_index)) * 0.05, rng.normal(size=len(det.distance_index)) * 0.1
    det.breakthrough = bt
    model_path = tmp_path / "model.json"
    det.save(model_path)

    audio, _ = make_saw_run(seed=40, sr=SR, fault="thump")
    windows = split_windows(audio, SR)
    (tmp_path / "windows.bin").write_bytes(windows.astype(np.float32).tobytes())
    py_feats = extract_all(windows, SR).astype(np.float64)
    py_scores = det.score(py_feats)
    py_phases = label_phases(py_feats, thr)
    py_verdicts = det.verdicts(py_scores, py_phases)

    out = subprocess.run([NODE, str(ROOT / "tests" / "parity_runner.mjs"), str(model_path),
                          str(tmp_path / "windows.bin"), str(windows.shape[1])],
                         capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    js_feats = np.array(js["features"])

    assert js_feats.shape == py_feats.shape
    # features: float32 in Python, float64 in JS -> allow float32-level differences
    np.testing.assert_allclose(js_feats, py_feats, rtol=2e-4, atol=1e-3)
    np.testing.assert_allclose(np.array(js["scores"]), py_scores, rtol=1e-3, atol=2e-3)
    assert js["phases"] == list(py_phases)
    assert js["verdicts"] == list(py_verdicts)
    Z = (py_feats[:, det.distance_index] - det.mean) / det.std
    py_load, py_events = bt.track(bt.load(Z), py_phases)
    py_fire = fired_index(py_events)
    js_load = np.array([np.nan if v is None else v for v in js["loads"]], dtype=float)
    np.testing.assert_allclose(js_load, py_load, rtol=1e-3, atol=2e-3, equal_nan=True)
    assert js["through"] == py_fire
