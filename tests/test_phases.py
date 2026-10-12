import numpy as np

from src import config
from src.audio_io import split_windows
from src.features import extract_all, feature_names
from src.phases import OFF, RUNNING, TRANSITION, cut_done_events, label_phases, motor_on_threshold, on_off_times
from src.synth import make_saw_run

SR = 22050


def _run(seed, fault=None):
    audio, sample_labels = make_saw_run(seed=seed, sr=SR, fault=fault)
    windows = split_windows(audio, SR)
    feats = extract_all(windows, SR)
    centre = (np.arange(len(windows)) * config.HOP_SECONDS * SR + SR // 2).astype(int)
    truth = sample_labels[centre]
    return feats, truth


def test_threshold_sits_between_off_and_on():
    feats, truth = _run(3)
    t = motor_on_threshold(feats[:, 0])
    off_level = feats[truth == "off", 0].max()
    on_level = feats[np.isin(truth, ["idle", "cutting"]), 0].min()
    assert off_level < t < on_level


def test_phases_match_synthetic_truth():
    feats, truth = _run(4)
    labels = label_phases(feats, motor_on_threshold(feats[:, 0]))
    # every window the synthetic saw was truly idle/cutting (not spinning up/down) is judged
    steady = np.isin(truth, ["idle", "cutting"])
    spinup = int(round(config.SPINUP_S / config.HOP_SECONDS))
    first_on = np.flatnonzero(labels != OFF)[0]
    judged = steady.copy()
    judged[: first_on + spinup] = False            # allow the spin-up grace period
    judged[-(config.SPINDOWN_WINDOWS + 6):] = False
    assert np.all(labels[judged] == RUNNING)
    assert np.all(labels[truth == "off"] == OFF)
    (on, off), = on_off_times(labels)
    true_on = np.flatnonzero(truth != "off")[0] * config.HOP_SECONDS
    true_off = (np.flatnonzero(truth != "off")[-1] + 1) * config.HOP_SECONDS
    assert abs(on - true_on) <= 2.0 and abs(off - true_off) <= 2.0
    assert cut_done_events(labels) == [off] if off - on >= config.MIN_CUT_S else cut_done_events(labels) == []


def test_bang_while_off_does_not_switch_on():
    feats, _ = _run(5)
    t = motor_on_threshold(feats[:, 0])
    quiet = feats[:6].copy()
    quiet[:, 0] = t - 5
    bang = quiet.copy()
    bang[2:5, 0] = t + 10                            # loud for 3 windows...
    bang[2:5, feature_names().index("pulse_depth")] = 0.9   # ...but it's a thump, not a motor
    assert np.all(label_phases(bang, t) == OFF)
