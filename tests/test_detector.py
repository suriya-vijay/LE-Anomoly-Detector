import numpy as np
import pytest

from src import config
from src.audio_io import split_windows
from src.detector import GOOD, ODD, STOP, SawDetector, smooth
from src.features import extract_all
from src.phases import OFF, RUNNING, label_phases, motor_on_threshold
from src.synth import make_saw_run

SR = 22050


@pytest.fixture(scope="module")
def runs():
    """Features + phase labels for 4 normal synthetic runs and 2 faulty ones."""
    out = {}
    for name, seed, fault in [("n0", 10, None), ("n1", 11, None), ("n2", 12, None), ("n3", 13, None),
                              ("thump", 20, "thump"), ("bind", 21, "bind")]:
        audio, _ = make_saw_run(seed=seed, sr=SR, fault=fault)
        out[name] = extract_all(split_windows(audio, SR), SR)
    t = motor_on_threshold(np.concatenate([out[n][:, 0] for n in ("n0", "n1", "n2", "n3")]))
    return {n: (f, label_phases(f, t)) for n, f in out.items()}, t


def _running(run):
    f, labels = run
    return f[labels == RUNNING]


def _fit(runs, names):
    det = SawDetector().fit([_running(runs[n]) for n in names])
    return det


def test_faults_score_higher_than_unseen_normal(runs):
    data, _ = runs
    det = _fit(data, ["n0", "n1", "n2"])
    unseen_normal = det.score(_running(data["n3"]))
    for fault in ("thump", "bind"):
        faulty = det.score(_running(data[fault]))
        assert np.percentile(faulty, 90) > np.percentile(unseen_normal, 99), fault


def test_json_round_trip_gives_identical_scores(runs, tmp_path):
    data, t = runs
    det = _fit(data, ["n0", "n1", "n2"])
    det.yellow, det.red, det.motor_threshold_db = 3.0, 4.0, t
    det.save(tmp_path / "m.json")
    back = SawDetector.load(tmp_path / "m.json")
    f = _running(data["n3"])
    np.testing.assert_allclose(back.score(f), det.score(f), rtol=1e-9)
    assert back.yellow == 3.0 and back.motor_threshold_db == t
    d = det.to_dict()
    assert len(d["mel"]) == config.N_MELS and all(len(m["weights"]) > 0 for m in d["mel"])


def test_verdict_rules(monkeypatch):
    monkeypatch.setattr(config, "STOP_CONFIRM_SECONDS", 0.0)        # the instant rules; the wait is tested below
    det = SawDetector()
    det.yellow, det.red = 3.0, 5.0
    red_run = int(round(config.RED_AFTER_YELLOW_SECONDS / config.HOP_SECONDS))
    scores = np.array([1, 4, 1, 6, 1] + [4] * red_run + [1])
    phases = np.array([RUNNING] * len(scores), dtype=object)
    phases[2] = OFF
    v = det.verdicts(scores, phases)
    assert list(v[:5]) == [GOOD, ODD, OFF, STOP, GOOD]
    assert v[5 + red_run - 1] == STOP and v[5] == ODD  # odd for 3 s in a row becomes stop
    assert v[-1] == GOOD


def test_too_loud_rule(monkeypatch):
    monkeypatch.setattr(config, "STOP_CONFIRM_SECONDS", 0.0)
    from src.detector import loud_windows
    hold, base_n, sm_n = loud_windows()
    det = SawDetector()
    det.yellow, det.red = 3.0, 5.0
    n = base_n + 3 * sm_n
    scores = np.ones(n)                                                    # the score never looks weird
    phases = np.array([RUNNING] * n, dtype=object)
    flat = np.full(n, -20.0)
    assert set(det.verdicts(scores, phases, flat)) == {GOOD}               # steady loudness: nothing
    loud = flat.copy(); loud[base_n:] = -20.0 + config.LOUD_RISE_DB + 1    # then 7 dB louder than its own first minute
    v = det.verdicts(scores, phases, loud)
    assert set(v[:base_n]) == {GOOD}
    first = int(np.flatnonzero(v == STOP)[0])
    assert base_n + sm_n <= first <= base_n + sm_n + hold                  # STOP once the 10 s mean is over, held 2 s
    assert set(v[first:]) == {STOP}
    small = flat.copy(); small[base_n:] = -20.0 + config.LOUD_RISE_DB - 1  # 5 dB louder is still cutting
    assert set(det.verdicts(scores, phases, small)) == {GOOD}
    assert set(det.verdicts(scores, phases)) == {GOOD}                     # no loudness given: rule off


def test_stop_waits_for_confirmation():
    """A STOP reason shows yellow first; STOP only once it has stood STOP_CONFIRM_SECONDS with the motor running.
    A short burst that ends with the motor stopping (winding down after switch-off) never shows STOP."""
    from src.detector import stop_confirm_windows
    n = stop_confirm_windows()
    assert n > 0
    det = SawDetector()
    det.yellow, det.red = 3.0, 5.0
    scores = np.array([1.0] * 5 + [6.0] * (n + 3) + [1.0] * 3)
    phases = np.array([RUNNING] * len(scores), dtype=object)
    v = det.verdicts(scores, phases)
    assert set(v[5:5 + n]) == {ODD}                                        # waiting: yellow
    assert set(v[5 + n:5 + n + 3]) == {STOP}                               # still there after the wait: STOP
    assert v[-1] == GOOD
    short = np.array([1.0] * 5 + [6.0] * (n - 2) + [6.0] * 3)               # motor goes off before the wait is over
    ph = np.array([RUNNING] * (5 + n - 2) + [OFF] * 3, dtype=object)
    assert STOP not in set(det.verdicts(short, ph))


def test_smooth_is_trailing():
    np.testing.assert_allclose(smooth(np.array([3.0, 0, 0, 0]), 3), [3, 1.5, 1, 0])
