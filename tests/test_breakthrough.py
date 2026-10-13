"""The blade break-through indicator: learns 'in the steel' vs 'through', fires only after a real cut."""
import numpy as np

from src import config
from src.breakthrough import BreakthroughModel, segments
from src.phases import OFF, RUNNING, TRANSITION

RNG = np.random.default_rng(7)
D = 8


def fake_cut(n_idle=60, n_loaded=300, n_tail=40):
    """Standardised features for one cut: idle motor, blade in steel (shifted on 3 axes), through again."""
    idle = RNG.normal(size=(n_idle, D))
    loaded = RNG.normal(size=(n_loaded, D)) + np.array([0, -2, 1.5, 0, 0, 0, -1, 0])
    tail = RNG.normal(size=(n_tail, D)) + np.array([0, -0.2, 0.1, 0, 0, 0, 0, 0])
    Z = np.vstack([idle, loaded, tail])
    phases = np.array([TRANSITION] * 20 + [RUNNING] * (len(Z) - 22) + [TRANSITION] * 2, dtype=object)
    return Z, phases, n_idle + n_loaded


def test_fires_after_the_cut_not_during_idle():
    cuts = [fake_cut() for _ in range(3)]
    segs = [segments(Z, ph) for Z, ph, _ in cuts]
    m = BreakthroughModel().fit([s[0] for s in segs], [s[1] for s in segs])
    Z, ph, true_through = fake_cut()
    load, fired = m.track(m.load(Z), ph)
    assert fired is not None
    assert true_through <= fired <= true_through + config.BT_HOLD_WINDOWS + config.BT_SMOOTH_WINDOWS
    assert np.nanmean(load[100:300]) < config.BT_LOADED_LEVEL          # reads "in the steel" while cutting
    assert np.isnan(load[0]) and np.isnan(load[-1])                   # transition windows are not judged


def test_no_fire_without_a_cut():
    cuts = [fake_cut() for _ in range(3)]
    segs = [segments(Z, ph) for Z, ph, _ in cuts]
    m = BreakthroughModel().fit([s[0] for s in segs], [s[1] for s in segs])
    Z = RNG.normal(size=(400, D))                                      # idle motor the whole time
    ph = np.array([RUNNING] * 400, dtype=object)
    assert m.track(m.load(Z), ph)[1] is None


def test_user_mark_overrides_weak_labels():
    Z, ph, true_through = fake_cut()
    ph[:5] = OFF
    mark_s = (true_through - 5) * config.HOP_SECONDS                   # the page counts from the first non-off window
    loaded, unloaded = segments(Z, ph, through_at_s=mark_s)
    assert len(unloaded) == 40 - int(5 / config.HOP_SECONDS) - 2      # tail after the mark, minus 5 s and the 2 transition windows
    assert loaded.shape[1] == D and len(loaded) > 100


def test_round_trip():
    Z, ph, _ = fake_cut()
    seg = segments(Z, ph)
    m = BreakthroughModel().fit([seg[0]], [seg[1]])
    back = BreakthroughModel.from_dict(m.to_dict())
    np.testing.assert_allclose(back.load(Z), m.load(Z), rtol=1e-6, atol=1e-6)
