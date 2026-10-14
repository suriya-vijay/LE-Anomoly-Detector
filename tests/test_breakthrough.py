"""The blade break-through indicator: fires on the rise out of the steel, relative to the cut's own
sound, so a phone that sits somewhere else (every number shifted) still works; takes a guess back."""
import numpy as np

from src import config
from src.breakthrough import BreakthroughModel, fired_index, segments, taken_back
from src.phases import OFF, RUNNING, TRANSITION

RNG = np.random.default_rng(7)
D = 8
SHIFT = np.array([0, -2, 1.5, 0, 0, 0, -1, 0])          # what "in the steel" does to the numbers


def fake_cut(n_idle=60, n_loaded=400, n_tail=60, offset=0.0, bump_at=None):
    """Standardised features for one cut: idle motor, blade in steel, through again.
    offset shifts EVERY number (phone somewhere else); bump_at adds a 30-window 'through-like'
    wobble in the middle of the cut."""
    idle = RNG.normal(size=(n_idle, D))
    loaded = RNG.normal(size=(n_loaded, D)) + SHIFT
    if bump_at is not None:
        loaded[bump_at:bump_at + 30] -= SHIFT * 0.9
    tail = RNG.normal(size=(n_tail, D)) + SHIFT * 0.1
    Z = np.vstack([idle, loaded, tail]) + offset
    phases = np.array([TRANSITION] * 20 + [RUNNING] * (len(Z) - 22) + [TRANSITION] * 2, dtype=object)
    return Z, phases, n_idle + n_loaded


def trained(n=3):
    cuts = [fake_cut(offset=RNG.normal() * 2) for _ in range(n)]      # each training cut has its own offset
    segs = [segments(Z, ph) for Z, ph, _ in cuts]
    return BreakthroughModel().fit([s[0] for s in segs], [s[1] for s in segs])


def test_fires_after_the_cut_not_during_idle():
    m = trained()
    Z, ph, true_through = fake_cut()
    load, events = m.track(m.load(Z), ph)
    fired = fired_index(events)
    assert fired is not None and taken_back(events) == 0
    assert true_through <= fired <= true_through + config.BT_HOLD_WINDOWS + config.BT_SMOOTH_WINDOWS
    assert np.nanmean(load[100:400]) + 1.0 < np.nanmean(load[true_through + 30:])   # lower in the steel than through
    assert np.isnan(load[0]) and np.isnan(load[-1])                                # transition windows are not judged


def test_shifted_phone_still_fires():
    """Every number shifted by a constant (phone in another place): the absolute level is useless,
    the rise relative to the cut's own low still works."""
    m = trained()
    Z, ph, true_through = fake_cut(offset=3.0)
    _, events = m.track(m.load(Z), ph)
    fired = fired_index(events)
    assert fired is not None and taken_back(events) == 0
    assert true_through <= fired <= true_through + config.BT_HOLD_WINDOWS + config.BT_SMOOTH_WINDOWS


def test_no_fire_without_a_cut():
    m = trained()
    Z = RNG.normal(size=(600, D))                                      # idle motor the whole time
    ph = np.array([RUNNING] * 600, dtype=object)
    assert fired_index(m.track(m.load(Z), ph)[1]) is None


def test_wobble_is_taken_back_then_real_through_fires():
    m = trained()
    Z, ph, true_through = fake_cut(bump_at=250)
    _, events = m.track(m.load(Z), ph)
    assert taken_back(events) == 1                                     # the wobble fired and was withdrawn
    bump_fire, bump_back = events[0]
    assert 60 + 250 <= bump_fire <= 60 + 280 + config.BT_HOLD_WINDOWS
    assert bump_back is not None and bump_back <= 60 + 280 + 2 * config.BT_HOLD_WINDOWS + config.BT_SMOOTH_WINDOWS
    fired = fired_index(events)
    assert fired is not None and true_through <= fired <= true_through + config.BT_HOLD_WINDOWS + config.BT_SMOOTH_WINDOWS


def test_user_mark_overrides_weak_labels():
    Z, ph, true_through = fake_cut()
    ph[:5] = OFF
    mark_s = (true_through - 5) * config.HOP_SECONDS                   # the page counts from the first non-off window
    loaded, unloaded = segments(Z, ph, through_at_s=mark_s)
    assert len(unloaded) == 60 - int(5 / config.HOP_SECONDS) - 2      # tail after the mark, minus 5 s and the 2 transition windows
    assert loaded.shape[1] == D and len(loaded) > 100


def test_round_trip():
    m = trained()
    Z, ph, _ = fake_cut()
    back = BreakthroughModel.from_dict(m.to_dict())
    np.testing.assert_allclose(back.load(Z), m.load(Z), rtol=1e-6, atol=1e-6)
    assert back.track(back.load(Z), ph)[1] == m.track(m.load(Z), ph)[1]
