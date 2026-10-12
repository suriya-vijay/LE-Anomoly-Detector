import numpy as np

from src.features import extract_all, extract_features, feature_names
from src.synth import make_saw_run
from src.audio_io import split_windows

SR = 22050


def _tone(freq, amp=0.5, seconds=1.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_shape_and_names():
    f = extract_features(_tone(1000), SR)
    assert f.shape == (len(feature_names()),) == (70,)
    assert np.all(np.isfinite(f))


def test_pure_tone_frequencies():
    f = dict(zip(feature_names(), extract_features(_tone(1000), SR)))
    assert abs(f["dominant_hz"] - 1000) < 1.0  # sub-bin accuracy (bins are ~10.8 Hz)
    assert abs(f["centroid_hz"] - 1000) < 50
    assert abs(f["rolloff_hz"] - 1000) < 50


def test_pulsing_detected():
    names = feature_names()
    t = np.arange(SR) / SR
    throb = 0.5 * (1 + np.sin(2 * np.pi * 4 * t)) / 2         # loudness rises/falls 4x per second
    pulsing = (throb * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    f_pulse = dict(zip(names, extract_features(pulsing, SR)))
    f_steady = dict(zip(names, extract_features(_tone(300), SR)))
    assert abs(f_pulse["pulse_hz"] - 4) < 0.5
    assert f_pulse["pulse_depth"] > 0.4
    assert f_steady["pulse_depth"] < 0.05


def test_louder_is_louder():
    quiet = extract_features(_tone(500, amp=0.05), SR)[0]
    loud = extract_features(_tone(500, amp=0.5), SR)[0]
    assert abs((loud - quiet) - 20) < 0.5  # 10x amplitude = +20 dB


def test_synthetic_cutting_lowers_pitch_and_brightens():
    audio, labels = make_saw_run(seed=1, sr=SR)
    w = split_windows(audio, SR)
    feats = extract_all(w, SR)
    names = feature_names()
    starts = (np.arange(len(w)) * 0.5 * SR).astype(int)
    lab = labels[starts + SR // 2]  # phase at the centre of each window
    idle = feats[lab == "idle"]
    cut = feats[lab == "cutting"]
    c = names.index("centroid_hz")
    d = names.index("dominant_hz")
    assert cut[:, c].mean() > idle[:, c].mean() * 1.5   # scraping brightens the sound
    assert cut[:, 0].mean() > idle[:, 0].mean()          # and makes it louder
    assert np.median(cut[:, d]) < np.median(idle[:, d]) * 0.96  # motor slows under load
