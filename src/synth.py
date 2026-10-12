"""Fake "saw-like" recordings for testing the pipeline before real data exists.

This is NOT a physical model of a saw. It just reproduces the main things we
expect to hear: a motor hum with harmonics, a pitch drop + scraping noise while
cutting, and a few planted faults. Every run is slightly different (random seed).

Faults you can plant:
  "bind"       - blade bogs down: much bigger pitch drop, louder, rougher
  "chipped"    - a click every blade revolution while cutting
  "no_coolant" - extra high-pitched screech while cutting
  "thump"      - "duh duh duh": loudness throbs a few times per second, in
                 bursts that come and go (e.g. blade deflecting / sample shifting)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

FAULTS = ("bind", "chipped", "no_coolant", "thump")


def _bandnoise(rng, n, sr, lo, hi):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.std(x) + 1e-12)


def make_saw_run(seed: int = 0, sr: int = config.SAMPLE_RATE, fault: str | None = None):
    """Return (audio, phase_labels_per_sample) for one synthetic run."""
    rng = np.random.default_rng(seed)
    segs = [  # (phase name, seconds)
        ("off", rng.uniform(1.5, 2.5)),
        ("spinup", 1.5),
        ("idle", rng.uniform(3, 5)),
        ("cutting", rng.uniform(7, 11)),
        ("idle", rng.uniform(2.5, 4)),
        ("spindown", 1.5),
        ("off", rng.uniform(1.5, 2.5)),
    ]
    f_idle = 100 * rng.uniform(0.98, 1.02)        # motor fundamental, Hz
    cut_drop = rng.uniform(0.90, 0.93)            # pitch falls ~8% while cutting
    if fault == "bind":
        cut_drop = 0.70

    f0, amp, cut_env, labels = [], [], [], []
    for name, secs in segs:
        n = int(secs * sr)
        ramp = np.linspace(0, 1, n)
        if name == "off":
            f0.append(np.zeros(n)); amp.append(np.zeros(n)); cut_env.append(np.zeros(n))
        elif name == "spinup":
            f0.append(f_idle * ramp); amp.append(ramp); cut_env.append(np.zeros(n))
        elif name == "spindown":
            f0.append(f_idle * (1 - ramp)); amp.append(1 - ramp); cut_env.append(np.zeros(n))
        elif name == "idle":
            f0.append(np.full(n, f_idle)); amp.append(np.ones(n)); cut_env.append(np.zeros(n))
        else:  # cutting: smooth pitch drop in, then back out at break-through
            edge = np.clip(np.minimum(ramp, 1 - ramp) * secs / 0.4, 0, 1)
            f0.append(f_idle * (1 - (1 - cut_drop) * edge)); amp.append(np.ones(n)); cut_env.append(edge)
        labels += [name] * n

    f0, amp, cut_env = map(np.concatenate, (f0, amp, cut_env))
    n = len(f0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    hum = sum((0.6 / k) * np.sin(k * phase + rng.uniform(0, 6.28)) for k in range(1, 8))
    whine = 0.08 * np.sin(20 * phase)              # blade-tooth whine (~2 kHz)
    scrape = 0.35 * _bandnoise(rng, n, sr, 800, 6000)
    audio = amp * (0.25 * hum + whine) + cut_env * scrape * (1.8 if fault == "bind" else 1.0)

    if fault == "chipped":
        rev = np.diff(np.floor(phase / (2 * np.pi)), prepend=0) > 0
        clicks = np.convolve(rev.astype(float), np.exp(-np.arange(80) / 10), mode="same")
        audio += cut_env * 1.2 * clicks * _bandnoise(rng, n, sr, 2000, 9000)
    elif fault == "thump":
        t = np.arange(n) / sr
        rate = rng.uniform(3, 5)                                  # thumps per second
        throb = 0.5 * (1 + np.sin(2 * np.pi * rate * t)) ** 2 / 2  # sharp-ish pulses
        bursts = (np.sin(2 * np.pi * 0.25 * t + rng.uniform(0, 6.28)) > -0.3).astype(float)
        bursts = np.convolve(bursts, np.ones(int(0.2 * sr)) / int(0.2 * sr), mode="same")
        on = cut_env * bursts                                     # only while cutting, on and off
        audio = audio * (1 - 0.6 * on) + on * 1.2 * throb * (0.25 * hum + scrape)
    elif fault == "no_coolant":
        audio += cut_env * 0.25 * np.sin(2 * np.pi * rng.uniform(6500, 7500) * np.arange(n) / sr)

    audio += 0.003 * rng.standard_normal(n)        # room / mic noise
    audio = 0.25 * audio / (np.max(np.abs(audio)) + 1e-12) * (1.3 if fault == "bind" else 1.0)
    return audio.astype(np.float32), np.array(labels)


def write_dataset(root: str | Path, n_normal: int = 20, n_per_fault: int = 3, sr: int = 44100) -> Path:
    """Write synthetic WAVs + a runs.csv into root/normal, root/abnormal (44.1 kHz like a phone)."""
    import pandas as pd

    root = Path(root)
    (root / "normal").mkdir(parents=True, exist_ok=True)
    (root / "abnormal").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_normal):
        name = f"synth_normal_{i:02d}.wav"
        sf.write(root / "normal" / name, make_saw_run(seed=i, sr=sr)[0], sr)
        rows.append(dict(filename=name, label="normal", material="synthetic", fault=""))
    for j, fault in enumerate(FAULTS):
        for k in range(n_per_fault):
            name = f"synth_{fault}_{k:02d}.wav"
            sf.write(root / "abnormal" / name, make_saw_run(seed=1000 + 10 * j + k, sr=sr, fault=fault)[0], sr)
            rows.append(dict(filename=name, label="abnormal", material="synthetic", fault=fault))
    pd.DataFrame(rows).to_csv(root / "runs.csv", index=False)
    return root
