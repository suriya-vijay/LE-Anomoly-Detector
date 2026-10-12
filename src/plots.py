"""Reusable plots for the notebooks (so each notebook cell stays short)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config
from .audio_io import list_audio, load_audio, split_windows, window_times
from .features import N_SCALARS, extract_all, feature_names


def find_recordings(root: str | Path) -> list[tuple[str, Path | None]]:
    """[(label, path)] for every recording in data/normal, data/abnormal and data/test.

    If there are none yet, returns two synthetic stand-ins (path=None)."""
    root = Path(root)
    found = [(label, p) for label in ("normal", "abnormal", "test") for p in list_audio(root / "data" / label)]
    return found or [("normal", None), ("abnormal", None)]


def load_recording(label: str, path: Path | None, sr: int = config.SAMPLE_RATE):
    """(audio, title). Synthetic runs are used when path is None."""
    if path is not None:
        return load_audio(path, sr), f"{label}: {path.name}"
    from .synth import make_saw_run

    fault = None if label == "normal" else "thump"
    audio, _ = make_saw_run(seed=1 if fault is None else 7, sr=sr, fault=fault)
    return audio, f"{label}: SYNTHETIC saw run{' with thump fault' if fault else ''} (no real recordings yet)"


def plot_recording(audio: np.ndarray, title: str, sr: int = config.SAMPLE_RATE):
    """Waveform (loudness over time) + spectrogram (pitch over time)."""
    import librosa
    import librosa.display

    fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    t = np.arange(len(audio)) / sr
    ax[0].plot(t, audio, lw=0.3)
    ax[0].set_ylabel("amplitude")
    ax[0].set_title(title)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(audio, n_fft=config.N_FFT, hop_length=config.FRAME_HOP)), ref=np.max)
    librosa.display.specshow(S, sr=sr, hop_length=config.FRAME_HOP, x_axis="time", y_axis="log", ax=ax[1])
    ax[1].set_ylabel("frequency (Hz, log)")
    _minor_seconds(ax[1], len(audio) / sr)
    fig.tight_layout()
    return fig


def plot_features(audio: np.ndarray, title: str, sr: int = config.SAMPLE_RATE):
    """The easy-to-read features over time, plus the 64-band tone fingerprint."""
    feats = extract_all(split_windows(audio, sr), sr)
    names = feature_names()
    t = window_times(len(feats)) + config.WINDOW_SECONDS / 2
    rows = {"rms_db": "loudness (dB)", "dominant_hz": "main pitch (Hz)",
            "centroid_hz": "brightness (Hz)", "pulse_depth": "pulsing strength"}
    fig, ax = plt.subplots(len(rows) + 1, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw=dict(height_ratios=[1] * len(rows) + [2]))
    for a, (key, lab) in zip(ax, rows.items()):
        a.plot(t, feats[:, names.index(key)], ".-")
        a.set_ylabel(lab, fontsize=9)
        a.grid(alpha=0.3)
    ax[1].set_yscale("log")
    mel = feats[:, N_SCALARS:].T
    ax[-1].imshow(mel, aspect="auto", origin="lower", extent=[t[0], t[-1], 0, mel.shape[0]], cmap="magma")
    ax[-1].set_ylabel("tone band (low -> high)")
    ax[-1].set_xlabel("time (s)")
    _minor_seconds(ax[-1], t[-1])
    ax[0].set_title(title)
    fig.tight_layout()
    return fig, feats


def _minor_seconds(ax, total_s: float):
    """Tick every 5 s (every 10 s for long recordings) so you can match events by ear."""
    step = 5 if total_s <= 120 else 10
    ax.set_xticks(np.arange(0, total_s + step, step))
