"""Step 3: turn each ~1 s audio window into a short list of numbers ("features").

Plain language: for every window we measure
  - loudness (RMS, in decibels)
  - dominant frequency (the strongest pitch, e.g. motor/blade speed)
  - spectral centroid ("brightness": scraping/grinding pushes it up)
  - spectral roll-off (frequency below which 85% of the energy sits)
  - a 64-band log-mel spectrum (a "tone fingerprint", bands spaced like human hearing)

Implementation note: everything is plain numpy (no padding tricks) so that the
phone web page (web/features.js) can reproduce the exact same numbers. Only the
mel filterbank comes from librosa, and it is exported into the model file.
"""
from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from . import config

EPS = 1e-10
ROLLOFF_FRACTION = 0.85
SCALAR_NAMES = ["rms_db", "dominant_hz", "centroid_hz", "rolloff_hz", "pulse_depth", "pulse_hz"]
N_SCALARS = len(SCALAR_NAMES)

# "Pulsing" (the "duh duh duh" sound): loudness measured every 20 ms inside the window
ENVELOPE_FRAME_S = 0.020
PULSE_MIN_HZ, PULSE_MAX_HZ = 1.0, 20.0
PULSE_FFT = 256  # zero-padded so the pulse rate is estimated finer than 1 Hz


def feature_names(n_mels: int = config.N_MELS) -> list[str]:
    return SCALAR_NAMES + [f"mel_{i:02d}" for i in range(n_mels)]


@lru_cache(maxsize=8)
def mel_filterbank(sr: int, n_fft: int = config.N_FFT, n_mels: int = config.N_MELS) -> np.ndarray:
    """Matrix (n_mels, n_fft//2+1) that groups FFT bins into mel bands."""
    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20.0, fmax=sr / 2)


@lru_cache(maxsize=8)
def _hann(n_fft: int) -> np.ndarray:
    # periodic Hann window (same formula used in JS)
    return (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float64)


def power_spectrogram(window: np.ndarray, n_fft: int = config.N_FFT, hop: int = config.FRAME_HOP) -> np.ndarray:
    """Short frames -> FFT -> power. Returns (n_frames, n_fft//2+1). No padding."""
    x = np.asarray(window, dtype=np.float64)
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * _hann(n_fft)
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2


def _peak_hz(mag: np.ndarray, sr: int, n_fft: int) -> float:
    """Strongest frequency, refined between FFT bins by fitting a parabola
    through the peak and its two neighbours (log magnitude). FFT bins are
    ~11 Hz apart, too coarse to see a motor slowing from 100 to 92 Hz."""
    k = 1 + int(np.argmax(mag[1:-1]))                  # skip DC and the last bin
    a, b, c = np.log(mag[k - 1:k + 2] + EPS)
    denom = a - 2 * b + c
    offset = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return float((k + np.clip(offset, -0.5, 0.5)) * sr / n_fft)


def pulse_features(x: np.ndarray, sr: int) -> tuple[float, float]:
    """How strongly, and how fast, the loudness throbs inside this window.

    depth: spread of the 20 ms loudness values divided by their average
           (steady hum ~0, a "duh duh duh" thump pattern is much higher)
    rate:  the throb frequency in Hz (strongest rhythm between 1 and 20 Hz)
    """
    frame = int(round(ENVELOPE_FRAME_S * sr))
    n = len(x) // frame
    env = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    mean = env.mean()
    depth = float(env.std() / (mean + EPS))
    spec = np.abs(np.fft.rfft(env - mean, PULSE_FFT))
    freqs = np.arange(len(spec)) / (PULSE_FFT * ENVELOPE_FRAME_S)
    band = (freqs >= PULSE_MIN_HZ) & (freqs <= PULSE_MAX_HZ)
    rate = float(freqs[band][np.argmax(spec[band])])
    return depth, rate


def extract_features(window: np.ndarray, sr: int = config.SAMPLE_RATE,
                     n_fft: int = config.N_FFT, hop: int = config.FRAME_HOP,
                     n_mels: int = config.N_MELS) -> np.ndarray:
    """One window of audio -> 1-D feature vector (6 + n_mels numbers)."""
    x = np.asarray(window, dtype=np.float64)
    power = power_spectrogram(x, n_fft, hop)          # (frames, bins)
    mean_power = power.mean(axis=0)                    # average spectrum of this window
    mag = np.sqrt(mean_power)
    freqs = np.arange(len(mean_power)) * sr / n_fft

    rms_db = 10 * np.log10(np.mean(x ** 2) + EPS)
    dominant_hz = _peak_hz(mag, sr, n_fft)
    total = mag.sum() + EPS
    centroid_hz = float((freqs * mag).sum() / total)
    cumulative = np.cumsum(mag)
    rolloff_hz = freqs[min(np.searchsorted(cumulative, ROLLOFF_FRACTION * cumulative[-1]), len(freqs) - 1)]

    mel = mel_filterbank(sr, n_fft, n_mels) @ power.T  # (n_mels, frames)
    log_mel = np.log10(mel + EPS).mean(axis=1) * 10    # average dB per band

    pulse_depth, pulse_hz = pulse_features(x, sr)

    scalars = [rms_db, dominant_hz, centroid_hz, rolloff_hz, pulse_depth, pulse_hz]
    return np.concatenate([scalars, log_mel]).astype(np.float32)


def extract_all(windows: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """(n_windows, samples) -> (n_windows, n_features)."""
    if len(windows) == 0:
        return np.empty((0, len(feature_names())), dtype=np.float32)
    return np.stack([extract_features(w, sr) for w in windows])
