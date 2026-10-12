"""Step 2: read WAV files and cut them into short overlapping windows.

Plain language: a recording is just a long list of numbers (air-pressure samples).
We (1) read it, (2) make it mono, (3) resample it to one fixed rate so every
recording is comparable, and (4) chop it into ~1-second slices ("windows").
Each window later gets its own anomaly score.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import config


AUDIO_EXTENSIONS = (".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".mp4", ".3gp")
_SOUNDFILE_EXTENSIONS = (".wav", ".flac", ".ogg")


def list_audio(folder: str | Path) -> list[Path]:
    """All audio files in a folder (any supported type), sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)


def load_audio(path: str | Path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Read an audio file and return mono float32 samples at sample rate `sr`.

    - WAV/FLAC/OGG are read directly; phone formats like M4A/MP3 are decoded
      with ffmpeg (bundled free by the `imageio-ffmpeg` package).
    - Stereo / multi-channel files are averaged to mono.
    - Any input rate is resampled to `sr` (e.g. 44,100 -> 22,050 Hz).
    - Values are floats roughly in [-1, 1]. We do NOT rescale loudness,
      because loudness itself is a useful clue (a jammed blade is louder).
    """
    path = Path(path)
    if path.suffix.lower() not in _SOUNDFILE_EXTENSIONS:
        return _load_with_ffmpeg(path, sr)
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # (samples, channels) -> mono
    if file_sr != sr:
        audio = resample(audio, file_sr, sr)
    return audio.astype(np.float32)


load_wav = load_audio  # old name, kept so earlier code/notebooks still work


def _load_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """Decode any format ffmpeg understands straight to mono float32 at `sr`."""
    import subprocess

    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    result = subprocess.run(cmd, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Could not decode {path.name}: {result.stderr.decode(errors='ignore').strip()}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Change sample rate with a polyphase filter (fast and accurate)."""
    if orig_sr == target_sr:
        return audio
    g = np.gcd(int(orig_sr), int(target_sr))
    return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)


def split_windows(
    audio: np.ndarray,
    sr: int = config.SAMPLE_RATE,
    window_s: float = config.WINDOW_SECONDS,
    hop_s: float = config.HOP_SECONDS,
) -> np.ndarray:
    """Cut audio into overlapping windows. Returns an array (n_windows, window_len).

    A leftover piece at the end that is shorter than one window is dropped.
    Window i starts at time i * hop_s seconds.
    """
    win = int(round(window_s * sr))
    hop = int(round(hop_s * sr))
    if len(audio) < win:
        return np.empty((0, win), dtype=np.float32)
    n = 1 + (len(audio) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return audio[idx]


def window_times(n_windows: int, hop_s: float = config.HOP_SECONDS) -> np.ndarray:
    """Start time (seconds) of each window, handy for plots."""
    return np.arange(n_windows) * hop_s
