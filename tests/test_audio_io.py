import numpy as np
import soundfile as sf

from src.audio_io import load_wav, split_windows, window_times


def _sine(freq, seconds, sr, channels=1):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def test_load_wav_resamples_and_makes_mono(tmp_path):
    path = tmp_path / "stereo_44k.wav"
    sf.write(path, _sine(1000, 2.0, 44100, channels=2), 44100)
    audio = load_wav(path, sr=22050)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert abs(len(audio) - 2 * 22050) <= 1
    # still a 1 kHz tone after resampling
    spectrum = np.abs(np.fft.rfft(audio))
    peak_hz = np.argmax(spectrum) * 22050 / len(audio)
    assert abs(peak_hz - 1000) < 5


def test_load_m4a_like_a_phone_recording(tmp_path):
    """Phones save .m4a; make one with ffmpeg and check it loads as the same tone."""
    import subprocess

    import imageio_ffmpeg

    wav, m4a = tmp_path / "tone.wav", tmp_path / "tone.m4a"
    sf.write(wav, _sine(440, 2.0, 44100), 44100)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(wav),
                    "-c:a", "aac", str(m4a)], check=True)
    audio = load_wav(m4a, sr=22050)
    assert abs(len(audio) / 22050 - 2.0) < 0.1  # AAC may add a few ms of padding
    spectrum = np.abs(np.fft.rfft(audio))
    assert abs(np.argmax(spectrum) * 22050 / len(audio) - 440) < 5


def test_split_windows_count_and_overlap():
    sr = 1000
    audio = np.arange(int(3.2 * sr), dtype=np.float32)  # 3.2 s
    w = split_windows(audio, sr=sr, window_s=1.0, hop_s=0.5)
    # starts at 0, 0.5, ..., 2.0  -> 5 full windows; the 0.2 s tail is dropped
    assert w.shape == (5, 1000)
    assert w[1, 0] == 500  # second window starts half a second in
    np.testing.assert_allclose(window_times(5, 0.5), [0, 0.5, 1.0, 1.5, 2.0])


def test_split_windows_too_short():
    assert split_windows(np.zeros(100, np.float32), sr=1000).shape == (0, 1000)
