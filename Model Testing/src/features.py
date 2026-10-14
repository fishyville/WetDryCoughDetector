"""
Feature extraction, written from scratch (PyTorch + Python standard library only).

Pipeline (same in training and in the real-time app):

    waveform (16 kHz, mono, float in [-1, 1])
      -> pre-emphasis            y[n] = x[n] - 0.97 * x[n-1]
      -> framing + Hann window   25 ms windows, 10 ms hop       (inside torch.stft)
      -> FFT -> power spectrum   |X(k)|^2
      -> mel filterbank          64 triangular filters, 20 Hz - 8 kHz   (built by hand below)
      -> log                     log-mel spectrogram  [n_mels, n_frames]   <- CNN input
      -> DCT-II (optional)       MFCC                 [n_mfcc, n_frames]

IMPORTANT: this file is copied unchanged into the app (Final Project/app/features.py).
Change settings only through FeatureConfig, which is saved in config.json at export time.
"""

import math
import wave
from dataclasses import dataclass, asdict

import torch


# ---------------------------------------------------------------- settings
@dataclass
class FeatureConfig:
    sample_rate: int = 16000
    n_fft: int = 512          # FFT size (32 ms), next power of two above win_length
    win_length: int = 400     # 25 ms analysis window
    hop_length: int = 160     # 10 ms step -> 100 frames per second
    n_mels: int = 64
    f_min: float = 20.0
    f_max: float = 8000.0     # Nyquist frequency for 16 kHz
    pre_emphasis: float = 0.97
    n_mfcc: int = 13
    clip_seconds: float = 1.0  # length of the window the model sees
    log_eps: float = 1e-6

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------- audio I/O
def load_wav(path):
    """Read a 16-bit PCM WAV with Python's `wave` module -> (float tensor [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        sr = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0
    if channels > 1:                         # average channels -> mono
        x = x.view(-1, channels).mean(dim=1)
    return x, sr


# ---------------------------------------------------------------- mel scale
def hz_to_mel(f):
    """HTK mel formula: m = 2595 * log10(1 + f / 700)."""
    return 2595.0 * math.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig):
    """
    Build the [n_mels, n_fft//2 + 1] matrix of triangular filters.

    n_mels + 2 points are spaced evenly on the mel scale between f_min and f_max.
    Filter i rises linearly from point i to point i+1 and falls back to 0 at point i+2.
    """
    n_freqs = cfg.n_fft // 2 + 1
    fft_freqs = torch.linspace(0.0, cfg.sample_rate / 2.0, n_freqs)

    mel_points = torch.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_mels + 2)
    hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])

    fb = torch.zeros(cfg.n_mels, n_freqs)
    for i in range(cfg.n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


def dct_matrix(n_mfcc, n_mels):
    """Orthonormal DCT-II matrix [n_mfcc, n_mels]: turns log-mel energies into MFCCs."""
    n = torch.arange(n_mels).float()
    k = torch.arange(n_mfcc).float().unsqueeze(1)
    D = torch.cos(math.pi / n_mels * (n + 0.5) * k)
    D[0] *= 1.0 / math.sqrt(2.0)
    return D * math.sqrt(2.0 / n_mels)


# ---------------------------------------------------------------- extractor
class FeatureExtractor:
    """Holds the precomputed window, filterbank and DCT so each call is fast (needed for real time)."""

    def __init__(self, cfg: FeatureConfig = None):
        self.cfg = cfg or FeatureConfig()
        self.window = torch.hann_window(self.cfg.win_length)
        self.mel_fb = mel_filterbank(self.cfg)
        self.dct = dct_matrix(self.cfg.n_mfcc, self.cfg.n_mels)

    def to(self, device):
        """Move the precomputed window/filterbank/DCT to a device (e.g. 'cuda') for fast batches."""
        self.window = self.window.to(device)
        self.mel_fb = self.mel_fb.to(device)
        self.dct = self.dct.to(device)
        return self

    # All methods accept one clip [time] or a batch [batch, time].
    def pre_emphasize(self, x):
        if self.cfg.pre_emphasis <= 0:
            return x
        return torch.cat([x[..., :1], x[..., 1:] - self.cfg.pre_emphasis * x[..., :-1]], dim=-1)

    def power_spectrogram(self, x):
        """[time] -> [n_fft//2 + 1, n_frames] power spectrum (or [batch, ...] for a batch)."""
        X = torch.stft(x, n_fft=self.cfg.n_fft, hop_length=self.cfg.hop_length,
                       win_length=self.cfg.win_length, window=self.window,
                       center=True, return_complex=True)
        return X.real ** 2 + X.imag ** 2

    def log_mel(self, x):
        """Waveform [time] -> log-mel [n_mels, n_frames]; batch [B, time] -> [B, n_mels, n_frames]."""
        x = self.pre_emphasize(x)
        power = self.power_spectrogram(x)
        mel = self.mel_fb @ power
        return torch.log(mel + self.cfg.log_eps)

    def mfcc(self, x):
        """Waveform [time] -> MFCC [n_mfcc, n_frames]."""
        return self.dct @ self.log_mel(x)


# ---------------------------------------------------------------- cropping
def frame_energy(x, frame=400, hop=160):
    """Short-time energy per frame (used to find where the cough is)."""
    if x.numel() < frame:
        x = torch.nn.functional.pad(x, (0, frame - x.numel()))
    frames = x.unfold(0, frame, hop)
    return (frames ** 2).mean(dim=1)


def loudest_window(x, cfg: FeatureConfig):
    """
    Return the clip_seconds window with the highest energy, padded if the audio is shorter.
    COUGHVID recordings are ~10 s with silence around the cough; the app also sees ~1 s windows,
    so training on the loudest 1 s keeps training and real-time input consistent.
    """
    n = int(cfg.clip_seconds * cfg.sample_rate)
    if x.numel() <= n:
        return torch.nn.functional.pad(x, (0, n - x.numel()))
    hop = cfg.hop_length
    e = frame_energy(x, cfg.win_length, hop)
    frames_per_win = max(1, n // hop)
    # moving sum of energy over a 1 s span, computed with a cumulative sum
    c = torch.cat([torch.zeros(1), torch.cumsum(e, 0)])
    sums = c[frames_per_win:] - c[:-frames_per_win] if e.numel() > frames_per_win else c[-1:]
    start = int(torch.argmax(sums).item()) * hop
    start = min(start, x.numel() - n)
    return x[start:start + n]


def peak_normalize(x, target=0.9):
    """Scale so the loudest sample is at `target`; removes differences in recording volume."""
    peak = x.abs().max()
    return x if peak < 1e-8 else x * (target / peak)


if __name__ == "__main__":
    # quick self-test: python src/features.py data/wav/<some uuid>.wav
    import sys
    fx = FeatureExtractor()
    if len(sys.argv) > 1:
        x, sr = load_wav(sys.argv[1])
        assert sr == fx.cfg.sample_rate, f"expected {fx.cfg.sample_rate} Hz, got {sr}"
        print(f"loaded {x.numel() / sr:.2f} s")
    else:
        x = torch.randn(fx.cfg.sample_rate * 2) * 0.1
        print("no file given, using 2 s of noise")
    clip = peak_normalize(loudest_window(x, fx.cfg))
    print("log-mel:", tuple(fx.log_mel(clip).shape), " mfcc:", tuple(fx.mfcc(clip).shape))
