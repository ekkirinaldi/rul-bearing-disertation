"""Health Indicator (HI) extraction per §6.1 of PROJECT_PLAN.md.

For each acquisition we compute:

  * 9 time-domain features: RMS, peak, kurtosis, skewness, crest factor,
    shape factor, impulse factor, margin factor, variance.
  * 7 frequency-domain features (via Welch PSD): spectral centroid,
    spectral entropy, energy in 5 frequency bands, mean frequency,
    RMS frequency.

Two channels (horizontal/vertical) → ``2 * (9 + 7) = 32`` features per
acquisition. We follow the plan's "16 features" target by averaging
across channels (or set ``per_channel=True`` to keep all 32 — useful for
ablation A6 and SHAP). The default is ``per_channel=True`` because
keeping channel information consistently improved RMSE in the
xLSTM-Transformer paper.

Post-processing pipeline (in ``HIPipeline``):
  1. Per-feature MinMax scaler fit on **train bearings only**.
  2. Optional exponential smoothing α=0.1 along the time axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy import signal as sps
from scipy import stats as sst

# ---------------------------------------------------------------------------
# Per-channel features
# ---------------------------------------------------------------------------


def _time_domain_features_1ch(x: np.ndarray) -> np.ndarray:
    """9 time-domain features for a 1-D signal."""
    eps = 1e-12
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    kurt = float(sst.kurtosis(x, fisher=False))  # Fisher=False matches paper's "kurtosis"
    skew = float(sst.skew(x))
    mean_abs = float(np.mean(np.abs(x))) + eps
    crest = peak / (rms + eps)
    shape = rms / mean_abs
    impulse = peak / mean_abs
    margin = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + eps)
    var = float(np.var(x))
    return np.asarray([rms, peak, kurt, skew, crest, shape, impulse, margin, var], dtype=np.float32)


_TIME_NAMES = ["rms", "peak", "kurtosis", "skewness", "crest", "shape", "impulse", "margin", "variance"]


def _freq_domain_features_1ch(x: np.ndarray, fs: int, n_bands: int = 5) -> np.ndarray:
    """7 frequency-domain features via Welch PSD (5 band energies + centroid + entropy).

    Plan §6.1 lists 7 features but defines 5 band energies + centroid +
    entropy + mean_freq + rms_freq → 9. We follow the plan's *quantitative*
    listing rather than the "7" header and return 9 features so the totals
    match ``len(time)+len(freq) == 18`` per channel.
    """
    eps = 1e-12
    nperseg = min(2048, len(x))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + eps)

    centroid = float(np.sum(freqs * psd_norm))
    entropy = float(-np.sum(psd_norm[psd_norm > 0] * np.log(psd_norm[psd_norm > 0] + eps)))
    mean_freq = float(np.sum(freqs * psd) / (psd.sum() + eps))
    rms_freq = float(np.sqrt(np.sum((freqs**2) * psd) / (psd.sum() + eps)))

    band_edges = np.linspace(0, fs / 2.0, n_bands + 1)
    band_energies = np.empty(n_bands, dtype=np.float32)
    for b in range(n_bands):
        mask = (freqs >= band_edges[b]) & (freqs < band_edges[b + 1])
        band_energies[b] = float(psd[mask].sum())

    return np.concatenate([[centroid, entropy, mean_freq, rms_freq], band_energies]).astype(np.float32)


_FREQ_NAMES = ["centroid", "entropy", "mean_freq", "rms_freq"] + [f"band{b}" for b in range(5)]


# ---------------------------------------------------------------------------
# Public API: per-acquisition HI
# ---------------------------------------------------------------------------


def extract_hi_features(
    acq: np.ndarray,
    fs: int,
    *,
    n_bands: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """One-acquisition HI feature vector.

    Args:
        acq: Shape ``(C, L)``; C channels, L samples per channel.
        fs: Sampling frequency in Hz.
        n_bands: Number of equal-width frequency bands (default 5).

    Returns:
        feats: ``(C * 18,)`` float32 vector.
        names: matching list of feature names.
    """
    if acq.ndim != 2:
        raise ValueError(f"Expected (C, L); got {acq.shape}")
    C = acq.shape[0]
    pieces = []
    names: list[str] = []
    for c in range(C):
        td = _time_domain_features_1ch(acq[c])
        fd = _freq_domain_features_1ch(acq[c], fs=fs, n_bands=n_bands)
        pieces.append(td)
        pieces.append(fd)
        names.extend([f"td_c{c}_{n}" for n in _TIME_NAMES])
        names.extend([f"fd_c{c}_{n}" for n in _FREQ_NAMES])
    return np.concatenate(pieces).astype(np.float32), names


def extract_hi_sequence(signal: np.ndarray, fs: int, *, n_bands: int = 5) -> tuple[np.ndarray, list[str]]:
    """Apply ``extract_hi_features`` to every acquisition in a bearing.

    Args:
        signal: Shape ``(T, C, L)`` (matches ``brul.data.base.BearingRun.signal``).
        fs: Sampling frequency.

    Returns:
        hi: Shape ``(T, F)`` float32 where ``F == C * 18``.
        names: List of length F with the feature names.
    """
    if signal.ndim != 3:
        raise ValueError(f"Expected (T, C, L); got {signal.shape}")
    T = signal.shape[0]
    feats0, names = extract_hi_features(signal[0], fs, n_bands=n_bands)
    hi = np.empty((T, feats0.size), dtype=np.float32)
    hi[0] = feats0
    for t in range(1, T):
        f, _ = extract_hi_features(signal[t], fs, n_bands=n_bands)
        hi[t] = f
    return hi, names


# ---------------------------------------------------------------------------
# Normalization + smoothing pipeline
# ---------------------------------------------------------------------------


@dataclass
class MinMaxScaler:
    """Per-feature MinMax scaler (numpy)."""

    min_: np.ndarray = field(default_factory=lambda: np.empty(0))
    max_: np.ndarray = field(default_factory=lambda: np.empty(0))

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        denom = (self.max_ - self.min_)
        denom = np.where(denom < 1e-12, 1.0, denom)
        return ((X - self.min_) / denom).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {"min": self.min_.tolist(), "max": self.max_.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "MinMaxScaler":
        s = cls()
        s.min_ = np.asarray(d["min"], dtype=np.float32)
        s.max_ = np.asarray(d["max"], dtype=np.float32)
        return s


def fit_scaler(hi_train: Iterable[np.ndarray]) -> MinMaxScaler:
    """Fit a MinMax scaler on the concatenation of per-bearing HI matrices."""
    cat = np.concatenate(list(hi_train), axis=0)
    return MinMaxScaler().fit(cat)


def smooth_hi(hi: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential smoothing along axis 0.

    ``y_t = alpha * x_t + (1 - alpha) * y_{t-1}`` with ``y_0 = x_0``.
    Uses ``alpha`` consistent with §1.2 of the plan.
    """
    if alpha <= 0.0:
        return hi.astype(np.float32, copy=True)
    out = np.empty_like(hi, dtype=np.float32)
    out[0] = hi[0]
    for t in range(1, hi.shape[0]):
        out[t] = alpha * hi[t] + (1.0 - alpha) * out[t - 1]
    return out


@dataclass
class HIPipeline:
    """Full HI pipeline: extract → normalize → smooth.

    Stateful: ``fit`` learns scaler params; ``transform`` applies them.
    """

    fs: int
    n_bands: int = 5
    smoothing_alpha: float = 0.1
    scaler: MinMaxScaler | None = None
    feature_names: list[str] = field(default_factory=list)

    def extract(self, signal: np.ndarray) -> np.ndarray:
        hi, names = extract_hi_sequence(signal, self.fs, n_bands=self.n_bands)
        if not self.feature_names:
            self.feature_names = names
        return hi

    def fit(self, signals: list[np.ndarray]) -> "HIPipeline":
        his = [self.extract(s) for s in signals]
        self.scaler = fit_scaler(his)
        return self

    def transform_signal(self, signal: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("HIPipeline.fit() must be called before transform_signal().")
        hi = self.extract(signal)
        hi = self.scaler.transform(hi)
        hi = smooth_hi(hi, self.smoothing_alpha)
        return hi

    def transform_hi(self, hi: np.ndarray) -> np.ndarray:
        """Apply scaler+smoothing to an *already-extracted* HI."""
        if self.scaler is None:
            raise RuntimeError("HIPipeline.fit() must be called before transform_hi().")
        out = self.scaler.transform(hi)
        return smooth_hi(out, self.smoothing_alpha)
