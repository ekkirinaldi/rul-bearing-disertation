"""PHM2012 preprocessing aligned with Liu et al. (Sensors 2026, 26, 1578).

Horizontal channel only; 34 engineered features; degradation metric ``T_o``;
ISOMAP to a scalar health index. Triggered by ``hi_pipeline: liu2026_phm`` in
``configs/data/phm2012_paper.yaml`` (or ``phm2012_liu2026_strict.yaml``).

The same §3.2 feature + ISOMAP machinery is reused for XJTU-SY via
:class:`mxlstm.data.liu2026_xjtu.Liu2026XjtuHiPipeline` (``hi_pipeline: liu2026_xjtu``),
which appends a normalized time column per §3.3.1.

Where the PDF equations for monotonicity/robustness were incomplete in text extraction,
``_mon_metric`` / ``_rob_metric`` use standard degradation-feature proxies; the
spectral-kurtosis block uses distribution moments of the magnitude spectrum rather
than a full SK estimator. Selection still follows the paper's weights and ``T_o>0.6``
rule with a top-feature fallback when nothing passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps
from scipy import stats as sst
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler

from mxlstm.data.hi import MinMaxScaler as NumpyMinMaxScaler, smooth_hi

try:
    import pywt
except ImportError as e:  # pragma: no cover
    pywt = None  # type: ignore[misc, assignment]
    _PYWT_ERR = e
else:
    _PYWT_ERR = None

# Paper §3.2: ω1=0.2, ω2=0.5, ω3=0.3; keep features with T_o > 0.6
_W1, _W2, _W3 = 0.2, 0.5, 0.3
_TO_THRESHOLD = 0.6
_ISOMAP_NEIGHBORS = 10


def _require_pywt() -> None:
    if pywt is None:
        raise ImportError(
            "Liu et al. (2026) PHM pipeline requires PyWavelets (wavelet packet energies). "
            "Install with: pip install PyWavelets"
        ) from _PYWT_ERR


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(3, int(win))
    win = min(win, len(x))
    k = np.ones(win, dtype=np.float64) / float(win)
    return sps.convolve(x.astype(np.float64), k, mode="same")


def _time_domain_17(x: np.ndarray) -> np.ndarray:
    """Seventeen time-domain scalars for one acquisition (1-D float array)."""
    eps = 1e-12
    x = np.asarray(x, dtype=np.float64).ravel()
    mx = float(np.max(x))
    mn = float(np.min(x))
    med = float(np.median(x))
    p2p = mx - mn
    mean_abs = float(np.mean(np.abs(x))) + eps
    var = float(np.var(x))
    std = float(np.std(x))
    kurt = float(sst.kurtosis(x, fisher=False))
    skew = float(sst.skew(x))
    rms = float(np.sqrt(np.mean(x**2)))
    msq = float(np.mean(x**2))
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(x)))) + eps
    sra = float(mean_sqrt_abs**2)
    waveform = rms / mean_abs
    crest = mx / (rms + eps)
    impulse = mx / mean_abs
    margin = mx / (mean_sqrt_abs**2 + eps)
    energy = float(np.sum(x**2))
    return np.asarray(
        [
            mx,
            mn,
            med,
            p2p,
            mean_abs,
            var,
            std,
            kurt,
            skew,
            rms,
            msq,
            sra,
            waveform,
            crest,
            impulse,
            margin,
            energy,
        ],
        dtype=np.float32,
    )


def _freq_domain_5(x: np.ndarray, fs: int) -> np.ndarray:
    """Five frequency-domain features (Welch PSD)."""
    eps = 1e-12
    x = np.asarray(x, dtype=np.float64).ravel()
    nperseg = min(2048, len(x))
    freqs, psd = sps.welch(x, fs=fs, nperseg=nperseg)
    psd = np.maximum(psd, 0.0)
    psd_sum = psd.sum() + eps
    psd_n = psd / psd_sum
    centroid = float(np.sum(freqs * psd_n))
    msf = float(np.sum((freqs**2) * psd_n))
    rmsf = float(np.sqrt(np.sum((freqs**2) * psd) / psd_sum))
    freq_var = float(np.sum(((freqs - centroid) ** 2) * psd_n))
    total_power = float(np.sum(psd))
    return np.asarray([centroid, msf, rmsf, freq_var, total_power], dtype=np.float32)


def _spectral_shape_4(x: np.ndarray, fs: int) -> np.ndarray:
    """Four statistics of the magnitude spectrum (paper: spectral kurtosis-related)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    mag = np.abs(np.fft.rfft(x))
    if mag.size < 4:
        return np.zeros(4, dtype=np.float32)
    mag = mag.astype(np.float64)
    m = float(np.mean(mag))
    s = float(np.std(mag))
    sk = float(sst.skew(mag))
    ku = float(sst.kurtosis(mag, fisher=False))
    return np.asarray([m, s, sk, ku], dtype=np.float32)


def _wpt_energy_8(x: np.ndarray) -> np.ndarray:
    """Eight level-3 wavelet packet energy proportions (paper §3.2.1)."""
    _require_pywt()
    x = np.asarray(x, dtype=np.float64).ravel()
    wp = pywt.WaveletPacket(data=x, wavelet="db4", mode="symmetric", maxlevel=3)
    nodes = wp.get_level(3, order="freq")
    energies = np.array([float(np.sum(n.data**2)) for n in nodes], dtype=np.float64)
    s = energies.sum() + 1e-12
    return (energies / s).astype(np.float32)


def extract_liu2026_feature_vector(x_1ch: np.ndarray, fs: int) -> np.ndarray:
    """Single acquisition, one channel → (34,) float32."""
    td = _time_domain_17(x_1ch)
    fd = _freq_domain_5(x_1ch, fs)
    sp = _spectral_shape_4(x_1ch, fs)
    wp = _wpt_energy_8(x_1ch)
    return np.concatenate([td, fd, sp, wp]).astype(np.float32)


def extract_liu2026_feature_matrix(signal: np.ndarray, fs: int) -> np.ndarray:
    """``signal`` shape (T, 1, L) → (T, 34)."""
    if signal.ndim != 3 or signal.shape[1] != 1:
        raise ValueError(f"Liu2026 PHM expects signal (T, 1, L); got {signal.shape}")
    T = signal.shape[0]
    out = np.empty((T, 34), dtype=np.float32)
    for t in range(T):
        out[t] = extract_liu2026_feature_vector(signal[t, 0], fs)
    return out


def _corr_time(x: np.ndarray) -> float:
    T = len(x)
    if T < 3:
        return 0.0
    t = np.arange(T, dtype=np.float64)
    r = np.corrcoef(x.astype(np.float64), t)[0, 1]
    if np.isnan(r):
        return 0.0
    return float(np.clip(abs(r), 0.0, 1.0))


def _mon_metric(x: np.ndarray) -> float:
    """Local smooth monotonic trend strength in [0, 1] (paper Eq. 21 spirit)."""
    x = x.astype(np.float64)
    if len(x) < 2:
        return 0.0
    d = np.abs(np.diff(x))
    ptp = float(np.ptp(x)) + 1e-12
    return float(np.clip(np.mean(d) / ptp, 0.0, 1.0))


def _rob_metric(x: np.ndarray) -> float:
    """Robustness via inverse relative noise after short MA detrend (paper Eq. 22 spirit)."""
    x = x.astype(np.float64)
    T = len(x)
    if T < 5:
        return 0.0
    win = max(5, min(51, T // 20 * 2 + 1))
    xt = _moving_average(x, win)
    xr = x - xt
    rel = float(np.std(xr) / (np.mean(np.abs(xt)) + 1e-12))
    return float(np.clip(1.0 / (1.0 + rel), 0.0, 1.0))


def degradation_metric_to(x: np.ndarray) -> float:
    """Weighted ``T_o`` from paper Eq. (23)."""
    c = _corr_time(x)
    m = _mon_metric(x)
    r = _rob_metric(x)
    return float(_W1 * c + _W2 * m + _W3 * r)


def select_feature_indices(feature_matrix_train_concat: np.ndarray) -> np.ndarray:
    """Return indices of features with ``T_o > 0.6`` (paper §3.2.2).

    If none pass, keep the top three features by ``T_o`` for numerical stability.
    """
    X = feature_matrix_train_concat.astype(np.float64)
    F = X.shape[1]
    scores = np.array([degradation_metric_to(X[:, j]) for j in range(F)], dtype=np.float64)
    idx = np.where(scores > _TO_THRESHOLD)[0]
    if idx.size == 0:
        idx = np.argsort(-scores)[: max(3, min(8, F))]
    return np.sort(idx.astype(np.int64))


FEATURE_NAMES_34: list[str] = (
    [f"td_{i}" for i in range(17)]
    + [f"fd_{i}" for i in range(5)]
    + [f"sp_{i}" for i in range(4)]
    + [f"wp_{i}" for i in range(8)]
)


@dataclass
class Liu2026PhmHiPipeline:
    """Fit on train bearings only (features → selection → ISOMAP → 1-D HI)."""

    fs: int
    smoothing_alpha: float = 0.1
    selected_indices: np.ndarray = field(default_factory=lambda: np.arange(34, dtype=np.int64))
    _feat_scaler: MinMaxScaler | None = None
    _isomap: Isomap | None = None
    _hi_scaler: NumpyMinMaxScaler | None = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES_34))

    def extract(self, signal: np.ndarray) -> np.ndarray:
        return extract_liu2026_feature_matrix(signal, self.fs)

    def fit(self, train_signals: list[np.ndarray]) -> Liu2026PhmHiPipeline:
        raw_list = [self.extract(sig) for sig in train_signals]
        cat = np.concatenate(raw_list, axis=0)
        self.selected_indices = select_feature_indices(cat)
        X = cat[:, self.selected_indices]
        self._feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        Xn = self._feat_scaler.fit_transform(X)
        self._isomap = Isomap(n_neighbors=_ISOMAP_NEIGHBORS, n_components=1)
        emb = self._isomap.fit_transform(Xn).astype(np.float64).ravel()
        self._hi_scaler = NumpyMinMaxScaler().fit(emb.reshape(-1, 1))
        return self

    def transform_hi_raw(self, raw_feats: np.ndarray) -> np.ndarray:
        """``(T, 34)`` raw → ``(T, 1)`` normalized ISOMAP HI (no smoothing)."""
        if self._feat_scaler is None or self._isomap is None or self._hi_scaler is None:
            raise RuntimeError("Call fit() before transform_hi_raw().")
        X = raw_feats[:, self.selected_indices].astype(np.float64)
        Xn = self._feat_scaler.transform(X)
        emb = self._isomap.transform(Xn).astype(np.float64).ravel()
        return self._hi_scaler.transform(emb.reshape(-1, 1))

    def transform_signal(self, signal: np.ndarray) -> np.ndarray:
        """``(T, 1, L)`` → ``(T, 1)`` smoothed HI for the network."""
        raw = self.extract(signal)
        hi = self.transform_hi_raw(raw)
        return smooth_hi(hi, self.smoothing_alpha)

    def to_dict(self) -> dict:
        return {
            "fs": self.fs,
            "smoothing_alpha": self.smoothing_alpha,
            "selected_indices": self.selected_indices.tolist(),
            "feat_min": self._feat_scaler.min_.tolist() if self._feat_scaler is not None else [],
            "feat_scale": self._feat_scaler.scale_.tolist() if self._feat_scaler is not None else [],
            "hi_min": float(self._hi_scaler.min_[0]) if self._hi_scaler is not None else 0.0,
            "hi_max": float(self._hi_scaler.max_[0]) if self._hi_scaler is not None else 1.0,
        }
