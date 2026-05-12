import numpy as np

from mxlstm.data.hi import HIPipeline, extract_hi_features, extract_hi_sequence, smooth_hi
from mxlstm.data.labels import detect_degradation_onset, make_rul_labels


def _synthetic_bearing(T: int = 50, fs: int = 25_600, L: int = 2560, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = rng.normal(scale=0.1, size=(T, channels, L)).astype(np.float32)
    # Inject increasing impulses to simulate degradation
    for t in range(T):
        amp = 0.05 + 0.05 * (t / T)
        base[t] += amp * rng.normal(scale=1.0, size=(channels, L)).astype(np.float32)
    return base


def test_extract_hi_features_shape():
    rng = np.random.default_rng(0)
    acq = rng.normal(size=(2, 2560)).astype(np.float32)
    feats, names = extract_hi_features(acq, fs=25_600)
    assert feats.shape == (2 * 18,)
    assert len(names) == feats.size


def test_extract_hi_sequence_shape():
    sig = _synthetic_bearing(T=20)
    hi, names = extract_hi_sequence(sig, fs=25_600)
    assert hi.shape == (20, 2 * 18)
    assert len(names) == 2 * 18


def test_smoothing_reduces_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    s = smooth_hi(x, alpha=0.1)
    assert s.shape == x.shape
    assert s.var() < x.var()


def test_pipeline_fit_and_transform():
    sigs = [_synthetic_bearing(T=15) for _ in range(2)]
    pipe = HIPipeline(fs=25_600, smoothing_alpha=0.1)
    pipe.fit(sigs)
    out = pipe.transform_signal(sigs[0])
    assert out.shape == (15, 2 * 18)
    assert out.min() >= -0.01 and out.max() <= 1.01  # MinMax + smoothing keeps roughly in [0, 1]


def test_make_rul_labels_linear_endpoints():
    y = make_rul_labels(10, scheme="linear")
    assert y.shape == (10,)
    assert abs(y[0] - 1.0) < 1e-6
    assert abs(y[-1] - 0.0) < 1e-6


def test_make_rul_labels_piecewise():
    y = make_rul_labels(20, scheme="piecewise", degradation_onset=10)
    assert y[5] == 1.0
    assert y[-1] == 0.0
    assert y[10] == 1.0


def test_detect_degradation_onset_clean():
    # Stable baseline followed by a clear excursion
    rms = np.concatenate([np.full(100, 0.1), np.full(50, 1.0)])
    onset = detect_degradation_onset(rms, k=3.0, n0=50, min_consecutive=5)
    assert 100 <= onset <= 110
