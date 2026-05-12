"""Tests for Liu et al. (2026) PHM strict preprocessing."""

from __future__ import annotations

import numpy as np

from mxlstm.data.labels import detect_degradation_onset_liu2026
from mxlstm.data.liu2026_phm import (
    extract_liu2026_feature_matrix,
    extract_liu2026_feature_vector,
    select_feature_indices,
)


def test_feature_vector_length():
    x = np.random.randn(2560).astype(np.float32)
    v = extract_liu2026_feature_vector(x, fs=25_600)
    assert v.shape == (34,)


def test_feature_matrix_shape():
    T = 20
    sig = np.random.randn(T, 1, 2560).astype(np.float32)
    m = extract_liu2026_feature_matrix(sig, fs=25_600)
    assert m.shape == (T, 34)


def test_select_feature_indices_fallback():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 34)).astype(np.float64)
    idx = select_feature_indices(X)
    assert idx.size >= 1
    assert idx.max() < 34


def test_liu2026_onset_basic():
    T = 100
    hi = np.concatenate([np.zeros(20), np.linspace(0, 3, T - 20)]).astype(np.float64)
    onset = detect_degradation_onset_liu2026(hi, healthy_frac=0.2, k_sigma=0.5, min_consecutive=3)
    assert 0 <= onset < T
