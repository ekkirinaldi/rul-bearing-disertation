"""Tests for Liu et al. (2026) XJTU-SY strict preprocessing (ISOMAP HI + time index)."""

from __future__ import annotations

import numpy as np

from mxlstm.data.liu2026_xjtu import Liu2026XjtuHiPipeline


def test_xjtu_pipeline_two_channel_output():
    rng = np.random.default_rng(1)
    T, L = 30, 4096
    sig_a = rng.standard_normal((T, 1, L)).astype(np.float32)
    sig_b = rng.standard_normal((T, 1, L)).astype(np.float32)

    pipe = Liu2026XjtuHiPipeline(fs=25_600, smoothing_alpha=0.1)
    pipe.fit([sig_a, sig_b])
    out = pipe.transform_signal(sig_a)
    assert out.shape == (T, 2)
    assert np.allclose(out[:, 1], np.linspace(0.0, 1.0, T, dtype=np.float32))


def test_xjtu_time_column_endpoints():
    rng = np.random.default_rng(2)
    T = 24
    sig = rng.standard_normal((T, 1, 2560)).astype(np.float32)
    pipe = Liu2026XjtuHiPipeline(fs=25_600, smoothing_alpha=0.0)
    pipe.fit([sig])
    out = pipe.transform_signal(sig)
    assert float(out[0, 1]) == 0.0
    assert abs(float(out[-1, 1]) - 1.0) < 1e-5
