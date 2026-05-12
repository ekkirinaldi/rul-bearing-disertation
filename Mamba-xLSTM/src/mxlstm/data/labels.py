"""RUL label construction per §6.2.

  * ``linear``    : RUL[t] = (T - 1 - t) / (T - 1)   — straight line in [0, 1]
  * ``piecewise`` : RUL[t] = 1 for t < T_degrade, then linear decay to 0 at EOL
                    where T_degrade is detected by an RMS-threshold rule
                    ``RMS[t] > k * mean(RMS[0:n0])`` (Jiang et al. 2023).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

LabelScheme = Literal["linear", "piecewise", "piecewise_liu2026"]


def detect_degradation_onset_liu2026(
    hi_iso: np.ndarray,
    *,
    healthy_frac: float = 0.2,
    k_sigma: float = 3.0,
    min_consecutive: int = 5,
) -> int:
    """First prediction time from Liu et al. (2026) §3.2.2 (ISOMAP HI + 3σ after healthy head).

    The first ``healthy_frac`` fraction of acquisitions defines the healthy-stage baseline
    (mean/std). The onset is the first index after that head where ``hi_iso`` exceeds
    ``mean + k_sigma * std`` for ``min_consecutive`` steps.
    """
    x = np.asarray(hi_iso, dtype=np.float64).ravel()
    T = x.size
    if T < 10:
        return max(0, T - 2)
    n0 = max(3, min(int(T * healthy_frac), T - 3))
    baseline = x[:n0]
    mu = float(np.mean(baseline))
    sig = float(np.std(baseline) + 1e-12)
    thresh = mu + k_sigma * sig
    start = n0
    run = 0
    for t in range(start, T):
        run = run + 1 if x[t] > thresh else 0
        if run >= min_consecutive:
            return t - min_consecutive + 1
    return max(0, T - 2)


def detect_degradation_onset(
    rms_series: np.ndarray,
    k: float = 3.0,
    n0: int = 50,
    min_consecutive: int = 5,
) -> int:
    """Find first index where RMS persistently exceeds ``k * mean(RMS[0:n0])``.

    Args:
        rms_series: 1-D array of per-acquisition RMS values.
        k: Threshold multiplier (plan suggests 2.5–3).
        n0: Number of initial healthy acquisitions used for the baseline mean.
        min_consecutive: How many acquisitions in a row must exceed the threshold.

    Returns:
        Integer index of the onset, clamped to ``[n0, T-1]``. If no
        crossing is detected, returns ``T-1`` (means: linear-only).
    """
    T = len(rms_series)
    if T <= n0 + min_consecutive:
        return max(0, T - 1)
    baseline = float(np.mean(rms_series[:n0]))
    threshold = k * baseline
    above = rms_series > threshold
    run = 0
    for t in range(n0, T):
        run = run + 1 if above[t] else 0
        if run >= min_consecutive:
            return t - min_consecutive + 1
    return T - 1


def make_rul_labels(
    n_acquisitions: int,
    *,
    scheme: LabelScheme = "linear",
    degradation_onset: int | None = None,
) -> np.ndarray:
    """Build RUL labels in [0, 1] for a single bearing.

    For ``piecewise`` you must pass ``degradation_onset`` (call
    ``detect_degradation_onset`` first using the un-smoothed RMS series).
    """
    T = int(n_acquisitions)
    if T < 2:
        return np.ones(max(T, 0), dtype=np.float32)
    eol = T - 1
    t_idx = np.arange(T, dtype=np.float32)
    if scheme == "linear":
        rul = (eol - t_idx) / eol
    elif scheme == "piecewise":
        if degradation_onset is None:
            raise ValueError("piecewise labels require degradation_onset.")
        ds = max(0, min(int(degradation_onset), eol))
        rul = np.ones(T, dtype=np.float32)
        if ds < eol:
            decay_len = eol - ds
            rul[ds:] = (eol - np.arange(ds, T, dtype=np.float32)) / decay_len
    else:
        raise ValueError(f"Unknown label scheme: {scheme}")
    return np.clip(rul, 0.0, 1.0).astype(np.float32)
