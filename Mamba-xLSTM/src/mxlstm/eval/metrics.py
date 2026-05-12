"""Regression + RUL-specific metrics.

Two score conventions are exposed because the literature is split:

1. ``phm_score`` — PROJECT_PLAN.md §1.5 convention (mean of asymmetric
   exponential of percentage error, **higher is better**, max = 1)::

        Er_i = (RUL_actual_i - RUL_pred_i) / RUL_actual_i
        A_i  = exp(-ln(0.5) * (Er_i / 5))   if Er_i <= 0
        A_i  = exp( ln(0.5) * (Er_i / 20))  if Er_i >  0
        Score = mean(A_i)

2. ``phm_score_paper`` — Liu et al. (Sensors 2026, 26, 1578) Eq. 26-28,
   the variant the baseline paper actually reports (sum of asymmetric
   exponential of *signed* error, **lower is better**)::

        Error_i = pred_i - true_i                           (signed, same units)
        s_i = exp(-Error_i / 13) - 1   if Error_i <  0      (early, lenient)
        s_i = exp( Error_i / 10) - 1   if Error_i >= 0      (late,  harsh)
        Score = sum(s_i)

The plan stresses (§1.5 + §11) that errors must be in **physical RUL
units** (e.g. seconds), not normalized [0, 1] RUL. Both functions take
an optional ``horizon`` to scale predictions/targets back to physical
units. If ``horizon`` is None, normalized RUL is used — useful for
sanity checks but **not** for thesis numbers.

All functions accept numpy arrays or torch tensors; tensors are detached
and moved to CPU before computation.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor


def _as_np(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        # bf16 / fp16 tensors are not directly convertible to NumPy; promote first.
        return x.detach().float().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Standard regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a, b = _as_np(y_true), _as_np(y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a, b = _as_np(y_true), _as_np(y_pred)
    return float(np.mean(np.abs(a - b)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    a, b = _as_np(y_true), _as_np(y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# PHM Score (asymmetric exponential)
# ---------------------------------------------------------------------------


def phm_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    horizon: float | None = None,
    eps: float = 1e-6,
) -> float:
    """PHM Score per §1.5; higher is better, max = 1.

    Args:
        y_true: True RUL (normalized [0, 1] if ``horizon`` given, else physical).
        y_pred: Predicted RUL in the same convention.
        horizon: If provided, multiplies both ``y_true`` and ``y_pred`` to
            convert normalized [0, 1] back to physical units (e.g. seconds).
            Strongly recommended for thesis-quality numbers.
        eps: Numerical floor on ``y_true`` to avoid division by zero at EOL.

    Returns:
        Mean asymmetric exponential score over the dataset.
    """
    a = _as_np(y_true)
    b = _as_np(y_pred)
    if horizon is not None:
        a = a * float(horizon)
        b = b * float(horizon)

    a_safe = np.where(np.abs(a) < eps, np.sign(a) * eps + (a == 0) * eps, a)
    er = (a - b) / a_safe  # signed percentage error

    half = math.log(0.5)
    score = np.where(
        er <= 0.0,
        np.exp(-half * (er / 5.0)),   # early prediction (er<=0): lenient
        np.exp(half * (er / 20.0)),   # late prediction (er>0): harsh
    )
    return float(np.mean(score))


def phm_score_paper(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    horizon: float | None = None,
) -> float:
    """PHM Score per Liu et al. Eq. 26-28; lower is better.

    Implements ``s_i = exp(-Err/13) - 1`` for early predictions
    (``Err < 0``) and ``s_i = exp(Err/10) - 1`` for late predictions
    (``Err >= 0``), summed across all samples.

    Args:
        y_true: True RUL.
        y_pred: Predicted RUL.
        horizon: If provided, scales both arrays to physical units before
            computing the signed error (recommended for thesis numbers).
    """
    a = _as_np(y_true)
    b = _as_np(y_pred)
    if horizon is not None:
        a = a * float(horizon)
        b = b * float(horizon)
    err = b - a  # signed: positive => late prediction
    s = np.where(
        err < 0.0,
        np.exp(-err / 13.0) - 1.0,   # early: lenient (decay constant 13)
        np.exp(err / 10.0) - 1.0,    # late:  harsh   (decay constant 10)
    )
    return float(np.sum(s))


# ---------------------------------------------------------------------------
# Per-bearing aggregation helper
# ---------------------------------------------------------------------------


def per_bearing_metrics(
    y_true_by_bearing: dict[str, ArrayLike],
    y_pred_by_bearing: dict[str, ArrayLike],
    horizons: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute RMSE/MAE/R^2/PHM for each bearing in the dict."""
    out: dict[str, dict[str, float]] = {}
    horizons = horizons or {}
    for bid in sorted(y_true_by_bearing):
        a = _as_np(y_true_by_bearing[bid])
        b = _as_np(y_pred_by_bearing[bid])
        out[bid] = {
            "rmse": rmse(a, b),
            "mae": mae(a, b),
            "r2": r2(a, b),
            "phm_score": phm_score(a, b, horizon=horizons.get(bid)),
            "phm_score_paper": phm_score_paper(a, b, horizon=horizons.get(bid)),
            "n": int(a.size),
        }
    return out


def aggregate_metrics(per_bearing: dict[str, dict[str, float]]) -> dict[str, float]:
    """Mean across bearings (equal-weighted, the convention in the paper)."""
    if not per_bearing:
        return {
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "phm_score": float("nan"),
            "phm_score_paper": float("nan"),
        }
    keys: Iterable[str] = ("rmse", "mae", "r2", "phm_score", "phm_score_paper")
    return {k: float(np.mean([m[k] for m in per_bearing.values()])) for k in keys}
