"""SHAP attribution for trained RUL models.

Uses ``shap.GradientExplainer`` (works on PyTorch nn.Module) when
available; falls back to ``shap.KernelExplainer`` for small backgrounds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _wrap_for_shap(model: nn.Module, device: torch.device | str) -> nn.Module:
    """Return a small wrapper that takes a (B, L, F) tensor and returns (B,)."""

    class _Wrapper(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.m = m

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.m(x)
            if isinstance(out, tuple):
                out = out[0]
            return out.reshape(-1, 1)

    w = _Wrapper(model).to(device).eval()
    for p in w.parameters():
        p.requires_grad = False
    return w


def compute_shap_values(
    model: nn.Module,
    background: np.ndarray,
    test: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    explainer: str = "gradient",
) -> np.ndarray:
    """Return SHAP values for ``test`` of shape ``(N, L, F)``.

    Output shape matches input: ``(N, L, F)``.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is required: pip install shap") from e

    wrapped = _wrap_for_shap(model, device)
    bg = torch.from_numpy(background.astype(np.float32)).to(device)
    ts = torch.from_numpy(test.astype(np.float32)).to(device)

    if explainer == "gradient":
        ex = shap.GradientExplainer(wrapped, bg)
        shap_values = ex.shap_values(ts)
    elif explainer == "deep":
        ex = shap.DeepExplainer(wrapped, bg)
        shap_values = ex.shap_values(ts)
    else:
        raise ValueError(f"Unknown explainer: {explainer}")

    # shap returns list-of-array for multi-output models; we have 1 output.
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    arr = np.asarray(shap_values)
    # GradientExplainer pads scalar outputs → (N, L, F, 1); strip trailing singletons.
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def aggregate_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Mean absolute SHAP per feature (averaged over time + samples)."""
    arr = np.abs(np.asarray(shap_values))
    while arr.ndim > 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    axes_reduce = tuple(range(arr.ndim - 1))
    abs_mean = np.mean(arr, axis=axes_reduce).reshape(-1)
    flat = abs_mean.astype(np.float64)
    names = feature_names[: flat.size]
    vals = flat[: len(names)]
    return {name: float(v) for name, v in zip(names, vals.tolist(), strict=False)}


def plot_global_importance(
    importances: dict[str, float],
    save_path: str | Path,
    top_n: int = 20,
) -> Path:
    import matplotlib.pyplot as plt

    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names = [k for k, _ in items][::-1]
    vals = [v for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(6, 0.3 * len(items) + 1), dpi=120)
    ax.barh(names, vals, color="#1f77b4")
    ax.set_xlabel("mean |SHAP|")
    ax.set_title("Global SHAP feature importance")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = Path(save_path)
    fig.savefig(out)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return out.resolve()


def plot_time_feature_heatmap(
    shap_values_one: np.ndarray,
    feature_names: list[str],
    save_path: str | Path,
) -> Path:
    """Heatmap of SHAP attributions for one window: shape (L, F)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.25 * len(feature_names) + 1), dpi=120)
    im = ax.imshow(shap_values_one.T, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("timestep within window")
    ax.set_title("SHAP attribution heatmap (one prediction)")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = Path(save_path)
    fig.savefig(out)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return out.resolve()
