"""Integrated Gradients along the time axis (Captum-backed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
    n_steps: int = 64,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Return IG attributions matching ``inputs`` shape ``(N, L, F)``."""
    try:
        from captum.attr import IntegratedGradients
    except ImportError as e:
        raise ImportError("captum is required: pip install captum") from e

    inputs = inputs.to(device)
    if baseline is None:
        baseline = torch.zeros_like(inputs)
    else:
        baseline = baseline.to(device)

    def _wrap(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        return out.unsqueeze(-1)

    ig = IntegratedGradients(_wrap)
    attr = ig.attribute(inputs, baselines=baseline, n_steps=n_steps)
    return attr.detach().cpu().numpy()


def plot_time_attribution(
    attr_one: np.ndarray,
    save_path: str | Path,
    feature_names: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.25 * (attr_one.shape[1]) + 1), dpi=120)
    im = ax.imshow(attr_one.T, aspect="auto", cmap="RdBu_r", origin="lower")
    if feature_names:
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_xlabel("timestep within window")
    ax.set_title("Integrated Gradients attribution")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    out = Path(save_path)
    fig.savefig(out)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return out.resolve()
