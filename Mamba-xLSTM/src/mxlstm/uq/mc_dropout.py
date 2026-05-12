"""Monte-Carlo Dropout uncertainty quantification per §5 Phase 6."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _enable_dropout_inference(model: nn.Module) -> None:
    """Toggle every Dropout/Dropout1d/etc. into train() while keeping the rest in eval()."""
    for m in model.modules():
        if isinstance(m, nn.modules.dropout._DropoutNd):
            m.train()


def predict_with_uncertainty(
    model: nn.Module,
    x: torch.Tensor,
    *,
    n_samples: int = 100,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run N stochastic forward passes; return (mean, std, samples) as numpy arrays.

    Output shapes::
        mean    : (B,)
        std     : (B,)
        samples : (n_samples, B)
    """
    model = model.to(device).eval()
    _enable_dropout_inference(model)
    x = x.to(device)
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)  # (N, B)
    samples = preds.cpu().numpy()
    return samples.mean(0), samples.std(0), samples


def reliability_diagram(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    *,
    n_bins: int = 10,
    save_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """Calibration of predicted CIs vs empirical coverage.

    For each predicted std bin, compute the empirical fraction of
    samples where ``|y_pred_mean - y_true| <= z * y_pred_std`` for
    z = 1.96 (95% CI).
    """
    z = 1.96
    in_ci = (np.abs(y_pred_mean - y_true) <= z * y_pred_std).astype(np.float32)
    bins = np.linspace(0, np.percentile(y_pred_std, 99), n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred_std, bins) - 1, 0, n_bins - 1)
    coverage = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int32)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            coverage[b] = float(in_ci[mask].mean())

    if save_path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        centers = 0.5 * (bins[:-1] + bins[1:])
        ax.bar(centers, coverage, width=(bins[1] - bins[0]) * 0.9, alpha=0.7, label="empirical 95% coverage")
        ax.axhline(0.95, color="red", lw=1, ls="--", label="target 95%")
        ax.set_xlabel("predicted std")
        ax.set_ylabel("coverage")
        ax.set_title("MC Dropout reliability diagram")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        import matplotlib.pyplot as _plt
        _plt.close(fig)

    return {"bin_edges": bins, "coverage": coverage, "counts": counts}
