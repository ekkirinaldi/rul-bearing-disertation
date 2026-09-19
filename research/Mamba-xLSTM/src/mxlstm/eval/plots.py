"""Prediction-curve and residual plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_prediction_curve(
    bearing_id: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    save_path: str | Path | None = None,
    title_suffix: str = "",
) -> plt.Figure:
    """Plot true vs predicted RUL over time for one bearing."""
    t = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=120)
    ax.plot(t, y_true, label="true RUL", color="#1f77b4", lw=1.5)
    ax.plot(t, y_pred, label="pred RUL", color="#d62728", lw=1.0, alpha=0.85)
    ax.set_xlabel("acquisition index")
    ax.set_ylabel("RUL")
    ax.set_title(f"{bearing_id} {title_suffix}".strip())
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    res = y_pred - y_true
    ax.scatter(y_true, res, s=4, alpha=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("true RUL")
    ax.set_ylabel("residual (pred - true)")
    ax.set_title("residuals vs true RUL")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
    return fig
