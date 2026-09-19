"""Losses: MSE + monotonicity penalty per §7.2."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def monotonicity_penalty(pred: torch.Tensor, t_indices: torch.Tensor) -> torch.Tensor:
    """L_mono per §7.2: penalize *increasing* RUL across consecutive timesteps.

    Operates within a batch by sorting predictions by ``t_indices`` (per
    bearing if grouped) and summing ReLU of positive deltas. Since the
    DataLoader is shuffled by default, the penalty is computed within
    each mini-batch on the natural order — a small but useful regularizer.
    """
    if pred.numel() < 2:
        return pred.new_zeros(())
    order = torch.argsort(t_indices)
    p = pred[order]
    deltas = p[1:] - p[:-1]
    return torch.clamp(deltas, min=0.0).mean()
