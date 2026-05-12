"""Regression heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

PoolKind = Literal["last", "flatten", "mean"]


class SigmoidRegressionHead(nn.Module):
    """LayerNorm -> MLP -> Sigmoid regression head.

    Pooling modes (controlled by ``pool``):

      * ``"last"``   - take only the last timestep ``seq[:, -1]``. Default.
        Used by the proposed Mamba-xLSTM-Net (matches the "predict the RUL
        at the end of the window" convention).
      * ``"flatten"`` - flatten ``(B, L, D) -> (B, L*D)``. Used by the
        paper-faithful baseline (Liu et al. §2.4: "the decoder results
        are flattened and fed into a linear layer to produce the
        predicted RUL"). Requires ``context_length`` so the input
        Linear has a fixed size.
      * ``"mean"``   - mean-pool over the time axis.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int = 64,
        dropout: float = 0.1,
        *,
        pool: PoolKind = "last",
        context_length: int | None = None,
    ) -> None:
        super().__init__()
        self.pool = pool
        if pool == "flatten":
            if context_length is None or context_length <= 0:
                raise ValueError("pool='flatten' requires a positive context_length")
            in_features = d_model * context_length
            self.norm = nn.LayerNorm(d_model)
        else:
            in_features = d_model
            self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def _pool(self, seq: torch.Tensor) -> torch.Tensor:
        if seq.dim() != 3:
            raise ValueError(f"Expected (B, L, D); got {seq.shape}")
        if self.pool == "last":
            return self.norm(seq[:, -1])
        if self.pool == "mean":
            return self.norm(seq.mean(dim=1))
        # flatten: norm per-timestep then concat across time
        return self.norm(seq).reshape(seq.size(0), -1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self._pool(seq)
        x = self.drop(self.act(self.fc1(x)))
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(-1)
