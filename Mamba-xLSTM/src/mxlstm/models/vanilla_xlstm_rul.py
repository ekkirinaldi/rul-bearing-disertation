"""Plain xLSTM stack for RUL (no Transformer encoder/decoder).

Contrasts with ``XLSTMTransformer`` (Liu et al. 2026): this baseline is only
``Linear(F → d_model)`` → ``XLSTMStack`` → ``SigmoidRegressionHead`` on the
last timestep, matching the project's other windowed HI → scalar RUL models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mxlstm.models.heads import SigmoidRegressionHead
from mxlstm.models.xlstm_blocks import XLSTMStack


class VanillaXLSTMRUL(nn.Module):
    """``(B, L, F) -> (B,)`` RUL in ``[0, 1]`` via stacked xLSTM blocks only."""

    def __init__(
        self,
        n_features: int,
        context_length: int,
        *,
        d_model: int = 64,
        num_blocks: int = 3,
        slstm_positions: list[int] | None = None,
        n_heads: int = 4,
        head_hidden: int = 64,
        dropout: float = 0.1,
        xlstm_force_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.input_proj = nn.Linear(int(n_features), int(d_model))
        self.xlstm = XLSTMStack(
            d_model=int(d_model),
            context_length=int(context_length),
            num_blocks=int(num_blocks),
            slstm_positions=list(slstm_positions) if slstm_positions is not None else None,
            n_heads=int(n_heads),
            dropout=float(dropout),
            force_fallback=bool(xlstm_force_fallback),
        )
        self.head = SigmoidRegressionHead(
            int(d_model),
            hidden=int(head_hidden),
            dropout=float(dropout),
            pool="last",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, L, F); got {x.shape}")
        if x.size(1) != self.context_length:
            raise ValueError(f"Expected L={self.context_length}; got {x.size(1)}")
        h = self.input_proj(x)
        h = self.xlstm(h)
        return self.head(h)
