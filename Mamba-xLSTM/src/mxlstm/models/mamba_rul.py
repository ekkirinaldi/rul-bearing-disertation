"""Selective SSM (Mamba-3) stack only for RUL — no xLSTM branch.

Uses ``MambaStack`` from ``mamba_blocks`` (same kernels as Mamba-xLSTM-Net)
with a last-timestep sigmoid head. Intended as a strong nonlinear sequence
baseline on HI windows.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mxlstm.models.heads import SigmoidRegressionHead
from mxlstm.models.mamba_blocks import MambaBackendChoice, MambaStack


class MambaRUL(nn.Module):
    """``(B, L, F) -> (B,)`` RUL in ``[0, 1]`` via bidirectional Mamba stack."""

    def __init__(
        self,
        n_features: int,
        context_length: int,
        *,
        d_model: int = 96,
        mamba_blocks: int = 3,
        mamba_d_state: int = 64,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_headdim: int = 32,
        mamba_n_groups: int = 1,
        mamba_rope_fraction: float = 0.5,
        mamba_is_mimo: bool = False,
        mamba_mimo_rank: int = 4,
        mamba_chunk_size: int = 64,
        mamba_is_outproj_norm: bool = False,
        mamba_bidirectional: bool = True,
        mamba_backend: MambaBackendChoice = "auto",
        head_hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.input_proj = nn.Linear(int(n_features), int(d_model))
        self.backbone = MambaStack(
            int(d_model),
            num_blocks=int(mamba_blocks),
            d_state=int(mamba_d_state),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
            headdim=int(mamba_headdim),
            n_groups=int(mamba_n_groups),
            rope_fraction=float(mamba_rope_fraction),
            is_mimo=bool(mamba_is_mimo),
            mimo_rank=int(mamba_mimo_rank),
            chunk_size=int(mamba_chunk_size),
            is_outproj_norm=bool(mamba_is_outproj_norm),
            bidirectional=bool(mamba_bidirectional),
            dropout=float(dropout),
            backend=mamba_backend,
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
        h = self.backbone(h)
        return self.head(h)
