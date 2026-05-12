"""SparseGate-TCN adapter for the shared RUL training pipeline (scalar median + pinball loss).

This module is intentionally separate from ``sparse_gate_tcn_core.py`` so baseline
models stay decoupled from the pinball / gate-specific training path.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from mxlstm.models.sparse_gate_tcn_core import SparseGateTCN, SparseTCNConfig, SparseTCNLoss


class SparseGateTCNRUL(nn.Module):
    """``(B, L, F) -> (B,)`` median RUL; training loss via ``SparseTCNLoss`` when enabled."""

    def __init__(
        self,
        *,
        n_features: int,
        tcn_channels: Sequence[int] = (64, 64, 128, 128),
        tcn_kernel: int = 3,
        gate_hidden: int = 32,
        gate_context: int = 5,
        attn_d_model: int = 32,
        attn_heads: int = 4,
        head_hidden: int = 64,
        dropout: float = 0.1,
        lambda_sparse: float = 1e-3,
        lambda_entropy: float = 1e-3,
    ) -> None:
        super().__init__()
        nf = int(n_features)
        default_names = SparseTCNConfig().feature_names
        if nf == len(default_names):
            feat_names: tuple[str, ...] = default_names
        else:
            feat_names = tuple(f"hi_{i}" for i in range(nf))
        cfg = SparseTCNConfig(
            n_features=nf,
            feature_names=feat_names,
            tcn_channels=tuple(int(c) for c in tcn_channels),
            tcn_kernel=int(tcn_kernel),
            gate_hidden=int(gate_hidden),
            gate_context=int(gate_context),
            attn_d_model=int(attn_d_model),
            attn_heads=int(attn_heads),
            head_hidden=int(head_hidden),
            dropout=float(dropout),
        )
        self.backbone = SparseGateTCN(cfg)
        self._loss = SparseTCNLoss(
            lambda_sparse=float(lambda_sparse),
            lambda_entropy=float(lambda_entropy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x, return_attn=False)["rul"]

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x, return_attn=False)
        return self._loss(out, y)["total"]
