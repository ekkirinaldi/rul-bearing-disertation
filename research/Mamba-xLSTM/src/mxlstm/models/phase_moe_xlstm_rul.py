"""PhaseMoE-xLSTM adapter: scalar ``forward`` + dense ``PhysicsInformedLoss`` when ``rul_window`` is provided."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mxlstm.models.phase_moe_xlstm_core import PhaseMoEConfig, PhaseMoExLSTM, PhysicsInformedLoss


class PhaseMoExLSTMRUL(nn.Module):
    """``(B, L, F) -> (B,)`` last-step RUL; dense supervision via ``rul_window`` in ``compute_loss``."""

    def __init__(
        self,
        *,
        n_features: int,
        d_model: int = 128,
        n_phases: int = 3,
        dropout: float = 0.1,
        hi_index: int = 0,
        kurt_index: int = 2,
        healthy_window: int = 8,
        lambda_mono: float = 0.1,
        lambda_paris: float = 0.05,
        lambda_phase: float = 0.1,
        healthy_threshold: float = 0.7,
        prefailure_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        cfg = PhaseMoEConfig(
            n_features=int(n_features),
            d_model=int(d_model),
            n_phases=int(n_phases),
            dropout=float(dropout),
            hi_index=int(hi_index),
            kurt_index=int(kurt_index),
            healthy_window=int(healthy_window),
        )
        self.core = PhaseMoExLSTM(cfg)
        self._loss = PhysicsInformedLoss(
            lambda_mono=float(lambda_mono),
            lambda_paris=float(lambda_paris),
            lambda_phase=float(lambda_phase),
            healthy_threshold=float(healthy_threshold),
            prefailure_threshold=float(prefailure_threshold),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(x)["rul"]

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        rul_window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.core(x)
        if rul_window is None:
            return F.mse_loss(out["rul"], y)
        return self._loss(out, rul_window)["total"]
