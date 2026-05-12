"""Proposed architecture: Mamba-xLSTM-Net per §2.2 of PROJECT_PLAN.md.

  Input HI window (B, L, F)
    -> Linear projection (F -> d_model)
    -> Branch A: xLSTM stack (mLSTM, sLSTM, mLSTM)
    -> Branch B: Bidirectional Mamba-3 stack (2 blocks)
    -> Fusion (GatedFusion default; CrossAttentionFusion / ConcatFusion swappable)
    -> SigmoidRegressionHead -> scalar RUL in [0, 1]

The forward signature is ``(B, L, F) -> (B,)``. With ``return_hidden=True``
it additionally returns the post-fusion hidden states (used by the SAE),
the per-timestep gate values (when GatedFusion is used), and the two
branches' outputs (useful for SHAP / per-branch ablation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

from mxlstm.models.fusion import GatedFusion, build_fusion
from mxlstm.models.heads import SigmoidRegressionHead
from mxlstm.models.lstm_blocks import PlainLSTMStack
from mxlstm.models.mamba_blocks import MambaBackendChoice, MambaStack
from mxlstm.models.xlstm_blocks import XLSTMStack

FusionName = Literal["gated", "cross_attention", "concat"]


@dataclass
class MambaXLSTMConfig:
    n_features: int
    d_model: int = 128
    context_length: int = 64
    # xLSTM branch
    xlstm_blocks: int = 3
    slstm_positions: list[int] = field(default_factory=lambda: [1])
    xlstm_heads: int = 4
    xlstm_force_fallback: bool = False
    # If True, replace the xLSTM stack with a plain ``nn.LSTM`` stack of
    # equivalent depth (used by ablation A5 to isolate the contribution of
    # exponential gating + matrix memory).
    use_vanilla_lstm: bool = False
    # Mamba-3 branch (state-spaces/mamba ``Mamba3`` block).
    mamba_blocks: int = 2
    mamba_d_state: int = 128
    mamba_d_conv: int = 4              # kept for backward yaml compat; unused in Mamba-3
    mamba_expand: int = 2
    mamba_headdim: int = 64
    mamba_n_groups: int = 1            # forwarded as ``ngroups`` to mamba_ssm.Mamba3
    mamba_rope_fraction: float = 0.5   # 0.5 = partial RoPE (paper/official default); 1.0 = full RoPE
    mamba_is_mimo: bool = False        # SISO by default; True enables MIMO on mamba_ssm backend
    mamba_mimo_rank: int = 4
    mamba_chunk_size: int = 64         # SSD chunk size (mamba_ssm backend; 64/mimo_rank for MIMO)
    mamba_is_outproj_norm: bool = False
    mamba_bidirectional: bool = True
    mamba_backend: MambaBackendChoice = "auto"
    # Fusion + head
    fusion: FusionName = "gated"
    head_hidden: int = 64
    dropout: float = 0.1


class MambaXLSTMNet(nn.Module):
    """Hybrid xLSTM + Bidirectional Mamba network with gated/cross-attn fusion."""

    def __init__(self, cfg: MambaXLSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)

        if cfg.use_vanilla_lstm:
            self.xlstm = PlainLSTMStack(
                d_model=cfg.d_model,
                num_blocks=cfg.xlstm_blocks,
                dropout=cfg.dropout,
            )
        else:
            self.xlstm = XLSTMStack(
                d_model=cfg.d_model,
                context_length=cfg.context_length,
                num_blocks=cfg.xlstm_blocks,
                slstm_positions=cfg.slstm_positions,
                n_heads=cfg.xlstm_heads,
                dropout=cfg.dropout,
                force_fallback=cfg.xlstm_force_fallback,
            )
        self.mamba = MambaStack(
            d_model=cfg.d_model,
            num_blocks=cfg.mamba_blocks,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
            headdim=cfg.mamba_headdim,
            n_groups=cfg.mamba_n_groups,
            rope_fraction=cfg.mamba_rope_fraction,
            is_mimo=cfg.mamba_is_mimo,
            mimo_rank=cfg.mamba_mimo_rank,
            chunk_size=cfg.mamba_chunk_size,
            is_outproj_norm=cfg.mamba_is_outproj_norm,
            bidirectional=cfg.mamba_bidirectional,
            dropout=cfg.dropout,
            backend=cfg.mamba_backend,
        )
        self.fusion = build_fusion(cfg.fusion, d_model=cfg.d_model, n_heads=cfg.xlstm_heads, dropout=cfg.dropout)
        self.head = SigmoidRegressionHead(cfg.d_model, hidden=cfg.head_hidden, dropout=cfg.dropout)

    # ------------------------------------------------------------------

    def _fuse(self, h_a: torch.Tensor, h_b: torch.Tensor, return_gate: bool):
        if isinstance(self.fusion, GatedFusion):
            return self.fusion(h_a, h_b, return_gate=return_gate)
        out = self.fusion(h_a, h_b)
        return (out, None) if return_gate else out

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        h = self.input_proj(x)
        h_a = self.xlstm(h)
        h_b = self.mamba(h)
        if return_hidden:
            fused, gate = self._fuse(h_a, h_b, return_gate=True)
            pred = self.head(fused)
            return pred, {"branch_a": h_a, "branch_b": h_b, "fused": fused, "gate": gate}
        fused = self._fuse(h_a, h_b, return_gate=False)
        return self.head(fused)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Convenience builders for ablation configs
# ---------------------------------------------------------------------------


def build_mamba_xlstm(cfg: dict) -> MambaXLSTMNet:
    """Construct a MambaXLSTMNet from a flat dict (e.g. parsed YAML)."""
    config_kwargs = {k: v for k, v in cfg.items() if k in MambaXLSTMConfig.__dataclass_fields__}
    return MambaXLSTMNet(MambaXLSTMConfig(**config_kwargs))
