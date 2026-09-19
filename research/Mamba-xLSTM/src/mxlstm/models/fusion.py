"""Fusion modules combining xLSTM (branch A) and Mamba (branch B) outputs.

Both modules accept ``(B, L, d_model)`` tensors and return the same shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """``g = σ(W [h_A; h_B]); out = g·h_A + (1-g)·h_B``.

    The gate is computed elementwise on ``d_model`` and applied per
    timestep. Returns the fused tensor and (optionally) the gate values
    so we can plot how the model balances the two branches over time.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(
        self, h_a: torch.Tensor, h_b: torch.Tensor, *, return_gate: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        g = torch.sigmoid(self.gate(torch.cat([h_a, h_b], dim=-1)))
        out = g * h_a + (1.0 - g) * h_b
        if return_gate:
            return out, g
        return out


class CrossAttentionFusion(nn.Module):
    """Q from branch A, K/V from branch B, output projected back to d_model."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(h_a)
        kv = self.norm_kv(h_b)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        return h_a + self.proj(attn_out)


class ConcatFusion(nn.Module):
    """Baseline ablation: concatenate then linearly project back to d_model."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([h_a, h_b], dim=-1))


def build_fusion(name: str, d_model: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ("gated", "gate", "gatedfusion"):
        return GatedFusion(d_model)
    if name in ("cross", "cross_attention", "crossattention"):
        return CrossAttentionFusion(d_model, **kwargs)
    if name in ("concat", "concatfusion"):
        return ConcatFusion(d_model)
    raise ValueError(f"Unknown fusion: {name}")
