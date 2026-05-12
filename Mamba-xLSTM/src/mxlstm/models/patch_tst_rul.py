"""Patch time-series encoder for RUL (PatchTST-style).

Unfolds non-overlapping/overlapping temporal patches from ``(B, L, F)``,
projects each patch to ``d_model``, applies a Transformer encoder on patch
tokens, mean-pools, then a sigmoid RUL head. No variate-wise RevIN (kept
minimal for dissertation baselines).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mxlstm.models.heads import SigmoidRegressionHead


class _SinusoidalPatchPos(nn.Module):
    def __init__(self, d_model: int, max_patches: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_patches, d_model)
        pos = torch.arange(0, max_patches, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        else:
            pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, D)
        return x + self.pe[:, : x.size(1)]


def _pad_time_to_patch_grid(length: int, patch_len: int, stride: int) -> int:
    if length < patch_len:
        return patch_len
    rem = (length - patch_len) % stride
    if rem == 0:
        return length
    return length + (stride - rem)


class PatchTSTRUL(nn.Module):
    """``(B, L, F) -> (B,)`` RUL in ``[0, 1]`` via patch self-attention."""

    def __init__(
        self,
        n_features: int,
        context_length: int,
        *,
        d_model: int = 96,
        n_heads: int = 4,
        n_encoder_layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        ffn_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_features = int(n_features)
        self.context_length = int(context_length)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        patch_dim = self.patch_len * self.n_features
        self._pad_len = _pad_time_to_patch_grid(self.context_length, self.patch_len, self.stride)
        self._n_patches = max(1, (self._pad_len - self.patch_len) // self.stride + 1)

        self.patch_proj = nn.Linear(patch_dim, d_model)
        self.pos = _SinusoidalPatchPos(d_model, max_patches=self._n_patches + 8)
        ff = int(ffn_dim) if ffn_dim is not None else 4 * d_model
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=int(n_encoder_layers),
            enable_nested_tensor=False,
        )
        self.head = SigmoidRegressionHead(
            d_model,
            hidden=max(32, d_model // 2),
            dropout=dropout,
            pool="mean",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, L, F); got {x.shape}")
        b, l, f = x.shape
        if f != self.n_features:
            raise ValueError(f"Expected F={self.n_features}; got {f}")
        if l != self.context_length:
            raise ValueError(f"Expected L={self.context_length}; got {l}")
        pad_len = _pad_time_to_patch_grid(l, self.patch_len, self.stride)
        if pad_len > l:
            pad = x.new_zeros(b, pad_len - l, f)
            x = torch.cat([x, pad], dim=1)
        # (B, pad_len, F) -> unfold time: (B, n_patches, patch_len, F)
        x = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        n_p = x.size(1)
        x = x.reshape(b, n_p, self.patch_len * f)
        h = self.patch_proj(x)
        h = self.pos(h)
        h = self.encoder(h)
        return self.head(h)
