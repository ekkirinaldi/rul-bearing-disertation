"""xLSTM block stack with NX-AI library + pure-PyTorch fallback.

We try to use NX-AI's official ``xlstm`` package (vanilla backend on Mac,
CUDA-fused backend on NVIDIA hardware). If the package is unavailable we
fall back to a self-contained implementation of mLSTM and sLSTM that
keeps the *spirit* of Beck et al. (2024):

  * Exponential input gate (sLSTM, mLSTM)
  * Sigmoid forget gate
  * Matrix memory + outer product update (mLSTM)
  * Scalar memory with normalizer (sLSTM)

The fallback is intentionally not as fast as the CUDA kernels — its
purpose is to keep the codebase runnable on Mac without exotic CUDA
toolchains. It matches the math in §6.5 of PROJECT_PLAN.md.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Pure-PyTorch fallbacks
# ---------------------------------------------------------------------------


class _ExpInputGate(nn.Module):
    """Stabilized exponential gate: ``i_t = exp(W x + b - m_t)`` with
    running-max stabilization for numerical safety (Beck et al. §3)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, prev_max: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.proj(x)
        max_state = z if prev_max is None else torch.maximum(z, prev_max)
        i = torch.exp(z - max_state)
        return i, max_state


class VanillaSLSTMCell(nn.Module):
    """Scalar-memory xLSTM cell (sLSTM) with exponential input gate."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.i_gate = _ExpInputGate(d_model, d_model)
        self.f_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.z_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.f_proj.bias, 1.0)  # forget gate bias init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        c = x.new_zeros(B, D)
        n = x.new_zeros(B, D)
        h = x.new_zeros(B, D)
        m: torch.Tensor | None = None
        outs = []
        for t in range(L):
            xt = x[:, t]
            i, m = self.i_gate(xt, m)
            f = torch.sigmoid(self.f_proj(xt))
            o = torch.sigmoid(self.o_proj(xt))
            z = torch.tanh(self.z_proj(xt))
            c = f * c + i * z
            n = f * n + i
            h = o * (c / (n.abs().clamp(min=1.0)))
            outs.append(h)
        return torch.stack(outs, dim=1)


class VanillaMLSTMBlock(nn.Module):
    """Matrix-memory xLSTM block (mLSTM).

    Uses query/key/value projections with an exponential input gate and
    sigmoid forget gate. Memory is a (d_head x d_head) matrix per head.
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.i_gate = _ExpInputGate(d_model, n_heads)
        self.f_proj = nn.Linear(d_model, n_heads)
        self.o_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.f_proj.bias, 1.0)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_head
        q = self.q(x).view(B, L, H, Dh)
        k = self.k(x).view(B, L, H, Dh) / math.sqrt(Dh)
        v = self.v(x).view(B, L, H, Dh)
        f_logits = self.f_proj(x)  # (B, L, H)
        o = torch.sigmoid(self.o_proj(x)).view(B, L, H, Dh)

        C = x.new_zeros(B, H, Dh, Dh)
        n = x.new_zeros(B, H, Dh)
        m: torch.Tensor | None = None
        outs = []
        for t in range(L):
            qt = q[:, t]    # (B, H, Dh)
            kt = k[:, t]
            vt = v[:, t]
            ot = o[:, t]
            ft = torch.sigmoid(f_logits[:, t])  # (B, H)
            it, m = self.i_gate(x[:, t], m)     # (B, H)

            # Outer product update of matrix memory
            outer = vt.unsqueeze(-1) * kt.unsqueeze(-2)  # (B, H, Dh, Dh)
            C = ft[..., None, None] * C + it[..., None, None] * outer
            n = ft.unsqueeze(-1) * n + it.unsqueeze(-1) * kt

            # Read out
            num = torch.einsum("bhij,bhj->bhi", C, qt)  # (B, H, Dh)
            denom = torch.einsum("bhi,bhi->bh", n, qt).abs().clamp(min=1.0)
            h_t = ot * (num / denom.unsqueeze(-1))
            outs.append(h_t.reshape(B, D))
        return self.out_proj(torch.stack(outs, dim=1))


# ---------------------------------------------------------------------------
# Block stack (NX-AI when available, fallback otherwise)
# ---------------------------------------------------------------------------


BlockKind = Literal["mlstm", "slstm"]


class _FallbackXLSTMStack(nn.Module):
    def __init__(self, d_model: int, kinds: list[BlockKind], n_heads: int, dropout: float) -> None:
        super().__init__()
        layers = []
        for kind in kinds:
            if kind == "mlstm":
                layers.append(VanillaMLSTMBlock(d_model, n_heads=n_heads))
            elif kind == "slstm":
                layers.append(VanillaSLSTMCell(d_model))
            else:
                raise ValueError(f"Unknown xLSTM block kind: {kind}")
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in layers])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.layers, self.norms, strict=False):
            out = norm(out + self.dropout(layer(out)))
        return out


def _try_build_nx_xlstm(
    d_model: int,
    context_length: int,
    num_blocks: int,
    slstm_at: list[int],
    n_heads: int,
    dropout: float,
):
    """Attempt to construct NX-AI's xLSTMBlockStack with the vanilla backend."""
    try:
        from xlstm import (
            FeedForwardConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
        )
    except ImportError:
        return None

    try:
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=n_heads,
                    backend="vanilla",
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=n_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=d_model,
            slstm_at=list(slstm_at),
            dropout=dropout,
        )
        return xLSTMBlockStack(cfg)
    except Exception:
        # Some xlstm versions require CUDA even with backend='vanilla' for sLSTM.
        return None


class XLSTMStack(nn.Module):
    """Drop-in xLSTM block stack with automatic backend selection.

    Args:
        d_model: Hidden width.
        context_length: Maximum sequence length (some xlstm versions need it).
        num_blocks: Total number of stacked blocks.
        slstm_positions: 0-indexed positions where sLSTM is used; remainders are mLSTM.
        n_heads: Heads per mLSTM block.
        dropout: Block-level dropout.
        force_fallback: Skip the NX-AI library and always use the pure-PyTorch impl.
    """

    def __init__(
        self,
        d_model: int,
        context_length: int = 64,
        num_blocks: int = 3,
        slstm_positions: list[int] | None = None,
        n_heads: int = 4,
        dropout: float = 0.1,
        force_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.slstm_positions = list(slstm_positions or [num_blocks // 2])

        if not force_fallback:
            stack = _try_build_nx_xlstm(
                d_model=d_model,
                context_length=context_length,
                num_blocks=num_blocks,
                slstm_at=self.slstm_positions,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:
            stack = None

        if stack is not None:
            self.backend = "nxai"
            self.stack = stack
        else:
            self.backend = "fallback"
            kinds: list[BlockKind] = [
                "slstm" if i in self.slstm_positions else "mlstm" for i in range(num_blocks)
            ]
            self.stack = _FallbackXLSTMStack(d_model, kinds=kinds, n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stack(x)
