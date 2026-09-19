"""Mamba-3 block wrapper with two backends and a Bidirectional variant.

This module implements the **Mamba-3** block, following Lahoti et al.,
"Mamba-3: Improved Sequence Modeling using State Space Principles"
(arXiv:2603.15569, 2026), using the parameterisation released in
``state-spaces/mamba`` (``mamba_ssm.modules.mamba3.Mamba3``).

Mamba-3 vs Mamba-2 (paper §3.4 + official ``mamba3.py``)
--------------------------------------------------------
The selective SSM keeps the SSD per-head, scalar-A layout from Mamba-2 but
adds the following changes:

1. **Exponential-trapezoidal discretisation** (paper Eq. 5/6).

       alpha_t = exp(dt_t * A_t)                        # decay
       gamma_t = lambda_t * dt_t                        # current input weight
       beta_t  = (1 - lambda_t) * dt_t * alpha_t        # previous input weight
       h_t     = alpha_t * h_{t-1}
                 + gamma_t * B_t      * x_t
                 + beta_t  * B_{t-1}  * x_{t-1}

   ``lambda_t = sigmoid(trap_t)`` is a per-(token, head) interpolation
   that the model learns; ``lambda_t == 1`` recovers the Mamba-2 update.

2. **Complex-valued state via the RoPE trick** (paper Eq. 11). State
   channels are processed in pairs; each pair is rotated by an angle
   ``psi_t = cumsum(dt_t * theta_t)``. By pre-rotating both ``B`` and
   ``C`` with ``R(-psi_t)`` we keep the scan associative *and* let the
   SSM model oscillatory dynamics. Following the official block we use
   **partial RoPE**: only ``rope_fraction`` of the state channels (the
   first ``floor(d_state * rope_fraction)`` rounded down to even) are
   rotated; the remainder pass through unchanged. The rotation angles
   are produced by a single per-token projection (``num_rope_angles``
   entries) and broadcast across heads.

3. **Data-dependent ``A``** (paper Remark 1, default in
   ``mamba_ssm.Mamba3``). ``A_t = -softplus(dd_A_t)`` clamped from
   above by ``-A_floor``, where ``dd_A_t`` is part of the per-token
   ``in_proj`` output.

4. **BC / QK normalisation** (paper §3.4). ``RMSNorm`` is applied on the
   state dimension of ``B`` and ``C`` after the projection, before the
   bias is added.

5. **Per-head learnable B/C bias** (paper §3.4 + official source). Shape
   ``(nheads, d_state)``, **initialised to ones** so ``B`` and ``C``
   start with a non-trivial DC component (acts like a built-in
   convolution and removes the need for a short causal conv).

6. **No short causal conv** (paper §3.4). The combination of (1)+(4)+(5)
   makes the Mamba-2 ``Conv1d`` redundant.

7. **Optional post-gate ``RMSNorm``** (``is_outproj_norm``, default
   ``False`` per the official block). When False the output is just
   ``y * SiLU(z)`` (SwiGLU-style); when True, the Mamba-2 gated RMSNorm
   ``RMSNorm(y) * SiLU(z)`` is applied (recommended for hybrid models).

The optional MIMO formulation from §3.3 is exposed for interface
compatibility with ``mamba_ssm.Mamba3`` but the **vanilla scan in this
module is SISO only** - dissertation models use small ``d_model`` where
SISO is already plenty expressive, and a faithful MIMO scan would
require the Triton/Tilelang kernels shipped in ``state-spaces/mamba``.
``is_mimo``, ``mimo_rank``, ``chunk_size`` are accepted and forwarded
to the CUDA backend, but ignored by the vanilla path.

Backends
--------

  * ``mamba_ssm`` - NVIDIA's CUDA-only ``Mamba3`` block (fastest;
                    requires building ``mamba-ssm`` from source per the
                    official README:
                    ``MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir
                    --force-reinstall git+https://github.com/state-spaces/mamba.git
                    --no-build-isolation``).
  * ``vanilla``   - pure-PyTorch SISO reference implementing the seven
                    Mamba-3 details above. Sequential per-token scan;
                    correct on any device but slower than the Triton
                    kernels. Used for CPU/MPS runs, unit tests, and as
                    a deterministic reference.

The legacy ``mambapy`` backend is kept as a no-op alias that
transparently falls back to ``vanilla`` because ``mambapy`` does not
yet ship a Mamba-3 implementation (only Mamba-1 / Mamba-2).

All backends present the same forward signature
``(B, L, d_model) -> (B, L, d_model)``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mxlstm.compute import _detect_mamba_backend


# ---------------------------------------------------------------------------
# Vanilla Mamba-3 (SISO reference) - works anywhere
# ---------------------------------------------------------------------------


class _VanillaMamba3(nn.Module):
    """Single Mamba-3 SISO block (sequential reference scan).

    Follows the parameterisation of ``mamba_ssm.modules.mamba3.Mamba3``
    with ``is_mimo=False``.

    ``in_proj`` produces eight concatenated tensors per token:
    ``[z, x, B, C, dd_dt, dd_A, trap, angles]`` of widths
    ``[d_inner, d_inner, ngroups*d_state, ngroups*d_state, nheads,
    nheads, nheads, num_rope_angles]``.

    The recurrence below uses the cached ``prev_BX`` to apply the
    second-order trapezoidal term ``beta_t * B_{t-1} * x_{t-1}`` exactly
    as in paper Eq. (5).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
        A_floor: float = 1e-4,
        is_outproj_norm: bool = False,
        # MIMO kwargs accepted for interface parity with mamba_ssm.Mamba3
        # but ignored by the SISO vanilla scan.
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        **_unused: object,
    ) -> None:
        super().__init__()
        del is_mimo, mimo_rank, chunk_size, _unused

        if rope_fraction not in (0.5, 1.0):
            raise ValueError(
                f"rope_fraction must be 0.5 or 1.0 (got {rope_fraction})"
            )

        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.num_bc_heads = n_groups
        self.A_floor = A_floor
        self.is_outproj_norm = is_outproj_norm

        if self.d_inner % self.headdim != 0:
            for cand in (self.headdim, 64, 32, 16, 8, 4, 2, 1):
                if self.d_inner % cand == 0:
                    self.headdim = cand
                    break
        self.nheads = self.d_inner // self.headdim
        if self.nheads % self.num_bc_heads != 0:
            self.num_bc_heads = 1

        # Partial RoPE (paper §3.2 + official source).
        self.rotary_dim_divisor = int(2 / rope_fraction)  # 2 for 1.0, 4 for 0.5
        split = int(d_state * rope_fraction)
        if split % 2 != 0:
            split -= 1
        if split <= 0:
            # Tiny d_state: still need at least one pair to rotate.
            split = 2 if d_state >= 2 else 0
        self.split_tensor_size = split
        self.num_rope_angles = max(split // 2, 1)

        # in_proj output:  [z, x, B, C, dd_dt, dd_A, trap, angles]
        d_in_proj = (
            2 * self.d_inner
            + 2 * self.d_state * self.num_bc_heads
            + 3 * self.nheads
            + self.num_rope_angles
        )
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        # dt bias: inverse-softplus init so softplus(dt_bias) ~ U(dt_min, dt_max)
        # before any data-dependent contribution.
        _dt = torch.exp(
            torch.empty(self.nheads).uniform_(math.log(dt_min), math.log(dt_max))
        ).clamp(min=dt_init_floor)
        inv_dt = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(inv_dt)

        # Per-head learnable B/C bias, init to ones (matches official source).
        self.B_bias = nn.Parameter(torch.ones(self.nheads, self.d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, self.d_state))

        # BCNorm (RMSNorm on the state dim of B, C).
        rms = nn.RMSNorm if hasattr(nn, "RMSNorm") else nn.LayerNorm
        self.B_norm = rms(self.d_state)
        self.C_norm = rms(self.d_state)

        # Per-head residual ("D" in the paper).
        self.D = nn.Parameter(torch.ones(self.nheads))

        # Optional post-gate RMSNormGated (Mamba-2 style); off by default.
        if self.is_outproj_norm:
            self.norm = rms(self.d_inner)
        self.act = nn.SiLU()

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    @staticmethod
    def _rotate_pairs(
        v: torch.Tensor, cos_psi: torch.Tensor, sin_psi: torch.Tensor
    ) -> torch.Tensor:
        """Rotate consecutive ``(2i, 2i+1)`` pairs of the last dim by ``psi``.

        ``v`` has shape ``(..., S)`` with ``S`` even.
        ``cos_psi``/``sin_psi`` broadcast against ``v[..., 0::2]``.
        Implements the 2-D rotation
        ``[ cos -sin ; sin cos ] @ [v0 ; v1]``; pass ``-sin_psi`` for the
        inverse rotation ``R(-psi)``.
        """
        v_pair = v.unflatten(-1, (-1, 2))           # (..., S/2, 2)
        v0 = v_pair[..., 0]
        v1 = v_pair[..., 1]
        r0 = v0 * cos_psi - v1 * sin_psi
        r1 = v0 * sin_psi + v1 * cos_psi
        out = torch.stack((r0, r1), dim=-1)         # (..., S/2, 2)
        return out.flatten(-2, -1)                  # (..., S)

    def _apply_partial_rotation(
        self, v: torch.Tensor, cos_psi: torch.Tensor, sin_psi: torch.Tensor
    ) -> torch.Tensor:
        """Rotate the first ``split_tensor_size`` channels of ``v``.

        The remainder of the state is left untouched (partial RoPE).
        """
        S = self.split_tensor_size
        if S <= 0:
            return v
        if S == v.shape[-1]:
            return self._rotate_pairs(v, cos_psi, sin_psi)
        v_rot = v[..., :S]
        v_pass = v[..., S:]
        return torch.cat([self._rotate_pairs(v_rot, cos_psi, sin_psi), v_pass], dim=-1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B_sz, L, _ = u.shape
        H = self.nheads
        P = self.headdim
        N = self.d_state
        G = self.num_bc_heads
        S_half = self.num_rope_angles

        proj = self.in_proj(u)
        sizes = [
            self.d_inner,           # z
            self.d_inner,           # x
            G * N,                  # B
            G * N,                  # C
            H,                      # dd_dt  (for dt = softplus(dd_dt + dt_bias))
            H,                      # dd_A   (for A_t = -softplus(dd_A) clamp)
            H,                      # trap   (for lambda = sigmoid(trap))
            S_half,                 # angles (shared across heads)
        ]
        z, x, B_param, C_param, dd_dt, dd_A, trap, angles = torch.split(
            proj, sizes, dim=-1
        )

        # Reshape per-head / per-group tensors.
        x = x.view(B_sz, L, H, P)
        B_param = B_param.view(B_sz, L, G, N)
        C_param = C_param.view(B_sz, L, G, N)

        # 1) BC normalisation (paper §3.4: RMSNorm follows the B/C projection).
        B_param = self.B_norm(B_param)
        C_param = self.C_norm(C_param)

        # 2) Broadcast B, C to per-head tensors (multi-value attention layout).
        heads_per_group = H // G
        if heads_per_group > 1:
            B_param = B_param.repeat_interleave(heads_per_group, dim=2)
            C_param = C_param.repeat_interleave(heads_per_group, dim=2)
        else:
            B_param = B_param.expand(-1, -1, H, -1)
            C_param = C_param.expand(-1, -1, H, -1)

        # 3) Per-head BC bias (init to ones; broadcasts over (B, L)).
        B_param = B_param + self.B_bias.unsqueeze(0).unsqueeze(0)
        C_param = C_param + self.C_bias.unsqueeze(0).unsqueeze(0)

        # 4) dt, lambda (Mamba-3 trapezoidal interpolation), data-dependent A.
        dt = F.softplus(dd_dt + self.dt_bias)                  # (B, L, H), > 0
        lam = torch.sigmoid(trap)                              # (B, L, H), in (0,1)
        A_t = -F.softplus(dd_A.float())                        # (B, L, H), < 0
        A_t = torch.clamp(A_t, max=-self.A_floor).to(dt.dtype)

        # 5) Cumulative rotation angle psi_t = sum_{i<=t} dt_i * angles_i,
        #    with angles shared across heads (broadcast).
        # angles: (B, L, S_half) -> add head dim for broadcast.
        phi = dt.unsqueeze(-1) * angles.unsqueeze(-2)          # (B, L, H, S_half)
        psi = torch.cumsum(phi, dim=1)                         # (B, L, H, S_half)
        cos_psi = torch.cos(psi)
        sin_psi = torch.sin(psi)

        # Pre-rotate B, C by R(-psi_t) on the first split_tensor_size channels.
        B_rot = self._apply_partial_rotation(B_param, cos_psi, -sin_psi)
        C_rot = self._apply_partial_rotation(C_param, cos_psi, -sin_psi)

        # 6) Discretisation (paper Eq. 5).
        alpha = torch.exp(dt * A_t)                            # (B, L, H)
        gamma = lam * dt
        beta = (1.0 - lam) * dt * alpha

        # Sequential exp-trapezoidal recurrence; state shape (B, H, P, N).
        h = u.new_zeros(B_sz, H, P, N)
        prev_BX = u.new_zeros(B_sz, H, P, N)                   # B_rot_{t-1} * x_{t-1}
        ys: list[torch.Tensor] = []
        for t in range(L):
            cur_BX = B_rot[:, t, :, None, :] * x[:, t, :, :, None]   # (B, H, P, N)
            h = (
                alpha[:, t, :, None, None] * h
                + gamma[:, t, :, None, None] * cur_BX
                + beta[:, t, :, None, None] * prev_BX
            )
            yt = torch.einsum("bhpn,bhn->bhp", h, C_rot[:, t])
            ys.append(yt)
            prev_BX = cur_BX
        y = torch.stack(ys, dim=1)                             # (B, L, H, P)

        # Per-head residual through D, then merge heads back to d_inner.
        y = y + x * self.D.view(1, 1, -1, 1)
        y = y.reshape(B_sz, L, self.d_inner)

        # 7) Optional post-gate RMSNorm (off by default per official source).
        if self.is_outproj_norm:
            y = self.norm(y) * self.act(z)
        else:
            y = y * self.act(z)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Library-backed Mamba-3 block
# ---------------------------------------------------------------------------


class _MambaSSMBlock(nn.Module):
    """``mamba_ssm.Mamba3`` wrapper (CUDA + Triton/CuTe/Tilelang)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        expand: int = 2,
        headdim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
    ) -> None:
        super().__init__()
        from mamba_ssm import Mamba3

        # Note: official kwarg is ``ngroups`` (not ``n_groups``).
        self.block = Mamba3(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            headdim=headdim,
            ngroups=n_groups,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# Backward-compatible aliases. External code that imported the older block
# names keeps working transparently after the Mamba-2 -> Mamba-3 swap.
_VanillaMamba2 = _VanillaMamba3
_VanillaMamba = _VanillaMamba3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


MambaBackendChoice = Literal["auto", "mamba_ssm", "mambapy", "vanilla"]


def _build_mamba(
    d_model: int,
    d_state: int,
    d_conv: int,                                          # noqa: ARG001 (kept for API parity)
    expand: int,
    backend: MambaBackendChoice,
    *,
    headdim: int = 64,
    n_groups: int = 1,
    rope_fraction: float = 0.5,
    is_mimo: bool = False,
    mimo_rank: int = 4,
    chunk_size: int = 64,
    is_outproj_norm: bool = False,
) -> nn.Module:
    """Build a single Mamba-3 block.

    ``d_conv`` is accepted for backward compatibility with older Mamba-2
    yaml configs but is unused (Mamba-3 has no short causal conv).
    """
    common_kwargs = dict(
        headdim=headdim,
        n_groups=n_groups,
        rope_fraction=rope_fraction,
        is_mimo=is_mimo,
        mimo_rank=mimo_rank,
        chunk_size=chunk_size,
        is_outproj_norm=is_outproj_norm,
    )

    if backend == "auto":
        detected = _detect_mamba_backend()
        if detected == "mamba_ssm":
            try:
                return _MambaSSMBlock(d_model, d_state, expand, **common_kwargs)
            except (ImportError, AttributeError):
                # mamba_ssm older than 2.3 lacks Mamba3; fall back gracefully.
                return _VanillaMamba3(d_model, d_state, expand, **common_kwargs)
        # mambapy does not ship Mamba-3; route everything else to vanilla.
        return _VanillaMamba3(d_model, d_state, expand, **common_kwargs)
    if backend == "mamba_ssm":
        return _MambaSSMBlock(d_model, d_state, expand, **common_kwargs)
    if backend == "mambapy":
        # No Mamba-3 in mambapy yet; transparent downgrade to vanilla.
        return _VanillaMamba3(d_model, d_state, expand, **common_kwargs)
    if backend == "vanilla":
        return _VanillaMamba3(d_model, d_state, expand, **common_kwargs)
    raise ValueError(f"Unknown Mamba backend: {backend}")


class BidirectionalMamba(nn.Module):
    """Wraps two Mamba-3 blocks (forward + reverse), concatenates, projects back to d_model."""

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
        backend: MambaBackendChoice = "auto",
    ) -> None:
        super().__init__()
        common_kwargs = dict(
            headdim=headdim,
            n_groups=n_groups,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
        )
        self.fwd = _build_mamba(d_model, d_state, d_conv, expand, backend, **common_kwargs)
        self.bwd = _build_mamba(d_model, d_state, d_conv, expand, backend, **common_kwargs)
        self.proj = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_f = self.fwd(x)
        h_b = self.bwd(x.flip(dims=[1])).flip(dims=[1])
        return self.proj(torch.cat([h_f, h_b], dim=-1))


class MambaStack(nn.Module):
    """N stacked (Bi)Mamba-3 blocks with residual + LayerNorm, dropout."""

    def __init__(
        self,
        d_model: int,
        *,
        num_blocks: int = 2,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        n_groups: int = 1,
        rope_fraction: float = 0.5,
        is_mimo: bool = False,
        mimo_rank: int = 4,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
        bidirectional: bool = True,
        dropout: float = 0.1,
        backend: MambaBackendChoice = "auto",
    ) -> None:
        super().__init__()
        common_kwargs = dict(
            headdim=headdim,
            n_groups=n_groups,
            rope_fraction=rope_fraction,
            is_mimo=is_mimo,
            mimo_rank=mimo_rank,
            chunk_size=chunk_size,
            is_outproj_norm=is_outproj_norm,
        )
        if bidirectional:
            self.layers = nn.ModuleList([
                BidirectionalMamba(
                    d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    backend=backend,
                    **common_kwargs,
                )
                for _ in range(num_blocks)
            ])
        else:
            self.layers = nn.ModuleList([
                _build_mamba(d_model, d_state, d_conv, expand, backend, **common_kwargs)
                for _ in range(num_blocks)
            ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_blocks)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer, norm in zip(self.layers, self.norms, strict=False):
            out = norm(out + self.dropout(layer(out)))
        return out
