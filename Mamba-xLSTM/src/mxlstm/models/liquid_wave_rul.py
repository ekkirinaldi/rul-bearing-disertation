"""LiquidWave-RUL model adapted to the HI-window training contract.

The original prototype consumes explicit wavelet-band features. The packaged
version receives the existing ``(B, L, F)`` health-indicator windows and forms
interpretable pseudo-bands by grouping HI features, then summarising each group
with differentiable statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LTCConfig:
    input_dim: int
    hidden_dim: int
    num_unfolds: int = 4
    epsilon: float = 1e-3
    tau_max: float = 1e3


class LTCCell(nn.Module):
    """Liquid Time-Constant cell with the fused semi-implicit update."""

    def __init__(self, cfg: LTCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.w_x = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.w_h = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.tau_x = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.tau_h = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        # Non-zero init so the fused update does not stay at the zero state when
        # paired with h0 = 0. With a = 0 the cell is stuck at zero until `a`
        # drifts through gradient, which is what caused the flat RMSE at start.
        self.a = nn.Parameter(torch.empty(cfg.hidden_dim).uniform_(-0.1, 0.1))
        self.b = nn.Parameter(torch.zeros(cfg.hidden_dim))

    def forward(self, x: torch.Tensor, h: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        drive = self.w_x(x) + self.w_h(h)
        gate = torch.sigmoid(drive)
        tau = F.softplus(self.tau_x(x) + self.tau_h(h)) + self.cfg.epsilon
        tau = tau.clamp(max=self.cfg.tau_max)
        sub_dt = dt / self.cfg.num_unfolds
        source = gate * self.a + self.b
        for _ in range(self.cfg.num_unfolds):
            h = (h + sub_dt * source) / (1.0 + sub_dt * (1.0 / tau + gate))
        return h


class LTCLayer(nn.Module):
    def __init__(self, cfg: LTCConfig) -> None:
        super().__init__()
        self.cell = LTCCell(cfg)
        self.hidden_dim = cfg.hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        h = x.new_zeros(batch_size, self.hidden_dim)
        outs = []
        for t in range(seq_len):
            h = self.cell(x[:, t, :], h)
            outs.append(h)
        return torch.stack(outs, dim=1)


class PseudoBandFeatures(nn.Module):
    """Convert HI features into per-band summary features at each timestep."""

    def __init__(self, n_features: int, n_bands: int, n_band_feats: int = 4) -> None:
        super().__init__()
        if n_bands < 1:
            raise ValueError("n_bands must be >= 1")
        self.n_features = int(n_features)
        self.n_bands = int(n_bands)
        self.n_band_feats = int(n_band_feats)
        self.group_width = math.ceil(self.n_features / self.n_bands)
        self.padded_features = self.group_width * self.n_bands
        if n_band_feats != 4:
            self.proj = nn.Linear(4, n_band_feats)
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pad = self.padded_features - self.n_features
        if pad:
            x = F.pad(x, (0, pad))
        bands = x.view(batch_size, seq_len, self.n_bands, self.group_width)
        mean = bands.mean(dim=-1)
        std = bands.std(dim=-1, unbiased=False)
        peak = bands.abs().amax(dim=-1)
        energy = bands.square().mean(dim=-1).sqrt()
        features = torch.stack([mean, std, peak, energy], dim=-1)
        return self.proj(features)


class MultiHeadBandAttention(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int = 1) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = hidden_dim // n_heads
        self.hidden_dim = hidden_dim
        self.query = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, band_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, n_bands, _ = band_hidden.shape
        keys = self.k_proj(band_hidden).view(
            batch_size, seq_len, n_bands, self.n_heads, self.head_dim
        )
        values = self.v_proj(band_hidden).view(
            batch_size, seq_len, n_bands, self.n_heads, self.head_dim
        )
        keys = keys.permute(0, 1, 3, 2, 4)
        values = values.permute(0, 1, 3, 2, 4)
        scores = (keys * self.query.view(1, 1, self.n_heads, 1, self.head_dim)).sum(dim=-1)
        weights = F.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        out = (weights.unsqueeze(-1) * values).sum(dim=3)
        out = out.reshape(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(out), weights.mean(dim=2)


class LiquidWaveRUL(nn.Module):
    """Per-band LTC encoders + cross-band attention + temporal LTC RUL head."""

    def __init__(
        self,
        n_features: int,
        *,
        n_bands: int = 6,
        n_band_feats: int = 4,
        hidden_dim: int = 32,
        attn_heads: int = 4,
        ltc_unfolds: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_bands = int(n_bands)
        self.hidden_dim = int(hidden_dim)
        self.band_features = PseudoBandFeatures(n_features, n_bands, n_band_feats)
        self.band_encoders = nn.ModuleList(
            [
                LTCLayer(
                    LTCConfig(
                        input_dim=n_band_feats,
                        hidden_dim=hidden_dim,
                        num_unfolds=ltc_unfolds,
                    )
                )
                for _ in range(n_bands)
            ]
        )
        self.band_norm = nn.LayerNorm(hidden_dim)
        self.band_attn = MultiHeadBandAttention(hidden_dim, n_heads=attn_heads)
        self.temporal_ltc = LTCLayer(
            LTCConfig(input_dim=hidden_dim, hidden_dim=hidden_dim, num_unfolds=ltc_unfolds)
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        # Residual direct path: pool per-band summary stats across bands and
        # feed them straight to the head. Gives the model a working baseline
        # before the LTC dynamics fully converge so the early-epoch RMSE
        # reflects real signal rather than the zero-state degenerate output.
        self.residual_proj = nn.Sequential(
            nn.Linear(n_band_feats * n_bands, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _forward_core(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        band_x = self.band_features(x)
        band_hidden = torch.stack(
            [enc(band_x[:, :, i, :]) for i, enc in enumerate(self.band_encoders)],
            dim=2,
        )
        band_hidden = self.band_norm(band_hidden)
        fused, weights = self.band_attn(band_hidden)
        temporal = self.temporal_ltc(fused)
        temporal = self.temporal_norm(temporal)
        batch_size, seq_len, _, _ = band_hidden.shape
        residual = self.residual_proj(
            band_x[:, -1, :, :].reshape(batch_size, -1)
        )
        return temporal, residual, weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal, residual, _weights = self._forward_core(x)
        pooled = temporal[:, -1, :] + residual
        return self.head(pooled).squeeze(-1)

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal, residual, weights = self._forward_core(x)
        pooled = temporal[:, -1, :] + residual
        rul = self.head(pooled).squeeze(-1)
        return {"rul": rul.cpu(), "band_weights": weights.cpu()}
