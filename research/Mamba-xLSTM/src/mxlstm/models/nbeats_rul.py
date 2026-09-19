"""N-BEATS-style decomposable model for scalar bearing RUL prediction."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NBeatsBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_theta_b: int,
        n_theta_f: int,
        context_length: int,
        n_features: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.n_features = int(n_features)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.theta_b = nn.Linear(hidden_dim, n_theta_b)
        self.theta_f = nn.Linear(hidden_dim, n_theta_f)
        gb, gf = self.make_basis(n_theta_b, n_theta_f, context_length, n_features)
        if isinstance(gb, nn.Parameter):
            self.gb = gb
        else:
            self.register_buffer("gb", gb)
        if isinstance(gf, nn.Parameter):
            self.gf = gf
        else:
            self.register_buffer("gf", gf)

    def make_basis(
        self,
        n_theta_b: int,
        n_theta_f: int,
        context_length: int,
        n_features: int,
    ) -> tuple[torch.Tensor | nn.Parameter, torch.Tensor | nn.Parameter]:
        raise NotImplementedError

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(x_flat)
        theta_b = self.theta_b(hidden)
        theta_f = self.theta_f(hidden)
        backcast = theta_b @ self.gb
        forecast = (theta_f @ self.gf).squeeze(-1)
        return backcast, forecast, theta_f


class TrendBlock(_NBeatsBlock):
    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int = 128,
        poly_degree: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=context_length * n_features,
            hidden_dim=hidden_dim,
            n_theta_b=poly_degree + 1,
            n_theta_f=poly_degree + 1,
            context_length=context_length,
            n_features=n_features,
            dropout=dropout,
        )

    def make_basis(
        self,
        n_theta_b: int,
        n_theta_f: int,
        context_length: int,
        n_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(0, 1, context_length)
        gb = torch.stack(
            [(t**p).unsqueeze(-1).expand(context_length, n_features).reshape(-1) for p in range(n_theta_b)],
            dim=0,
        )
        gf = torch.ones(n_theta_f, 1)
        return gb, gf


class WearBlock(_NBeatsBlock):
    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int = 128,
        n_harmonics: int = 8,
        dropout: float = 0.1,
    ) -> None:
        n_theta = 2 * n_harmonics
        super().__init__(
            input_dim=context_length * n_features,
            hidden_dim=hidden_dim,
            n_theta_b=n_theta,
            n_theta_f=n_theta,
            context_length=context_length,
            n_features=n_features,
            dropout=dropout,
        )

    def make_basis(
        self,
        n_theta_b: int,
        n_theta_f: int,
        context_length: int,
        n_features: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(0, 1, context_length)
        rows = []
        forecast_rows = []
        for k in range(1, n_theta_b // 2 + 1):
            rows.append(torch.cos(2 * math.pi * k * t).unsqueeze(-1).expand(context_length, n_features).reshape(-1))
            rows.append(torch.sin(2 * math.pi * k * t).unsqueeze(-1).expand(context_length, n_features).reshape(-1))
            forecast_rows.extend([1.0, 0.0])
        return torch.stack(rows, dim=0), torch.tensor(forecast_rows).view(n_theta_f, 1)


class ShockBlock(_NBeatsBlock):
    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int = 128,
        n_basis: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            input_dim=context_length * n_features,
            hidden_dim=hidden_dim,
            n_theta_b=n_basis,
            n_theta_f=n_basis,
            context_length=context_length,
            n_features=n_features,
            dropout=dropout,
        )

    def make_basis(
        self,
        n_theta_b: int,
        n_theta_f: int,
        context_length: int,
        n_features: int,
    ) -> tuple[nn.Parameter, nn.Parameter]:
        input_dim = context_length * n_features
        gb = nn.Parameter(torch.randn(n_theta_b, input_dim) * (1.0 / math.sqrt(input_dim)))
        gf = nn.Parameter(torch.randn(n_theta_f, 1) * 0.1)
        return gb, gf


class _Stack(nn.Module):
    def __init__(self, blocks: list[_NBeatsBlock]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x_flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        residual = x_flat
        forecast_sum = x_flat.new_zeros(x_flat.size(0))
        thetas = []
        for block in self.blocks:
            backcast, forecast, theta = block(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
            thetas.append(theta)
        return residual, forecast_sum, thetas


class NBeatsRUL(nn.Module):
    """Additive trend/wear/shock decomposition for normalized scalar RUL."""

    def __init__(
        self,
        context_length: int,
        n_features: int,
        *,
        hidden_dim: int = 128,
        trend_blocks: int = 2,
        wear_blocks: int = 2,
        shock_blocks: int = 2,
        poly_degree: int = 3,
        n_harmonics: int = 6,
        n_shock_basis: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.n_features = int(n_features)
        self.trend_stack = _Stack(
            [
                TrendBlock(context_length, n_features, hidden_dim, poly_degree, dropout)
                for _ in range(trend_blocks)
            ]
        )
        self.wear_stack = _Stack(
            [
                WearBlock(context_length, n_features, hidden_dim, n_harmonics, dropout)
                for _ in range(wear_blocks)
            ]
        )
        self.shock_stack = _Stack(
            [
                ShockBlock(context_length, n_features, hidden_dim, n_shock_basis, dropout)
                for _ in range(shock_blocks)
            ]
        )
        self.residual_head = nn.Linear(context_length * n_features, 1)
        self.bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, *, return_parts: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        batch_size = x.size(0)
        x_flat = x.reshape(batch_size, -1)
        residual, trend, theta_trend = self.trend_stack(x_flat)
        residual, wear, theta_wear = self.wear_stack(residual)
        residual, shock, theta_shock = self.shock_stack(residual)
        residual_correction = 0.01 * self.residual_head(residual).squeeze(-1)
        raw = trend + wear + shock + residual_correction + self.bias
        rul = raw.clamp(0.0, 1.0)
        if not return_parts:
            return rul
        return rul, {
            "trend": trend,
            "wear": wear,
            "shock": shock,
            "residual_correction": residual_correction,
            "bias": self.bias,
            "raw_sum": raw,
            "residual_norm": residual.norm(dim=-1),
            "theta_trend": theta_trend,
            "theta_wear": theta_wear,
            "theta_shock": theta_shock,
        }

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor, out_of_range_weight: float = 0.1) -> torch.Tensor:
        _rul, parts = self.forward(x, return_parts=True)
        raw = parts["raw_sum"]
        main = F.smooth_l1_loss(raw, y)
        penalty = F.relu(raw - 1.0).pow(2).mean() + F.relu(-raw).pow(2).mean()
        return main + out_of_range_weight * penalty

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rul, parts = self.forward(x, return_parts=True)
        return {
            "rul": rul.cpu(),
            "trend_contribution": parts["trend"].cpu(),
            "wear_contribution": parts["wear"].cpu(),
            "shock_contribution": parts["shock"].cpu(),
            "residual_correction": parts["residual_correction"].cpu(),
            "raw_sum": parts["raw_sum"].cpu(),
            "unexplained_residual": parts["residual_norm"].cpu(),
        }
