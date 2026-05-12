"""Physics-conditioned N-BEATS variants for bearing RUL.

``PhysicsNBeatsRUL`` — additive decomposition with bearing-frequency wear basis,
FiLM operating-condition modulation, and kurtosis-gated shocks (see
``physics_nbeats_core.py`` for per-block rationale).

``NBeatsXLSTMRUL`` — same stacks, but stacked mLSTM blocks read long-range
temporal context before the per-stack depthwise conv encoder. Motivation:
XJTU-SY windows span 60 s between acquisitions; a single conv receptive field
may miss slow cross-window drifts that still affect the label.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from mxlstm.models.physics_nbeats_core import (
    BernsteinTrendBlock,
    CharacteristicFrequencyWearBlock,
    GaborShockBlock,
    LearnedPhysShockBlock,
    _PhysStack,
    _dataset_key,
)
from mxlstm.models.xlstm_blocks import VanillaMLSTMBlock


def _default_condition_ids(x: torch.Tensor) -> torch.Tensor:
    """Fallback when the Lightning batch has no ``condition`` meta (should not happen)."""
    return torch.ones(x.size(0), device=x.device, dtype=torch.long)


class PhysicsNBeatsRUL(nn.Module):
    """Dataset-aware N-BEATS with physics wear basis and FiLM."""

    def __init__(
        self,
        context_length: int,
        n_features: int,
        dataset: str,
        *,
        hidden_dim: int = 96,
        trend_blocks: int = 2,
        wear_blocks: int = 2,
        shock_blocks: int = 2,
        poly_degree: int = 4,
        n_shock_basis: int = 14,
        film_num_embeddings: int = 4,
        kurt_index: int = 8,
        dropout: float = 0.15,
        lambda_wear_sparse: float = 1e-3,
        use_xlstm_front: bool = False,
        xlstm_d_model: int = 64,
        xlstm_heads: int = 4,
        xlstm_num_blocks: int = 1,
        xlstm_inter_dropout: float = 0.0,
        encoder_kernel_size: int | None = None,
        monotone_trend: bool = True,
        hybrid_learned_shock: bool = False,
        n_learned_shock_basis: int = 8,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.n_features = int(n_features)
        self.dataset = _dataset_key(str(dataset))
        self.kurt_index = int(kurt_index)
        self.lambda_wear_sparse = float(lambda_wear_sparse)
        self.use_xlstm_front = bool(use_xlstm_front)
        self.xlstm_d_model = int(xlstm_d_model)
        self.xlstm_inter_dropout = float(xlstm_inter_dropout)
        self.monotone_trend = bool(monotone_trend)
        self.hybrid_learned_shock = bool(hybrid_learned_shock)
        self.n_learned_shock_basis = int(n_learned_shock_basis)
        if encoder_kernel_size is not None:
            self._encoder_kernel = int(encoder_kernel_size)
        else:
            self._encoder_kernel = 7 if self.dataset == "xjtusy" else 5

        if self.use_xlstm_front:
            if xlstm_d_model % xlstm_heads != 0:
                raise ValueError("xlstm_d_model must be divisible by xlstm_heads")
            nb = max(1, int(xlstm_num_blocks))
            self._feat_to_d = nn.Linear(self.n_features, self.xlstm_d_model)
            self._xlstm_blocks = nn.ModuleList(
                VanillaMLSTMBlock(self.xlstm_d_model, n_heads=int(xlstm_heads)) for _ in range(nb)
            )
            self._xlstm_norms = nn.ModuleList(nn.LayerNorm(self.xlstm_d_model) for _ in range(nb))
            enc_in = self.xlstm_d_model
        else:
            self._feat_to_d = None  # type: ignore[assignment]
            self._xlstm_blocks = None  # type: ignore[assignment]
            self._xlstm_norms = None  # type: ignore[assignment]
            enc_in = self.n_features

        def trend_stack() -> _PhysStack:
            blocks = [
                BernsteinTrendBlock(
                    self.context_length,
                    enc_in,
                    hidden_dim,
                    poly_degree,
                    film_num_embeddings,
                    dropout,
                    encoder_kernel_size=self._encoder_kernel,
                    monotone_trend=self.monotone_trend,
                )
                for _ in range(int(trend_blocks))
            ]
            return _PhysStack(blocks, self.context_length, enc_in)

        def wear_stack() -> _PhysStack:
            blocks = [
                CharacteristicFrequencyWearBlock(
                    self.context_length,
                    enc_in,
                    hidden_dim,
                    film_num_embeddings,
                    dropout,
                    str(self.dataset),
                    encoder_kernel_size=self._encoder_kernel,
                )
                for _ in range(int(wear_blocks))
            ]
            return _PhysStack(blocks, self.context_length, enc_in)

        def shock_stack() -> _PhysStack:
            blocks: list[nn.Module] = [
                GaborShockBlock(
                    self.context_length,
                    enc_in,
                    hidden_dim,
                    int(n_shock_basis),
                    film_num_embeddings,
                    self.kurt_index,
                    dropout,
                    str(self.dataset),
                    encoder_kernel_size=self._encoder_kernel,
                )
                for _ in range(int(shock_blocks))
            ]
            if self.hybrid_learned_shock:
                blocks.append(
                    LearnedPhysShockBlock(
                        self.context_length,
                        enc_in,
                        hidden_dim,
                        int(self.n_learned_shock_basis),
                        film_num_embeddings,
                        dropout,
                        encoder_kernel_size=self._encoder_kernel,
                    )
                )
            return _PhysStack(blocks, self.context_length, enc_in)

        self.trend_stack = trend_stack()
        self.wear_stack = wear_stack()
        self.shock_stack = shock_stack()
        lf = self.context_length * enc_in
        self.residual_head = nn.Linear(lf, 1)
        self.bias = nn.Parameter(torch.tensor(0.5))

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, F) -> (B, L, enc_in) for stacks."""
        if not self.use_xlstm_front:
            return x
        z = self._feat_to_d(x)
        assert self._xlstm_blocks is not None and self._xlstm_norms is not None
        n_blk = len(self._xlstm_blocks)
        for i, (blk, ln) in enumerate(zip(self._xlstm_blocks, self._xlstm_norms, strict=True)):
            z = ln(z + blk(z))
            if self.xlstm_inter_dropout > 0.0 and i + 1 < n_blk:
                z = F.dropout(z, p=self.xlstm_inter_dropout, training=self.training)
        return z

    def forward(
        self,
        x: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
        *,
        return_parts: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        if condition_ids is None:
            condition_ids = _default_condition_ids(x)
        b = x.size(0)
        x_enc = self._encode_input(x)
        c_enc = int(x_enc.size(-1))
        lf = self.context_length * c_enc
        res_flat = x_enc.reshape(b, lf)
        res_flat, trend_f, t_ex = self.trend_stack(res_flat, res_flat, condition_ids)
        res_flat, wear_f, w_ex = self.wear_stack(res_flat, res_flat, condition_ids)
        res_flat, shock_f, s_ex = self.shock_stack(res_flat, res_flat, condition_ids)

        residual_correction = 0.01 * self.residual_head(res_flat).squeeze(-1)
        raw = trend_f + wear_f + shock_f + residual_correction + self.bias
        rul = raw.clamp(0.0, 1.0)
        if not return_parts:
            return rul
        wear_named: dict[str, torch.Tensor] = {}
        wear_theta_cat: list[torch.Tensor] = []
        for d in w_ex:
            wear_named.update(d.get("wear_named", {}))
            if "wear_theta" in d:
                wear_theta_cat.append(d["wear_theta"])
        shock_gates = [d["shock_gate"] for d in s_ex if "shock_gate" in d]
        parts: dict[str, Any] = {
            "trend": trend_f,
            "wear": wear_f,
            "shock": shock_f,
            "residual_correction": residual_correction,
            "bias": self.bias,
            "raw_sum": raw,
            "residual_norm": res_flat.norm(dim=-1),
            "stack_extras": {"trend": t_ex, "wear": w_ex, "shock": s_ex},
            "wear_theta_per_freq": wear_named,
            "wear_theta": torch.cat(wear_theta_cat, dim=-1) if wear_theta_cat else None,
            "shock_gate": torch.stack(shock_gates, dim=-1).mean(dim=-1) if shock_gates else None,
        }
        return rul, parts

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        condition_ids: torch.Tensor | None = None,
        rul_window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del rul_window  # unused; accepted for Lightning signature compatibility
        if condition_ids is None:
            condition_ids = _default_condition_ids(x)
        _rul, parts = self.forward(x, condition_ids, return_parts=True)
        raw = parts["raw_sum"]
        main = F.smooth_l1_loss(raw, y)
        penalty = F.relu(raw - 1.0).pow(2).mean() + F.relu(-raw).pow(2).mean()
        wear_sp = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        wt = parts.get("wear_theta")
        if wt is not None:
            wear_sp = wt.abs().mean()
        return main + 0.1 * penalty + self.lambda_wear_sparse * wear_sp

    @torch.no_grad()
    def explain(
        self,
        x: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if condition_ids is None:
            condition_ids = _default_condition_ids(x)
        rul, parts = self.forward(x, condition_ids, return_parts=True)
        out: dict[str, torch.Tensor] = {
            "rul": rul.cpu(),
            "trend_contribution": parts["trend"].cpu(),
            "wear_contribution": parts["wear"].cpu(),
            "shock_contribution": parts["shock"].cpu(),
            "residual_correction": parts["residual_correction"].cpu(),
            "raw_sum": parts["raw_sum"].cpu(),
            "unexplained_residual": parts["residual_norm"].cpu(),
        }
        if parts.get("shock_gate") is not None:
            out["shock_gate"] = parts["shock_gate"].cpu()
        for k, v in parts["wear_theta_per_freq"].items():
            out[f"wear_{k}"] = v.cpu()
        return out


class NBeatsXLSTMRUL(PhysicsNBeatsRUL):
    """Physics-N-BEATS with a stacked mLSTM temporal front-end."""

    def __init__(
        self,
        context_length: int,
        n_features: int,
        dataset: str,
        *,
        hidden_dim: int = 96,
        trend_blocks: int = 2,
        wear_blocks: int = 2,
        shock_blocks: int = 2,
        poly_degree: int = 4,
        n_shock_basis: int = 14,
        film_num_embeddings: int = 4,
        kurt_index: int = 8,
        dropout: float = 0.15,
        lambda_wear_sparse: float = 1e-3,
        xlstm_d_model: int = 64,
        xlstm_heads: int = 4,
        xlstm_num_blocks: int = 2,
        xlstm_inter_dropout: float = 0.0,
        encoder_kernel_size: int | None = None,
        monotone_trend: bool = True,
        hybrid_learned_shock: bool = False,
        n_learned_shock_basis: int = 8,
    ) -> None:
        super().__init__(
            context_length,
            n_features,
            dataset,
            hidden_dim=hidden_dim,
            trend_blocks=trend_blocks,
            wear_blocks=wear_blocks,
            shock_blocks=shock_blocks,
            poly_degree=poly_degree,
            n_shock_basis=n_shock_basis,
            film_num_embeddings=film_num_embeddings,
            kurt_index=kurt_index,
            dropout=dropout,
            lambda_wear_sparse=lambda_wear_sparse,
            use_xlstm_front=True,
            xlstm_d_model=xlstm_d_model,
            xlstm_heads=xlstm_heads,
            xlstm_num_blocks=xlstm_num_blocks,
            xlstm_inter_dropout=xlstm_inter_dropout,
            encoder_kernel_size=encoder_kernel_size,
            monotone_trend=monotone_trend,
            hybrid_learned_shock=hybrid_learned_shock,
            n_learned_shock_basis=n_learned_shock_basis,
        )
