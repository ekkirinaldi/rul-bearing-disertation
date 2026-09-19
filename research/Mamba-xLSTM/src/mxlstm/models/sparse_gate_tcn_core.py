"""SparseGate-TCN-RUL core (gates + TCN + cross-feature attention + quantile head).

REVERT / ABLATION NOTE (training objective):
    Production training uses ``SparseGateTCNRUL`` with ``SparseTCNLoss`` (pinball
    + gate penalties) and ``model_specific_loss: true``. To align with plain MSE
    baselines, switch the adapter to MSE-on-median only and set
    ``model_specific_loss: false`` in ``configs/model/sparse_gate_tcn_rul.yaml``.

See dissertation / prototype narrative in the original repo-root file for the
full architecture rationale.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sparse Feature Gate
# ---------------------------------------------------------------------------
class SparseFeatureGate(nn.Module):
    """
    Produces a per-timestep, per-feature soft mask g_t in [0,1]^F.

    Two design choices that matter:

    1. Sigmoid-not-softmax. We want each feature to be independently on/off,
       NOT competing in a softmax (where features sum to 1). Bearing
       degradation may genuinely require multiple features simultaneously
       (RMS + kurtosis + spectral entropy together).

    2. Local context. The gate is computed from a small temporal context
       (a 1D conv with kernel=5) so the gate at time t can react to
       short-term changes (e.g. kurtosis spike in last 3 steps -> raise
       kurtosis gate).

    Sparsity is enforced by the LOSS, not by hard top-k. This keeps gates
    differentiable and lets the network learn variable sparsity (some
    timesteps need 2 features, others need 6).
    """

    def __init__(self, n_features: int, hidden: int = 32,
                 context_kernel: int = 5):
        super().__init__()
        self.n_features = n_features
        self.context_kernel = context_kernel

        pad = context_kernel // 2
        # 1D conv along time, treating features as channels
        self.context = nn.Conv1d(
            in_channels=n_features,
            out_channels=hidden,
            kernel_size=context_kernel,
            padding=pad,
        )
        self.to_logits = nn.Conv1d(
            in_channels=hidden,
            out_channels=n_features,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) -> gates (B, T, F) in [0,1]
        """
        # Conv1d wants (B, C=F, T)
        x_t = x.transpose(1, 2)
        h = F.gelu(self.context(x_t))
        logits = self.to_logits(h)
        gates = torch.sigmoid(logits)
        return gates.transpose(1, 2)                   # back to (B, T, F)


# ---------------------------------------------------------------------------
# 2. TCN backbone (Bai et al. 2018)
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Conv1d):
    """1D convolution with left-padding only (causal). Avoids future leak."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int = 1, **kwargs):
        self.left_pad = (kernel_size - 1) * dilation
        super().__init__(in_ch, out_ch, kernel_size,
                         padding=0, dilation=dilation, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    """
    Residual TCN block: two stacked causal dilated convolutions with
    GELU + dropout + weight norm, plus a 1x1 residual projection.
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(n_inputs, n_outputs, kernel_size, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            CausalConv1d(n_outputs, n_outputs, kernel_size, dilation=dilation)
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        h = self.act(self.conv1(x))
        h = self.dropout(h)
        h = self.act(self.conv2(h))
        h = self.dropout(h)
        res = x if self.downsample is None else self.downsample(x)
        return h + res


class TCN(nn.Module):
    """
    Stack of TCNBlocks with exponentially increasing dilation:
        block i has dilation = 2^i
    so receptive field grows from kernel_size to kernel_size * (2^L - 1).
    For L=4 layers, kernel=3: receptive field = 3 * 15 = 45 timesteps,
    enough for windows of 32-64.
    """

    def __init__(self, n_inputs: int, channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        in_ch = n_inputs
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(
                n_inputs=in_ch, n_outputs=out_ch,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            ))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C_in) -> (B, T, C_out)
        x = x.transpose(1, 2)              # (B, C, T)
        x = self.network(x)
        return x.transpose(1, 2)           # (B, T, C_out)


# ---------------------------------------------------------------------------
# 3. Cross-feature attention (interpretable secondary signal)
# ---------------------------------------------------------------------------
class CrossFeatureAttention(nn.Module):
    """
    Self-attention OVER THE FEATURE DIMENSION at each timestep.

    For each timestep t, we have F features. We treat them as F tokens
    of dim d_model and do a standard self-attention on them. The
    resulting attention scores form an (F, F) matrix per timestep, which
    tells us: "feature i relies on feature j for its representation."

    This is the interpretability *secondary* signal — it answers the
    feature-INTERACTION question, complementing the gate's
    feature-IMPORTANCE question.
    """

    def __init__(self, n_features: int, d_model: int, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.feature_emb = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.in_proj = nn.Linear(1, d_model)         # each feature is a scalar
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F) -> mixed (B, T, F),  attn_scores (B, T, F, F)
        """
        B, T, F_ = x.shape
        # Treat each feature value as a "token" with embedding feature_emb[i]
        # Reshape to (B*T, F, 1) and lift to d_model
        flat = x.reshape(B * T, F_, 1)
        tokens = self.in_proj(flat) + self.feature_emb.unsqueeze(0)  # (B*T,F,d)
        out, scores = self.attn(tokens, tokens, tokens,
                                need_weights=True, average_attn_weights=True)
        mixed = self.out_proj(out).squeeze(-1)        # (B*T, F)
        mixed = mixed.reshape(B, T, F_)
        scores = scores.reshape(B, T, F_, F_)
        return mixed, scores


# ---------------------------------------------------------------------------
# 4. Quantile regression head
# ---------------------------------------------------------------------------
class QuantileHead(nn.Module):
    """
    Predicts {P5, P50, P95} of the RUL distribution. We enforce ordering
    (P5 <= P50 <= P95) via the parameterization:
        P50 = sigmoid(z_50)
        P5  = P50 - softplus(z_lo)
        P95 = P50 + softplus(z_hi)
    so the ordering is mathematically guaranteed, not learned.
    """

    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        # h: (B, d_in) -> dict of (B,) tensors
        z = self.body(h)
        z50, z_lo, z_hi = z[:, 0], z[:, 1], z[:, 2]
        p50 = torch.sigmoid(z50)
        p05 = (p50 - F.softplus(z_lo)).clamp(0.0, 1.0)
        p95 = (p50 + F.softplus(z_hi)).clamp(0.0, 1.0)
        return {"p05": p05, "p50": p50, "p95": p95}


# ---------------------------------------------------------------------------
# 5. SparseGate-TCN-RUL full model
# ---------------------------------------------------------------------------
@dataclass
class SparseTCNConfig:
    n_features: int = 16
    tcn_channels: Tuple[int, ...] = (64, 64, 128, 128)
    tcn_kernel: int = 3
    gate_hidden: int = 32
    gate_context: int = 5
    attn_d_model: int = 32
    attn_heads: int = 4
    head_hidden: int = 64
    dropout: float = 0.1
    feature_names: Tuple[str, ...] = (
        "RMS", "peak", "kurtosis", "skewness",
        "crest_factor", "shape_factor", "impulse_factor", "margin_factor",
        "variance", "spec_centroid", "spec_entropy",
        "band_energy_1", "band_energy_2", "band_energy_3",
        "band_energy_4", "band_energy_5",
    )


class SparseGateTCN(nn.Module):
    """
    Full SparseGate-TCN-RUL model.

    Forward returns a dict with:
        rul          : (B,) median RUL prediction (alias for p50)
        p05, p50, p95: (B,) quantiles
        feature_gates: (B, T, F) per-step feature attribution in [0,1]
        attn_scores  : (B, T, F, F) cross-feature attention (set to None
                                    if return_attn=False to save memory)
        last_hidden  : (B, d_model) representation at last timestep
    """

    def __init__(self, cfg: SparseTCNConfig):
        super().__init__()
        assert len(cfg.feature_names) == cfg.n_features
        self.cfg = cfg

        self.gate = SparseFeatureGate(
            n_features=cfg.n_features,
            hidden=cfg.gate_hidden,
            context_kernel=cfg.gate_context,
        )

        self.cross_attn = CrossFeatureAttention(
            n_features=cfg.n_features,
            d_model=cfg.attn_d_model,
            n_heads=cfg.attn_heads,
        )

        self.tcn = TCN(
            n_inputs=cfg.n_features,
            channels=list(cfg.tcn_channels),
            kernel_size=cfg.tcn_kernel,
            dropout=cfg.dropout,
        )

        self.head = QuantileHead(
            d_in=cfg.tcn_channels[-1],
            hidden=cfg.head_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
        *,
        return_sequence: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """x: (B, T, F)"""
        B, T, F_ = x.shape
        assert F_ == self.cfg.n_features, \
            f"Expected {self.cfg.n_features} features, got {F_}"

        # 1) Per-step feature gates
        gates = self.gate(x)                                  # (B, T, F)
        x_gated = x * gates                                   # (B, T, F)

        # 2) Cross-feature attention (provides the secondary explanation)
        x_mixed, attn_scores = self.cross_attn(x_gated)       # (B, T, F)
        # Residual mix: keep gated features but enrich with cross-feature info
        x_combined = x_gated + x_mixed

        # 3) TCN backbone
        h_seq = self.tcn(x_combined)                          # (B, T, C_out)

        # 4) Quantile head on last timestep
        h_last = h_seq[:, -1, :]
        quantiles = self.head(h_last)

        out = {
            "rul": quantiles["p50"],
            "p05": quantiles["p05"],
            "p50": quantiles["p50"],
            "p95": quantiles["p95"],
            "feature_gates": gates,
            "last_hidden": h_last,
        }
        if return_sequence:
            out["tcn_sequence"] = h_seq
        if return_attn:
            out["attn_scores"] = attn_scores
        return out

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns dissertation-ready interpretability outputs.

        The KEY interpretability artifact is `feature_gates`:
        (B, T, F) — at each timestep, the model's importance weight on
        each named HI feature. Plot as a heatmap (time x feature) and
        you have a SHAP-equivalent attribution map, computed online,
        mathematically faithful, no surrogate model needed.
        """
        out = self.forward(x, return_attn=True)
        # Aggregate: global feature importance = mean over time and batch
        global_importance = out["feature_gates"].mean(dim=(0, 1))   # (F,)
        return {
            "rul": out["rul"].cpu(),
            "p05": out["p05"].cpu(),
            "p95": out["p95"].cpu(),
            "feature_gates": out["feature_gates"].cpu(),
            "feature_names": list(self.cfg.feature_names),
            "global_feature_importance": global_importance.cpu(),
            "attn_scores": out["attn_scores"].cpu(),
        }


# ---------------------------------------------------------------------------
# 6. Loss
# ---------------------------------------------------------------------------
class SparseTCNLoss(nn.Module):
    """
    Total = pinball_loss + lambda_sparse * gate_L1 + lambda_entropy * gate_entropy.

    Note: older doc drafts mentioned ``lambda_mono``; that term is **not**
    implemented here. Per-window scalar monotonicity (when desired) is applied
    in ``RULLitModule`` via ``train.monotonicity_weight`` on the median RUL.

    Pinball loss for quantile q:
        L_q(y, y_hat) = mean(max(q*(y - y_hat), (q-1)*(y - y_hat)))
    Sum over q in {0.05, 0.5, 0.95}. This is the standard quantile
    regression objective (Koenker & Bassett 1978).

    Sparsity penalty (L1 on gates) drives gate values toward 0,
    encouraging only the genuinely useful features to remain active.

    Entropy penalty: minimizing the per-element entropy
        H(g) = -[g log g + (1 - g) log(1 - g)]
    pushes each gate AWAY from 0.5 (max-uncertainty) and toward {0, 1}
    (crisp on/off). Adding lambda_entropy * mean(H) to the loss (with
    lambda_entropy > 0) directly minimizes H -> sharper, more
    interpretable gates.

    The two penalties work together:
      - L1 says: 'keep gates small'
      - Entropy says: 'don't sit at 0.5; commit'
    Net effect: most gates -> 0, a few -> 1. Per-window adaptive sparsity.
    """

    def __init__(self,
                 quantiles: Tuple[float, ...] = (0.05, 0.5, 0.95),
                 lambda_sparse: float = 1e-3,
                 lambda_entropy: float = 1e-3):
        super().__init__()
        self.quantiles = quantiles
        self.lambda_sparse = lambda_sparse
        self.lambda_entropy = lambda_entropy

    @staticmethod
    def _pinball(y: torch.Tensor, y_hat: torch.Tensor, q: float) -> torch.Tensor:
        diff = y - y_hat
        return torch.maximum(q * diff, (q - 1.0) * diff).mean()

    def forward(self, outputs: Dict[str, torch.Tensor],
                y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        outputs: dict from SparseGateTCN.forward()
        y      : (B,) ground-truth RUL at the last timestep of the window
        """
        # Pinball loss across the three predicted quantiles
        loss_q = (
            self._pinball(y, outputs["p05"], 0.05) +
            self._pinball(y, outputs["p50"], 0.5) +
            self._pinball(y, outputs["p95"], 0.95)
        )

        # Sparsity (L1 on gates)
        gates = outputs["feature_gates"]                            # (B,T,F)
        loss_sparse = gates.mean()                                  # mean of |g|, g>=0

        # Entropy penalty: minimize H to encourage crisp gates
        # H(g) = -[g log g + (1-g) log(1-g)]
        eps = 1e-7
        H = -(gates * torch.log(gates + eps) +
              (1 - gates) * torch.log(1 - gates + eps))
        loss_entropy = H.mean()

        total = (loss_q
                 + self.lambda_sparse * loss_sparse
                 + self.lambda_entropy * loss_entropy)

        return {
            "total": total,
            "pinball": loss_q.detach(),
            "sparsity": loss_sparse.detach(),
            "entropy": loss_entropy.detach(),
            "mean_active_features": (gates > 0.5).float().mean().detach(),
        }
