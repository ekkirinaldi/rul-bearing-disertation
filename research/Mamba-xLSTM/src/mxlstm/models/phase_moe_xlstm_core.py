"""
PhaseMoE-xLSTM-RUL core (REVIEWED; packaged under ``mxlstm.models``).
==============================
A novel architecture for bearing RUL prediction designed to beat the
xLSTM-Transformer baseline on XJTU-SY and PHM2012.

Core idea
---------
Bearing degradation has *qualitatively different dynamics* in each life phase:
  - HEALTHY      : near-stationary HI, tiny stochastic fluctuations
  - WEAR         : slow monotonic drift, mild kurtosis growth
  - PRE-FAILURE  : explosive non-linear acceleration, impulsive shocks

xLSTM-Transformer treats all phases with one shared parameter set, which
is the architectural reason it has trouble with late-life saturation and
early-life over-smoothing simultaneously. We address this with:

  1. **Phase-aware Mixture-of-Experts** — three small expert sub-networks,
     each an mLSTM + tiny self-attention block, specializing on one phase.
  2. **Physics-informed router** — soft phase probabilities derived from
     engineered HI statistics (RMS slope/curvature, kurtosis trajectory,
     z-score vs. healthy baseline). The router output IS the
     interpretability surface — no SHAP needed.
  3. **Physics-informed loss** — encodes RUL monotonicity, Paris-law-style
     late-life acceleration, and a self-supervisory phase-consistency
     signal derived from RUL-bin labels.
  4. **Split-conformal prediction wrapper** — calibrated 90/95% prediction
     intervals with finite-sample coverage guarantees, no retraining.

Why this should beat xLSTM-Transformer
--------------------------------------
- The MoE structure lets each expert specialize without compromising on
  a generic representation that fits all phases poorly.
- Soft routing means the model can blend across phase transitions, which
  is exactly when xLSTM-T struggles most.
- The physics-informed loss is free regularization — gradient pressure
  from prior knowledge — which materially helps on small datasets like
  XJTU-SY (15 bearings) and PHM2012 (17 bearings).
- Conformal UQ adds calibrated uncertainty the baseline lacks, at zero
  training-time cost.

Novelty (verified vs. published bearing-RUL literature, May 2026):
  - Phase-MoE on bearings: not published
  - Physics-informed router using HI monotonicity: not published
  - Conformal prediction wrapping an xLSTM expert: not published
  - The combination is genuinely new.

Input  : (B, T, F)    F = number of HI features (default 16)
Outputs (forward):
  rul          : (B,)        in [0, 1]
  rul_per_step : (B, T)      RUL prediction at every timestep
  phase_probs  : (B, T, 3)   per-step soft phase assignment
  per_expert_rul: (B, T, 3)  each expert's RUL curve
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Lightweight mLSTM cell (matrix-memory LSTM, exponential input gate)
# ---------------------------------------------------------------------------
# Faithful re-implementation of the core mLSTM block from Beck et al.
# (NeurIPS 2024), self-contained so the file does not depend on the
# `xlstm` package. The exponential input gate is the key to bearing
# late-life prediction: it does NOT saturate when input magnitude grows.
#
# We use the log-stabilization trick from Beck §3.1 / Eq. 13:
#     m_t = max(log f_t + m_{t-1}, log i_t)
#     i_t = exp(log i_t - m_t)
#     f_t = exp(log f_t + m_{t-1} - m_t)
# which keeps the matrix memory in the dynamic range of float32.
class mLSTMCell(nn.Module):
    """
    Matrix-memory LSTM cell with stabilized exponential input gating.

    Per timestep:
        q_t, k_t, v_t = projections of x_t
        i_t (scalar), f_t (scalar)         # Beck uses scalar gates
        o_t (vector)                       # standard sigmoid output gate
        m_t = max(log_f + m_prev, log_i)   # stabilizer
        i = exp(log_i - m), f = exp(log_f + m_prev - m)
        C_t = f * C_{t-1} + i * (v outer k)
        n_t = f * n_{t-1} + i * k
        h_t = o * (C_t @ q) / max(|n.q|, 1)

    Note: This is sequential (Python loop in the layer). For our window
    sizes (T=64, 3 experts) this is acceptable on GPU.
    """

    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.d_in, self.d_h = d_in, d_h
        self.W_q = nn.Linear(d_in, d_h)
        self.W_k = nn.Linear(d_in, d_h)
        self.W_v = nn.Linear(d_in, d_h)
        self.W_i = nn.Linear(d_in, 1)        # scalar gate per Beck §3.1
        self.W_f = nn.Linear(d_in, 1)
        self.W_o = nn.Linear(d_in, d_h)
        self._scale = 1.0 / math.sqrt(d_h)

    def init_state(self, B: int, device, dtype):
        C = torch.zeros(B, self.d_h, self.d_h, device=device, dtype=dtype)
        n = torch.zeros(B, self.d_h, device=device, dtype=dtype)
        m = torch.zeros(B, 1, device=device, dtype=dtype)
        return C, n, m

    def forward(self, x_t: torch.Tensor,
                C: torch.Tensor, n: torch.Tensor, m: torch.Tensor):
        q = self.W_q(x_t)
        k = self.W_k(x_t) * self._scale
        v = self.W_v(x_t)
        i_pre = self.W_i(x_t)                # (B, 1) -- log of i_t
        f_pre = self.W_f(x_t)

        log_f = F.logsigmoid(f_pre)          # in (-inf, 0]
        log_i = i_pre

        m_new = torch.maximum(log_f + m, log_i)
        i_t = torch.exp(log_i - m_new)        # (B, 1)
        f_t = torch.exp(log_f + m - m_new)    # (B, 1)

        outer = v.unsqueeze(2) * k.unsqueeze(1)              # (B, d_h, d_h)
        C_new = f_t.unsqueeze(2) * C + i_t.unsqueeze(2) * outer
        n_new = f_t * n + i_t * k                            # (B, d_h)

        Cq = torch.bmm(C_new, q.unsqueeze(2)).squeeze(2)     # (B, d_h)
        nq = (n_new * q).sum(dim=-1, keepdim=True).abs().clamp(min=1.0)
        o_t = torch.sigmoid(self.W_o(x_t))
        h = o_t * (Cq / nq)
        return h, C_new, n_new, m_new


class mLSTMLayer(nn.Module):
    """Sequence wrapper around mLSTMCell."""

    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.cell = mLSTMCell(d_in, d_h)
        self.d_h = d_h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_in) -> (B, T, d_h)
        B, T, _ = x.shape
        C, n, m = self.cell.init_state(B, x.device, x.dtype)
        outs = []
        for t in range(T):
            h, C, n, m = self.cell(x[:, t], C, n, m)
            outs.append(h)
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# 2. Compact Expert: mLSTM + small self-attention
# ---------------------------------------------------------------------------
class PhaseExpert(nn.Module):
    """
    A specialized small network: mLSTM trunk + 1-layer self-attention + FFN.
    Each expert is intentionally compact (~50–100k params); the diversity
    across the three experts is what gives the MoE its capacity.

    Output shape: (B, T) — per-timestep RUL *logit* (NOT yet squashed to [0,1]).
    The squashing is done at the model level after expert mixing.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, n_heads: int = 4):
        super().__init__()
        assert d_model % n_heads == 0
        self.mlstm = mLSTMLayer(d_model, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(),
            nn.Linear(2 * d_model, d_model), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.mlstm(x)                          # (B, T, d_model)
        a, _ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h))
        h = h + a
        h = h + self.ff(self.ln2(h))
        return self.head(h).squeeze(-1)            # (B, T) RUL logits


# ---------------------------------------------------------------------------
# 3. Physics-informed router
# ---------------------------------------------------------------------------
class PhysicsInformedRouter(nn.Module):
    """
    Produces calibrated soft phase probabilities at every timestep from
    physics-informed engineered features:

        - rms_slope        : rolling slope of HI/RMS (drift indicator)
        - rms_curv         : rolling curvature of HI (acceleration -> pre-failure)
        - kurt             : current kurtosis (impulsiveness)
        - kurt_slope       : rolling slope of kurtosis (early-fault indicator)
        - rms_zscore       : (HI_t - mean(HI_healthy)) / std(HI_healthy)

    The router output IS the model's interpretability surface: at any
    timestep we read off P(healthy), P(wear), P(pre-failure).

    IMPORTANT: assumes feature index `hi_index` is RMS/HI and
    `kurt_index` is kurtosis. These defaults match the standard 16-feature
    HI vector defined in the dissertation §6.1: index 0 = RMS, index 2 =
    kurtosis. Override if your HI layout differs.
    """

    def __init__(self, hi_index: int = 0, kurt_index: int = 2,
                 healthy_window: int = 8, slope_window: int = 5):
        super().__init__()
        self.hi_index = hi_index
        self.kurt_index = kurt_index
        self.healthy_window = healthy_window
        self.slope_window = slope_window

        # 5 engineered features -> 3 phase logits
        self.mlp = nn.Sequential(
            nn.Linear(5, 32), nn.GELU(),
            nn.Linear(32, 32), nn.GELU(),
            nn.Linear(32, 3),
        )

    @staticmethod
    def _rolling_slope(x: torch.Tensor, win: int = 5) -> torch.Tensor:
        """
        x: (B, T) -> (B, T) least-squares slope of last `win` points.
        Uses 1-D conv with kernel = idx / sum(idx**2), where
        idx = [-2,-1,0,1,2] for win=5.  This is the closed-form OLS
        slope for a unit-spaced regression.
        """
        B, T = x.shape
        pad = win // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")  # (B,1,T+2p)
        idx = torch.arange(win, device=x.device, dtype=x.dtype) - (win - 1) / 2
        denom = (idx ** 2).sum().clamp(min=1e-6)
        kernel = (idx / denom).view(1, 1, win)
        slope = F.conv1d(x_pad, kernel)                              # (B,1,T)
        return slope.squeeze(1)

    def compute_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> (B, T, 5)."""
        B, T, _ = x.shape
        rms = x[:, :, self.hi_index]
        kurt = x[:, :, self.kurt_index]

        rms_slope = self._rolling_slope(rms, win=self.slope_window)
        rms_curv = self._rolling_slope(rms_slope, win=self.slope_window)
        kurt_slope = self._rolling_slope(kurt, win=self.slope_window)

        # Healthy baseline = mean/std of first `healthy_window` steps
        h = min(self.healthy_window, T)
        rms_baseline = rms[:, :h].mean(dim=1, keepdim=True)
        # Use unbiased=False to avoid NaN when h==1
        rms_std = rms[:, :h].std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        rms_zscore = (rms - rms_baseline) / rms_std

        feats = torch.stack(
            [rms_slope, rms_curv, kurt, kurt_slope, rms_zscore],
            dim=-1,
        )
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.compute_features(x)
        logits = self.mlp(feats)                                     # (B,T,3)
        return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# 4. PhaseMoE-xLSTM full model
# ---------------------------------------------------------------------------
@dataclass
class PhaseMoEConfig:
    n_features: int = 16
    d_model: int = 128
    n_phases: int = 3
    dropout: float = 0.1
    hi_index: int = 0
    kurt_index: int = 2
    healthy_window: int = 8
    phase_names: Tuple[str, ...] = ("Healthy", "Wear", "Pre-Failure")


class PhaseMoExLSTM(nn.Module):
    """
    Full PhaseMoE-xLSTM model.

    Forward returns dict with:
      rul           : (B,)         RUL at the last timestep, in [0,1]
      rul_per_step  : (B, T)       per-timestep mixed RUL, in [0,1]
      phase_probs   : (B, T, 3)    per-step soft phase assignment
      per_expert_rul: (B, T, 3)    per-expert RUL prediction (post-sigmoid)
    """

    def __init__(self, cfg: PhaseMoEConfig):
        super().__init__()
        self.cfg = cfg

        self.in_proj = nn.Linear(cfg.n_features, cfg.d_model)

        self.experts = nn.ModuleList([
            PhaseExpert(cfg.d_model, cfg.dropout)
            for _ in range(cfg.n_phases)
        ])

        self.router = PhysicsInformedRouter(
            hi_index=cfg.hi_index,
            kurt_index=cfg.kurt_index,
            healthy_window=cfg.healthy_window,
        )

        # Learnable temperature for the final sigmoid.
        # We parameterize as raw_param and use softplus to keep T > 0,
        # avoiding accidental sigmoid inversion.
        self._raw_temperature = nn.Parameter(torch.tensor(0.5))

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self._raw_temperature) + 1e-3

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, F_ = x.shape
        assert F_ == self.cfg.n_features, \
            f"Expected {self.cfg.n_features} features, got {F_}"

        # Phase probs from physics-informed router (sees RAW HI features)
        phase_probs = self.router(x)                                 # (B,T,3)

        # Project HI features to d_model for the experts
        h = self.in_proj(x)                                          # (B,T,d_model)

        # All experts in parallel; each returns (B, T) logits
        per_expert_logits = torch.stack(
            [expert(h) for expert in self.experts],
            dim=-1,
        )                                                            # (B,T,3)

        # Mixture: weighted sum of expert logits, then sigmoid
        # (Mixing in logit space is more stable than mixing post-sigmoid)
        T_temp = self.temperature
        mixed_logit = (per_expert_logits * phase_probs).sum(dim=-1)  # (B,T)
        rul_per_step = torch.sigmoid(mixed_logit / T_temp)           # (B,T)
        per_expert_rul = torch.sigmoid(per_expert_logits / T_temp)   # (B,T,3)

        return {
            "rul": rul_per_step[:, -1],
            "rul_per_step": rul_per_step,
            "phase_probs": phase_probs,
            "per_expert_rul": per_expert_rul,
        }

    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns dissertation-ready interpretability outputs."""
        out = self.forward(x)
        return {
            "rul": out["rul"].cpu(),
            "rul_per_step": out["rul_per_step"].cpu(),
            "phase_probs": out["phase_probs"].cpu(),
            "phase_names": list(self.cfg.phase_names),
            "per_expert_rul": out["per_expert_rul"].cpu(),
            "router_features": self.router.compute_features(x).cpu(),
        }


# ---------------------------------------------------------------------------
# 5. Physics-informed loss
# ---------------------------------------------------------------------------
class PhysicsInformedLoss(nn.Module):
    """
    Total loss = MSE + λ_mono · L_mono + λ_paris · L_paris + λ_phase · L_phase

    L_mono : penalize positive jumps in predicted RUL (RUL must not increase)
             L_mono = mean(ReLU(rul[t+1] - rul[t]) ** 2)

    L_paris: penalize positive 2nd derivative in last 30% of window
             (Paris-law-style late-life acceleration -> RUL is concave-down)
             L_paris = mean(ReLU(d2_rul on last 30%) ** 2)

    L_phase: cross-entropy on phase router output, with self-supervised
             targets derived from ground-truth RUL bins:
                 y >  healthy_threshold    -> Healthy (0)
                 prefailure_thr < y <= healthy_thr -> Wear (1)
                 y <= prefailure_threshold -> Pre-Failure (2)

             This gives the router a teacher signal grounded in the
             actual bearing labels rather than asking the network to
             discover phases from scratch.
    """

    def __init__(self,
                 lambda_mono: float = 0.1,
                 lambda_paris: float = 0.05,
                 lambda_phase: float = 0.1,
                 healthy_threshold: float = 0.7,
                 prefailure_threshold: float = 0.3):
        super().__init__()
        self.lambda_mono = lambda_mono
        self.lambda_paris = lambda_paris
        self.lambda_phase = lambda_phase
        self.healthy_threshold = healthy_threshold
        self.prefailure_threshold = prefailure_threshold

    def _phase_targets(self, y: torch.Tensor) -> torch.Tensor:
        """y: (B, T) RUL in [0,1] -> (B, T) phase labels in {0,1,2}."""
        labels = torch.full(y.shape, 1, dtype=torch.long, device=y.device)
        labels[y > self.healthy_threshold] = 0
        labels[y <= self.prefailure_threshold] = 2
        return labels

    def forward(self, outputs: Dict[str, torch.Tensor],
                y_per_step: torch.Tensor) -> Dict[str, torch.Tensor]:
        rul_pred = outputs["rul_per_step"]
        phase_probs = outputs["phase_probs"]

        # Main MSE across all timesteps (denser supervision than last-only)
        loss_mse = F.mse_loss(rul_pred, y_per_step)

        # Monotonicity
        diffs = rul_pred[:, 1:] - rul_pred[:, :-1]
        loss_mono = F.relu(diffs).pow(2).mean()

        # Paris-law concavity in last 30%
        T = rul_pred.size(1)
        late_start = int(T * 0.7)
        late = rul_pred[:, late_start:]
        if late.size(1) >= 3:
            d2 = late[:, 2:] - 2 * late[:, 1:-1] + late[:, :-2]
            loss_paris = F.relu(d2).pow(2).mean()
        else:
            loss_paris = rul_pred.new_zeros(())

        # Phase consistency (self-supervised CE)
        phase_targets = self._phase_targets(y_per_step)
        log_probs = torch.log(phase_probs.clamp(min=1e-8))
        loss_phase = F.nll_loss(
            log_probs.reshape(-1, 3),
            phase_targets.reshape(-1),
        )

        total = (loss_mse
                 + self.lambda_mono * loss_mono
                 + self.lambda_paris * loss_paris
                 + self.lambda_phase * loss_phase)

        return {
            "total": total,
            "mse": loss_mse.detach(),
            "mono": loss_mono.detach(),
            "paris": loss_paris.detach(),
            "phase": loss_phase.detach(),
        }


# ---------------------------------------------------------------------------
# 6. Split-conformal prediction wrapper
# ---------------------------------------------------------------------------
class ConformalPhaseMoE(nn.Module):
    """
    Wraps a trained PhaseMoExLSTM with split-conformal prediction
    (Vovk et al., 2005; Angelopoulos & Bates, 2023).

    Usage:
        model.eval()
        wrapper = ConformalPhaseMoE(model)
        wrapper.calibrate(calib_loader, alpha=0.1)        # 90% intervals
        out = wrapper.predict_with_interval(x)
        # out['rul'], out['lower'], out['upper']

    Coverage guarantee:
        For exchangeable calibration + test data, the interval contains
        the true RUL with probability >= 1 - alpha.

    CAVEAT for bearing data: the standard conformal exchangeability
    assumption is violated when calibration and test sets contain
    DIFFERENT BEARINGS (different operating conditions, different wear
    rates). The coverage guarantee then becomes APPROXIMATE rather than
    exact. In the dissertation, report empirical coverage on held-out
    bearings as a sanity check; if observed coverage drifts more than
    a few percent from nominal, switch to Mondrian or weighted conformal.
    """

    def __init__(self, model: PhaseMoExLSTM):
        super().__init__()
        self.model = model
        self.q_hat: Optional[float] = None
        self.alpha: Optional[float] = None

    @torch.no_grad()
    def calibrate(self, calib_loader, alpha: float = 0.1,
                  device: Optional[str] = None) -> float:
        """
        calib_loader yields (x, y_last) batches where y_last is the
        ground-truth RUL at the last timestep of the input window.
        """
        self.alpha = alpha
        self.model.eval()
        if device is None:
            device = next(self.model.parameters()).device

        residuals: List[float] = []
        for x, y_last in calib_loader:
            x = x.to(device)
            y_last = y_last.to(device)
            out = self.model(x)
            r = (y_last - out["rul"]).abs()
            residuals.extend(r.cpu().tolist())

        residuals = np.asarray(residuals)
        n = len(residuals)
        if n < 1:
            raise ValueError("Calibration set is empty.")
        # Conformal quantile w/ finite-sample correction (Angelopoulos & Bates §3)
        q_level = math.ceil((n + 1) * (1 - alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(residuals, q_level, method="higher"))
        return self.q_hat

    @torch.no_grad()
    def predict_with_interval(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.q_hat is None:
            raise RuntimeError("Call .calibrate() before .predict_with_interval().")
        out = self.model(x)
        rul = out["rul"]
        lower = (rul - self.q_hat).clamp(0.0, 1.0)
        upper = (rul + self.q_hat).clamp(0.0, 1.0)
        return {
            "rul": rul,
            "lower": lower,
            "upper": upper,
            "phase_probs": out["phase_probs"],
            "interval_half_width": self.q_hat,
        }
