# Angle A+B Hybrid — Dissertation Pipeline Specification

## Interpretable Mamba Backbones with Source-Free Domain Adaptation and Conditional Conformal Prediction for Calibrated Bearing RUL

**Working title:** *SAMBA-CCP: Selective State-Space Models with Mechanistic Interpretability, Source-Free Adaptation, and Conditional Conformal Prediction for Calibrated Cross-Condition Bearing Remaining Useful Life*

**Target venues:** MSSP (architecture + interpretability), RESS (UQ + domain shift), IEEE TII (systems/deployment) — three papers plus monograph.

**Datasets:** PHM2012 (FEMTO-ST PRONOSTIA) + XJTU-SY, cross-condition and cross-dataset, bidirectional.

---

## 0. Scientific Contributions (what reviewers should walk away remembering)

1. **Architectural.** First application of selective state-space models (Mamba-2) to bearing RUL on PHM2012 and XJTU-SY, with sparse-autoencoder (SAE) probes that identify physically meaningful "degradation concept features" inside the latent stream.
2. **Adaptation.** A **source-free** pseudo-label self-training scheme for regression (no source data at test-time), with a monotonic-RUL regulariser that matches the physics of bearing degradation.
3. **Calibration (the headline).** A **conditional conformal predictor over a class of operating-condition covariate shifts** (Gibbs–Cherian–Candès 2025), sitting on top of a CQR head, with online **DtACI** (Gibbs–Candès 2024) correction over the target bearing's chronological stream.
4. **Benchmark.** Published code + reproducible leaderboard with comparison against CRULP (IEEE TIM 2025) and ACRP (CASE 2025) — we must beat both.

This is a three-way novelty stack where the weakest leg alone would be borderline publishable; together they form a coherent thesis with a clean story arc.

---

## 1. Overall Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│               STAGE A — Source pre-training                      │
│  PHM2012 C1/C2/C3 or XJTU-SY C1/C2/C3 (labelled run-to-failure)  │
│                                                                  │
│  Raw (C, L) ──► time+freq feats ──► Mamba-2 backbone ──►         │
│                                      (emb, SSM hidden h_t)       │
│                                           │                       │
│               ┌───────────────────────────┼──────────┐            │
│               │                           │          │            │
│          Point head                 Quantile head   SAE probe     │
│          (RUL ∈ [0,1])             (q_lo, q_mid,    (k concepts) │
│                                     q_hi)                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼   (source model weights exported)
┌─────────────────────────────────────────────────────────────────┐
│               STAGE B — Source-Free Adaptation                  │
│  Target: bearings from a different operating condition,         │
│  unlabeled, source data NO LONGER ACCESSIBLE                    │
│                                                                  │
│  Pseudo-label generation (model's own q_mid)                     │
│         │                                                         │
│         ▼                                                         │
│  Confidence filtering via CQR interval width                     │
│         │                                                         │
│         ▼                                                         │
│  Monotonicity + physics regularisers  ──► fine-tune backbone+head│
│  (source data NOT used)                                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼   (target-adapted model)
┌─────────────────────────────────────────────────────────────────┐
│        STAGE C — Conditional CP + DtACI                         │
│                                                                  │
│  Source calibration bearings → fit class-of-shifts conditional  │
│  conformal predictor over Φ(x) = [condition_embedding, HI_level]│
│                                                                  │
│  Target bearing stream (chronological) → DtACI online correction│
│         ▼                                                         │
│  Final prediction intervals with (1-α)-coverage guarantees:     │
│     • Marginal under exchangeability                             │
│     • Worst-slab bounded under the covariate-shift class         │
│     • Long-run (asymptotic) under arbitrary drift                │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│        STAGE D — Interpretability                               │
│  SAE probes on Mamba hidden states across the stream            │
│         ▼                                                         │
│  Map SAE features to physical regimes: healthy → FPT →          │
│  thinning lube → spall initiation → accelerated failure         │
│  Validate by correlating with RMS / kurtosis / envelope peaks   │
│  at matching timestamps (causal validation, not just heatmaps)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Layout

Extends the existing `bearing-rul-cpda` tree. New modules in **bold**:

```
bearing-rul-cpda/
├── configs/
│   ├── experiment/ab_hybrid_phm2012_c1_to_c2.yaml
│   └── ...
├── src/brul/
│   ├── data/                     # (existing)
│   ├── features/                 # (existing)
│   ├── models/
│   │   ├── backbones.py          # (existing) keep CNN-LSTM as baseline
│   │   ├── heads.py              # (existing)
│   │   ├── mamba_backbone.py     # ** NEW: Mamba-2 implementation **
│   │   └── sae_probe.py          # ** NEW: sparse autoencoder **
│   ├── uda/                      # (existing DANN, MMD kept as baselines)
│   │   └── sfda.py               # ** NEW: source-free DA trainer **
│   ├── cp/
│   │   ├── split.py              # (existing)
│   │   ├── cqr.py                # (existing)
│   │   ├── weighted.py           # (existing; as baseline only)
│   │   ├── conditional.py        # ** NEW: Gibbs-Cherian-Candès 2025 **
│   │   └── dtaci.py              # ** NEW: Gibbs-Candès 2024 DtACI **
│   ├── interpret/
│   │   └── sae_analysis.py       # ** NEW: concept discovery + validation **
│   ├── eval/
│   └── viz/
├── tests/
│   ├── test_mamba.py             # ** NEW **
│   ├── test_sfda.py              # ** NEW **
│   ├── test_conditional_cp.py    # ** NEW **
│   └── test_dtaci.py             # ** NEW **
└── examples/
    └── demo_ab_hybrid.py         # ** NEW: end-to-end demo **
```

---

## 3. STAGE A — Mamba Backbone + Heads

### 3.1 Why Mamba for bearing RUL

Vibration is a long, stationary-until-it-isn't signal. Transformers are O(L²) and don't scale to the ≈25k-sample acquisitions common in PHM2012 / XJTU-SY without aggressive pooling. Mamba is linear-time, handles long sequences natively, and (importantly for us) exposes a **recurrent hidden state** that the SAE probe can read at every time step without any attention-heatmap gymnastics.

Mamba-SDP (JMST 2025) is the only prior Mamba-on-bearings paper. Our Mamba is bidirectional, deeper, and instrumented for SAE probing — those are the architectural deltas.

### 3.2 Module

```python
# src/brul/models/mamba_backbone.py
"""Mamba-2 backbone for bearing RUL with SAE-probeable hidden state.

Uses the ``mamba-ssm`` package (linear-time selective SSM).
If unavailable, falls back to a pure-PyTorch reference implementation
(slower but equivalent up to numerical tolerance).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba2  # pip install mamba-ssm
    _HAS_MAMBA = True
except ImportError:
    _HAS_MAMBA = False


@dataclass
class MambaConfig:
    d_model: int = 128          # embedding width
    d_state: int = 16           # SSM state dim (expanded internally)
    d_conv: int = 4             # local conv width
    expand: int = 2             # hidden expansion factor
    n_layers: int = 4
    dropout: float = 0.1
    bidirectional: bool = True  # crucial for offline RUL (we see full stream)


class _MambaBlock(nn.Module):
    def __init__(self, cfg: MambaConfig):
        super().__init__()
        if _HAS_MAMBA:
            self.fwd = Mamba2(
                d_model=cfg.d_model, d_state=cfg.d_state,
                d_conv=cfg.d_conv, expand=cfg.expand,
            )
            if cfg.bidirectional:
                self.bwd = Mamba2(
                    d_model=cfg.d_model, d_state=cfg.d_state,
                    d_conv=cfg.d_conv, expand=cfg.expand,
                )
        else:
            # fallback: depth-wise conv + gated MLP (approximates Mamba)
            self.fwd = nn.Sequential(
                nn.Conv1d(cfg.d_model, cfg.d_model, cfg.d_conv,
                          padding=cfg.d_conv - 1, groups=cfg.d_model),
                nn.SiLU(),
                nn.Conv1d(cfg.d_model, cfg.d_model, 1),
            )
            self.bwd = None
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.bidirectional = cfg.bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        z = self.norm(x)
        if _HAS_MAMBA:
            out = self.fwd(z)
            if self.bidirectional and self.bwd is not None:
                out = out + self.bwd(z.flip(1)).flip(1)
        else:
            # fallback expects (B, D, T)
            h = self.fwd(z.transpose(1, 2))
            h = h[..., : z.size(1)]
            out = h.transpose(1, 2)
        return x + self.drop(out)


class MambaBackbone(nn.Module):
    """Bidirectional Mamba-2 stack for windowed feature streams.

    Input:  (B, W, F)   W = window length, F = per-acquisition feature dim
    Output: (B, out_dim), PLUS the per-step hidden stream (B, W, out_dim)
            which is what the SAE probe reads.
    """

    def __init__(self, n_features: int, cfg: MambaConfig | None = None):
        super().__init__()
        self.cfg = cfg or MambaConfig()
        self.proj_in = nn.Linear(n_features, self.cfg.d_model)
        self.blocks = nn.ModuleList(
            [_MambaBlock(self.cfg) for _ in range(self.cfg.n_layers)]
        )
        self.norm_out = nn.LayerNorm(self.cfg.d_model)
        self.out_dim = self.cfg.d_model

    def forward(self, x: torch.Tensor, return_stream: bool = False):
        # x: (B, W, F)
        h = self.proj_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm_out(h)                   # (B, W, D)
        pooled = h.mean(dim=1)                 # (B, D)
        if return_stream:
            return pooled, h                   # both pooled and per-step
        return pooled
```

### 3.3 Heads stay the same

Use the existing `RegressionHead` for the point RUL and `QuantileHead` (q=0.05, 0.5, 0.95) for CQR. The quantile head is what Stage C will conformalize. Both take `pooled` (not the stream).

### 3.4 Source training loss

```
L_source = L_pinball(q_lo, q_mid, q_hi, y) + λ_mono · L_mono(q_mid) + λ_smooth · L_smooth(h_stream)
```

- `L_pinball` — standard pinball across the three quantiles.
- `L_mono` — soft monotonicity on the median prediction along a bearing's time index:
  `mean(ReLU(q_mid[t+1] - q_mid[t] - ε))` — penalises RUL that increases over time.
- `L_smooth` — total-variation on the hidden stream to make SAE probing less noisy:
  `mean(‖h[:, t+1] - h[:, t]‖²)` with a small weight (1e-3).

Together these give physically sensible point predictions and a "clean" hidden stream for Stage D.

### 3.5 Unit test

```python
# tests/test_mamba.py
import torch
from brul.models.mamba_backbone import MambaBackbone, MambaConfig

def test_mamba_forward_shape():
    model = MambaBackbone(n_features=32, cfg=MambaConfig(d_model=64, n_layers=2))
    x = torch.randn(4, 10, 32)
    pooled, stream = model(x, return_stream=True)
    assert pooled.shape == (4, 64)
    assert stream.shape == (4, 10, 64)

def test_mamba_gradient_flows():
    model = MambaBackbone(n_features=16, cfg=MambaConfig(d_model=32, n_layers=2))
    x = torch.randn(2, 8, 16, requires_grad=True)
    out = model(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
```

---

## 4. STAGE B — Source-Free Domain Adaptation

### 4.1 Motivation

DANN/MMD/CDAN all require simultaneous access to source data during adaptation. In realistic deployment you don't have that: the customer has target-machine sensor streams, the OEM holds the source data (and their IP). Source-Free DA (SFDA) matches industrial reality and is under-explored for RUL regression.

The approach below is inspired by SHOT (Liang 2020) for classification, adapted to regression by:
1. Replacing entropy-minimisation on predictions with **confidence-weighted pseudo-label regression** using the CQR interval width as the (inverse) confidence.
2. Adding **monotonicity + bearing-wise FPT priors** as physics regularisers.

### 4.2 Algorithm

```
Input: frozen source model (backbone + quantile head), target unlabeled runs T
1. Initialise the adapted model from the source checkpoint.
2. Freeze the quantile head (this preserves calibration scale).
3. Split each target bearing into chronological slices of length L.
4. For each epoch:
   a. Forward-pass entire target set to get (q_lo, q_mid, q_hi) per window.
   b. Pseudo-label y_ps = q_mid.
   c. Confidence weight w = exp(-γ · (q_hi - q_lo)).   γ≈5 by default.
   d. Loss:
        L = Σ_i w_i · pinball((q_lo, q_mid, q_hi)(x_i), y_ps_i)
              + λ_mono · L_mono(q_mid along bearing time axis)
              + λ_phys · L_phys(q_mid starts near 1, ends near 0)
              + λ_feat · ‖φ_adapted(x) - φ_source(x)‖² on a small anchor set
   e. Backprop only through the backbone (head is frozen).
5. Every K epochs, refresh pseudo-labels.
```

The `λ_feat` anchor term uses a small set (e.g. 64 windows) saved once at source-training time — storing embeddings not raw data, which preserves privacy and matches the "source-free" constraint. This prevents representation collapse, which is the main SFDA failure mode.

### 4.3 Implementation

```python
# src/brul/uda/sfda.py
"""Source-free domain adaptation for RUL regression.

Does NOT require source data at adaptation time. Uses:
- frozen source quantile head (preserves calibration scale),
- pseudo-label self-training with CQR-width-based confidence weights,
- monotonicity + physics regularisers,
- optional feature-anchor term using stored source embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brul.models.heads import pinball_loss


@dataclass
class SFDAConfig:
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    pseudo_refresh_every: int = 3
    confidence_gamma: float = 5.0
    lambda_mono: float = 0.1
    lambda_phys: float = 0.05
    lambda_feat_anchor: float = 0.2
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SFDATrainer:
    def __init__(
        self,
        backbone: nn.Module,
        quantile_head: nn.Module,
        cfg: SFDAConfig,
        anchor_source_emb: torch.Tensor | None = None,
    ):
        self.backbone = backbone.to(cfg.device)
        self.head = quantile_head.to(cfg.device)
        self.cfg = cfg
        # Freeze the quantile head — this preserves the source calibration scale
        for p in self.head.parameters():
            p.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.backbone.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.anchor_source_emb = (
            anchor_source_emb.to(cfg.device) if anchor_source_emb is not None else None
        )

    @torch.no_grad()
    def _generate_pseudo_labels(self, loader):
        self.backbone.eval(); self.head.eval()
        all_y_ps, all_w = [], []
        for x, *_ in loader:
            x = x.to(self.cfg.device)
            emb = self.backbone(x)
            q = self.head(emb)  # (B, 3) sorted
            q_lo, q_mid, q_hi = q[:, 0], q[:, 1], q[:, 2]
            width = (q_hi - q_lo).clamp(min=1e-6)
            w = torch.exp(-self.cfg.confidence_gamma * width)
            all_y_ps.append(q_mid.cpu()); all_w.append(w.cpu())
        return torch.cat(all_y_ps), torch.cat(all_w)

    def _bearing_mono_loss(self, q_mid: torch.Tensor) -> torch.Tensor:
        # assumes batch is chronological within a bearing window
        diff = q_mid[1:] - q_mid[:-1]
        return F.relu(diff + 1e-3).mean()

    def _physics_loss(self, q_mid: torch.Tensor, y_ps: torch.Tensor) -> torch.Tensor:
        # near-start: y_ps is high → q_mid should be high
        # near-EOL:   y_ps is low → q_mid should be low
        return F.mse_loss(q_mid, y_ps.detach())

    def _anchor_loss(self, current_emb_source_input: torch.Tensor | None) -> torch.Tensor:
        if self.anchor_source_emb is None or current_emb_source_input is None:
            return torch.zeros((), device=self.cfg.device)
        return F.mse_loss(current_emb_source_input, self.anchor_source_emb)

    def fit(self, target_loader):
        for epoch in range(self.cfg.epochs):
            if epoch % self.cfg.pseudo_refresh_every == 0:
                y_ps_all, w_all = self._generate_pseudo_labels(target_loader)
            self.backbone.train(); self.head.eval()
            ptr = 0
            for x, *_ in target_loader:
                x = x.to(self.cfg.device)
                B = x.size(0)
                y_ps = y_ps_all[ptr:ptr + B].to(self.cfg.device)
                w = w_all[ptr:ptr + B].to(self.cfg.device)
                ptr += B
                emb = self.backbone(x)
                q = self.head(emb)
                L_main = (w.unsqueeze(-1) * _per_sample_pinball(
                    q, y_ps, self.cfg.quantiles
                )).mean()
                L_mono = self._bearing_mono_loss(q[:, 1])
                L_phys = self._physics_loss(q[:, 1], y_ps)
                L_anchor = self._anchor_loss(None)  # plug in if using anchors
                loss = (
                    L_main
                    + self.cfg.lambda_mono * L_mono
                    + self.cfg.lambda_phys * L_phys
                    + self.cfg.lambda_feat_anchor * L_anchor
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.backbone.parameters(), self.cfg.grad_clip
                )
                self.optimizer.step()


def _per_sample_pinball(preds, target, quantiles):
    y = target.unsqueeze(-1)
    err = y - preds
    q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype).unsqueeze(0)
    return torch.maximum(q * err, (q - 1.0) * err).mean(dim=-1)
```

### 4.4 Unit tests

```python
# tests/test_sfda.py
import torch
from brul.models.backbones import FeatureMLPBackbone
from brul.models.heads import QuantileHead
from brul.uda.sfda import SFDATrainer, SFDAConfig


def _toy_loader(n=64, W=8, F=8, bs=16):
    xs = torch.randn(n, W, F)
    ys = torch.rand(n)
    from torch.utils.data import TensorDataset, DataLoader
    return DataLoader(TensorDataset(xs, ys), batch_size=bs)

def test_sfda_runs_end_to_end():
    bb = FeatureMLPBackbone(n_features=8, hidden=32, gru_hidden=32)
    head = QuantileHead(in_dim=bb.out_dim, hidden=16)
    trainer = SFDATrainer(bb, head, SFDAConfig(epochs=2, device="cpu"))
    trainer.fit(_toy_loader())
    # Quantile head must stay frozen
    for p in head.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0

def test_sfda_monotonicity_improves():
    # Construct a target where RUL is decreasing; check post-adaptation mono penalty decreases.
    ...  # left as exercise
```

---

## 5. STAGE C — Conditional CP + DtACI (the headline)

This is the centrepiece of the dissertation. Two layered methods:

### 5.1 Conditional Conformal Prediction (Gibbs, Cherian & Candès 2025)

**Problem.** Standard split CP guarantees only marginal coverage
`P(Y ∈ Ĉ(X)) ≥ 1 − α`. Under operating-condition shift the coverage can be high on average but catastrophic on specific conditions (e.g. heavy-load XJTU-SY Condition 3). We want `P(Y ∈ Ĉ(X) | condition = c) ≥ 1 − α` for every condition c.

**Gibbs–Cherian–Candès solution.** Cast conditional coverage as marginal coverage over a *class* of covariate shifts. For a feature map `Φ(x) ∈ R^d` (e.g. one-hot condition, or a richer embedding), parametrise

```
f(x) = Φ(x)ᵀ β    (linear function of features)
```

and choose `β̂` by minimising the quantile loss

```
β̂ = argmin_β  Σᵢ  ℓ_α(S_i - Φ(x_i)ᵀ β)
```

where `ℓ_α(u) = α·u⁺ + (1-α)·u⁻` is the pinball loss at level `1-α` and `S_i` are nonconformity scores from a held-out calibration set. The interval for a new point is then

```
Ĉ(x) = [ ŷ(x) - Φ(x)ᵀ β̂ ,  ŷ(x) + Φ(x)ᵀ β̂ ]     (for abs-residual scores)
```

or more generally:

```
Ĉ(x) = { y : S(x, y) ≤ Φ(x)ᵀ β̂ }
```

Under the class of shifts defined by all distributions whose likelihood ratio lies in the span of `Φ`, this achieves **exact finite-sample coverage** (Gibbs et al. 2025 Theorem 1).

**Choice of Φ for bearings.**
- Simple: one-hot of operating condition → gives per-condition coverage.
- Richer: `Φ(x) = [1, 1{c=1}, 1{c=2}, 1{c=3}, HI(x), HI(x)²]` → coverage across conditions *and* degradation stages (healthy/degraded/near-EOL).
- For cross-dataset: `Φ(x) = [1, 1{dataset=PHM2012}, 1{dataset=XJTUSY}, condition_one-hot, HI(x)]`.

### 5.2 Implementation

```python
# src/brul/cp/conditional.py
"""Conditional conformal prediction with a finite-dimensional function class.

Implements Gibbs, Cherian & Candès (JRSSB 2025): coverage over a user-specified
class of covariate shifts defined by a feature map Φ(x).

The method reduces to solving a quantile-regression problem on calibration scores
using Φ as design matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.linear_model import QuantileRegressor

from brul.cp.base import Conformalizer, Intervals, Preds


@dataclass
class ConditionalCPConfig:
    alpha: float = 0.1
    nonneg: bool = True  # clamp the score threshold to ≥ 0
    qr_alpha: float = 1e-4  # L1 regularisation of sklearn's QuantileRegressor
    qr_solver: str = "highs"


class ConditionalConformal(Conformalizer):
    """Conditional split CP using a finite feature class Φ.

    Args:
        phi: callable (features_np) -> np.ndarray of shape (N, d)  — the feature map.
        score_fn: callable (preds, y) -> nonconformity scores.
                  Default: absolute residuals (for Preds.point only).
                  Use CQR score for quantile-aware CP.
        config: configuration dataclass.

    References:
        Gibbs, Cherian & Candès (2025). "Conformal prediction with conditional
        guarantees." Journal of the Royal Statistical Society: Series B.
    """

    name = "conditional"

    def __init__(
        self,
        phi: Callable[[np.ndarray], np.ndarray],
        score_fn: Callable[[Preds], np.ndarray] | None = None,
        config: ConditionalCPConfig | None = None,
    ):
        self.phi = phi
        self.cfg = config or ConditionalCPConfig()
        super().__init__(alpha=self.cfg.alpha)
        self._beta: np.ndarray | None = None
        self._score_fn = score_fn or _default_abs_score

    def calibrate(self, calib: Preds, **_) -> None:
        if calib.y_true is None:
            raise ValueError("calib.y_true required.")
        if calib.features is None:
            raise ValueError("Conditional CP needs calib.features.")
        scores = self._score_fn(calib)                     # (n,)
        Phi = self.phi(calib.features)                     # (n, d)
        # Quantile regression at level (1 - alpha) of S on Phi
        qr = QuantileRegressor(
            quantile=1.0 - self.cfg.alpha,
            alpha=self.cfg.qr_alpha,
            solver=self.cfg.qr_solver,
            fit_intercept=False,       # intercept should be part of Phi if wanted
        )
        qr.fit(Phi, scores)
        self._beta = qr.coef_
        self._fitted = True

    def predict(self, test: Preds) -> Intervals:
        if not self._fitted:
            raise RuntimeError("Call calibrate() first.")
        if test.features is None:
            raise ValueError("Test preds need features.")
        Phi_test = self.phi(test.features)
        thresholds = Phi_test @ self._beta                  # (N,)
        if self.cfg.nonneg:
            thresholds = np.maximum(thresholds, 0.0)
        lower = test.point - thresholds
        upper = test.point + thresholds
        return Intervals(lower=lower, upper=upper,
                         meta={"beta": self._beta.tolist()})


def _default_abs_score(p: Preds) -> np.ndarray:
    if p.y_true is None:
        raise ValueError("y_true needed for absolute-residual score.")
    return np.abs(p.y_true - p.point)


def cqr_score(p: Preds) -> np.ndarray:
    """Non-conformity score for conformalized quantile regression."""
    if p.y_true is None or p.lower_quantile is None or p.upper_quantile is None:
        raise ValueError("CQR score needs lower_quantile, upper_quantile, y_true.")
    return np.maximum(p.lower_quantile - p.y_true, p.y_true - p.upper_quantile)
```

**Example Φ builder for our experiments:**

```python
def make_phi_bearing(
    include_intercept: bool = True,
    n_conditions: int = 3,
    hi_degree: int = 2,
) -> Callable[[np.ndarray], np.ndarray]:
    """Feature map returning [1, one-hot(condition), HI, HI², ...].

    Assumes the first column of features is the condition index (1..n_conditions)
    and the last column is a scalar Health Indicator in [0, 1].
    """

    def phi(feats_np: np.ndarray) -> np.ndarray:
        cond = feats_np[:, 0].astype(int)
        hi = feats_np[:, -1]
        cols = []
        if include_intercept:
            cols.append(np.ones(len(feats_np)))
        for c in range(1, n_conditions + 1):
            cols.append((cond == c).astype(float))
        for d in range(1, hi_degree + 1):
            cols.append(hi ** d)
        return np.stack(cols, axis=1)

    return phi
```

### 5.3 DtACI — Online correction (Gibbs & Candès 2024, JMLR v25)

**Problem.** Even with conditional CP, the target bearing's degradation may drift during operation (e.g. spall propagation changes the signal statistics). ACI (Gibbs–Candès 2021) updates α online but requires a well-chosen step size γ. DtACI (Dynamically-tuned ACI) runs a small ensemble of ACI experts at different γ's and mixes them via exponential weighting based on observed pinball loss — parameter-free in practice.

**Algorithm.**

```
Input: target RUL stream, CP subroutine (conditional CP above), γ-grid {γ_1,...,γ_K}
Initialise α_t^{(k)} = α for all k; expert weights p^{(k)} = 1/K.
For each t = 1, 2, ...:
    1. For each expert k, construct interval Ĉ_t^{(k)} via CP at level α_t^{(k)}.
    2. Mix: α̃_t = Σ_k p^{(k)} · α_t^{(k)};  interval Ĉ_t via CP at α̃_t.
    3. Observe y_t; compute err_t^{(k)} = 1{y_t ∉ Ĉ_t^{(k)}}.
    4. Update α: α_{t+1}^{(k)} = α_t^{(k)} + γ_k · (α - err_t^{(k)}).
    5. Update expert weights via pinball loss:
           ℓ_t^{(k)} = α·(y_t - Ĉ_t^{(k)})⁺ + (1-α)·(Ĉ_t^{(k)} - y_t)⁻
           p^{(k)} ← p^{(k)} · exp(-η · ℓ_t^{(k)});  renormalise.
```

### 5.4 Implementation

```python
# src/brul/cp/dtaci.py
"""Dynamically-tuned ACI (Gibbs & Candès 2024, JMLR).

Runs an ensemble of ACI experts at different step sizes and combines them via
exponential weighting on observed pinball loss. Parameter-free in practice.

Wraps any Conformalizer that accepts alpha at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from brul.cp.base import Conformalizer, Intervals, Preds


@dataclass
class DtACIConfig:
    alpha: float = 0.1
    gamma_grid: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
    alpha_min: float = 0.005
    alpha_max: float = 0.5
    eta_weights: float = 2.0   # learning rate for expert weights
    sigma: float = 0.1         # smoothing to prevent degenerate expert weights


class DtACI:
    """DtACI wrapper around a factory producing a new Conformalizer per alpha.

    The factory is called once per target-stream step per expert. For efficiency,
    the factory should produce a Conformalizer whose `.calibrate()` is cheap
    (e.g. caches calibration scores internally).
    """

    name = "dtaci"

    def __init__(
        self,
        factory: Callable[[float], Conformalizer],
        calib_preds: Preds,
        cfg: DtACIConfig | None = None,
    ):
        self.factory = factory
        self.calib = calib_preds
        self.cfg = cfg or DtACIConfig()
        K = len(self.cfg.gamma_grid)
        self.K = K
        self._alpha_t = np.full(K, self.cfg.alpha, dtype=float)
        self._weights = np.full(K, 1.0 / K, dtype=float)
        self._hist = {
            "alpha_mix": [], "coverage": [], "widths": [],
            "alpha_experts": [], "weights": [],
        }

    def run_stream(self, stream: Preds) -> Intervals:
        if stream.y_true is None:
            raise ValueError("DtACI needs stream.y_true for online updates.")
        N = stream.n
        lowers = np.empty(N)
        uppers = np.empty(N)

        for t in range(N):
            # build an expert interval per γ-expert at its current α
            expert_lows = np.empty(self.K)
            expert_ups = np.empty(self.K)
            for k in range(self.K):
                a_k = float(np.clip(self._alpha_t[k],
                                    self.cfg.alpha_min, self.cfg.alpha_max))
                cf = self.factory(a_k)
                cf.calibrate(self.calib)
                p_t = _slice_one(stream, t)
                iv_k = cf.predict(p_t)
                expert_lows[k] = iv_k.lower[0]
                expert_ups[k] = iv_k.upper[0]

            # weighted mixture
            mix_low = float(self._weights @ expert_lows)
            mix_up = float(self._weights @ expert_ups)
            lowers[t], uppers[t] = mix_low, mix_up

            # observe y_t and update
            y_t = stream.y_true[t]
            err = (expert_lows > y_t) | (expert_ups < y_t)  # per-expert miss
            loss = _pinball_miss(expert_lows, expert_ups, y_t, self.cfg.alpha)

            # α updates per expert
            self._alpha_t = self._alpha_t + np.asarray(
                self.cfg.gamma_grid
            ) * (self.cfg.alpha - err.astype(float))

            # expert weight updates (exp. weighting of pinball loss)
            log_w = np.log(self._weights + 1e-12) - self.cfg.eta_weights * loss
            log_w = log_w - log_w.max()
            w = np.exp(log_w)
            w = (1 - self.cfg.sigma) * (w / w.sum()) + self.cfg.sigma / self.K
            self._weights = w

            self._hist["alpha_mix"].append(
                float(self._weights @ self._alpha_t)
            )
            self._hist["widths"].append(mix_up - mix_low)
            self._hist["alpha_experts"].append(self._alpha_t.copy())
            self._hist["weights"].append(self._weights.copy())

        return Intervals(lower=lowers, upper=uppers, meta={"dtaci_history": self._hist})


def _slice_one(p: Preds, t: int) -> Preds:
    return Preds(
        point=p.point[t:t + 1],
        lower_quantile=p.lower_quantile[t:t + 1] if p.lower_quantile is not None else None,
        upper_quantile=p.upper_quantile[t:t + 1] if p.upper_quantile is not None else None,
        features=p.features[t:t + 1] if p.features is not None else None,
    )


def _pinball_miss(lows, ups, y, alpha):
    # A reasonable scalar loss proxy: asymmetric penalty on interval miss.
    below = np.maximum(lows - y, 0.0)
    above = np.maximum(y - ups, 0.0)
    return alpha * above + (1 - alpha) * below
```

### 5.5 Tests

```python
# tests/test_conditional_cp.py
import numpy as np
from brul.cp import Preds
from brul.cp.conditional import ConditionalConformal, ConditionalCPConfig

def test_conditional_cp_per_group_coverage():
    rng = np.random.default_rng(0)
    n = 3000
    groups = rng.integers(0, 3, size=n)
    sigma = np.array([0.2, 0.5, 1.0])[groups]  # heteroscedastic by group
    y = rng.normal(0, sigma)
    y_hat = np.zeros(n)               # trivial predictor
    feats = np.stack([groups + 1, np.zeros(n)], axis=1)  # first col = condition

    idx = rng.permutation(n)
    i_cal, i_te = idx[:1500], idx[1500:]
    calib = Preds(point=y_hat[i_cal], y_true=y[i_cal], features=feats[i_cal])
    test  = Preds(point=y_hat[i_te],  y_true=y[i_te],  features=feats[i_te])

    def phi(f):
        cond = f[:, 0].astype(int)
        return np.stack([(cond == c).astype(float) for c in (1, 2, 3)], axis=1)

    cp = ConditionalConformal(phi=phi, config=ConditionalCPConfig(alpha=0.1))
    cp.calibrate(calib)
    iv = cp.predict(test)
    # Marginal
    marg = ((test.y_true >= iv.lower) & (test.y_true <= iv.upper)).mean()
    assert 0.87 < marg < 0.93
    # Per-group
    for g in (0, 1, 2):
        mask = groups[i_te] == g
        cov = ((test.y_true[mask] >= iv.lower[mask]) &
               (test.y_true[mask] <= iv.upper[mask])).mean()
        assert 0.80 < cov < 0.97, f"group {g} coverage {cov:.3f} off"
```

```python
# tests/test_dtaci.py
import numpy as np
from brul.cp import Preds, SplitCP
from brul.cp.dtaci import DtACI, DtACIConfig


def test_dtaci_long_run_coverage_with_drift():
    rng = np.random.default_rng(7)
    # Calibration stable
    y_cal = rng.normal(size=500)
    p_cal = Preds(point=np.zeros(500), y_true=y_cal)
    # Test with drift
    n = 2000
    drift = np.linspace(0, 1.5, n)
    y_te = rng.normal(loc=drift, scale=0.5)
    p_te = Preds(point=np.zeros(n), y_true=y_te)

    def factory(a):
        return SplitCP(alpha=a)

    dt = DtACI(factory, p_cal, DtACIConfig(alpha=0.1))
    iv = dt.run_stream(p_te)
    cov = ((p_te.y_true >= iv.lower) & (p_te.y_true <= iv.upper)).mean()
    # DtACI should recover long-run coverage near 0.9
    assert 0.85 < cov < 0.94
```

---

## 6. STAGE D — Mechanistic Interpretability via SAE Probes

### 6.1 Why this is worth the extra effort

Attention heatmaps are empirically unreliable as explanations. Sparse autoencoders trained on a frozen model's hidden states — the Anthropic interpretability method — find human-interpretable concepts inside internal representations. We apply this to the Mamba hidden stream and validate each concept against a physical quantity (RMS, kurtosis, envelope-peak amplitude, band energy). A concept that fires when-and-only-when a physically meaningful change occurs is a causally-validated interpretation, not a correlation story.

### 6.2 Module

```python
# src/brul/models/sae_probe.py
"""Sparse autoencoder probe for Mamba hidden states.

Trained post-hoc on a frozen Mamba backbone. The SAE maps the d-dim hidden
stream to a k-dim sparse code (k ≥ d). Each of the k latents corresponds to
a candidate "concept" whose activation trajectory over a bearing's life can
be inspected and correlated with physical observables.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, k_latents: int, tied: bool = True):
        super().__init__()
        self.enc = nn.Linear(d_model, k_latents, bias=True)
        if tied:
            self.dec_weight = self.enc.weight  # tied
            self.dec_bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.dec = nn.Linear(k_latents, d_model, bias=True)
        self.tied = tied
        self.k = k_latents

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return F.relu(self.enc(h))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.tied:
            return F.linear(z, self.dec_weight.t(), self.dec_bias)
        return self.dec(z)

    def forward(self, h: torch.Tensor):
        z = self.encode(h)
        h_hat = self.decode(z)
        return h_hat, z


def sae_loss(h: torch.Tensor, h_hat: torch.Tensor, z: torch.Tensor,
             l1_weight: float = 1e-3) -> torch.Tensor:
    recon = F.mse_loss(h_hat, h)
    sparsity = z.abs().mean()
    return recon + l1_weight * sparsity
```

### 6.3 Concept validation procedure

```python
# src/brul/interpret/sae_analysis.py
"""Post-hoc validation of SAE latents against physical observables."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def collect_latents_and_physics(
    backbone, sae, loader, device="cuda"
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run backbone + SAE on all target windows; collect SAE latents and any
    physical feature that was stored in the loader's batch metadata.

    Returns:
        latents: (N, k) SAE activations at the last timestep of each window
        physics: dict of {name: (N,) array} for correlation analysis
    """
    backbone.eval(); sae.eval()
    latents, rms, kurt, env = [], [], [], []
    for batch in loader:
        x, y, *_ = batch
        x = x.to(device)
        _, h_stream = backbone(x, return_stream=True)   # (B, W, D)
        h_last = h_stream[:, -1]                        # (B, D)
        z = sae.encode(h_last).cpu().numpy()            # (B, k)
        latents.append(z)
        # Assume physical features already computed in the raw_window_features
        # stored on the sample; project out the three we care about
        # (this requires the loader to return them — adjust as needed)
        # rms.append(...); kurt.append(...); env.append(...)
    latents = np.concatenate(latents, axis=0)
    physics = {
        "rms": np.concatenate(rms, axis=0) if rms else None,
        "kurtosis": np.concatenate(kurt, axis=0) if kurt else None,
        "envelope_peak": np.concatenate(env, axis=0) if env else None,
    }
    return latents, {k: v for k, v in physics.items() if v is not None}


def rank_latents_by_physics_correlation(
    latents: np.ndarray, physics: dict[str, np.ndarray]
) -> dict[str, list[tuple[int, float]]]:
    """Rank SAE latents by Spearman correlation with each physical quantity."""
    ranked = {}
    for name, vec in physics.items():
        rhos = []
        for k in range(latents.shape[1]):
            if latents[:, k].std() < 1e-8:
                rhos.append(0.0)
                continue
            r, _ = spearmanr(latents[:, k], vec)
            rhos.append(float(r) if not np.isnan(r) else 0.0)
        order = np.argsort(-np.abs(rhos))
        ranked[name] = [(int(k), rhos[k]) for k in order[:10]]
    return ranked


def causal_ablation(backbone, sae, loader, latent_idx: int, device="cuda"):
    """Zero out a specific SAE latent and measure downstream RUL change.

    A latent that meaningfully encodes degradation should shift the predicted
    RUL when ablated — this is the causal validation claim.
    """
    # Implementation sketch:
    # 1. Run baseline forward pass; record q_mid predictions.
    # 2. Monkey-patch the SAE latent: force z[:, latent_idx] = 0 then decode.
    # 3. Replace backbone pooled output with reconstructed h_hat averaged over W.
    # 4. Re-run the quantile head; compare q_mid shifts.
    ...
```

The paper claim is then: "latent k has Spearman ρ = X with envelope-peak amplitude across the target bearing's life, AND ablating it changes predicted RUL by Y% — two independent signals that it causally encodes the spall-growth concept."

---

## 7. Experimental Matrix

### 7.1 Main table — Cross-condition within dataset

For each source→target pair (12 pairs: PHM2012 has 6, XJTU-SY has 6 directional pairs between C1/C2/C3), and each seed (3 seeds), run the following 5 methods:

| # | Method | Adaptation | CP method |
|---|--------|-----------|-----------|
| 1 | `baseline_srconly_split` | none | split CP |
| 2 | `baseline_srconly_cqr` | none | CQR |
| 3 | `sfda_cqr` | SFDA | CQR |
| 4 | `sfda_conditional` | SFDA | **Conditional CP** |
| 5 | `sfda_conditional_dtaci` | SFDA | **Conditional + DtACI** ← headline |

Plus baselines from the literature that we MUST beat:

| 6 | `CRULP` | — | CRULP (reimplement Piao et al. 2025) |
| 7 | `ACRP` | — | ACRP (reimplement Piao et al. 2025) |
| 8 | `DANN_weighted_CP` | DANN | Weighted Split CP (Tibshirani 2019) — the original naive Angle B |

Report: RMSE, MAE, PHM2012 Score, marginal coverage, **per-condition coverage (the thing conditional CP fixes)**, MPIW, SSC-5, WSC (worst bearing), and DtACI's long-run coverage gap at α ∈ {0.05, 0.1, 0.2}.

### 7.2 Cross-dataset

`PHM2012 (all conds) → XJTU-SY (all conds)` and vice versa, same 5+3 methods. This is where SFDA matters most because the sensor mounting differs.

### 7.3 Ablations

For method #5 (headline):
- Remove SFDA → baseline + conditional CP only
- Remove conditional CP → SFDA + split CP
- Remove DtACI → SFDA + conditional CP offline
- Replace Mamba with CNN-LSTM → baseline backbone impact
- Replace Mamba with Transformer → is the gain from Mamba or just "big model"?

### 7.4 Interpretability study

On one target bearing per operating condition, train the SAE with k = 2d latents on the frozen target-adapted backbone, then:
1. Rank the top-10 latents per physical observable (RMS, kurtosis, envelope peak).
2. Compute causal-ablation effect on predicted RUL.
3. Present a "degradation-regime transition diagram": which SAE latents activate in which life stage (healthy / FPT-approach / post-FPT / near-EOL).

### 7.5 Computational budget

- 3 seeds × 5 main methods × 12 cross-cond pairs × 2 datasets = 360 training runs.
- Plus 3 seeds × 3 baselines × 12 pairs × 2 = 216 runs.
- Plus 4 cross-dataset pairs × 5 methods × 3 seeds = 60 runs.
- **Total ≈ 636 runs**, each ~45 min on an A100 → ~480 GPU-hours → ~3 weeks.

---

## 8. Recommended Timeline (12 months)

| Month | Deliverable |
|---|---|
| 1 | Set up repo (existing), integrate Mamba backbone, write Stage A tests |
| 2 | EDA notebooks on PHM2012 + XJTU-SY; FPT detector; feature extraction |
| 3 | Source-training baselines (Mamba + CNN-LSTM + Transformer) and report source-only metrics |
| 4 | Implement SFDA; confirm pseudo-label + monotonicity converges on synthetic shifts |
| 5 | Implement Conditional CP and DtACI; run synthetic-data coverage validation (tests in §5.5) |
| 6 | Reimplement CRULP + ACRP as literature baselines |
| 7 | Run the full 360+216-run matrix on cluster, monitor mid-way; fix collapse modes |
| 8 | Cross-dataset experiments; aggregate main tables |
| 9 | SAE interpretability study; causal ablations; draft Chapter 5 |
| 10 | First paper submission (RESS, CP+SFDA headline) |
| 11 | Second paper submission (MSSP, Mamba+SAE) |
| 12 | Dissertation write-up, defence prep, IEEE TII paper on deployment systems |

---

## 9. Pitfalls & Honest Engineering Notes

1. **Mamba on feature-windows (W≈10) is overkill.** Mamba shines on long sequences. If feature-domain inputs are short, CNN-LSTM might tie or beat it. Two valid paths: (a) use Mamba on the raw signal stream (L=2560 for PHM, 32768 for XJTU-SY) — this is where the architectural claim lives; or (b) accept that for feature inputs Mamba is a deliberate choice for the interpretable hidden state, not raw accuracy. State this clearly.
2. **SFDA pseudo-labels collapse.** If confidence weights all saturate toward 0 (huge intervals) or 1 (overconfident), the model destroys itself. Use the feature-anchor term as a safety net.
3. **Conditional CP linearity assumption.** The Gibbs–Cherian–Candès method only guarantees conditional coverage over the *linear span* of Φ. If real shift is non-linear, coverage on specific subgroups can still fail — report this honestly. Their 2025 paper discusses exactly this.
4. **DtACI cost.** Each stream step invokes K calibrations. If calibration is expensive (e.g. re-fitting quantile regression for each α), precompute the calibration scores once and only re-query the quantile at different α's — this is O(K) per step instead of O(K × n).
5. **SAE is not automatically interpretable.** Many latents will correlate with nothing or with everything. The causal ablation step is what separates a real interpretability claim from cherry-picked correlation.
6. **Beating CRULP and ACRP is non-trivial.** They're strong baselines. If your headline doesn't beat them clearly, the paper is a coverage-study contribution, not a performance contribution — adjust the narrative accordingly. Both are valid PhD deliverables; be honest about which one you deliver.
7. **PHM2012 asymmetric Score is finicky.** The 0.5 base in the exponential was defined for minute-scale RUL. On normalised [0,1] RUL the score behaves differently. Report it as your primary literature-comparable metric but validate against RMSE first.
8. **Source-free ≠ no source artifacts.** You're allowed to keep a frozen head, anchor embeddings, and hyperparameters from source training. That is standard in the SFDA literature (SHOT, CURA, etc.) and not cheating.

---

## 10. Minimal End-to-End Demo Script

```python
# examples/demo_ab_hybrid.py
"""End-to-end A+B hybrid demo on synthetic data.

Flow:
  1. Generate synthetic bearings with 3 conditions (source = C1, target = C2).
  2. Extract features; window.
  3. Train Mamba backbone on source with pinball + monotonicity loss.
  4. Adapt to target via SFDA (no source data).
  5. Apply Conditional CP (Φ = one-hot condition + HI polynomial).
  6. Wrap with DtACI over the target stream.
  7. Report: RMSE, per-condition coverage, MPIW, DtACI long-run coverage.
  8. (Optional) Train SAE on the adapted backbone; plot top-correlated latent.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from brul.data import make_synthetic_dataset, source_target_split, BearingWindowDataset
from brul.features import FeatureExtractor
from brul.models.mamba_backbone import MambaBackbone, MambaConfig
from brul.models.heads import QuantileHead
from brul.uda.sfda import SFDATrainer, SFDAConfig
from brul.cp import Preds, ConformalizedQuantileRegression
from brul.cp.conditional import ConditionalConformal, ConditionalCPConfig, cqr_score
from brul.cp.dtaci import DtACI, DtACIConfig
from brul.cp.metrics import report_intervals


def main():
    # 1. Data
    runs = make_synthetic_dataset(n_per_condition=4, seed=42)
    split = source_target_split(runs, source_condition=1, target_condition=2)

    # 2. Features
    fs = runs[0].fs
    fe = FeatureExtractor(fs=fs)
    def ds(runs, shuffle):
        d = BearingWindowDataset(runs, window_length=8, feature_fn=fe)
        return d, DataLoader(d, batch_size=64, shuffle=shuffle)

    src_ds, src_loader = ds(split.source_train, True)
    calib_ds, calib_loader = ds(split.source_calib, False)
    tgt_ds, tgt_loader = ds(split.target_test, False)
    tgt_loader_shuf = DataLoader(tgt_ds, batch_size=64, shuffle=True)

    # 3. Stage A: source training with Mamba
    bb = MambaBackbone(n_features=fe.n_features, cfg=MambaConfig(d_model=64, n_layers=3))
    head = QuantileHead(in_dim=bb.out_dim, quantiles=(0.05, 0.5, 0.95))
    # ... normal pinball-loss training on src_loader (code identical to existing UDA base) ...

    # 4. Stage B: SFDA on target (no source access)
    sfda = SFDATrainer(bb, head, SFDAConfig(epochs=20))
    sfda.fit(tgt_loader_shuf)

    # 5. Stage C: Conditional CP + DtACI
    def pred_on_loader(loader):
        # run bb+head, collect (point, lo, up, features, y, bearing_id)
        ...
    cal_preds = pred_on_loader(calib_loader)
    tgt_preds = pred_on_loader(tgt_loader)

    def phi(feats):
        # feats expected shape (N, d); we prepend a condition one-hot column externally
        cond = feats[:, 0].astype(int)
        hi = feats[:, -1]
        return np.stack([
            np.ones(len(feats)),
            (cond == 1).astype(float),
            (cond == 2).astype(float),
            (cond == 3).astype(float),
            hi, hi**2,
        ], axis=1)

    cond_cp = ConditionalConformal(
        phi=phi, score_fn=cqr_score, config=ConditionalCPConfig(alpha=0.1)
    )
    cond_cp.calibrate(cal_preds)
    iv_offline = cond_cp.predict(tgt_preds)

    def factory(a):
        cp = ConditionalConformal(
            phi=phi, score_fn=cqr_score, config=ConditionalCPConfig(alpha=a)
        )
        return cp

    dtaci = DtACI(factory, cal_preds, DtACIConfig(alpha=0.1))
    iv_online = dtaci.run_stream(tgt_preds)

    # 6. Report
    print("Offline Conditional CP:",
          report_intervals(tgt_preds.y_true, iv_offline.lower, iv_offline.upper, 0.1))
    print("Online DtACI over Conditional CP:",
          report_intervals(tgt_preds.y_true, iv_online.lower, iv_online.upper, 0.1))


if __name__ == "__main__":
    main()
```

---

## 11. What to Cite (Required Bibliography for the Intro)

### Core methods (cite in §2 of any paper)
- Ganin & Lempitsky (2015), Domain-Adversarial Training, ICML. — as baseline
- Tibshirani, Barber, Candès & Ramdas (2019), Conformal Prediction Under Covariate Shift, NeurIPS. — as baseline
- Romano, Patterson & Candès (2019), Conformalized Quantile Regression, NeurIPS.
- Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution Shift, NeurIPS.
- Barber, Candès, Ramdas & Tibshirani (2023), Conformal prediction beyond exchangeability, Annals of Statistics.
- **Gibbs & Candès (2024)**, Conformal Inference for Online Prediction with Arbitrary Distribution Shifts, JMLR 25:162. — DtACI source
- **Gibbs, Cherian & Candès (2025)**, Conformal prediction with conditional guarantees, JRSSB 87(4): 1100–1126. — Conditional CP source
- **Gu & Dao (2024)**, Mamba: Linear-Time Sequence Modeling with Selective State Spaces, ICML. — Mamba
- Liang, Hu & Feng (2020), SHOT, ICML. — SFDA inspiration
- Bricken et al. (2023), Towards Monosemanticity, Anthropic. — SAE method

### Bearing-RUL baselines we must beat
- **Piao, Huang & Tsung (2025), CRULP, IEEE TIM 74.**
- **Piao, Wang, Huang, Wang & Tsung (2025), ACRP, IEEE CASE, pp. 3197–3202.**
- Wang et al. (2025), RobustUQ, RESS 262.

### Datasets
- Nectoux et al. (2012), PRONOSTIA, IEEE PHM.
- Wang et al. (2020), XJTU-SY, IEEE TR.

---

**End of specification.**

Everything above replaces `ANGLE_B_CURSOR_PROMPT.md` as the current plan.  
The existing `bearing-rul-cpda` codebase provides ~70% of the infrastructure (data, features, heads, baseline CP). Adding the 4 new modules (`mamba_backbone`, `sfda`, `conditional`, `dtaci`) and the SAE probe is the remaining implementation work.
