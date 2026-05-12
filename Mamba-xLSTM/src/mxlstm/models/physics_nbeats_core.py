"""Building blocks for Physics-N-BEATS / N-BEATS-xLSTM (bearing RUL).

Each public class starts with a short *why this exists for bearing RUL* note.
Geometry for BPFO/BPFI/BSF/FTF follows the dissertation domain rule
(``15-domain-rul-bearings.mdc``): deep-groove ball bearing, contact angle
``theta = 0``, ``f_r`` in Hz from shaft speed.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

DatasetKey = Literal["phm2012", "xjtusy", "xjtu_sy"]

# NSK 6804 (PHM2012 / PRONOSTIA) and LDK UER204 (XJTU-SY), mm, theta = 0.
_NSK6804 = {"n": 13, "d_mm": 3.50, "D_mm": 25.50}
_LDKUER204 = {"n": 8, "d_mm": 7.92, "D_mm": 34.55}

_PHM_ACQ_INTERVAL_S = 10.0
_XJTU_ACQ_INTERVAL_S = 60.0

_PHM_RPM = {1: 1800.0, 2: 1650.0, 3: 1500.0}
_XJTU_RPM = {1: 2100.0, 2: 2250.0, 3: 2400.0}


def _dataset_key(ds: str) -> DatasetKey:
    d = str(ds).lower().replace("-", "_")
    if d in ("phm2012",):
        return "phm2012"
    if d in ("xjtusy", "xjtu_sy", "xjtu"):
        return "xjtusy"
    raise ValueError(f"Unknown dataset for physics N-BEATS: {ds!r}")


def bearing_fault_freqs_hz(
    fr_hz: torch.Tensor,
    *,
    dataset: str,
    theta_rad: float = 0.0,
) -> torch.Tensor:
    """Return (B, 6) fault-related frequencies: BPFO, BPFI, BSF, FTF, 2*BPFO, 2*BPFI.

    Why Hz from ``f_r`` (not dimensionless time harmonics):

    1. Spalling / pitting excites impulses at BPFO/BPFI/BSF/FTF (domain rule).
    2. ``f_r`` comes from operating condition (rpm), which differs between PHM2012
       conditions and XJTU-SY folders — the same architecture must switch basis
       when ``condition_id`` switches.
    """
    key = _dataset_key(dataset)
    spec = _NSK6804 if key == "phm2012" else _LDKUER204
    n = float(spec["n"])
    d = float(spec["d_mm"])
    D = float(spec["D_mm"])
    c = math.cos(float(theta_rad))
    dd = (d / D) * c
    fr = fr_hz.clamp(min=1e-3)
    bpfo = (n / 2.0) * fr * (1.0 - dd)
    bpfI = (n / 2.0) * fr * (1.0 + dd)
    bsf = (D / (2.0 * d)) * fr * (1.0 - dd**2)
    ftf = 0.5 * fr * (1.0 - dd)
    two_bpfo = 2.0 * bpfo
    two_bpfI = 2.0 * bpfI
    return torch.stack([bpfo, bpfI, bsf, ftf, two_bpfo, two_bpfI], dim=-1)


def shaft_hz_from_conditions(
    condition_ids: torch.Tensor,
    *,
    dataset: str,
) -> torch.Tensor:
    """Map batch condition ids (1..3) to ``f_r`` in Hz."""
    key = _dataset_key(dataset)
    table = _PHM_RPM if key == "phm2012" else _XJTU_RPM
    cid = condition_ids.long().clamp(min=1, max=3)
    rpms = torch.tensor([table[1], table[2], table[3]], device=cid.device, dtype=torch.float32)
    r = rpms[cid - 1]
    return r / 60.0


def acquisition_dt_s(dataset: str) -> float:
    key = _dataset_key(dataset)
    return _PHM_ACQ_INTERVAL_S if key == "phm2012" else _XJTU_ACQ_INTERVAL_S


def expand_basis_lf(basis_l: torch.Tensor, n_features: int) -> torch.Tensor:
    """(K, L) -> (K, L * n_features) with the same layout as ``TrendBlock`` in ``nbeats_rul``."""
    k, L = basis_l.shape
    return (
        basis_l.unsqueeze(-1)
        .expand(k, L, n_features)
        .reshape(k, L * n_features)
    )


def bernstein_basis_matrix(degree: int, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(degree+1, L) matrix B_{degree,k}(t) on t in [0, 1]."""
    t = torch.linspace(0.0, 1.0, L, device=device, dtype=dtype)
    n = int(degree)
    rows = []
    for k in range(n + 1):
        c = math.comb(n, k)
        row = c * (1.0 - t).pow(n - k) * t.pow(k)
        rows.append(row)
    return torch.stack(rows, dim=0)


def _monotone_bernstein_coeffs(raw: torch.Tensor) -> torch.Tensor:
    """Map (B, degree+1) unconstrained raw to non-increasing control points.

    Non-increasing RUL *trend* along acquisition index is a soft structural prior
    for run-to-failure curves after HI smoothing (see domain pitfalls: monotonicity
    is an explicit modelling choice — we encode a *weak* inductive bias here).
    """
    b, n1 = raw.shape
    c0 = F.softplus(raw[:, 0:1])
    prev = c0
    cols = [prev]
    for k in range(1, n1):
        prev = prev - F.softplus(raw[:, k : k + 1])
        cols.append(prev)
    return torch.cat(cols, dim=1)


class ConditionFiLM(nn.Module):
    """FiLM from discrete operating-condition id.

    Why FiLM (not ignoring ``condition``):

    XJTU-SY mixes three rpm/load regimes in one training pool; vanilla N-BEATS
    uses one trunk for all windows, which forces a single set of basis weights
    to explain incompatible operating points. Conditioning the trunk with
    ``(gamma, beta)`` per condition is a lightweight way to specialize basis
    scaling without duplicating the full network per regime.
    """

    def __init__(self, num_embeddings: int, dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(int(num_embeddings), int(dim) * 2)

    def forward(self, h: torch.Tensor, condition_ids: torch.Tensor) -> torch.Tensor:
        idx = condition_ids.long().clamp(min=0, max=self.emb.num_embeddings - 1)
        gb = self.emb(idx)
        gamma, beta = gb.chunk(2, dim=-1)
        return h * (1.0 + torch.tanh(gamma)) + beta


def make_physics_trunk(hidden_dim: int, dropout: float) -> nn.Sequential:
    """Four-layer MLP matching ``nbeats_rul._NBeatsBlock`` trunk depth (per block).

    Physics blocks feed ``hidden_dim`` from ``PerFeatureTemporalEncoder`` + FiLM,
    not the full flattened window — depth is aligned with vanilla, not width.
    """

    h = int(hidden_dim)
    d = float(dropout)
    return nn.Sequential(
        nn.Linear(h, h),
        nn.ReLU(),
        nn.Dropout(d),
        nn.Linear(h, h),
        nn.ReLU(),
        nn.Dropout(d),
        nn.Linear(h, h),
        nn.ReLU(),
        nn.Dropout(d),
        nn.Linear(h, h),
        nn.ReLU(),
    )


class PerFeatureTemporalEncoder(nn.Module):
    """Depthwise temporal conv + mean/max pool over acquisitions.

    Why not flatten ``(B, L, F)`` into one MLP (vanilla N-BEATS):

    HI windows carry *per-feature* transients (e.g. kurtosis spikes vs slow RMS
    creep). A depthwise conv preserves per-feature time structure cheaply before
    pooling to a fixed-width code for the N-BEATS trunk.
    """

    def __init__(self, n_channels: int, kernel_size: int, out_dim: int) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(
            int(n_channels),
            int(n_channels),
            kernel_size=int(kernel_size),
            padding=pad,
            groups=int(n_channels),
        )
        self.proj = nn.Linear(int(n_channels) * 2, int(out_dim))

    def forward(self, x_blf: torch.Tensor) -> torch.Tensor:
        # (B, L, C) -> (B, C, L)
        h = self.conv(x_blf.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=-1)
        return self.proj(pooled)


class BernsteinTrendBlock(nn.Module):
    """Bernstein-basis trend with optional monotone control points and FiLM."""

    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int,
        degree: int,
        film_num_embeddings: int,
        dropout: float,
        encoder_kernel_size: int = 5,
        monotone_trend: bool = True,
    ) -> None:
        super().__init__()
        self.L = int(context_length)
        self.F = int(n_features)
        self.degree = int(degree)
        self.monotone_trend = bool(monotone_trend)
        self.register_buffer("_bern", torch.empty(0), persistent=False)
        self.encoder = PerFeatureTemporalEncoder(
            n_features, kernel_size=int(encoder_kernel_size), out_dim=hidden_dim
        )
        self.film = ConditionFiLM(int(film_num_embeddings), hidden_dim)
        self.trunk = make_physics_trunk(hidden_dim, dropout)
        n_theta = self.degree + 1
        self.theta_b = nn.Linear(hidden_dim, n_theta)
        self.theta_f = nn.Linear(hidden_dim, n_theta)

    def _get_bern(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._bern.numel() == 0 or self._bern.device != device or self._bern.dtype != dtype:
            self._bern = bernstein_basis_matrix(self.degree, self.L, device, dtype)
        return self._bern

    def forward(
        self,
        res_flat: torch.Tensor,
        res_3d: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(res_3d)
        h = self.film(h, condition_ids)
        h = self.trunk(h)
        raw = self.theta_b(h)
        coeffs = _monotone_bernstein_coeffs(raw) if self.monotone_trend else raw
        Bmat = self._get_bern(res_flat.device, res_flat.dtype)
        gb = expand_basis_lf(Bmat, self.F)
        backcast = coeffs @ gb
        gf = torch.ones(self.degree + 1, 1, device=res_flat.device, dtype=res_flat.dtype)
        forecast = (self.theta_f(h) @ gf).squeeze(-1)
        return backcast, forecast, coeffs


class CharacteristicFrequencyWearBlock(nn.Module):
    """Wear basis tied to BPFO/BPFI/BSF/FTF (+ harmonics) at acquisition times.

    Why replace integer harmonics of *normalized* time:

    1. Bearing fault energy concentrates near characteristic frequencies that
       scale with ``f_r`` and geometry (domain rule), not with arbitrary indices
       along the HI window.
    2. Named ``theta`` rows map directly to interpretability plots (Bab V:
       compare ``|theta_{BPFO}|`` to envelope spectrum peaks at the same Hz).
    """

    _FREQ_NAMES = ("BPFO", "BPFI", "BSF", "FTF", "2BPFO", "2BPFI")

    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int,
        film_num_embeddings: int,
        dropout: float,
        dataset: str,
        encoder_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.L = int(context_length)
        self.F = int(n_features)
        self.dataset = str(dataset)
        self.dt_s = acquisition_dt_s(self.dataset)
        self.n_theta = 12  # 6 freqs * (cos + sin)
        self.encoder = PerFeatureTemporalEncoder(
            n_features, kernel_size=int(encoder_kernel_size), out_dim=hidden_dim
        )
        self.film = ConditionFiLM(int(film_num_embeddings), hidden_dim)
        self.trunk = make_physics_trunk(hidden_dim, dropout)
        self.theta_b = nn.Linear(hidden_dim, self.n_theta)
        self.theta_f = nn.Linear(hidden_dim, self.n_theta)

    def _basis_batched(self, fr_hz: torch.Tensor) -> torch.Tensor:
        """(B, n_theta, L*F) — one row per cos/sin harmonic, layout matches ``TrendBlock``."""
        freqs = bearing_fault_freqs_hz(fr_hz, dataset=self.dataset)  # (B, 6)
        device, dtype = fr_hz.device, fr_hz.dtype
        t = (torch.arange(self.L, device=device, dtype=dtype) * self.dt_s).view(1, 1, self.L)
        ang = (2.0 * math.pi) * freqs.unsqueeze(-1) * t  # (B, 6, L)
        cos_sin = torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)  # (B, 12, L)
        pieces: list[torch.Tensor] = []
        for i in range(self.n_theta):
            row = cos_sin[:, i, :]  # (B, L)
            pieces.append(
                row.unsqueeze(-1)
                .expand(-1, self.L, self.F)
                .reshape(row.size(0), self.L * self.F)
            )
        return torch.stack(pieces, dim=1)

    def forward(
        self,
        res_flat: torch.Tensor,
        res_3d: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        fr = shaft_hz_from_conditions(condition_ids, dataset=self.dataset)
        gb = self._basis_batched(fr)
        h = self.encoder(res_3d)
        h = self.film(h, condition_ids)
        h = self.trunk(h)
        tb = self.theta_b(h).unsqueeze(1)  # (B, 1, K)
        backcast = torch.bmm(tb, gb).squeeze(1)
        gf = torch.ones(self.n_theta, 1, device=res_flat.device, dtype=res_flat.dtype)
        tf = self.theta_f(h)
        forecast = (tf @ gf).squeeze(-1)
        named = {
            f"{n}_mag": tf[:, 2 * i : 2 * i + 2].norm(dim=-1)
            for i, n in enumerate(self._FREQ_NAMES)
        }
        return backcast, forecast, tf, named


class GaborShockBlock(nn.Module):
    """Localized impulse-like basis (Morlet-style) gated by window kurtosis.

    Why kurtosis-gated shocks:

    Spalling impulses are short-lived and raise kurtosis; a global shock stack
    without gating can overfit benign windows. The gate down-weights shocks when
    kurtosis is low, nudging the model to attribute slow wear to the physics wear
    basis instead.
    """

    def __init__(
        self,
        context_length: int,
        n_features: int,
        hidden_dim: int,
        n_basis: int,
        film_num_embeddings: int,
        kurt_index: int,
        dropout: float,
        dataset: str,
        encoder_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.L = int(context_length)
        self.F = int(n_features)
        self.n_basis = int(n_basis)
        self.kurt_index = int(kurt_index)
        self.dt_s = acquisition_dt_s(str(dataset))
        self.mu = nn.Parameter(torch.linspace(0.0, float(context_length - 1), n_basis))
        self.log_sigma = nn.Parameter(torch.full((n_basis,), math.log(2.0)))
        self.carrier_hz = nn.Parameter(torch.randn(n_basis) * 0.1 + 1.0)
        self.encoder = PerFeatureTemporalEncoder(
            n_features, kernel_size=int(encoder_kernel_size), out_dim=hidden_dim
        )
        self.film = ConditionFiLM(int(film_num_embeddings), hidden_dim)
        self.trunk = make_physics_trunk(hidden_dim, dropout)
        self.theta_b = nn.Linear(hidden_dim, n_basis)
        self.theta_f = nn.Linear(hidden_dim, n_basis)
        self.k_gate = nn.Linear(2, 1)

    def _morlet_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = torch.arange(self.L, device=device, dtype=dtype) * self.dt_s
        sig = torch.exp(self.log_sigma).clamp(min=1e-3)
        rows = []
        for k in range(self.n_basis):
            env = torch.exp(-0.5 * ((t - self.mu[k]) / sig[k]) ** 2)
            wav = torch.cos(2.0 * math.pi * self.carrier_hz[k] * (t - self.mu[k]))
            rows.append(expand_basis_lf((env * wav).unsqueeze(0), self.F).squeeze(0))
        return torch.stack(rows, dim=0)

    def forward(
        self,
        res_flat: torch.Tensor,
        res_3d: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gb = self._morlet_matrix(res_flat.device, res_flat.dtype)
        h = self.encoder(res_3d)
        h = self.film(h, condition_ids)
        h = self.trunk(h)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = torch.matmul(theta_b, gb)
        gf = torch.ones(self.n_basis, 1, device=res_flat.device, dtype=res_flat.dtype)
        shock_raw = (theta_f @ gf).squeeze(-1)
        ki = min(max(self.kurt_index, 0), res_3d.size(-1) - 1)
        k_col = res_3d[:, :, ki]
        kurt_feats = torch.cat([k_col.mean(dim=1, keepdim=True), k_col.amax(dim=1, keepdim=True)], dim=-1)
        gate = torch.sigmoid(self.k_gate(kurt_feats)).squeeze(-1)
        forecast = shock_raw * gate
        return backcast, forecast, theta_f, gate


class LearnedPhysShockBlock(nn.Module):
    """Learned shock basis (N-BEATS ``ShockBlock``-style ``gb``/``gf``) after physics Morlets.

    Captures impulse energy not aligned with fixed Gabor carriers while keeping the
    same encoder + FiLM interface as ``GaborShockBlock``.
    """

    def __init__(
        self,
        context_length: int,
        enc_channels: int,
        hidden_dim: int,
        n_basis: int,
        film_num_embeddings: int,
        dropout: float,
        encoder_kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.L = int(context_length)
        self.C = int(enc_channels)
        self.n_basis = int(n_basis)
        lf = self.L * self.C
        self.gb = nn.Parameter(torch.randn(self.n_basis, lf) * (1.0 / math.sqrt(float(lf))))
        self.gf = nn.Parameter(torch.randn(self.n_basis, 1) * 0.1)
        self.encoder = PerFeatureTemporalEncoder(
            enc_channels, kernel_size=int(encoder_kernel_size), out_dim=hidden_dim
        )
        self.film = ConditionFiLM(int(film_num_embeddings), hidden_dim)
        self.trunk = make_physics_trunk(hidden_dim, dropout)
        self.theta_b = nn.Linear(hidden_dim, self.n_basis)
        self.theta_f = nn.Linear(hidden_dim, self.n_basis)

    def forward(
        self,
        res_flat: torch.Tensor,
        res_3d: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(res_3d)
        h = self.film(h, condition_ids)
        h = self.trunk(h)
        theta_b = self.theta_b(h)
        theta_f = self.theta_f(h)
        backcast = torch.matmul(theta_b, self.gb)
        shock_raw = (theta_f @ self.gf).squeeze(-1)
        gate = torch.ones(res_flat.size(0), device=res_flat.device, dtype=res_flat.dtype)
        return backcast, shock_raw, theta_f, gate


class _PhysStack(nn.Module):
    """Residual stack; each block sees the **current** residual as ``(B, L, C)``."""

    def __init__(self, blocks: list[nn.Module], context_length: int, enc_channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.L = int(context_length)
        self.C = int(enc_channels)

    def forward(
        self,
        x_flat: torch.Tensor,
        _x_3d_unused: torch.Tensor,
        condition_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]]]:
        del _x_3d_unused
        b, lf = x_flat.shape
        if lf != self.L * self.C:
            raise ValueError(f"residual flat dim {lf} != L*C ({self.L}*{self.C})")
        residual = x_flat
        forecast_sum = x_flat.new_zeros(b)
        extras: list[dict[str, torch.Tensor]] = []
        for block in self.blocks:
            x_3d = residual.view(b, self.L, self.C)
            if isinstance(block, CharacteristicFrequencyWearBlock):
                bc, fc, theta_w, named = block(residual, x_3d, condition_ids)
                extras.append({"wear_theta": theta_w, "wear_named": named})
            elif isinstance(block, GaborShockBlock):
                bc, fc, theta_s, gate = block(residual, x_3d, condition_ids)
                extras.append({"shock_theta": theta_s, "shock_gate": gate})
            elif isinstance(block, LearnedPhysShockBlock):
                bc, fc, theta_s, gate = block(residual, x_3d, condition_ids)
                extras.append({"shock_theta": theta_s, "shock_gate": gate, "learned_shock": True})
            elif isinstance(block, BernsteinTrendBlock):
                bc, fc, coeffs = block(residual, x_3d, condition_ids)
                extras.append({"trend_coeffs": coeffs})
            else:
                raise TypeError(type(block))
            residual = residual - bc
            forecast_sum = forecast_sum + fc
        return residual, forecast_sum, extras
