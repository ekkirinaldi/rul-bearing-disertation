"""End-to-end smoke test (no dataset download required).

Run::

    python -m mxlstm.smoke

What it checks:
  1. Synthetic ``BearingRun``-shaped data → HI extraction → MinMax+EMA pipeline.
  2. Window dataset construction.
  3. Forward pass through XLSTMTransformer baseline.
  4. Forward pass through MambaXLSTMNet (vanilla backend; no extra deps).
  5. One AdamW optimization step on each model and asserts loss decreases.

Exits 0 on success. Useful as a CI gate or first-run sanity check on Mac.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from mxlstm.compute import get_compute_profile
from mxlstm.data.hi import HIPipeline
from mxlstm.data.labels import make_rul_labels
from mxlstm.models.baseline_xlstm_transformer import XLSTMTransformer
from mxlstm.models.mamba_xlstm_net import MambaXLSTMConfig, MambaXLSTMNet
from mxlstm.utils.logging import logger
from mxlstm.utils.seed import seed_everything


def _synthetic_signal(T: int = 40, channels: int = 2, samples: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = rng.normal(scale=0.05, size=(T, channels, samples)).astype(np.float32)
    for t in range(T):
        amp = 0.05 + 0.6 * (t / T) ** 2
        out[t] += amp * rng.normal(scale=1.0, size=(channels, samples)).astype(np.float32)
    return out


def main() -> int:
    seed_everything(42)
    profile = get_compute_profile()
    logger.info(f"Smoke test on device={profile.device} mamba_backend={profile.mamba_backend}")

    # --- HI pipeline ---
    sig = _synthetic_signal(T=40)
    pipe = HIPipeline(fs=25_600, smoothing_alpha=0.1)
    pipe.fit([sig])
    hi = pipe.transform_signal(sig)
    rul = make_rul_labels(hi.shape[0], scheme="linear")
    logger.info(f"HI shape={hi.shape} dtype={hi.dtype} rul[0]={rul[0]:.2f} rul[-1]={rul[-1]:.2f}")
    assert hi.shape[1] == 2 * 18, hi.shape

    # --- Build a window batch ---
    L, F = 16, hi.shape[1]
    windows = []
    targets = []
    for t_end in range(L - 1, hi.shape[0]):
        windows.append(hi[t_end - L + 1 : t_end + 1])
        targets.append(rul[t_end])
    x = torch.from_numpy(np.stack(windows)).float()  # (N, L, F)
    y = torch.from_numpy(np.asarray(targets)).float()
    logger.info(f"Window batch: x={tuple(x.shape)} y={tuple(y.shape)}")

    # --- Baseline forward + 1 step ---
    base = XLSTMTransformer(
        n_features=F, d_model=32, n_heads=4, encoder_layers=1,
        xlstm_blocks=2, slstm_positions=[1], context_length=L,
    )
    n_p = sum(p.numel() for p in base.parameters() if p.requires_grad)
    logger.info(f"XLSTMTransformer: {n_p:,} params")
    pred = base(x)
    assert pred.shape == y.shape
    opt = torch.optim.AdamW(base.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(base(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    logger.info(f"Baseline losses: {losses}")
    assert losses[-1] <= losses[0] + 1e-3, "baseline failed to decrease loss"

    # --- Proposed Mamba-xLSTM-Net forward + 1 step ---
    cfg = MambaXLSTMConfig(
        n_features=F, d_model=32, context_length=L,
        xlstm_blocks=2, slstm_positions=[1], xlstm_heads=4, xlstm_force_fallback=True,
        mamba_blocks=1, mamba_d_state=8, mamba_backend="vanilla",
        fusion="gated",
    )
    net = MambaXLSTMNet(cfg)
    n_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"MambaXLSTMNet: {n_p:,} params")
    pred = net(x)
    assert pred.shape == y.shape
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    logger.info(f"Mamba-xLSTM losses: {losses}")
    assert losses[-1] <= losses[0] + 1e-3, "Mamba-xLSTM failed to decrease loss"

    # Also exercise return_hidden + SAE shapes briefly
    pred, hidden = net(x, return_hidden=True)
    assert hidden["fused"].shape[-1] == cfg.d_model

    logger.info("Smoke test PASSED.")
    return 0


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"\nDone in {time.time() - t0:.1f}s.")
    sys.exit(rc)
