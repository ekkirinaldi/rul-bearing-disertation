"""Top-k Sparse Autoencoder per §8.2 of PROJECT_PLAN.md.

  encoder : Linear(d_model -> d_model * expansion)
  topk    : keep top-k activations per sample, zero the rest
  decoder : Linear(d_model * expansion -> d_model, no bias)
  pre_bias: shared bias subtracted before encoding, added after decoding

Loss = reconstruction MSE (top-k variant). An optional auxiliary loss
(controlled by ``aux_k``) reconstructs from "dead" features to revive
them, following the Anthropic top-k SAE recipe.

This is a self-contained module that can be trained against any
collection of hidden-state vectors via :func:`train_sae`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int
    expansion: int = 8
    k: int | None = None           # defaults to int(d_latent * 0.05)
    aux_k: int = 0                 # auxiliary k for dead-feature loss; 0 disables


class TopKSparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        d_latent = cfg.d_model * cfg.expansion
        self.cfg = cfg
        self.k = int(cfg.k) if cfg.k else max(1, int(d_latent * 0.05))
        self.encoder = nn.Linear(cfg.d_model, d_latent)
        self.decoder = nn.Linear(d_latent, cfg.d_model, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_model))
        # Tie initial decoder norm so reconstruction starts near identity-ish
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T.contiguous())

    @property
    def d_latent(self) -> int:
        return self.cfg.d_model * self.cfg.expansion

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.pre_bias)

    def topk(self, z: torch.Tensor, k: int | None = None) -> torch.Tensor:
        k = int(k or self.k)
        vals, idx = z.topk(k, dim=-1)
        out = torch.zeros_like(z)
        out.scatter_(-1, idx, F.relu(vals))
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        z_sparse = self.topk(z)
        x_hat = self.decoder(z_sparse) + self.pre_bias
        return x_hat, z_sparse

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_hat, z_sparse = self.forward(x)
        recon = F.mse_loss(x_hat, x)
        total = recon
        info = {"recon": float(recon.detach().cpu())}
        if self.cfg.aux_k > 0:
            # Aux loss: reconstruct using *only* the dead (never-active) features.
            with torch.no_grad():
                ever_active = (z_sparse.abs() > 0).any(dim=0)
            dead_mask = (~ever_active).float()
            if dead_mask.sum() > 0:
                z = self.encode(x)
                z_dead = z * dead_mask
                z_aux = self.topk(z_dead, k=self.cfg.aux_k)
                aux_recon = self.decoder(z_aux) + self.pre_bias
                aux_loss = F.mse_loss(aux_recon, x)
                total = total + 0.1 * aux_loss
                info["aux_recon"] = float(aux_loss.detach().cpu())
        return total, info


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def collect_hidden_states(
    model: nn.Module,
    dataloader: Iterable,
    *,
    device: torch.device | str = "cpu",
    layer: str = "fused",
    max_samples: int | None = None,
) -> np.ndarray:
    """Collect (N, d_model) hidden states from a Mamba-xLSTM-Net forward pass.

    Calls ``model(x, return_hidden=True)`` per batch and stacks the
    requested ``layer`` ('fused' | 'branch_a' | 'branch_b'). Flattens
    over the time axis so ``N = total_timesteps``.
    """
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                x = batch[0]
            else:
                x = batch
            x = x.to(device)
            _, hidden = model(x, return_hidden=True)
            h = hidden[layer]
            out.append(h.reshape(-1, h.size(-1)).cpu().numpy())
            if max_samples is not None and sum(a.shape[0] for a in out) >= max_samples:
                break
    arr = np.concatenate(out, axis=0)
    if max_samples is not None:
        arr = arr[:max_samples]
    return arr


def train_sae(
    sae: TopKSparseAutoencoder,
    hidden: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> list[dict[str, float]]:
    sae = sae.to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)
    h = torch.from_numpy(hidden.astype(np.float32))
    history = []
    n = h.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_recon = 0.0
        steps = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = h[idx].to(device)
            opt.zero_grad()
            loss, info = sae.loss(xb)
            loss.backward()
            opt.step()
            epoch_recon += info["recon"]
            steps += 1
        history.append({"epoch": epoch, "recon": epoch_recon / max(steps, 1)})
    return history


def top_activating_examples(
    sae: TopKSparseAutoencoder,
    hidden: np.ndarray,
    *,
    n_top: int = 20,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """For each latent feature, return the indices of the top-N activating
    samples in ``hidden``. Output shape: ``(d_latent, n_top)``.
    """
    sae = sae.to(device).eval()
    h = torch.from_numpy(hidden.astype(np.float32)).to(device)
    with torch.no_grad():
        z = sae.encode(h)                                    # (N, d_latent)
        z_sparse = sae.topk(z)
    z_sparse = z_sparse.cpu().numpy()
    return np.argsort(-z_sparse, axis=0)[:n_top].T            # (d_latent, n_top)


def save_sae(sae: TopKSparseAutoencoder, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": sae.state_dict(), "cfg": sae.cfg.__dict__}, path)


def load_sae(path: str | Path) -> TopKSparseAutoencoder:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Older PyTorch versions do not accept the weights_only keyword.
        payload = torch.load(path, map_location="cpu")  # type: ignore[call-overload]
    cfg_raw = payload["cfg"]
    cfg = cfg_raw if isinstance(cfg_raw, SAEConfig) else SAEConfig(**cfg_raw)
    sae = TopKSparseAutoencoder(cfg)
    sae.load_state_dict(payload["state_dict"])
    return sae
