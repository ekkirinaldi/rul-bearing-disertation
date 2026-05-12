"""Scalar-compatible diffusion-inspired RUL regressor (v5).

v5 vs v4 (surgical fixes targeting both PHM2012 and XJTU-SY weaknesses)
----------------------------------------------------------------------
1. **No score leak into RUL logits.** v4 added ``0.01 * score.mean()`` (an
   untrained, unbounded scalar) into the RUL pre-sigmoid logit. v5 decouples:
   ``score_head`` is now a *denoiser* trained via an auxiliary loss, and
   ``rul_head`` outputs RUL on its own.
2. **Auxiliary denoising loss.** ``compute_loss`` now adds
   ``lambda_denoise * MSE(score_head(encoded_noisy), clean_x)`` so the
   denoiser actually learns HI structure → better representation → lower
   RUL RMSE.
3. **Cosine noise schedule, skipping σ=0.** ``sigma_k = max_noise_std *
   sin(pi/2 * k / K)`` for ``k = 1..K``. The σ=0 branch in v4 was redundant
   (no noise → identical to clean view) and wasted compute.
4. **Stochastic inference ensembling.** v4 zeroed out test-time noise, so the
   ``n_noise_levels`` branches collapsed to near-duplicates. v5 keeps a small
   ``inference_noise_scale * sigma_k`` at eval and averages
   ``n_inference_samples`` realisations. ``inference_noise_scale = 0`` recovers
   the deterministic v4 behaviour for ablation.
5. **FiLM by ``condition_ids``.** Optional per-condition (γ, β) modulation
   inside every denoising block — helps XJTU-SY where 3 operating regimes
   change the HI dynamics. Disabled if ``film_num_embeddings <= 0``.
6. **Mean + last pooling.** RUL depends strongly on the *end* of the window;
   mean-only pooling dilutes it. v5 concats mean and last-token features
   before the RUL head.

The forward signature is ``(B, L, F) -> (B,)`` with optional
``condition_ids: (B,)`` to stay compatible with ``RULLitModule._shared_step``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_period = float(max_period)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=positions.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = positions.float().unsqueeze(-1) * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class _DenoisingBlock(nn.Module):
    """Pre-norm self-attn + cross-attn (to clean) + FFN with optional FiLM."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(d_model)
        self.ln_cross_q = nn.LayerNorm(d_model)
        self.ln_cross_kv = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        noisy: torch.Tensor,
        clean_context: torch.Tensor,
        film: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        h = self.ln_self(noisy)
        if film is not None:
            gamma, beta = film  # each (B, d_model)
            h = h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        attn, _ = self.self_attn(h, h, h, need_weights=False)
        noisy = noisy + attn
        q = self.ln_cross_q(noisy)
        kv = self.ln_cross_kv(clean_context)
        cross, _ = self.cross_attn(q, kv, kv, need_weights=False)
        noisy = noisy + cross
        return noisy + self.ff(self.ln_ff(noisy))


class DiffusionRUL(nn.Module):
    """Denoising-score ensemble for normalised scalar RUL with optional FiLM.

    Notes:
        * ``compute_loss`` is model-specific (RUL smooth-L1 + auxiliary denoising
          MSE). Activate it via ``train.model_specific_loss: true`` in YAML or
          ``configs/model/diffusion_rul.yaml``.
        * Set ``inference_noise_scale = 0`` to reproduce deterministic v4 eval.
        * ``film_num_embeddings = 0`` disables FiLM (single-condition runs).
    """

    def __init__(
        self,
        n_features: int,
        *,
        context_length: int,
        d_model: int = 96,
        n_heads: int = 4,
        n_layers: int = 2,
        n_noise_levels: int = 4,
        max_noise_std: float = 0.35,
        dropout: float = 0.1,
        # v5 additions (all optional, defaults keep older runs comparable)
        inference_noise_scale: float = 0.5,
        n_inference_samples: int = 4,
        lambda_denoise: float = 0.1,
        film_num_embeddings: int = 0,
        pool: str = "mean_last",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if n_noise_levels < 1:
            raise ValueError("n_noise_levels must be >= 1")
        if pool not in ("mean", "last", "mean_last"):
            raise ValueError(f"pool must be 'mean'|'last'|'mean_last'; got {pool!r}")
        self.context_length = int(context_length)
        self.n_features = int(n_features)
        self.n_noise_levels = int(n_noise_levels)
        self.max_noise_std = float(max_noise_std)
        self.inference_noise_scale = float(inference_noise_scale)
        self.n_inference_samples = max(1, int(n_inference_samples))
        self.lambda_denoise = float(lambda_denoise)
        self.film_num_embeddings = int(film_num_embeddings)
        self.pool = pool

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_emb = SinusoidalEmbedding(d_model)
        self.noise_emb = SinusoidalEmbedding(d_model)
        self.noise_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.blocks = nn.ModuleList([_DenoisingBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

        # Per-condition FiLM: (γ, β) per block, stored as a single embedding table.
        self._n_blocks = int(n_layers)
        if self.film_num_embeddings > 0:
            self.film_embed = nn.Embedding(self.film_num_embeddings, 2 * d_model * self._n_blocks)
            nn.init.zeros_(self.film_embed.weight)  # start as identity (γ=0, β=0)
        else:
            self.film_embed = None

        # Denoiser: encoded noisy -> reconstructed clean input features.
        self.score_head = nn.Linear(d_model, n_features)

        # RUL head: pool -> MLP -> sigmoid
        pool_factor = 2 if self.pool == "mean_last" else 1
        self.rul_head = nn.Sequential(
            nn.LayerNorm(pool_factor * d_model),
            nn.Linear(pool_factor * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    # ------------------------------------------------------------------ utils

    def _noise_sigmas(self, device: torch.device, training: bool) -> torch.Tensor:
        """Cosine schedule σ_k = max_noise_std · sin(π/2 · k/K), k=1..K."""
        k = torch.arange(1, self.n_noise_levels + 1, device=device, dtype=torch.float32)
        sigmas = self.max_noise_std * torch.sin(0.5 * math.pi * k / self.n_noise_levels)
        if not training:
            sigmas = sigmas * self.inference_noise_scale
        return sigmas  # (K,)

    def _film(
        self, condition_ids: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> list[tuple[torch.Tensor, torch.Tensor] | None]:
        if self.film_embed is None or condition_ids is None:
            return [None] * self._n_blocks
        cids = condition_ids.clamp(min=0, max=self.film_num_embeddings - 1).long()
        all_params = self.film_embed(cids).to(dtype)  # (B, 2*d*n_blocks)
        d_model = all_params.size(-1) // (2 * self._n_blocks)
        per_block = all_params.view(batch_size, self._n_blocks, 2, d_model)
        return [(per_block[:, i, 0], per_block[:, i, 1]) for i in range(self._n_blocks)]

    def _encode(
        self,
        x: torch.Tensor,
        noise_std: torch.Tensor,
        condition_ids: torch.Tensor | None,
        noise_level_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (encoded_noisy, noise_actually_added).

        ``noise_std``: (B,) per-sample std.
        ``noise_level_norm``: (B,) in [0, 1] for the noise embedding.
        """
        batch_size, seq_len, _ = x.shape
        pos = torch.arange(seq_len, device=x.device)
        clean_h = self.input_proj(x) + self.pos_emb(pos).unsqueeze(0)
        noise = torch.randn_like(x) * noise_std.view(-1, 1, 1)
        noisy_h = self.input_proj(x + noise) + self.pos_emb(pos).unsqueeze(0)
        noisy_h = noisy_h + self.noise_mlp(self.noise_emb(noise_level_norm)).unsqueeze(1)

        films = self._film(condition_ids, batch_size, x.device, noisy_h.dtype)
        for block, film in zip(self.blocks, films, strict=True):
            noisy_h = block(noisy_h, clean_h, film=film)
        return self.norm(noisy_h), noise

    def _pool(self, encoded: torch.Tensor) -> torch.Tensor:
        if self.pool == "mean":
            return encoded.mean(dim=1)
        if self.pool == "last":
            return encoded[:, -1]
        return torch.cat([encoded.mean(dim=1), encoded[:, -1]], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        x: torch.Tensor,
        condition_ids: torch.Tensor | None = None,
        *,
        return_score: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        sigmas = self._noise_sigmas(x.device, self.training)
        # K = n_noise_levels at train; n_inference_samples * K averaged at eval.
        n_repeat = 1 if self.training else self.n_inference_samples

        logits_acc: list[torch.Tensor] = []
        scores_acc: list[torch.Tensor] = []
        for _ in range(n_repeat):
            for k, sigma_scalar in enumerate(sigmas):
                level_norm = torch.full(
                    (batch_size,),
                    float((k + 1) / self.n_noise_levels),
                    device=x.device,
                    dtype=torch.float32,
                )
                sigma_b = sigma_scalar.expand(batch_size)
                encoded, _ = self._encode(x, sigma_b, condition_ids, level_norm)
                pooled = self._pool(encoded)
                logits_acc.append(self.rul_head(pooled).squeeze(-1))
                if return_score:
                    scores_acc.append(self.score_head(encoded))

        logits = torch.stack(logits_acc, dim=0).mean(dim=0)
        rul = torch.sigmoid(logits)
        if return_score:
            score = torch.stack(scores_acc, dim=0).mean(dim=0)
            return rul, score
        return rul

    # ------------------------------------------------------------------ losses

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        condition_ids: torch.Tensor | None = None,
        rul_window: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """RUL smooth-L1 plus auxiliary denoising MSE.

        For each training noise level σ_k > 0, draw fresh Gaussian noise,
        encode the noisy window, reconstruct the clean input features through
        ``score_head``, and minimise MSE against ``x``. This trains the
        denoiser to learn HI structure → richer features for ``rul_head``.
        """
        del rul_window  # unused; kept for signature compatibility
        batch_size = x.size(0)
        sigmas = self._noise_sigmas(x.device, training=True)

        rul_logits_acc: list[torch.Tensor] = []
        denoise_acc: list[torch.Tensor] = []
        for k, sigma_scalar in enumerate(sigmas):
            level_norm = torch.full(
                (batch_size,),
                float((k + 1) / self.n_noise_levels),
                device=x.device,
                dtype=torch.float32,
            )
            sigma_b = sigma_scalar.expand(batch_size)
            encoded, noise_added = self._encode(x, sigma_b, condition_ids, level_norm)
            # RUL branch
            pooled = self._pool(encoded)
            rul_logits_acc.append(self.rul_head(pooled).squeeze(-1))
            # Denoising branch: reconstruct clean x from encoded noisy.
            reco = self.score_head(encoded).to(x.dtype)
            denoise_acc.append(F.mse_loss(reco, x))

        rul = torch.sigmoid(torch.stack(rul_logits_acc, dim=0).mean(dim=0))
        main = F.smooth_l1_loss(rul, y)
        denoise = torch.stack(denoise_acc).mean() if denoise_acc else x.new_zeros(())
        return main + self.lambda_denoise * denoise

    # ------------------------------------------------------------------ explain

    @torch.no_grad()
    def explain(
        self, x: torch.Tensor, condition_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        rul, score = self.forward(x, condition_ids, return_score=True)
        return {
            "rul": rul.cpu(),
            "feature_score": score.abs().mean(dim=1).cpu(),
        }
