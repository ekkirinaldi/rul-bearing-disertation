"""Paper-faithful xLSTM-Transformer baseline.

Implements the architecture in Liu et al. (Sensors 2026, 26, 1578) §2.4,
which differs from the simple "Transformer-then-xLSTM" composition: the
xLSTM module *replaces* the position-wise FFN inside each Transformer
encoder/decoder layer.

Architecture (aligned with §2.4 Eqs (15)–(18))::

    Input (B, L, F)
        -> Linear(F, d_model) + sinusoidal positional encoding
        -> N x XLSTMTransformerEncoderLayer (pre-norm MHA):
              X_att     = MHA(LN(X), LN(X), LN(X))
              X_xlstm   = xLSTM(LN(X_att))            # xLSTM on **attention output only**;
                                                     # up/down projection lives inside the xLSTM stack
              X         = X_in + X_att + X_xlstm    # three-way residual (paper residual paths)
        -> N x XLSTMTransformerDecoderLayer:
              Y_self    = MHA(LN(Y), ...)           # full window, not causal (§2.4)
              Y_cross   = MHA(LN(Y), LN(enc), LN(enc))
              Y         = Y + self + cross          # standard two-attention residuals
              Y_xlstm   = xLSTM(LN(Y))              # third sublayer: same *slot* as Vaswani FFN
              Y         = Y + Y_xlstm; extra LN   # paper: additional decoder normalization
        -> SigmoidRegressionHead(pool='flatten') -> scalar RUL in [0, 1]

The decoder receives the same input embedding as the encoder
(per the paper: "the decoder shares the same input embedding as the
encoder, with the purpose of reconstructing and reintegrating the
encoded features rather than performing autoregressive prediction").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mxlstm.models.heads import SigmoidRegressionHead
from mxlstm.models.xlstm_blocks import XLSTMStack


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].size(1)])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Encoder / decoder layers — pre-norm, xLSTM as the nonlinear "FFN" sublayer
# ---------------------------------------------------------------------------


def _build_xlstm_block(
    d_model: int,
    *,
    context_length: int,
    xlstm_blocks: int,
    slstm_positions: list[int] | None,
    n_heads: int,
    dropout: float,
    force_fallback: bool,
) -> XLSTMStack:
    """Single xLSTM 'sub-stack' (e.g. 1 mLSTM + 1 sLSTM per the paper)."""
    return XLSTMStack(
        d_model=d_model,
        context_length=context_length,
        num_blocks=xlstm_blocks,
        slstm_positions=slstm_positions or [xlstm_blocks // 2],
        n_heads=n_heads,
        dropout=dropout,
        force_fallback=force_fallback,
    )


class XLSTMTransformerEncoderLayer(nn.Module):
    """One paper encoder layer: pre-norm MHA, then xLSTM on attention output, three-way residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        context_length: int,
        xlstm_blocks: int,
        slstm_positions: list[int] | None,
        force_fallback: bool,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_xlstm = nn.LayerNorm(d_model)
        self.xlstm = _build_xlstm_block(
            d_model,
            context_length=context_length,
            xlstm_blocks=xlstm_blocks,
            slstm_positions=slstm_positions,
            n_heads=n_heads,
            dropout=dropout,
            force_fallback=force_fallback,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Eq. (15): X_Att = Attention(...). Eqs. (16)–(18): xLSTM acts on X_Att (via internal
        # up-projection inside the xLSTM stack), not on (X_in + X_Att).
        h = self.norm_attn(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x_xlstm = self.xlstm(self.norm_xlstm(a))
        return x + self.dropout(a) + self.dropout(x_xlstm)


class XLSTMTransformerDecoderLayer(nn.Module):
    """Decoder layer: self-attn + cross-attn + xLSTM (FFN slot) + extra norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float,
        *,
        context_length: int,
        xlstm_blocks: int,
        slstm_positions: list[int] | None,
        force_fallback: bool,
    ) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_cross_q = nn.LayerNorm(d_model)
        self.norm_cross_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_xlstm = nn.LayerNorm(d_model)
        self.xlstm = _build_xlstm_block(
            d_model,
            context_length=context_length,
            xlstm_blocks=xlstm_blocks,
            slstm_positions=slstm_positions,
            n_heads=n_heads,
            dropout=dropout,
            force_fallback=force_fallback,
        )
        # Paper §2.4: "the proposed network incorporates an additional
        # normalization layer" beyond the standard Transformer decoder.
        self.norm_extra = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        # Paper §2.4: the decoder is **not autoregressive**; it
        # "reconstructs and reintegrates the encoded features". Self-attention
        # therefore runs over the full window without a causal mask.
        h = self.norm_self(y)
        a, _ = self.self_attn(h, h, h, need_weights=False)
        y = y + self.dropout(a)
        q = self.norm_cross_q(y)
        kv = self.norm_cross_kv(encoder_out)
        c, _ = self.cross_attn(q, kv, kv, need_weights=False)
        y = y + self.dropout(c)
        # §2.4 specifies Eqs. (15)–(18) for the **encoder** only. The decoder follows the usual
        # Transformer third sublayer: FFN — here xLSTM — on the state after self + cross
        # residuals (Vaswani et al.), not on the raw cross-attn delta alone.
        y_xlstm = self.xlstm(self.norm_xlstm(y))
        y = y + self.dropout(y_xlstm)
        return self.norm_extra(y)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class XLSTMTransformer(nn.Module):
    """Paper-faithful xLSTM-Transformer encoder-decoder baseline.

    Args:
        n_features: Input feature dim F.
        d_model: Hidden width. Paper uses 16-32 to keep param count small.
        n_heads: Heads in self/cross attention. ``d_model % n_heads == 0``.
        encoder_layers: ``N`` repetitions of the paper encoder layer.
        decoder_layers: ``N`` repetitions of the paper decoder layer. Set
            to 0 to skip the decoder entirely (lightweight encoder-only
            ablation).
        xlstm_blocks: Number of blocks inside each layer's xLSTM sub-stack.
            Paper uses 2 (one mLSTM + one sLSTM).
        slstm_positions: 0-indexed positions of sLSTM blocks within each
            sub-stack. Paper places sLSTM at position 1 (after mLSTM).
        head_hidden: Hidden dim of the regression MLP.
        head_pool: Pooling for the head: ``"flatten"`` (paper),
            ``"last"``, or ``"mean"``.
        context_length: Window length L. Required when ``head_pool="flatten"``.
        dropout: Shared dropout rate.
        ff_dim: Retained for backward-compatible YAMLs; unused (the paper's
            FFN sublayer is *replaced* by the xLSTM block).
        force_fallback: Bypass NX-AI's ``xlstm`` library and use the
            pure-PyTorch fallback (used by ablation A5 / Mac runs).
    """

    def __init__(
        self,
        n_features: int,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        xlstm_blocks: int = 2,
        slstm_positions: list[int] | None = None,
        head_hidden: int = 32,
        head_pool: str = "flatten",
        context_length: int = 10,
        dropout: float = 0.1,
        ff_dim: int = 64,  # noqa: ARG002  (kept for YAML compatibility)
        force_fallback: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)

        slstm_positions = slstm_positions if slstm_positions is not None else [xlstm_blocks // 2]

        self.encoder_layers = nn.ModuleList([
            XLSTMTransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                context_length=context_length,
                xlstm_blocks=xlstm_blocks,
                slstm_positions=slstm_positions,
                force_fallback=force_fallback,
            )
            for _ in range(encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            XLSTMTransformerDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                context_length=context_length,
                xlstm_blocks=xlstm_blocks,
                slstm_positions=slstm_positions,
                force_fallback=force_fallback,
            )
            for _ in range(decoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model) if encoder_layers > 0 else nn.Identity()

        self.head = SigmoidRegressionHead(
            d_model=d_model,
            hidden=head_hidden,
            dropout=dropout,
            pool=head_pool,
            context_length=context_length if head_pool == "flatten" else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: ``(B, L, F)`` input HI window.

        Returns:
            ``(B,)`` RUL predictions in ``[0, 1]``.
        """
        h = self.pos(self.input_proj(x))
        enc = h
        for layer in self.encoder_layers:
            enc = layer(enc)
        enc = self.encoder_norm(enc)
        if len(self.decoder_layers) == 0:
            dec = enc
        else:
            dec = h  # decoder shares input embedding with the encoder
            for layer in self.decoder_layers:
                dec = layer(dec, enc)
        return self.head(dec)

    @torch.no_grad()
    def encoder_first_layer_self_attention_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Average encoder layer-0 self-attention over batch: shape ``(L, L)``."""
        if not self.encoder_layers:
            raise ValueError("No encoder layers.")
        h = self.pos(self.input_proj(x))
        layer = self.encoder_layers[0]
        normed = layer.norm_attn(h)
        _attn_out, attn_w = layer.attn(
            normed,
            normed,
            normed,
            need_weights=True,
            average_attn_weights=True,
        )
        return attn_w.detach().mean(dim=0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
