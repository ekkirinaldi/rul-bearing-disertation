"""Per-backbone architecture diagrams for the JETS Q2 manuscript — horizontal,
neural-network style.

Generates three PNGs in the same dark-navy / soft-pastel palette as
``Mamba-xLSTM/results/_chapter_assets/diagrams/fig_proposed_architecture.png``,
but laid out **left → right** with per-layer sub-rectangles, tensor-shape
labels above the layer band, and explicit sub-blocks for every multi-step
process (Sigmoid Regression Head, Sparse Feature Gate, Order-preserving
Quantile Head, Trend / Wear / Shock stacks, etc.). Output files:

    * mamba_xlstm_net.png      — Branch A (xLSTM stack: mLSTM-mLSTM-sLSTM)
                                 ‖ Branch B (BiMamba-3 × 2) → Gated Fusion
                                 → Sigmoid Regression Head (LN ▸ Linear ▸ GELU
                                 ▸ Dropout ▸ Linear ▸ σ) → ŷ
    * nbeats_xlstm_rul.png     — xLSTM front-end (mLSTM × 2) → sequential
                                 Trend (Bernstein × 2) / Wear (Char-freq × 2)
                                 / Shock (Gabor × 2) decomposition →
                                 Forecast aggregation → Clamp[0,1] → ŷ
    * sparse_gate_tcn_rul.png  — parallel Sparse Feature Gate (Conv ▸ GELU
                                 ▸ Conv ▸ σ) ‖ Cross-Feature Attention →
                                 Gated input → 4 dilated TCN blocks (d = 1,
                                 2, 4, 8) → Last-step pool → Order-preserving
                                 Quantile Head (LN ▸ Linear ▸ GELU ▸ Linear)
                                 → ŷ = p₅₀

Run::

    python Mamba-xLSTM/scripts/journal_q2/draw_jets_backbone_architectures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIAGRAMS_DIR = _REPO_ROOT / "Mamba-xLSTM" / "scripts" / "chapters"
sys.path.insert(0, str(_DIAGRAMS_DIR))

from _diagrams import (  # type: ignore  # noqa: E402
    C_BLUE,
    C_DARK,
    C_GRAY,
    C_GREEN,
    C_LIGHT,
    C_LIGHT2,
    C_LIGHT3,
    C_ORANGE,
    C_PURPLE,
    C_RED,
    _arrow,
    _box,
)

OUT_DIR = _REPO_ROOT / "writings" / "journal-q2" / "jets-docs" / "figures" / "architectures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str, dpi: int = 220) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layout primitives tailored to a horizontal NN-style diagram
# ─────────────────────────────────────────────────────────────────────────────
def _shape(ax, x: float, y: float, text: str, *, color: str = C_GRAY) -> None:
    """Tensor-shape annotation (italic gray) anchored with va="bottom"."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=color,
        style="italic",
    )


def _flow(ax, x1: float, x2: float, y: float, *, color: str = C_DARK, lw: float = 1.3) -> None:
    """Short horizontal arrow between two adjacent layer rectangles."""
    arrow = FancyArrowPatch(
        (x1, y),
        (x2, y),
        arrowstyle="->",
        mutation_scale=11,
        color=color,
        linewidth=lw,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def _dashed_callout(ax, src: tuple[float, float], dst: tuple[float, float], *, color: str = C_GRAY) -> None:
    """Dashed arrow used for SAE-input callouts and forecast taps."""
    arrow = FancyArrowPatch(
        src,
        dst,
        arrowstyle="->",
        mutation_scale=10,
        color=color,
        linewidth=1.0,
        linestyle=(0, (4, 3)),
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arrow)


def _layer_seq(
    ax,
    x_left: float,
    y_center: float,
    layers: list[tuple[str, float]],
    *,
    header: str | None = None,
    sub_label: str | None = None,
    facecolor: str = C_LIGHT,
    edgecolor: str = C_DARK,
    fontsize: float = 9.0,
    box_h: float = 1.1,
    gap: float = 0.13,
    header_color: str | None = None,
    header_fontsize: float = 10.0,
    arrow_color: str | None = None,
) -> float:
    """Render a horizontal sequence of layer sub-rectangles.

    ``layers`` is a list of ``(label, width)`` tuples. The boxes are drawn
    in the given facecolor / edgecolor, separated by small arrows in
    ``arrow_color`` (defaults to ``edgecolor``). An optional header label
    sits above the row and an optional italic ``sub_label`` sits below it.

    Returns the x-coordinate of the right edge of the last sub-block.
    """
    x = x_left
    arr_color = arrow_color if arrow_color is not None else edgecolor
    for i, (lbl, w) in enumerate(layers):
        _box(
            ax,
            (x, y_center - box_h / 2),
            w,
            box_h,
            lbl,
            facecolor=facecolor,
            edgecolor=edgecolor,
            fontsize=fontsize,
            rounding=0.05,
        )
        if i < len(layers) - 1:
            x_from = x + w
            x_to = x + w + gap
            _flow(ax, x_from, x_to, y_center, color=arr_color, lw=0.9)
        x += w + gap
    x_right = x - gap

    total_w = x_right - x_left
    # Section headers are lifted well above the box top (offset 0.45) so they
    # do not collide with the tensor-shape labels that sit on the connecting
    # arrows just above the box tops (those labels live at +0.10 above the
    # box top, with text height ~0.16 → top at ~+0.26). 0.45 gives ~0.19
    # vertical clearance between shape-label tops and header bottoms.
    if header:
        ax.text(
            x_left + total_w / 2,
            y_center + box_h / 2 + 0.45,
            header,
            ha="center",
            va="bottom",
            fontsize=header_fontsize,
            color=header_color or edgecolor,
            fontweight="bold",
        )
    if sub_label:
        ax.text(
            x_left + total_w / 2,
            y_center - box_h / 2 - 0.18,
            sub_label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=C_DARK,
            style="italic",
        )
    return x_right


# ─────────────────────────────────────────────────────────────────────────────
# 1. Mamba-xLSTM-Net — horizontal NN-style with expanded regression head
# ─────────────────────────────────────────────────────────────────────────────
def mamba_xlstm_net() -> Path:
    fig, ax = plt.subplots(figsize=(16.5, 7.2))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    Y_main, Y_top, Y_bot = 3.4, 4.9, 1.9
    BOX_H = 1.1
    Y_shape = Y_main + BOX_H / 2 + 0.10  # 4.05 — clearly above box top at 3.95

    ax.text(
        8.25,
        6.85,
        "Mamba-xLSTM-Net (selective state-space + xLSTM, gated fusion)",
        ha="center",
        fontsize=12,
        color=C_DARK,
        fontweight="bold",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    _box(ax, (0.15, Y_main - BOX_H / 2), 1.15, BOX_H, "Input HI\nwindow",
         facecolor=C_LIGHT, fontsize=9.5, bold=True)
    _shape(ax, 1.55, Y_shape, r"$(B,L,F)$")
    _flow(ax, 1.30, 1.80, Y_main)

    # ── Linear projection F→128 ────────────────────────────────────────────
    _box(ax, (1.80, Y_main - BOX_H / 2), 1.15, BOX_H, "Linear\n$F\\!\\to\\!128$",
         facecolor=C_LIGHT, fontsize=9.5)
    _shape(ax, 3.27, Y_shape + 0.18, r"$(B,L,128)$")  # lifted further to clear the diverging arrows

    # ── Branch split ───────────────────────────────────────────────────────
    _flow(ax, 2.95, 3.60, Y_top, color=C_GREEN)
    _flow(ax, 2.95, 3.60, Y_bot, color=C_ORANGE)
    ax.plot([2.95, 2.95], [Y_main, Y_top], color=C_GREEN, lw=1.0, alpha=0.55)
    ax.plot([2.95, 2.95], [Y_main, Y_bot], color=C_ORANGE, lw=1.0, alpha=0.55)

    # ── Branch A — xLSTM stack: mLSTM ▸ mLSTM ▸ sLSTM ─────────────────────
    x_top_end = _layer_seq(
        ax, 3.60, Y_top,
        [("mLSTM", 0.92), ("mLSTM", 0.92), ("sLSTM", 0.92)],
        header="Branch A — xLSTM stack",
        sub_label=r"$n_{\mathrm{heads}}{=}4$, exp. gating, matrix memory",
        facecolor=C_LIGHT3, edgecolor=C_GREEN,
        fontsize=9.5, box_h=BOX_H, gap=0.14,
        header_color=C_GREEN,
    )

    # ── Branch B — BiMamba-3 × 2 (drawn manually so the BOLD-coloured branch
    #    tag sits BELOW the stack and the italic param-info ABOVE — the
    #    inverse of Branch A which is above the centre row) ───────────────
    x_bot = 3.60
    sub_w_b, sub_gap_b = 0.92, 0.14
    for i in range(2):
        _box(ax, (x_bot + i * (sub_w_b + sub_gap_b), Y_bot - BOX_H / 2),
             sub_w_b, BOX_H, "BiMamba-3",
             facecolor=C_LIGHT2, edgecolor=C_ORANGE, fontsize=9.5, rounding=0.05)
        if i < 1:
            x_from = x_bot + i * (sub_w_b + sub_gap_b) + sub_w_b
            x_to = x_bot + (i + 1) * (sub_w_b + sub_gap_b)
            _flow(ax, x_from, x_to, Y_bot, color=C_ORANGE, lw=1.0)
    bimamba_total_w = 2 * sub_w_b + sub_gap_b
    x_bot_end = x_bot + bimamba_total_w
    # Italic param-info ABOVE the stack (closer to centre row)
    ax.text(
        x_bot + bimamba_total_w / 2,
        Y_bot + BOX_H / 2 + 0.18,
        r"$d_{\mathrm{state}}{=}128,\;\mathrm{headdim}{=}64,\;O(L\!\cdot\!d)$",
        ha="center", va="bottom", fontsize=8.5, color=C_DARK, style="italic",
    )
    # Bold orange branch tag BELOW the stack
    ax.text(
        x_bot + bimamba_total_w / 2,
        Y_bot - BOX_H / 2 - 0.18,
        "Branch B — BiMamba-3 stack",
        ha="center", va="top", fontsize=10, color=C_ORANGE, fontweight="bold",
    )

    # ── Branch merge into Gated Fusion ─────────────────────────────────────
    x_fusion = max(x_top_end, x_bot_end) + 0.55
    ax.plot([x_top_end + 0.05, x_fusion - 0.05], [Y_top, Y_main + 0.4],
            color=C_GREEN, lw=1.2)
    ax.plot([x_bot_end + 0.05, x_fusion - 0.05], [Y_bot, Y_main - 0.4],
            color=C_ORANGE, lw=1.2)

    fusion_w = 2.55
    _box(
        ax,
        (x_fusion, Y_main - 0.7),
        fusion_w,
        1.4,
        "Gated Fusion\n"
        + r"$g_t = \sigma(W_g[h^{\mathrm{xLSTM}}_t;h^{\mathrm{Mamba}}_t])$"
        + "\n"
        + r"$h^{\mathrm{fused}}_t = g_t \odot h^{\mathrm{x}} + (1{-}g_t) \odot h^{\mathrm{M}}$",
        facecolor=C_LIGHT,
        edgecolor=C_PURPLE,
        fontsize=8.5,
        rounding=0.06,
    )
    x_fusion_end = x_fusion + fusion_w
    _shape(ax, x_fusion_end + 0.30, Y_shape, r"$(B,L,128)$")
    _flow(ax, x_fusion_end, x_fusion_end + 0.55, Y_main, color=C_PURPLE)

    # ── Sigmoid Regression Head — expanded into 6 explicit sub-blocks ─────
    head_x = x_fusion_end + 0.55
    head_x_end = _layer_seq(
        ax, head_x, Y_main,
        [
            ("LN", 0.45),
            ("Linear\n(64)", 0.65),
            ("GELU", 0.45),
            ("Dropout\n(0.1)", 0.65),
            ("Linear\n(1)", 0.55),
            (r"$\sigma$", 0.40),
        ],
        header="Sigmoid Regression Head",
        sub_label=None,
        facecolor=C_LIGHT, edgecolor=C_DARK,
        fontsize=8.5, box_h=BOX_H, gap=0.12,
        header_color=C_DARK,
    )

    # ── Output ─────────────────────────────────────────────────────────────
    _flow(ax, head_x_end, head_x_end + 0.40, Y_main)
    _box(ax, (head_x_end + 0.40, Y_main - 0.55), 1.20, BOX_H,
         r"$\hat{y}_t \in [0,1]$", facecolor=C_LIGHT, fontsize=10.5, bold=True)

    # ── SAE-input dashed callout from Gated Fusion ─────────────────────────
    callout_y = 0.55
    fusion_cx = x_fusion + fusion_w / 2
    _dashed_callout(ax, (fusion_cx, Y_main - 0.7), (fusion_cx, callout_y + 0.20))
    ax.text(
        fusion_cx,
        callout_y - 0.05,
        "$h^{\\mathrm{fused}}_t \\in \\mathbb{R}^{128}$ feeds the post-hoc Top-$k$ SAE",
        ha="center",
        va="top",
        fontsize=9,
        color=C_GRAY,
        style="italic",
    )

    return _save(fig, "mamba_xlstm_net.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. N-BEATS-xLSTM-RUL — horizontal NN-style with expanded decomposition stacks
# ─────────────────────────────────────────────────────────────────────────────
def nbeats_xlstm_rul() -> Path:
    fig, ax = plt.subplots(figsize=(17.5, 7.0))
    ax.set_xlim(0, 17.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    Y_main = 4.6   # main left → right flow row
    Y_agg = 1.7    # forecast-aggregation + clamp + output row
    BOX_H = 1.1
    Y_shape = Y_main + BOX_H / 2 + 0.10

    ax.text(
        8.75,
        6.65,
        "N-BEATS-xLSTM-RUL (basis-block decomposition with xLSTM front-end)",
        ha="center",
        fontsize=12,
        color=C_DARK,
        fontweight="bold",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    _box(ax, (0.15, Y_main - BOX_H / 2), 1.0, BOX_H, "Input HI\nwindow",
         facecolor=C_LIGHT, fontsize=9.5, bold=True)
    _shape(ax, 1.40, Y_shape, r"$(B,L,F)$")
    _flow(ax, 1.15, 1.65, Y_main)

    # ── Linear projection F→64 ─────────────────────────────────────────────
    _box(ax, (1.65, Y_main - BOX_H / 2), 1.0, BOX_H, "Linear\n$F\\!\\to\\!64$",
         facecolor=C_LIGHT, fontsize=9.5)
    _shape(ax, 2.90, Y_shape, r"$(B,L,64)$")
    _flow(ax, 2.65, 3.15, Y_main)

    # ── xLSTM front-end (2 × mLSTM, residual + LN) ─────────────────────────
    x_front_end = _layer_seq(
        ax, 3.15, Y_main,
        [("mLSTM", 0.85), ("mLSTM", 0.85)],
        header="xLSTM front-end",
        sub_label=r"residual + LN, $n_{\mathrm{heads}}{=}4$",
        facecolor=C_LIGHT3, edgecolor=C_GREEN,
        fontsize=9.5, box_h=BOX_H, gap=0.14,
        header_color=C_GREEN,
    )
    _shape(ax, x_front_end + 0.28, Y_shape, r"$z\in\mathbb{R}^{B\times L\times 64}$")
    _flow(ax, x_front_end, x_front_end + 0.55, Y_main, color=C_GREEN)

    # ── Three decomposition stacks ────────────────────────────────────────
    # Each stack rendered as 2 sub-blocks via _layer_seq.
    stacks = [
        {
            "name": "Trend stack",
            "labels": ("Bernstein", "Bernstein"),
            "sub": "$\\mathrm{deg}{=}4$, monotone\n(slow-aging baseline)",
            "fc": C_LIGHT2, "ec": C_ORANGE,
            "f": "$f_T$", "r": "$r_1$",
        },
        {
            "name": "Wear stack",
            "labels": ("Char-freq", "Char-freq"),
            "sub": "$\\{\\sin,\\cos\\}(2\\pi f_\\omega t)$\n$\\omega\\!\\in\\!\\{$BPFO,BPFI,BSF,FTF$\\}$",
            "fc": C_LIGHT, "ec": C_PURPLE,
            "f": "$f_W$", "r": "$r_2$",
        },
        {
            "name": "Shock stack",
            "labels": ("Gabor", "Gabor"),
            "sub": "$n_{\\mathrm{basis}}{=}14$, kurt-gated\n(transient bursts)",
            "fc": C_LIGHT3, "ec": C_BLUE,
            "f": "$f_S$", "r": "$r_3$",
        },
    ]
    sub_w, sub_gap, inter_gap = 1.0, 0.14, 0.65
    forecast_xs: list[float] = []
    x_stack = x_front_end + 0.55
    for i, s in enumerate(stacks):
        x_end = _layer_seq(
            ax, x_stack, Y_main,
            [(s["labels"][0], sub_w), (s["labels"][1], sub_w)],
            header=s["name"], sub_label=s["sub"],
            facecolor=s["fc"], edgecolor=s["ec"],
            fontsize=9, box_h=BOX_H, gap=sub_gap,
            header_color=s["ec"],
        )
        # Forecast tap (downward dashed arrow)
        cx = (x_stack + x_end) / 2
        _dashed_callout(ax, (cx, Y_main - BOX_H / 2),
                        (cx, Y_agg + 0.65), color=s["ec"])
        ax.text(cx + 0.06, (Y_main - BOX_H / 2 + Y_agg + 0.6) / 2,
                s["f"], ha="left", va="center",
                fontsize=10, color=s["ec"], fontweight="bold", style="italic")
        forecast_xs.append(cx)

        # Residual r_{i+1} flows right into the next stack (or the residual term)
        if i < len(stacks) - 1:
            x_from, x_to = x_end, x_end + inter_gap
            _flow(ax, x_from, x_to, Y_main, color=C_GRAY, lw=1.2)
            ax.text((x_from + x_to) / 2, Y_main + 0.18,
                    s["r"], ha="center", va="bottom",
                    fontsize=9, color=C_DARK, style="italic")
        x_stack = x_end + inter_gap
    x_shock_end = x_stack - inter_gap  # right edge of Shock stack

    # ── r_3 → 0.01·W·r_3 dashed callout from Shock stack down to aggregate ──
    r3_cx = x_shock_end + 0.05
    _dashed_callout(
        ax,
        (r3_cx, Y_main - BOX_H / 2),
        (r3_cx, Y_agg + 0.65),
        color=C_GRAY,
    )
    ax.text(
        r3_cx + 0.08,
        (Y_main - BOX_H / 2 + Y_agg + 0.6) / 2,
        "$0.01\\,Wr_3$",
        ha="left", va="center",
        fontsize=9, color=C_DARK, fontweight="bold", style="italic",
    )

    # ── Forecast aggregation (additive) — sits below, spans the 3 forecast taps ──
    agg_x_left = forecast_xs[0] - 1.4
    agg_x_right = max(r3_cx, forecast_xs[-1]) + 0.6
    agg_w = agg_x_right - agg_x_left
    _box(ax, (agg_x_left, Y_agg - 0.65), agg_w, 1.3,
         "Forecast aggregation (additive)\n"
         + r"$\hat{y}_{\mathrm{raw}} = f_T + f_W + f_S + 0.01\,Wr_3 + b$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=10, rounding=0.06)

    # ── Aggregate → Clamp → Output (on Y_agg row) ─────────────────────────
    _flow(ax, agg_x_right, agg_x_right + 0.55, Y_agg)
    clamp_x = agg_x_right + 0.55
    _box(ax, (clamp_x, Y_agg - 0.55), 1.6, BOX_H,
         "Clamp\n$\\hat{y}\\in[0,1]$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=9.5, rounding=0.05)
    clamp_end = clamp_x + 1.6
    _flow(ax, clamp_end, clamp_end + 0.40, Y_agg)
    _box(ax, (clamp_end + 0.40, Y_agg - 0.5), 1.0, 1.0,
         r"$\hat{y}_t$", facecolor=C_LIGHT, fontsize=10.5, bold=True)

    # ── SAE-input callout from xLSTM front-end output ─────────────────────
    sae_cx = x_front_end - 0.30
    callout_y = 0.55
    _dashed_callout(ax, (sae_cx, Y_main - BOX_H / 2),
                    (sae_cx, callout_y + 0.20))
    ax.text(
        sae_cx,
        callout_y - 0.05,
        "$z_t \\in \\mathbb{R}^{64}$ feeds the post-hoc Top-$k$ SAE",
        ha="center",
        va="top",
        fontsize=9,
        color=C_GRAY,
        style="italic",
    )

    return _save(fig, "nbeats_xlstm_rul.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. SparseGate-TCN-RUL — horizontal NN-style with expanded gate + quantile head
# ─────────────────────────────────────────────────────────────────────────────
def sparse_gate_tcn_rul() -> Path:
    fig, ax = plt.subplots(figsize=(19.6, 7.2))
    ax.set_xlim(0, 19.6)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    Y_main, Y_top, Y_bot = 3.4, 4.9, 1.9
    BOX_H = 1.1
    Y_shape = Y_main + BOX_H / 2 + 0.10

    ax.text(
        9.8,
        6.85,
        "SparseGate-TCN-RUL (sparse feature gate + dilated TCN + quantile head)",
        ha="center",
        fontsize=12,
        color=C_DARK,
        fontweight="bold",
    )

    # ── Input ──────────────────────────────────────────────────────────────
    _box(ax, (0.15, Y_main - BOX_H / 2), 1.15, BOX_H, "Input HI\nwindow",
         facecolor=C_LIGHT, fontsize=9.5, bold=True)
    _shape(ax, 1.55, Y_shape, r"$(B,T,F{=}16)$")

    # ── Branch split (above and below the centre row) ──────────────────────
    ax.plot([1.30, 1.85], [Y_main, Y_main], color=C_DARK, lw=1.0)
    ax.plot([1.85, 1.85], [Y_main, Y_top], color=C_GREEN, lw=1.1, alpha=0.7)
    ax.plot([1.85, 1.85], [Y_main, Y_bot], color=C_ORANGE, lw=1.1, alpha=0.7)
    _flow(ax, 1.85, 2.30, Y_top, color=C_GREEN)
    _flow(ax, 1.85, 2.30, Y_bot, color=C_ORANGE)

    # ── Sparse Feature Gate (top) — expanded into 4 sub-blocks ────────────
    gate_x_end = _layer_seq(
        ax, 2.30, Y_top,
        [
            ("Conv1D\n$F\\!\\to\\!32$\n$k{=}5$", 0.95),
            ("GELU", 0.45),
            ("Conv1D\n$32\\!\\to\\!F$\n$k{=}1$", 0.95),
            (r"$\sigma$", 0.40),
        ],
        header="Sparse Feature Gate",
        sub_label=r"$g_t \in [0,1]^F$ per timestep   (L1 + entropy penalty)",
        facecolor=C_LIGHT3, edgecolor=C_GREEN,
        fontsize=8.5, box_h=BOX_H, gap=0.13,
        header_color=C_GREEN,
    )

    # ── Cross-Feature Attention (bottom) — single MHSA block, width-matched ─
    attn_w = gate_x_end - 2.30
    _box(
        ax,
        (2.30, Y_bot - BOX_H / 2),
        attn_w,
        BOX_H,
        "Cross-Feature Attention\n"
        + r"per-timestep MHSA over $F$ tokens"
        + "\n"
        + r"$d_{\mathrm{model}}{=}32,\;H{=}4$",
        facecolor=C_LIGHT2,
        edgecolor=C_ORANGE,
        fontsize=9,
        rounding=0.06,
    )
    ax.text(
        2.30 + attn_w / 2,
        Y_bot + BOX_H / 2 + 0.18,
        "Cross-Feature Attention branch",
        ha="center", va="bottom",
        fontsize=10, color=C_ORANGE, fontweight="bold",
    )
    ax.text(
        2.30 + attn_w / 2,
        Y_bot - BOX_H / 2 - 0.18,
        r"scores $A_{t,i,j}$: feature interactions",
        ha="center", va="top",
        fontsize=8.5, color=C_DARK, style="italic",
    )

    # ── Branches merge into Gated Input ───────────────────────────────────
    x_branch_end = gate_x_end
    x_merge = x_branch_end + 0.55
    ax.plot([x_branch_end + 0.05, x_merge - 0.05], [Y_top, Y_main + 0.30],
            color=C_GREEN, lw=1.2)
    ax.plot([x_branch_end + 0.05, x_merge - 0.05], [Y_bot, Y_main - 0.30],
            color=C_ORANGE, lw=1.2)

    gated_w = 2.30
    _box(
        ax,
        (x_merge, Y_main - BOX_H / 2),
        gated_w,
        BOX_H,
        "Gated input\n"
        + r"$\tilde{x}_t = g_t \odot x_t + \mathrm{CrossAttn}(x_t)$",
        facecolor=C_LIGHT,
        edgecolor=C_PURPLE,
        fontsize=9,
        rounding=0.06,
        bold=True,
    )
    x_gated_end = x_merge + gated_w
    _shape(ax, x_gated_end + 0.30, Y_shape, r"$(B,T,F)$")
    _flow(ax, x_gated_end, x_gated_end + 0.55, Y_main, color=C_PURPLE)

    # ── 4 dilated TCN blocks (d = 1, 2, 4, 8) ─────────────────────────────
    tcn_x_end = _layer_seq(
        ax, x_gated_end + 0.55, Y_main,
        [
            ("TCN\n$d{=}1$\n$c{=}64$", 0.85),
            ("TCN\n$d{=}2$\n$c{=}64$", 0.85),
            ("TCN\n$d{=}4$\n$c{=}128$", 0.85),
            ("TCN\n$d{=}8$\n$c{=}128$", 0.85),
        ],
        header="Dilated causal TCN backbone (4 residual blocks)",
        sub_label=(
            r"WeightNorm-CausalConv $\to$ GELU $\to$ Dropout $\to$ residual"
            + "\n"
            + r"receptive field $\approx 45$ timesteps"
        ),
        facecolor=C_LIGHT, edgecolor=C_BLUE,
        fontsize=9, box_h=BOX_H, gap=0.13,
        header_color=C_BLUE,
    )
    _shape(ax, tcn_x_end + 0.30, Y_shape, r"$(B,T,128)$")
    _flow(ax, tcn_x_end, tcn_x_end + 0.55, Y_main, color=C_BLUE)

    # ── Last-step pool ────────────────────────────────────────────────────
    pool_x = tcn_x_end + 0.55
    pool_w = 1.20
    _box(
        ax,
        (pool_x, Y_main - BOX_H / 2),
        pool_w,
        BOX_H,
        "Last-step\npool",
        facecolor=C_LIGHT,
        edgecolor=C_DARK,
        fontsize=9,
        rounding=0.05,
    )
    pool_end = pool_x + pool_w
    _shape(ax, pool_end + 0.25, Y_shape, r"$h_T \in \mathbb{R}^{128}$")
    _flow(ax, pool_end, pool_end + 0.45, Y_main)

    # ── Order-preserving Quantile Head — expanded into 4 sub-blocks ───────
    head_x = pool_end + 0.45
    head_x_end = _layer_seq(
        ax, head_x, Y_main,
        [
            ("LN", 0.45),
            ("Linear\n(64)", 0.65),
            ("GELU", 0.45),
            ("Linear\n(3)", 0.65),
        ],
        header="Order-preserving Quantile Head",
        sub_label=r"$\to \{p_{05},p_{50},p_{95}\}$",
        facecolor=C_LIGHT, edgecolor=C_DARK,
        fontsize=8.5, box_h=BOX_H, gap=0.13,
        header_color=C_DARK,
    )

    # ── Output ────────────────────────────────────────────────────────────
    _flow(ax, head_x_end, head_x_end + 0.40, Y_main)
    _box(
        ax,
        (head_x_end + 0.40, Y_main - 0.55),
        1.20,
        BOX_H,
        r"$\hat{y}{=}p_{50}$",
        facecolor=C_LIGHT,
        fontsize=10.5,
        bold=True,
    )

    # ── SAE-input callout from last TCN block ─────────────────────────────
    sae_cx = tcn_x_end - 0.85 / 2  # mid of the last (d=8) TCN block
    callout_y = 0.55
    _dashed_callout(
        ax,
        (sae_cx, Y_main - BOX_H / 2),
        (sae_cx, callout_y + 0.20),
    )
    ax.text(
        sae_cx,
        callout_y - 0.05,
        "$h_t \\in \\mathbb{R}^{128}$ feeds the post-hoc Top-$k$ SAE",
        ha="center",
        va="top",
        fontsize=9,
        color=C_GRAY,
        style="italic",
    )

    return _save(fig, "sparse_gate_tcn_rul.png")


def main() -> None:
    paths = [
        mamba_xlstm_net(),
        nbeats_xlstm_rul(),
        sparse_gate_tcn_rul(),
    ]
    print("Saved (horizontal NN-style, expanded multi-step heads):")
    for p in paths:
        print(f"  {p}  ({p.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
