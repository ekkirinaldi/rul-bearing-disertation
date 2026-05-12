"""Architecture diagrams and analytical charts for Bab III.

All diagrams are rendered as PNG into
``Mamba-xLSTM/results/_chapter_assets/diagrams/`` and reused on subsequent runs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ASSET_ROOT = Path(__file__).resolve().parents[2] / "results" / "_chapter_assets" / "diagrams"
ASSET_ROOT.mkdir(parents=True, exist_ok=True)

# Color palette - desaturated, prints well
C_BLUE = "#3a6ea5"
C_DARK = "#1a3550"
C_GREEN = "#3a8a64"
C_ORANGE = "#c97b3a"
C_RED = "#b04a4a"
C_PURPLE = "#7c4f9b"
C_GRAY = "#666666"
C_LIGHT = "#eef2f7"
C_LIGHT2 = "#f6efe5"
C_LIGHT3 = "#ecf5ee"


def _save(fig: plt.Figure, name: str, dpi: int = 200) -> Path:
    out = ASSET_ROOT / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def _fit_text_in_box(
    ax,
    txt,
    box_w: float,
    box_h: float,
    *,
    min_fontsize: float = 7.0,
    pad_ratio: float = 0.92,
) -> None:
    """Shrink ``txt`` fontsize until its rendered bounding box fits the box.

    Box dimensions are in data units of ``ax``. The padding ratio leaves a small
    margin between text and box edge so the text never visually touches the
    border. Stops at ``min_fontsize`` to avoid unreadable shrinkage.
    """

    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    inv = ax.transData.inverted()
    target_w = box_w * pad_ratio
    target_h = box_h * pad_ratio
    fs = txt.get_fontsize()
    for _ in range(40):
        bb_disp = txt.get_window_extent(renderer=renderer)
        bb_data = bb_disp.transformed(inv)
        if bb_data.width <= target_w and bb_data.height <= target_h:
            return
        if fs <= min_fontsize:
            return
        fs = max(min_fontsize, fs - 0.5)
        txt.set_fontsize(fs)


def _box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    facecolor=C_LIGHT,
    edgecolor=C_DARK,
    fontsize=10,
    bold=False,
    rounding=0.04,
    fit_text: bool = True,
    min_fontsize: float = 7.0,
):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    txt = ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=C_DARK,
        wrap=True,
    )
    if fit_text:
        _fit_text_in_box(ax, txt, width, height, min_fontsize=min_fontsize)
    return (x + width / 2, y + height / 2)


def _arrow(ax, start, end, *, color=C_DARK, label=None, lw=1.4, style="->", shrink=18):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=lw,
        shrinkA=shrink,
        shrinkB=shrink,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx + 0.02, my + 0.02, label, fontsize=8, color=color, ha="left", va="bottom")


# ============================================================
# III.2.2  Evolution timeline (2014-2025)
# ============================================================
def evolution_timeline() -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.set_xlim(2013.3, 2026.4)
    ax.set_ylim(-0.4, 5.6)
    ax.axis("off")

    ax.annotate(
        "",
        xy=(2026.2, 0.6),
        xytext=(2013.6, 0.6),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.4),
    )

    # (x_event, x_label, label, sub, color, y_top, side)
    # x_label nudges the text horizontally so co-located events do not overlap
    points = [
        (2014.0, 2014.0, "GRU", "Cho dkk., 2014\n$O(L \\cdot d^2)$", C_GRAY, 1.5),
        (2017.0, 2017.0, "Transformer", "Vaswani dkk., 2017\n$O(L^2 \\cdot d)$", C_BLUE, 2.7),
        (2023.0, 2023.0, "Mamba (SSM)", "Gu & Dao, 2023\n$O(L \\cdot d)$", C_GREEN, 1.5),
        (2024.0, 2023.7, "xLSTM", "Beck dkk., 2024\nexp. gating + matrix mem.", C_PURPLE, 4.6),
        (2024.0, 2024.6, "Mamba-2 (SSD)", "Dao & Gu, 2024\nstate-space duality", C_ORANGE, 3.0),
        (2025.0, 2025.0, "xLSTM\u2013Transformer", "Liu dkk., 2025\nbearing RUL baseline", C_RED, 1.6),
        (2025.5, 2025.7, "Mamba-2-xLSTM-Net", "Disertasi ini\n(usulan)", C_DARK, 4.6),
    ]

    for x_event, x_label, label, sub, color, y in points:
        # vertical line from timeline up to the label cluster
        ax.plot([x_event, x_event], [0.6, y - 0.45], color=color, lw=1.0, alpha=0.5)
        # diagonal connector to the (possibly nudged) label position
        ax.plot([x_event, x_label], [y - 0.45, y - 0.05], color=color, lw=1.0, alpha=0.5)
        ax.scatter([x_event], [0.6], s=70, color=color, zorder=5, edgecolor="white", linewidth=1.4)
        ax.text(x_label, y + 0.18, label, ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=color)
        ax.text(x_label, y - 0.05, sub, ha="center", va="top", fontsize=8, color=C_DARK)

    for yr in range(2014, 2026):
        ax.text(yr, 0.18, str(yr), ha="center", va="center", fontsize=8, color=C_GRAY)

    ax.set_title(
        "Evolusi pemodelan sekuens dalam pembelajaran mendalam (2014\u20132025)",
        fontsize=11,
        color=C_DARK,
        pad=8,
    )
    return _save(fig, "fig_evolution_timeline.png")


# ============================================================
# III.2.2  Complexity comparison O(L^2) vs O(L)
# ============================================================
def complexity_chart() -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    L = np.arange(50, 3001, 50)
    d = 128
    cost_attn = (L ** 2) * d
    cost_ssm = L * d

    ax.plot(L, cost_attn, color=C_RED, lw=2.0, label=r"Self-Attention (Transformer): $O(L^2 \cdot d)$")
    ax.plot(L, cost_ssm, color=C_GREEN, lw=2.0, label=r"Selective SSM (Mamba): $O(L \cdot d)$")
    ax.fill_between(L, cost_attn, cost_ssm, color=C_RED, alpha=0.07)

    ax.axvline(2800, color=C_GRAY, ls="--", lw=1.0, alpha=0.7)
    ax.text(2820, cost_attn[-1] * 0.6, "PHM 2012\n(\u22482800 akuisisi)", fontsize=9, color=C_GRAY)

    ax.set_xlabel("Panjang sekuens $L$ (time step)", fontsize=10)
    ax.set_ylabel("Biaya komputasi (operasi, $d=128$)", fontsize=10)
    ax.set_title(
        "Skala biaya komputasi: kuadratik (Attention) vs linear (Selective SSM)",
        fontsize=11,
        color=C_DARK,
    )
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    return _save(fig, "fig_complexity_chart.png")


# ============================================================
# III.3.3  RUL labeling schemes (linear vs piecewise)
# ============================================================
def rul_label_schemes() -> Path:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    T = 100
    t = np.linspace(0, T, T + 1)

    y_linear = 1 - t / T
    t_d = 70
    y_pw = np.where(t <= t_d, 1.0, np.maximum(0.0, (T - t) / (T - t_d)))

    ax.plot(t, y_linear, color=C_BLUE, lw=2.2, label="Skema linier")
    ax.plot(t, y_pw, color=C_GREEN, lw=2.2, label="Skema $piecewise\\_liu2026$ (digunakan)")
    ax.axvline(t_d, color=C_GRAY, ls="--", lw=1.0)
    ax.text(t_d + 1, 0.5, r"$t_d$ (degradation onset)", fontsize=9, color=C_GRAY)
    ax.axvline(T, color=C_RED, ls=":", lw=1.0)
    ax.text(T - 1, 0.05, r"$t_f$ (EOL)", fontsize=9, color=C_RED, ha="right")

    ax.set_xlabel("Waktu $t$ (akuisisi)", fontsize=10)
    ax.set_ylabel(r"Target RUL ternormalisasi $y_t \in [0,1]$", fontsize=10)
    ax.set_title("Dua skema pelabelan RUL", fontsize=11, color=C_DARK)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    return _save(fig, "fig_rul_label_schemes.png")


# ============================================================
# III.4.1  sLSTM block diagram - simplified conceptual flow
# ============================================================
def slstm_block() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, (0.3, 2.6), 1.7, 0.8, r"$x_t$", facecolor=C_LIGHT, fontsize=12, bold=True)
    _box(ax, (0.3, 4.4), 1.7, 0.9,
         r"$h_{t-1}$" + "\n" + r"$c_{t-1},\ n_{t-1}$",
         facecolor=C_LIGHT, fontsize=10)

    # Four gates - exponential gating innovation highlighted
    gates = [
        ("Input gate",      r"$i_t = \exp(\cdot)$",   5.0, C_ORANGE),
        ("Forget gate",     r"$f_t = \exp(\cdot)$",   3.9, C_ORANGE),
        ("Candidate",       r"$z_t = \tanh(\cdot)$",  2.8, C_GREEN),
        ("Output gate",     r"$o_t = \sigma(\cdot)$", 1.7, C_BLUE),
    ]
    for name, formula, y, color in gates:
        _box(ax, (3.2, y - 0.35), 3.2, 0.8, f"{name}\n{formula}",
             facecolor=C_LIGHT, edgecolor=color, fontsize=9.5, rounding=0.04)

    # Composition box
    _box(
        ax,
        (8.0, 2.4),
        3.7,
        2.2,
        "Komposisi sel skalar\n"
        + r"$c_t = f_t\,c_{t-1} + i_t\,z_t$" + "\n"
        + r"$n_t = f_t\,n_{t-1} + i_t$" + "\n"
        + r"$h_t = o_t \cdot (c_t / n_t)$",
        facecolor=C_LIGHT3,
        edgecolor=C_DARK,
        fontsize=10,
        rounding=0.06,
    )

    _arrow(ax, (2.0, 3.0), (3.2, 3.0), shrink=4)
    _arrow(ax, (2.0, 4.9), (3.2, 4.6), shrink=4)
    # bundled arrow from gates -> composition
    _arrow(ax, (6.4, 3.4), (8.0, 3.5), color=C_GRAY, shrink=4)
    ax.text(7.2, 3.7, "gabung", fontsize=8, color=C_GRAY)

    ax.set_title("Blok sLSTM (scalar LSTM, Beck dkk., 2024)",
                 fontsize=12, color=C_DARK, pad=6)
    ax.text(6.0, 0.6,
            "Inovasi utama: gating eksponensial mengatasi saturasi gradien sigmoid pada LSTM klasik.",
            ha="center", fontsize=9, color=C_DARK, style="italic")
    return _save(fig, "fig_slstm_block.png")


# ============================================================
# III.4.2  mLSTM block diagram with matrix memory
# ============================================================
def mlstm_block() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _box(ax, (0.3, 2.6), 1.6, 0.8, r"$x_t$", facecolor=C_LIGHT, fontsize=12, bold=True)

    # Query / Key / Value projection block
    _box(ax, (2.6, 3.5), 3.4, 1.7,
         "Proyeksi Query/Key/Value\n"
         + r"$q_t = W_q x_t$" + "\n"
         + r"$k_t = W_k x_t$" + "\n"
         + r"$v_t = W_v x_t$",
         facecolor=C_LIGHT, edgecolor=C_PURPLE, fontsize=10, rounding=0.05)

    # Exponential gates block
    _box(ax, (2.6, 1.5), 3.4, 1.5,
         "Gating eksponensial\n"
         + r"$i_t = \exp(\cdot)$" + "\n"
         + r"$f_t = \exp(\cdot)$",
         facecolor=C_LIGHT, edgecolor=C_ORANGE, fontsize=10, rounding=0.05)

    _arrow(ax, (1.9, 3.0), (2.6, 4.3), shrink=4)
    _arrow(ax, (1.9, 3.0), (2.6, 2.2), shrink=4)

    # Matrix memory box
    _box(
        ax,
        (6.7, 2.4),
        4.9,
        2.4,
        r"Matrix memory $\;C_t \in \mathbb{R}^{d \times d}$" + "\n"
        + r"$C_t = f_t\,C_{t-1} + i_t \cdot v_t k_t^\top$" + "\n"
        + r"$n_t = f_t\,n_{t-1} + i_t\,k_t$" + "\n"
        + r"$h_t = o_t \odot (C_t q_t) / \max(|n_t^\top q_t|, 1)$",
        facecolor=C_LIGHT3,
        edgecolor=C_DARK,
        fontsize=10,
        rounding=0.06,
    )

    _arrow(ax, (6.0, 4.3), (6.7, 3.8), color=C_PURPLE, shrink=4)
    _arrow(ax, (6.0, 2.2), (6.7, 3.0), color=C_ORANGE, shrink=4)

    ax.text(
        6.0,
        0.6,
        r"Memori matriks menyimpan asosiasi $key\text{–}value$ dengan kapasitas $d \times d$,"
        + " jauh lebih besar dari memori vektor LSTM klasik.",
        ha="center",
        fontsize=9,
        color=C_DARK,
        style="italic",
    )

    ax.set_title("Blok mLSTM (matrix LSTM, Beck dkk., 2024)",
                 fontsize=12, color=C_DARK, pad=6)
    return _save(fig, "fig_mlstm_block.png")


# ============================================================
# III.4.4  Baseline xLSTM-Transformer encoder-decoder
# ============================================================
def baseline_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    _box(ax, (4.0, 8.0), 4.0, 0.7, r"Input HI window: $x \in \mathbb{R}^{B \times L \times F}$",
         facecolor=C_LIGHT, fontsize=10, bold=True)
    _box(ax, (4.0, 7.0), 4.0, 0.6,
         r"Linear projection $F \rightarrow d_{\mathrm{model}}{=}32$",
         facecolor=C_LIGHT, fontsize=9.5)
    _box(ax, (4.0, 6.0), 4.0, 0.6, "Sinusoidal Positional Encoding",
         facecolor=C_LIGHT, fontsize=9.5)

    # Transformer encoder branch (left)
    _box(ax, (0.4, 3.6), 4.6, 1.8,
         "Transformer Encoder (1 layer)\nMulti-Head Self-Attention ($H{=}4$)\nFFN (hidden 64)\n"
         + r"Kompleksitas: $O(L^2 \cdot d)$",
         facecolor=C_LIGHT2, edgecolor=C_RED, fontsize=10, rounding=0.06)
    # xLSTM stack branch (right)
    _box(ax, (7.0, 3.6), 4.6, 1.8,
         "xLSTM stack (3 blok)\n2 \u00d7 mLSTM + 1 \u00d7 sLSTM\nExp. gating, matrix memory",
         facecolor=C_LIGHT3, edgecolor=C_GREEN, fontsize=10, rounding=0.06)

    _arrow(ax, (6.0, 6.0), (2.7, 5.4))
    _arrow(ax, (6.0, 6.0), (9.3, 5.4))

    # Decoder: widened so the cross-attention text never overflows.
    _box(ax, (1.5, 1.6), 9.0, 1.4,
         "Transformer Decoder (1 layer)\n"
         "Cross-Attention: query = xLSTM stack, key/value = Transformer encoder",
         facecolor=C_LIGHT, edgecolor=C_BLUE, fontsize=10, rounding=0.06)
    _arrow(ax, (2.7, 3.6), (4.0, 3.0), color=C_RED)
    _arrow(ax, (9.3, 3.6), (8.0, 3.0), color=C_GREEN)

    # Regression head: widened + line break so the long pipeline text fits.
    _box(ax, (1.5, 0.2), 9.0, 1.0,
         "Regression head\n"
         "LN \u2192 Linear(32) \u2192 GELU \u2192 Dropout \u2192 Linear(1) "
         "\u2192 Sigmoid \u2192 $\\hat{y}_t$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=9.5, rounding=0.06)
    _arrow(ax, (6.0, 1.6), (6.0, 1.2))

    ax.text(6.0, 8.85, "Baseline xLSTM\u2013Transformer (Liu dkk., 2025)",
            ha="center", fontsize=12, color=C_DARK, fontweight="bold")
    ax.text(11.8, 0.0, "Total parameter \u2248 43,4 K",
            ha="right", fontsize=8, color=C_GRAY, style="italic")
    return _save(fig, "fig_baseline_architecture.png")


# ============================================================
# III.5.2  Mamba selective SSM block
# ============================================================
def mamba_ssm_block() -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    _box(ax, (0.3, 2.9), 1.6, 0.8, r"$x_t$", facecolor=C_LIGHT, fontsize=12, bold=True)

    # Selective parameter generation block
    _box(ax, (2.6, 3.6), 4.0, 2.1,
         "Parameter selektif (input-dependent)\n"
         + r"$\Delta_t = \mathrm{softplus}(W_\Delta x_t)$" + "\n"
         + r"$B_t = W_B\,x_t$" + "\n"
         + r"$C_t = W_C\,x_t$",
         facecolor=C_LIGHT2, edgecolor=C_ORANGE, fontsize=10, rounding=0.05)

    # Discretization block
    _box(ax, (2.6, 1.3), 4.0, 1.9,
         "Diskritisasi (ZOH)\n"
         + r"$\bar{A}_t = \exp(\Delta_t A)$" + "\n"
         + r"$\bar{B}_t \approx \Delta_t B_t$" + "\n"
         + "(dependen-input)",
         facecolor=C_LIGHT, edgecolor=C_PURPLE, fontsize=10, rounding=0.05)

    _arrow(ax, (1.9, 3.3), (2.6, 4.6), shrink=4)
    _arrow(ax, (1.9, 3.3), (2.6, 2.2), shrink=4)

    # SSM recurrence block
    _box(ax, (7.4, 3.3), 4.3, 2.3,
         "State-Space Recurrence\n"
         + r"$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$" + "\n"
         + r"$y_t = C_t h_t$" + "\n"
         + r"Parallel scan: $O(L \cdot d)$",
         facecolor=C_LIGHT3, edgecolor=C_GREEN, fontsize=10, rounding=0.05)

    _arrow(ax, (6.6, 4.6), (7.4, 4.6), color=C_ORANGE, shrink=4)
    _arrow(ax, (6.6, 2.2), (7.4, 4.0), color=C_PURPLE, shrink=4)

    ax.text(
        6.0, 0.5,
        r"Selektif: $\Delta_t, B_t, C_t$ bergantung pada input $x_t$ sehingga model dapat"
        + " memilih informasi yang dipertahankan, ditolak, atau dilewatkan.",
        ha="center", fontsize=9, color=C_DARK, style="italic",
    )

    ax.set_title("Blok Mamba (Selective State Space Model, Gu & Dao 2023)",
                 fontsize=12, color=C_DARK, pad=6)
    return _save(fig, "fig_mamba_ssm_block.png")


# ============================================================
# III.5.3  Mamba-1 vs Mamba-2 comparison
# ============================================================
def mamba1_vs_mamba2() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Mamba-1 column
    ax.text(2.7, 5.6, "Mamba-1 (Gu & Dao, 2023)", ha="center", fontsize=11, color=C_PURPLE, fontweight="bold")
    items_m1 = [
        "Matriks $A$ penuh, parameter per dimensi",
        r"Parameter $\Delta, B, C$ dihitung sekuensial",
        "Scan rekuren; throughput moderat",
        "Implementasi via custom CUDA kernel",
    ]
    for i, t in enumerate(items_m1):
        _box(ax, (0.4, 4.6 - i * 0.95), 4.6, 0.75, t,
             facecolor=C_LIGHT, edgecolor=C_PURPLE, fontsize=9.5, rounding=0.04)

    # Mamba-2 column
    ax.text(9.0, 5.6, "Mamba-2 (Dao & Gu, 2024)", ha="center", fontsize=11, color=C_ORANGE, fontweight="bold")
    items_m2 = [
        "Restriksi $A_t = a_t \\cdot I$ (skalar \u00d7 identitas)",
        r"Parameter $\Delta, A, B, C$ dihitung paralel di awal",
        "Block matrix multiply \u2192 throughput 2\u20138\u00d7 lebih tinggi",
        "Structured State Space Duality (SSD) dengan attention",
    ]
    for i, t in enumerate(items_m2):
        _box(ax, (6.7, 4.6 - i * 0.95), 4.9, 0.75, t,
             facecolor=C_LIGHT2, edgecolor=C_ORANGE, fontsize=9.5, rounding=0.04)

    ax.text(6.0, 0.4,
            "Konsekuensi: Mamba-2 lebih hemat parameter dan lebih cepat dilatih dibanding Mamba-1, "
            "sambil menjaga ekspresivitas selective SSM.",
            ha="center", fontsize=9, color=C_DARK, wrap=True, style="italic")
    return _save(fig, "fig_mamba1_vs_mamba2.png")


# ============================================================
# III.5.4  Bidirectional Mamba
# ============================================================
def bidirectional_mamba() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, (4.5, 3.8), 3.0, 0.8, r"Input $x_{1:L}$", facecolor=C_LIGHT, fontsize=10, bold=True)

    _box(ax, (0.4, 2.0), 4.6, 1.0,
         r"Mamba forward: $h^{\rightarrow}_t = \mathrm{Mamba}(x_{1:t})$",
         facecolor=C_LIGHT3, edgecolor=C_GREEN, fontsize=10, rounding=0.05)
    _box(ax, (7.0, 2.0), 4.6, 1.0,
         r"Mamba backward: $h^{\leftarrow}_t = \mathrm{Mamba}(x_{t:L})$",
         facecolor=C_LIGHT3, edgecolor=C_BLUE, fontsize=10, rounding=0.05)

    _arrow(ax, (5.5, 3.8), (3.0, 3.0), color=C_GREEN)
    _arrow(ax, (6.5, 3.8), (9.0, 3.0), color=C_BLUE)

    _box(ax, (3.5, 0.4), 5.0, 1.0,
         r"$h^{\mathrm{BiMamba}}_t = \mathrm{Linear}\!\left([h^{\rightarrow}_t \,;\, h^{\leftarrow}_t]\right)$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=11, rounding=0.05)
    _arrow(ax, (2.8, 2.0), (5.0, 1.4), color=C_GREEN)
    _arrow(ax, (9.0, 2.0), (7.5, 1.4), color=C_BLUE)

    ax.set_title("Blok Bidirectional Mamba: konkatenasi forward/backward scan",
                 fontsize=11, color=C_DARK, pad=4)
    return _save(fig, "fig_bimamba.png")


# ============================================================
# III.5.5  Gated fusion mechanism
# ============================================================
def gated_fusion() -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    _box(ax, (0.4, 3.6), 4.0, 1.0,
         "Cabang xLSTM\n" + r"$h^{\mathrm{xLSTM}}_t$",
         facecolor=C_LIGHT3, edgecolor=C_GREEN, fontsize=10, rounding=0.05)
    _box(ax, (7.6, 3.6), 4.0, 1.0,
         "Cabang BiMamba-2\n" + r"$h^{\mathrm{BiMamba}}_t$",
         facecolor=C_LIGHT3, edgecolor=C_ORANGE, fontsize=10, rounding=0.05)

    _box(ax, (2.0, 2.0), 8.0, 1.0,
         r"Gate: $g_t = \sigma\!\left(W_g\,[\,h^{\mathrm{xLSTM}}_t \,;\, h^{\mathrm{BiMamba}}_t\,] + b_g\right)$",
         facecolor=C_LIGHT, edgecolor=C_PURPLE, fontsize=10, rounding=0.05)
    _arrow(ax, (2.4, 3.6), (4.5, 3.0), color=C_GREEN)
    _arrow(ax, (9.6, 3.6), (7.5, 3.0), color=C_ORANGE)

    _box(ax, (1.5, 0.4), 9.0, 1.0,
         r"$h^{\mathrm{fused}}_t = g_t \odot h^{\mathrm{xLSTM}}_t + (1 - g_t) \odot h^{\mathrm{BiMamba}}_t$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=11, rounding=0.05, bold=True)
    _arrow(ax, (6.0, 2.0), (6.0, 1.4), color=C_PURPLE)

    ax.text(6.0, 4.85, "Mekanisme Gated Fusion (Dual-Branch)",
            ha="center", fontsize=11, color=C_DARK, fontweight="bold")
    return _save(fig, "fig_gated_fusion.png")


# ============================================================
# III.5.7  Mamba-2-xLSTM-Net full architecture
# ============================================================
def proposed_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11, 7.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, (4.0, 9.0), 4.0, 0.7, r"Input HI window: $x \in \mathbb{R}^{B \times L \times F}$",
         facecolor=C_LIGHT, fontsize=10, bold=True)
    _box(ax, (4.0, 8.0), 4.0, 0.6,
         r"Linear projection $F \rightarrow d_{\mathrm{model}}{=}128$",
         facecolor=C_LIGHT, fontsize=9.5)
    _arrow(ax, (6.0, 9.0), (6.0, 8.6))

    # Branch A: xLSTM
    _box(ax, (0.4, 4.6), 4.6, 2.6,
         "Cabang A \u2014 xLSTM stack\n3 blok (2 mLSTM + 1 sLSTM)\nExp. gating, matrix memory\n"
         + "Kapasitas dinamika lokal",
         facecolor=C_LIGHT3, edgecolor=C_GREEN, fontsize=10, rounding=0.06)
    # Branch B: BiMamba-2
    _box(ax, (7.0, 4.6), 4.6, 2.6,
         "Cabang B \u2014 BiMamba-2 stack\n2 blok bidirectional\n"
         + r"$d_{\mathrm{state}}{=}128,\;\mathrm{headdim}{=}32$" + "\n"
         + r"Kompleksitas $O(L \cdot d)$",
         facecolor=C_LIGHT2, edgecolor=C_ORANGE, fontsize=10, rounding=0.06)

    _arrow(ax, (5.0, 8.0), (2.7, 7.2), color=C_GREEN)
    _arrow(ax, (7.0, 8.0), (9.3, 7.2), color=C_ORANGE)

    # Gated fusion - widened so the long fused-state formula fits cleanly.
    _box(ax, (1.5, 2.6), 9.0, 1.6,
         "Gated Fusion\n"
         + r"$g_t = \sigma\!\left(W_g\,[\,h^{\mathrm{xLSTM}}_t \,;\, h^{\mathrm{BiMamba}}_t\,]\right)$" + "\n"
         + r"$h^{\mathrm{fused}}_t = g_t \odot h^{\mathrm{xLSTM}}_t + (1 - g_t) \odot h^{\mathrm{BiMamba}}_t$",
         facecolor=C_LIGHT, edgecolor=C_PURPLE, fontsize=10, rounding=0.06)
    _arrow(ax, (2.7, 4.6), (4.0, 4.2), color=C_GREEN)
    _arrow(ax, (9.3, 4.6), (8.0, 4.2), color=C_ORANGE)

    # Regression head - widened. Two-line layout keeps the math on one row each
    # so matplotlib mathtext does not have to break inside a $...$ block.
    _box(ax, (2.0, 0.4), 8.0, 1.4,
         "Regression head\n"
         + r"$\mathrm{LN} \rightarrow \mathrm{Linear}(64) \rightarrow \mathrm{GELU} \rightarrow \mathrm{Dropout}(0.1)$" + "\n"
         + r"$\rightarrow \mathrm{Linear}(1) \rightarrow \sigma \rightarrow \hat{y}_t$",
         facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=10, rounding=0.06)
    _arrow(ax, (6.0, 2.6), (6.0, 1.8))

    ax.text(6.0, 9.85, "Mamba-2-xLSTM-Net (arsitektur usulan)",
            ha="center", fontsize=12.5, color=C_DARK, fontweight="bold")
    ax.text(11.8, 0.0, "Total parameter \u2248 811,6 K (PHM 2012) / 811,7 K (XJTU-SY)",
            ha="right", fontsize=8, color=C_GRAY, style="italic")
    return _save(fig, "fig_proposed_architecture.png")


# ============================================================
# III.6.1  Loss + training protocol summary
# ============================================================
def training_protocol() -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    items = [
        ("Loss",         r"$\mathrm{MSE}(y,\hat{y})$"),
        ("Optimizer",    r"Adam, lr $= 10^{-3}$, wd $= 0$"),
        ("Schedule",     "Konstan, 50 epoch\n(tanpa early stop)"),
        ("Batch / Seed", r"$B = 32$, seed $= 42$"),
        ("Grad clip",    r"$\|\nabla\|_2 \leq 1.0$"),
        ("Precision",    "FP32, GPU NVIDIA + CUDA"),
    ]
    cols = 3
    box_w, box_h = 3.6, 1.3
    margin_x = (12 - cols * box_w - (cols - 1) * 0.2) / 2
    for i, (k, v) in enumerate(items):
        r = i // cols
        c = i % cols
        x = margin_x + c * (box_w + 0.2)
        y = 2.8 - r * 1.5
        _box(ax, (x, y), box_w, box_h, f"{k}\n{v}",
             facecolor=C_LIGHT, edgecolor=C_DARK, fontsize=10, rounding=0.05)

    ax.text(6.0, 4.6, "Protokol pelatihan (identik untuk baseline & usulan)",
            ha="center", fontsize=11.5, color=C_DARK, fontweight="bold")
    return _save(fig, "fig_training_protocol.png")


# ============================================================
# III.7.4  PHM scoring asymmetric curve
# ============================================================
def phm_scoring_curve() -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    Er = np.linspace(-30, 30, 600)
    A = np.where(
        Er <= 0,
        np.exp(-np.log(0.5) * (Er / 5.0)),
        np.exp(np.log(0.5) * (Er / 20.0)),
    )
    ax.plot(Er, A, color=C_BLUE, lw=2.0)
    ax.fill_between(Er, 0, A, where=Er <= 0, color=C_GREEN, alpha=0.18, label="Early prediction (lenient)")
    ax.fill_between(Er, 0, A, where=Er > 0, color=C_RED, alpha=0.18, label="Late prediction (harsh)")
    ax.axvline(0, color=C_GRAY, ls="--", lw=1.0)
    ax.set_xlabel(r"Persentase error $Er = 100 \cdot (\mathrm{RUL}_{\mathrm{actual}} - \mathrm{RUL}_{\mathrm{pred}})/\mathrm{RUL}_{\mathrm{actual}}$", fontsize=9.5)
    ax.set_ylabel(r"Skor per-sampel $A_i$", fontsize=10)
    ax.set_title("PHM Score asimetris (IEEE PHM 2012, Nectoux dkk., 2012)",
                 fontsize=11, color=C_DARK)
    ax.set_xlim(-30, 30)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    return _save(fig, "fig_phm_scoring.png")


# ============================================================
# III.9  Research workflow diagram
# ============================================================
def research_workflow() -> Path:
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    box_w, box_h = 3.6, 1.0
    nodes = [
        ("n1", "1. Identifikasi masalah\n& kajian literatur", 6.0, 11.5, C_DARK, C_LIGHT),
        ("n2", "2. Pra-pemrosesan data\n(PHM 2012, XJTU-SY)", 6.0, 9.8, C_BLUE, C_LIGHT),
        ("n3a", "3a. Implementasi baseline\nxLSTM\u2013Transformer", 2.2, 8.0, C_RED, C_LIGHT2),
        ("n3b", "3b. Implementasi usulan\nMamba-2-xLSTM-Net", 9.8, 8.0, C_GREEN, C_LIGHT3),
        ("n4", "4. Studi Kasus I\nReplikasi baseline", 2.2, 6.1, C_RED, C_LIGHT2),
        ("n5", "5. Studi Kasus II\nPerbandingan langsung", 6.0, 6.1, C_PURPLE, C_LIGHT),
        ("n6", "6. Studi Kasus III\nAblasi + interpretabilitas\n(SHAP, SAE, IG)", 9.8, 6.1, C_ORANGE, C_LIGHT2),
        ("n7", "7. Sintesis hasil\n& pembahasan", 6.0, 4.0, C_DARK, C_LIGHT),
        ("n8", "8. Studi Kasus IV (rencana akhir)\nTransfer ke PT SKF Indonesia", 6.0, 2.2, C_BLUE, C_LIGHT),
        ("n9", "9. Kesimpulan & saran", 6.0, 0.6, C_DARK, C_LIGHT),
    ]
    coords = {}
    for nid, label, x, y, edge, fill in nodes:
        _box(ax, (x - box_w / 2, y - box_h / 2), box_w, box_h, label,
             facecolor=fill, edgecolor=edge, fontsize=9.5, rounding=0.05)
        coords[nid] = {
            "c": (x, y),
            "top": (x, y + box_h / 2),
            "bot": (x, y - box_h / 2),
            "l": (x - box_w / 2, y),
            "r": (x + box_w / 2, y),
        }

    def edge(a_anchor, b_anchor, color=C_GRAY, lw=1.1):
        arrow = FancyArrowPatch(
            a_anchor, b_anchor,
            arrowstyle="->", mutation_scale=14, color=color, linewidth=lw,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arrow)

    # Vertical edges
    edge(coords["n1"]["bot"], coords["n2"]["top"])
    edge(coords["n3a"]["bot"], coords["n4"]["top"])
    edge(coords["n5"]["bot"], coords["n7"]["top"])
    edge(coords["n7"]["bot"], coords["n8"]["top"])
    edge(coords["n8"]["bot"], coords["n9"]["top"])

    # Diagonal edges from n2 -> n3a/n3b
    edge(coords["n2"]["bot"], coords["n3a"]["top"])
    edge(coords["n2"]["bot"], coords["n3b"]["top"])

    # Horizontal: n4 -> n5, n3b -> n5 (diagonal), n5 -> n6, n6 -> n7 (diagonal)
    edge(coords["n4"]["r"], coords["n5"]["l"])
    edge(coords["n3b"]["bot"], coords["n5"]["top"])
    edge(coords["n5"]["r"], coords["n6"]["l"])
    edge(coords["n6"]["bot"], coords["n7"]["top"])

    ax.text(6.0, 12.2, "Diagram alur penelitian",
            ha="center", fontsize=12, color=C_DARK, fontweight="bold")
    return _save(fig, "fig_research_workflow.png")


def render_all() -> dict[str, Path]:
    """Render every diagram and return mapping of name -> Path."""

    return {
        "evolution_timeline": evolution_timeline(),
        "complexity_chart": complexity_chart(),
        "rul_label_schemes": rul_label_schemes(),
        "slstm_block": slstm_block(),
        "mlstm_block": mlstm_block(),
        "baseline_architecture": baseline_architecture(),
        "mamba_ssm_block": mamba_ssm_block(),
        "mamba1_vs_mamba2": mamba1_vs_mamba2(),
        "bidirectional_mamba": bidirectional_mamba(),
        "gated_fusion": gated_fusion(),
        "proposed_architecture": proposed_architecture(),
        "training_protocol": training_protocol(),
        "phm_scoring_curve": phm_scoring_curve(),
        "research_workflow": research_workflow(),
    }


if __name__ == "__main__":
    paths = render_all()
    for name, p in paths.items():
        print(f"  {name:25s} -> {p.relative_to(p.parents[3])} ({p.stat().st_size // 1024} KB)")
