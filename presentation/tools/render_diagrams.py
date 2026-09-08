"""Render the paper-style algorithm diagrams for the defence deck.

Every method in the deck gets a block diagram in the visual idiom of
Jiang dkk. (2026), Sensors 26:1578 — rounded colour-coded blocks, dashed
repeat containers, zoom-in panels, and a legend row decoding the colours.
One master palette is shared across all figures so the family reads as one
system, harmonised with the deck theme (navy ink, gold accents).

Content is grounded in the V14 manuscript: every number drawn here appears
in V14 (Subbab IV.4, V.1, V.2, V.3, V.5.3). Structure-only detail follows
Gu dan Dao (2023), Beck dkk. (2024), Oreshkin dkk. (2020), Bai dkk. (2018).

Usage: python3 tools/render_diagrams.py [--only name1,name2]  (or ``make diagrams``)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "diagrams"

# --------------------------------------------------------------------------
# Palette — one master mapping from component function to pastel fill
# --------------------------------------------------------------------------
INK = "#16284B"      # deck navy: text, arrows, outlines
MUTED = "#6B7689"
GOLD = "#B8923B"
GOLD_FILL = "#E6D6A8"

KIND = {
    "input":  {"fc": "#FBF0D3", "ec": "#C9B06B"},   # kuning  — input / embedding
    "proj":   {"fc": "#E7E1F4", "ec": "#A79BCF"},   # ungu    — projection / convolution / FC
    "seq":    {"fc": "#D9EAF7", "ec": "#7FA3C4"},   # biru    — sequence modeling
    "mem":    {"fc": "#DFEEDF", "ec": "#7FAF8A"},   # hijau   — memory / state
    "gate":   {"fc": "#F8E1E6", "ec": "#CE9AA8"},   # merah muda — gating / normalization
    "out":    {"fc": "#CBE3D3", "ec": "#3E8E5B"},   # hijau tua — output / head
    "util":   {"fc": "#EEF1F5", "ec": "#9AA5B5"},   # abu     — pooling / utilitas
    "active": {"fc": GOLD_FILL, "ec": GOLD},        # emas    — active feature
}

FS_TITLE = 12.5   # panel / container titles
FS_BLOCK = 10.2   # block labels
FS_NOTE = 9.2     # annotations, legend
FS_TINY = 8.2


# --------------------------------------------------------------------------
# Foreign-term italics (ITB rule for research writing)
# --------------------------------------------------------------------------
# A dissertation italicises foreign terms, figure labels included, so every
# English term in these diagrams is marked [[like this]] and rendered italic.
# Exempt and therefore left upright: acronyms (SAE, WDCNN, RUL, BPFO), model
# and product names (Mamba, xLSTM, N-BEATS, XGBoost), code identifiers
# (Conv1D, ReLU, BatchNorm, Softmax), math symbols, and units.
_SPAN = re.compile(r"\[\[(.+?)\]\]")


def fr(text: str, bold: bool = False) -> str:
    """Render the ``[[marked]]`` foreign spans of `text` in italic.

    Marked-up strings become mathtext, which under the default dejavusans
    fontset is the italic cut of the very font the rest of the label uses, so
    the mix is seamless. A bold label needs ``\\boldsymbol`` instead of
    ``\\mathit`` or its foreign words would drop back to book weight. Hyphens
    and slashes stay upright outside the math span, which keeps
    ``cross-feature attention`` readable.
    """
    cmd = "\\boldsymbol" if bold else "\\mathit"

    def swap(match: "re.Match[str]") -> str:
        span = match.group(1)
        if not re.fullmatch(r"[A-Za-z0-9 \-/]+", span):
            raise ValueError(f"unsupported characters in a foreign span: {span!r}")
        out = []
        for chunk in re.split(r"([-/])", span):
            if chunk in "-/":
                out.append(chunk)
            elif chunk.strip():
                out.append("$%s{%s}$" % (cmd, chunk.strip().replace(" ", "\\ ")))
        return "".join(out)

    return _SPAN.sub(swap, text)


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def new_fig(w: float, h: float, pad_bottom: float = 0.0):
    """A canvas whose data units are inches, optionally with a band below y=0.

    `pad_bottom` grows the figure downwards and shifts the y limit into
    negative territory, so every coordinate already written in a draw function
    keeps its meaning and the input strip gets a lane of its own underneath.
    """
    fig = plt.figure(figsize=(w, h + pad_bottom))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0, w)
    ax.set_ylim(-pad_bottom, h)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, -pad_bottom), w, h + pad_bottom,
                           fc="white", ec="none", zorder=-10))
    # Every label in every diagram goes through ax.text (box, container and
    # legend_row included), so hooking it here italicises the foreign spans
    # once, for all nine figures, with no call site left to forget.
    plain = ax.text

    def text(x, y, s, *a, **k):
        bold = "bold" in str(k.get("fontweight", k.get("weight", "")))
        return plain(x, y, fr(s, bold=bold), *a, **k)

    ax.text = text
    return fig, ax


def box(ax, cx, cy, w, h, text, kind="seq", fs=FS_BLOCK, bold=False, sub=None, zorder=3):
    c = KIND[kind]
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.07",
        fc=c["fc"], ec=c["ec"], lw=1.2, zorder=zorder,
    ))
    ty = cy if sub is None else cy + 0.11
    ax.text(cx, ty, text, ha="center", va="center", fontsize=fs, color=INK,
            fontweight="bold" if bold else "normal", zorder=zorder + 1, linespacing=1.25)
    if sub:
        ax.text(cx, cy - 0.16, sub, ha="center", va="center", fontsize=FS_TINY,
                color=MUTED, zorder=zorder + 1, linespacing=1.2)


def arrow(ax, p1, p2, color=INK, lw=1.2, style="-|>", shrink=3, zorder=2, ls="solid"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
        shrinkA=shrink, shrinkB=shrink, zorder=zorder, linestyle=ls,
    ))


def container(ax, x, y, w, h, title=None, times=None, ec=MUTED, zorder=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.09",
        fc="none", ec=ec, lw=1.1, ls=(0, (5, 3)), zorder=zorder,
    ))
    if title:
        ax.text(x + 0.10, y + h - 0.06, title, ha="left", va="top",
                fontsize=FS_NOTE, color=MUTED, zorder=zorder + 1)
    if times:
        ax.text(x + w - 0.02, y + h + 0.07, times, ha="right", va="bottom",
                fontsize=FS_TITLE, color=INK, fontweight="bold", zorder=zorder + 1)


def gate_glyph(ax, cx, cy, symbol, kind="gate", r=0.15, fs=8):
    c = KIND[kind]
    ax.add_patch(Circle((cx, cy), r, fc=c["fc"], ec=c["ec"], lw=1.2, zorder=4))
    ax.text(cx, cy, symbol, ha="center", va="center", fontsize=fs, color=INK, zorder=5)


def matrix_glyph(ax, cx, cy, s=0.52, n=3, kind="mem", zorder=4):
    c = KIND[kind]
    ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s, fc=c["fc"], ec=c["ec"],
                           lw=1.2, zorder=zorder))
    for i in range(1, n):
        t = i * s / n
        ax.plot([cx - s / 2 + t] * 2, [cy - s / 2, cy + s / 2],
                color=c["ec"], lw=0.7, ls=(0, (2, 2)), zorder=zorder + 1)
        ax.plot([cx - s / 2, cx + s / 2], [cy - s / 2 + t] * 2,
                color=c["ec"], lw=0.7, ls=(0, (2, 2)), zorder=zorder + 1)


def zoom_link(ax, small_pt_top, small_pt_bot, panel_pt_top, panel_pt_bot):
    for a, b in ((small_pt_top, panel_pt_top), (small_pt_bot, panel_pt_bot)):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=MUTED, lw=0.9,
                ls=(0, (4, 3)), zorder=0)


def legend_row(ax, x, y, entries, dx=None, glyph_s=0.17):
    """entries: list of (kind-or-glyph, label). Laid out left-to-right from x."""
    cx = x
    for kind, text in entries:
        if kind in KIND:
            c = KIND[kind]
            ax.add_patch(Rectangle((cx, y - glyph_s / 2), glyph_s, glyph_s,
                                   fc=c["fc"], ec=c["ec"], lw=1.1, zorder=4))
            off = glyph_s + 0.09
        elif kind == "sigmoid":
            gate_glyph(ax, cx + 0.09, y, "σ", kind="proj", r=0.10, fs=7)
            off = 0.28
        elif kind == "exp":
            gate_glyph(ax, cx + 0.09, y, "e", kind="gate", r=0.10, fs=7)
            off = 0.28
        elif kind == "matrix":
            matrix_glyph(ax, cx + 0.11, y, s=0.22, n=3)
            off = 0.32
        else:
            off = 0.0
        ax.text(cx + off, y, text, ha="left", va="center", fontsize=FS_NOTE, color=INK)
        plain = _SPAN.sub(r"\1", text)          # markers must not widen the step
        cx += off + 0.13 * len(plain) * 0.55 + 0.42 if dx is None else dx
    return cx


def waveform(ax, x, y, w, h, seed=7, decay=False, color=INK, lw=0.8):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 400)
    sig = 0.5 * np.sin(2 * np.pi * 9 * t) + 0.35 * rng.standard_normal(t.size)
    if decay:
        sig *= 0.25 + 0.75 * t**2
    sig = sig / np.max(np.abs(sig))
    ax.plot(x + t * w, y + sig * h / 2, color=color, lw=lw, zorder=5)


def spectrum(ax, x, y, w, h, peaks, seed=3, color=INK, lw=0.9):
    """Baseline noise floor with peaks at given fractional positions."""
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 1, 500)
    s = 0.06 + 0.04 * rng.random(f.size)
    for p, amp in peaks:
        s += amp * np.exp(-((f - p) ** 2) / (2 * 0.006**2))
    s = s / s.max()
    ax.plot(x + f * w, y + s * h, color=color, lw=lw, zorder=5)
    ax.plot([x, x + w], [y, y], color=MUTED, lw=0.7, zorder=4)


# --------------------------------------------------------------------------
# What each algorithm actually eats
# --------------------------------------------------------------------------
# Read across two diagrams and the three fields tell them apart: WDCNN takes a
# waveform, the RUL backbones never see one, the SAE takes neither. Numbers are
# the ones in the run artifacts under Mamba-xLSTM/results/runs, which is also
# what V14 states. Edit the contract here, not inside a draw function.
#
#   name -> (besaran fisis, sensor dan sampling rate, bentuk tensor)
INPUT_SPEC: dict[str, tuple[str, str, str]] = {
    "mamba_xlstm_full": (
        "[[feature]] HI 36-D per rekaman, bukan [[waveform]]",
        "akselerometer horizontal dan vertikal, 25,6 kHz",
        "[[window]] 64 rekaman (PHM2012); 32 (XJTU-SY)",
    ),
    "nbeats_xlstm_full": (
        "[[feature]] HI 36-D per rekaman, bukan [[waveform]]",
        "akselerometer horizontal dan vertikal, 25,6 kHz",
        "[[window]] 64 rekaman (PHM2012); 32 (XJTU-SY)",
    ),
    "sparsegate_tcn_full": (
        "[[feature]] HI 36-D per rekaman, bukan [[waveform]]",
        "akselerometer horizontal dan vertikal, 25,6 kHz",
        "[[window]] 64 rekaman (PHM2012); 32 (XJTU-SY)",
    ),
    "wdcnn_full": (
        "amplitudo getaran domain waktu ([[raw signal]])",
        "akselerometer [[drive-end]] CWRU, 48 kHz",
        "1 × 2.048 titik, sekitar 42,67 ms",
    ),
    "shap_fsm_pipeline": (
        "amplitudo getaran domain waktu ([[raw signal]])",
        "akselerometer [[drive-end]] CWRU, 48 kHz",
        "2.048 titik per segmen",
    ),
    "classic_ml_trio": (
        "[[feature]] HI turunan sinyal, bukan [[waveform]]",
        "akselerometer CWRU, 48 kHz",
        "satu vektor per segmen 2.048 titik",
    ),
    "topk_sae": (
        "[[hidden state]] [[backbone]], bukan sinyal getaran",
        "[[output]] [[gated fusion]], [[weights]] dibekukan",
        "20.000 vektor $h \\in \\mathbb{R}^{128}$ per dataset",
    ),
    "sae_bpfx_pipeline": (
        "amplitudo getaran mentah, [[channel]] horizontal",
        "25,6 kHz; 0,1 s (PHM2012), 1,28 s (XJTU-SY)",
        "BPFx dari geometri [[bearing]]: rujukan validasi",
    ),
    "streaming_engine": (
        "dua [[channel]] percepatan per akuisisi",
        "akselerometer 25,6 kHz",
        "[[feature]] HI 36-D, [[buffer window]] 64 akuisisi",
    ),
    "hi_feature_pipeline": (
        "amplitudo getaran domain waktu ([[raw signal]])",
        "akselerometer [[drive-end]] dan [[fan-end]] CWRU, 48 kHz",
        "segmen 2.048 titik menjadi [[feature vector]] 36-D",
    ),
}

# These two sit in a narrow slide column, so they get bigger type; the rest
# read at 8.6 pt across the full slide width.
_STRIP_WIDE = {"mamba_xlstm_full", "sparsegate_tcn_full"}
_STRIP_INNER = 10.9      # inches of text the bar can hold on one line
_CHAR_IN = 0.0070        # rough inches per character per point of font size


def _strip_layout(name: str) -> tuple[str, float, bool]:
    """The bar's text, its font size, and whether it needs a second line."""
    besaran, sensor, tensor = INPUT_SPEC[name]
    head = f"[[Input]]:  {besaran}  ·  {sensor}"
    fs = 10.4 if name in _STRIP_WIDE else 8.6
    plain = _SPAN.sub(r"\1", f"{head}  ·  {tensor}")
    two_line = name in _STRIP_WIDE or len(plain) * fs * _CHAR_IN > _STRIP_INNER
    return f"{head}{'\n' if two_line else '  ·  '}{tensor}", fs, two_line


# A diagram that already ends well above y=0 hosts the bar in that gap rather
# than growing the canvas; the value is where the bar is centred.
_STRIP_INSIDE = {"streaming_engine": 0.40}


# The bar is kept as shallow as it can be: on a full-width slide figure every
# inch of canvas height costs an inch of the content band.
def _strip_height(name: str) -> float:
    return 0.84 if _strip_layout(name)[2] else 0.46


def strip_pad(name: str) -> float:
    """Height `new_fig` must reserve below y=0 for this diagram's strip."""
    if name in _STRIP_INSIDE:
        return 0.0
    return _strip_height(name) + 0.16


def input_strip(ax, w: float, name: str) -> None:
    """Draw the full-width bar stating what goes into the algorithm."""
    label, fs, _ = _strip_layout(name)
    cy = _STRIP_INSIDE.get(name, -strip_pad(name) / 2)
    box(ax, w / 2, cy, w - 0.80, _strip_height(name), label, kind="input", fs=fs)


# The two zoom panels of Mamba-xLSTM-Net are also the two "building block"
# charts shown before the backbones are introduced, so they are drawn by
# helpers that take the panel's box; the full diagram and the block charts
# call the same code.
def _mamba_panel(ax, p1x, p1y, p1w, p1h):
    container(ax, p1x, p1y, p1w, p1h, ec=KIND["seq"]["ec"])
    ax.add_patch(FancyBboxPatch((p1x, p1y), p1w, p1h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                fc=KIND["seq"]["fc"], ec="none", alpha=0.25, zorder=0))
    ax.text(p1x + p1w / 2, p1y + p1h - 0.22, "Blok Mamba ([[Selective State-Space Model]])",
            ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    my = p1y + 1.62                                   # main path
    gy = p1y + 0.62                                   # gate branch
    ax.plot([p1x + 0.28], [my], marker="o", ms=4, color=INK)
    arrow(ax, (p1x + 0.30, my), (p1x + 0.72, my))
    box(ax, p1x + 1.22, my, 0.96, 0.5, "[[Projection]]", kind="proj", fs=8.5)
    arrow(ax, (p1x + 1.72, my), (p1x + 2.06, my))
    box(ax, p1x + 2.56, my, 0.96, 0.5, "Conv1D", kind="proj", fs=8.5)
    arrow(ax, (p1x + 3.06, my), (p1x + 3.38, my))
    box(ax, p1x + 3.80, my, 0.80, 0.5, "SiLU", kind="util", fs=8.5)
    arrow(ax, (p1x + 4.22, my), (p1x + 4.54, my))
    box(ax, p1x + 5.32, my, 1.52, 0.72, "[[Selective]] SSM", kind="seq", fs=9,
        sub="B, C, Δ bergantung\npada [[input]]")
    matrix_glyph(ax, p1x + 5.32, my - 0.82, s=0.40)
    ax.text(p1x + 5.62, my - 0.82, "[[state]]", ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    arrow(ax, (p1x + 5.32, my - 0.60), (p1x + 5.32, my - 0.38), lw=0.9)
    arrow(ax, (p1x + 6.10, my), (p1x + 6.50, my))
    gate_glyph(ax, p1x + 6.66, my, "×", kind="gate")
    # gate branch: input -> projection+SiLU -> multiplicative gate
    ax.plot([p1x + 0.28, p1x + 0.28], [my - 0.04, gy], color=INK, lw=1.0)
    arrow(ax, (p1x + 0.28, gy), (p1x + 1.30, gy))
    box(ax, p1x + 2.14, gy, 1.66, 0.5, "[[Projection]] + SiLU", kind="proj", fs=8.5)
    ax.plot([p1x + 2.98, p1x + 6.66], [gy, gy], color=INK, lw=1.0)
    arrow(ax, (p1x + 6.66, gy), (p1x + 6.66, my - 0.16), shrink=1)
    arrow(ax, (p1x + 6.82, my), (p1x + 7.10, my))
    box(ax, p1x + 7.56, my, 0.86, 0.5, "[[Projection]]", kind="proj", fs=8.5)
    ax.text(p1x + p1w / 2, p1y + 0.16, "[[gate]] multiplikatif memilih informasi yang diteruskan",
            ha="center", va="bottom", fontsize=FS_TINY, color=MUTED, style="italic")



def _mlstm_panel(ax, p2x, p2y, p2w, p2h):
    container(ax, p2x, p2y, p2w, p2h, ec=KIND["mem"]["ec"])
    ax.add_patch(FancyBboxPatch((p2x, p2y), p2w, p2h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                fc=KIND["mem"]["fc"], ec="none", alpha=0.25, zorder=0))
    ax.text(p2x + p2w / 2, p2y + p2h - 0.22, "Sel mLSTM ([[matrix]] LSTM)",
            ha="center", va="center", fontsize=FS_TITLE, color=INK, fontweight="bold")
    cyc = p2y + 1.22
    ax.plot([p2x + 0.28], [cyc], marker="o", ms=4, color=INK)
    for lbl, dy in (("q", 0.85), ("k", 0.0), ("v", -0.85)):
        box(ax, p2x + 1.30, cyc + dy, 0.86, 0.46, f"[[Projection]] {lbl}", kind="proj", fs=8)
        ax.plot([p2x + 0.28, p2x + 0.60], [cyc, cyc], color=INK, lw=1.0)
        arrow(ax, (p2x + 0.60, cyc), (p2x + 0.85, cyc + dy), lw=0.9, shrink=1)
    Cx = p2x + 3.66
    matrix_glyph(ax, Cx, cyc - 0.20, s=0.94, n=4)
    ax.text(Cx, cyc - 0.98, "[[matrix memory]]  $C_t = f_t\\,C_{t-1} + i_t\\,v_t k_t^{\\top}$",
            ha="center", va="center", fontsize=8, color=INK)
    arrow(ax, (p2x + 1.74, cyc), (Cx - 0.52, cyc - 0.10), shrink=2)         # k
    arrow(ax, (p2x + 1.74, cyc - 0.85), (Cx - 0.52, cyc - 0.42), shrink=2)  # v
    gate_glyph(ax, Cx - 0.30, cyc + 0.60, "e", kind="gate")     # input gate (exp)
    gate_glyph(ax, Cx + 0.30, cyc + 0.60, "e", kind="gate")     # forget gate (exp)
    ax.text(Cx - 0.54, cyc + 0.60, "$i_t$", ha="right", va="center", fontsize=8, color=INK)
    ax.text(Cx + 0.54, cyc + 0.60, "$f_t$", ha="left", va="center", fontsize=8, color=INK)
    arrow(ax, (Cx - 0.30, cyc + 0.44), (Cx - 0.30, cyc + 0.29), lw=0.9, shrink=0)
    arrow(ax, (Cx + 0.30, cyc + 0.44), (Cx + 0.30, cyc + 0.29), lw=0.9, shrink=0)
    arrow(ax, (Cx + 0.48, cyc - 0.20), (Cx + 1.14, cyc - 0.20))
    box(ax, p2x + 5.34, cyc - 0.20, 1.14, 0.5, "[[Readout]] q", kind="util", fs=8.5,
        sub="[[normalization]]")
    ax.plot([p2x + 1.74, p2x + 5.34], [cyc + 0.85, cyc + 0.85], color=INK, lw=1.0)  # q path
    arrow(ax, (p2x + 5.34, cyc + 0.85), (p2x + 5.34, cyc + 0.14), lw=0.9, shrink=1)
    arrow(ax, (p2x + 5.92, cyc - 0.20), (p2x + 6.30, cyc - 0.20))
    gate_glyph(ax, p2x + 6.46, cyc - 0.20, "σ", kind="proj")
    ax.text(p2x + 6.46, cyc + 0.12, "$o_t$", ha="center", fontsize=8, color=INK)
    arrow(ax, (p2x + 6.62, cyc - 0.20), (p2x + 7.00, cyc - 0.20))
    box(ax, p2x + 7.42, cyc - 0.20, 0.72, 0.5, "$h_t$", kind="out", fs=9)



# --------------------------------------------------------------------------
# 1 · Mamba-xLSTM-Net (zoom-in idiom)
# --------------------------------------------------------------------------
def mamba_xlstm_full():
    fig, ax = new_fig(12.0, 6.9, strip_pad("mamba_xlstm_full"))

    # ---- right column: the stack -----------------------------------------
    sx, bw = 10.35, 2.55
    box(ax, sx, 1.28, bw, 0.56, "[[Input]] HI", kind="input",
        sub="[[window]] 64 rekaman (PHM2012)")
    arrow(ax, (sx, 1.58), (sx, 1.90))
    box(ax, sx, 2.16, bw, 0.52, "[[Linear projection]]\nke dimensi model", kind="proj", fs=8.5)
    arrow(ax, (sx, 2.44), (sx, 2.76))
    container(ax, sx - bw / 2 - 0.14, 2.78, bw + 0.28, 1.94, times="3×")
    box(ax, sx, 3.28, bw - 0.18, 0.62, "Blok Mamba", kind="seq", sub="[[Selective State-Space]]")
    arrow(ax, (sx, 3.62), (sx, 3.92))
    box(ax, sx, 4.26, bw - 0.18, 0.62, "Blok mLSTM", kind="mem", sub="[[matrix memory]]")
    arrow(ax, (sx, 4.74), (sx, 5.02))
    box(ax, sx, 5.28, bw, 0.52, "[[Gated fusion]]", kind="gate", sub="[[weights]] dapat dilatih")
    arrow(ax, (sx, 5.56), (sx, 5.88))
    ax.text(sx + 0.24, 5.72, r"$h \in \mathbb{R}^{128}$", ha="left", va="center",
            fontsize=8.5, color=MUTED)
    box(ax, sx, 6.18, bw, 0.60, "MLP dua [[layer]]", kind="out", sub="estimasi RUL", bold=True)
    ax.text(sx, 6.72, "Mamba-xLSTM-Net", ha="center", va="center",
            fontsize=FS_TITLE, color=INK, fontweight="bold")

    # ---- zoom panels (shared with the block charts) ----------------------
    p1x, p1y, p1w, p1h = 0.35, 0.72, 8.10, 2.70
    _mamba_panel(ax, p1x, p1y, p1w, p1h)
    p2x, p2y, p2w, p2h = 0.35, 3.72, 8.10, 2.70
    _mlstm_panel(ax, p2x, p2y, p2w, p2h)

    # ---- zoom links (Mamba block -> bottom panel, mLSTM block -> top) ----
    zoom_link(ax, (sx - bw / 2 + 0.09, 3.59), (sx - bw / 2 + 0.09, 2.97),
              (p1x + p1w, p1y + p1h), (p1x + p1w, p1y))
    zoom_link(ax, (sx - bw / 2 + 0.09, 4.57), (sx - bw / 2 + 0.09, 3.95),
              (p2x + p2w, p2y + p2h), (p2x + p2w, p2y))

    # ---- legend (two rows) ------------------------------------------------
    legend_row(ax, 0.45, 0.44, [
        ("input", "[[input]]"), ("proj", "[[projection]] / [[convolution]]"),
        ("seq", "[[sequence modeling]]"), ("mem", "[[memory]] / [[state]]"),
        ("gate", "[[gate]]"), ("out", "[[output]]"),
    ])
    legend_row(ax, 0.45, 0.15, [
        ("exp", "[[exponential gating]]"), ("sigmoid", "[[sigmoid gating]]"),
        ("matrix", "[[matrix memory]]"),
    ])
    input_strip(ax, 12.0, "mamba_xlstm_full")
    save(fig, "mamba_xlstm_full")


# --------------------------------------------------------------------------
# 2 · N-BEATS-xLSTM-RUL
# --------------------------------------------------------------------------
def nbeats_xlstm_full():
    fig, ax = new_fig(12.0, 3.9, strip_pad("nbeats_xlstm_full"))
    my = 1.78

    box(ax, 1.15, my, 1.75, 0.78, "[[Input]] HI", kind="input",
        sub="[[window]] 64 rekaman\n(PHM2012)")
    arrow(ax, (2.03, my), (2.42, my))
    container(ax, 2.46, my - 0.62, 1.62, 1.24, times="2×")
    box(ax, 3.27, my, 1.34, 0.66, "Blok xLSTM", kind="seq", sub="[[front-end]]")
    arrow(ax, (4.24, my), (4.62, my))
    ax.text(3.27, my - 0.92, "konteks [[sequence]] memodulasi\n[[output]] basis",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED, style="italic")

    # three basis blocks with mini curve glyphs
    basis = [
        (5.55, "Blok [[Trend]]", "polinomial Bernstein", "trend"),
        (7.55, "Blok [[Wear]]", "frekuensi karakteristik", "wear"),
        (9.55, "Blok [[Shock]]", "[[wavelet]] Gabor", "shock"),
    ]
    t = np.linspace(0, 1, 160)
    for bx, name, subname, glyph in basis:
        box(ax, bx, my, 1.72, 1.46, "", kind="seq")
        ax.text(bx, my + 0.50, name, ha="center", va="center", fontsize=FS_BLOCK,
                color=INK, fontweight="bold", zorder=6)
        ax.text(bx, my - 0.54, subname, ha="center", va="center",
                fontsize=FS_TINY, color=MUTED, zorder=6)
        gx, gy, gw, gh = bx - 0.62, my - 0.24, 1.24, 0.52
        if glyph == "trend":
            ax.plot(gx + t * gw, gy + gh * (0.92 - 0.7 * t - 0.15 * t**2),
                    color=INK, lw=1.3, zorder=6)
        elif glyph == "wear":
            ax.plot(gx + t * gw, gy + gh * (0.15 + 0.75 * t**4), color=INK, lw=1.3,
                    zorder=6)
        else:
            spike = 0.12 + 0.8 * np.exp(-((t - 0.62) ** 2) / (2 * 0.02**2))
            ax.plot(gx + t * gw, gy + gh * spike, color=INK, lw=1.3, zorder=6)

    # doubly-residual plumbing: backcast chain + forecast bus into the sum
    for bx, *_ in basis[:-1]:
        arrow(ax, (bx + 0.86, my), (bx + 2.00 - 0.86, my))
    ax.text(6.55, my - 0.98, "[[residual]] diteruskan: [[input]] berikutnya = [[input]] − [[backcast]]",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED, style="italic")
    sumx, sumy = 10.15, my + 1.25
    for bx, *_ in basis:
        ax.plot([bx, bx], [my + 0.73, sumy], color=INK, lw=1.0)
    ax.plot([basis[0][0], sumx - 0.20], [sumy, sumy], color=INK, lw=1.0)
    gate_glyph(ax, sumx, sumy, "+", kind="mem", r=0.20, fs=10)
    arrow(ax, (sumx - 0.30, sumy), (sumx - 0.21, sumy), shrink=0, lw=1.0)
    ax.text(5.68, sumy - 0.12, "[[forecast]] tiap blok", ha="left", va="top",
            fontsize=FS_TINY, color=MUTED, style="italic")
    arrow(ax, (sumx + 0.21, sumy), (10.66, sumy), shrink=0)
    box(ax, 11.28, sumy, 1.14, 0.80, "Estimasi\nRUL", kind="out", fs=9, bold=True,
        sub="[[clamp]] fraksi RUL")

    ax.text(0.45, 3.68, "N-BEATS-xLSTM-RUL", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(0.45, 3.38, "dekomposisi kurva degradasi ke tiga basis terstruktur "
                        "(agregasi aditif, [[doubly residual]])",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED)

    legend_row(ax, 0.45, 0.32, [
        ("input", "[[input]]"), ("seq", "[[sequence modeling]] / basis blok"),
        ("mem", "agregasi aditif"), ("out", "[[output]]"),
    ])
    input_strip(ax, 12.0, "nbeats_xlstm_full")
    save(fig, "nbeats_xlstm_full")


def _dilation_panel(ax, px, py, pw, ph):
    container(ax, px, py, pw, ph, ec=KIND["proj"]["ec"])
    ax.add_patch(FancyBboxPatch((px, py), pw, ph,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                fc=KIND["proj"]["fc"], ec="none", alpha=0.22, zorder=0))
    ax.text(px + pw / 2, py + ph - 0.20, "[[Dilated causal convolution]]: "
            "[[receptive field]] menutup seluruh [[window]]",
            ha="center", va="center", fontsize=FS_TITLE - 0.5, color=INK, fontweight="bold")
    n = 17
    xs = np.linspace(px + 0.85, px + pw - 0.30, n)
    rows = [(py + 0.52, 1), (py + 0.99, 2), (py + 1.46, 4), (py + 1.93, 8)]
    for ry, d in rows:
        ax.text(px + 0.14, ry, f"d = {d}", fontsize=FS_TINY, color=MUTED, va="center")
    ax.text(px + 0.14, py + 0.18, "[[input]]", fontsize=FS_TINY, color=MUTED, va="center")
    # receptive-field cone of the last output node
    active = {3: {n - 1}}
    for level in (3, 2, 1, 0):
        d = rows[level][1]
        prev = set()
        for j in active.get(level, set()):
            prev.update({j, j - d} & set(range(n)))
        if level > 0:
            active[level - 1] = prev
        for j in active.get(level, set()):
            for src in ({j, j - d} & set(range(n))):
                if level == 0:
                    ax.plot([xs[src], xs[j]], [rows[0][0] - 0.34, rows[0][0]],
                            color=KIND["proj"]["ec"], lw=0.9, zorder=2)
                else:
                    ax.plot([xs[src], xs[j]], [rows[level - 1][0], rows[level][0]],
                            color=KIND["proj"]["ec"], lw=0.9, zorder=2)
    base = active.get(0, set())
    inputs = set()
    for j in base:
        inputs.update({j, j - 1} & set(range(n)))
    for level, (ry, d) in enumerate(rows):
        for j in range(n):
            on = j in active.get(level, set())
            ax.plot([xs[j]], [ry], marker="o", ms=5 if on else 3.5,
                    color=INK if on else "#B9C2D2", zorder=3)
    for j in range(n):
        ax.plot([xs[j]], [rows[0][0] - 0.34], marker="s", ms=4,
                color=GOLD if j in inputs else "#D8DEE8", zorder=3)


# --------------------------------------------------------------------------
# 3 · SparseGate-TCN-RUL
# --------------------------------------------------------------------------
def sparsegate_tcn_full():
    fig, ax = new_fig(12.0, 5.6, strip_pad("sparsegate_tcn_full"))
    my = 1.30

    box(ax, 1.05, my, 1.65, 0.78, "[[Input]] HI", kind="input",
        sub="[[window]] 64 rekaman\n(PHM2012)")
    ax.plot([1.88, 2.16], [my, my], color=INK, lw=1.1)
    ax.plot([2.16, 2.16], [my - 0.55, my + 0.55], color=INK, lw=1.1)
    arrow(ax, (2.16, my + 0.55), (2.52, my + 0.55), shrink=0)
    arrow(ax, (2.16, my - 0.55), (2.52, my - 0.55), shrink=0)
    box(ax, 3.42, my + 0.55, 1.80, 0.5, "[[Sparse feature gate]]", kind="gate", fs=8.5)
    box(ax, 3.42, my - 0.55, 1.80, 0.5, "[[Cross-feature attention]]", kind="seq", fs=8.5)
    ax.plot([4.32, 4.62], [my + 0.55, my + 0.55], color=INK, lw=1.1)
    ax.plot([4.32, 4.62], [my - 0.55, my - 0.55], color=INK, lw=1.1)
    arrow(ax, (4.62, my + 0.55), (4.86, my + 0.10), shrink=1)
    arrow(ax, (4.62, my - 0.55), (4.86, my - 0.10), shrink=1)
    gate_glyph(ax, 4.94, my, "+", kind="mem", r=0.17, fs=10)
    ax.text(4.94, my - 0.60, "[[gated input]]", ha="center", va="top",
            fontsize=FS_TINY, color=MUTED)
    arrow(ax, (5.11, my), (5.42, my))

    container(ax, 5.46, my - 0.52, 3.92, 1.14, title="[[stack]] TCN")
    for i, d in enumerate((1, 2, 4, 8)):
        box(ax, 6.02 + i * 0.94, my, 0.84, 0.56, f"d = {d}", kind="proj", fs=8.5)
        if i < 3:
            arrow(ax, (6.44 + i * 0.94, my), (6.54 + i * 0.94, my), shrink=0, lw=0.9)
    arrow(ax, (9.40, my), (9.66, my))
    box(ax, 10.20, my, 1.04, 0.62, "Langkah\nterakhir", kind="util", fs=8.5)
    arrow(ax, (10.74, my), (10.92, my))
    box(ax, 11.42, my, 0.94, 0.78, "[[Quantile]]\n[[head]]", kind="out", fs=8.5, sub="median = RUL")

    # ---- zoom panel: dilated receptive field ------------------------------
    px, py, pw, ph = 1.60, 2.55, 8.60, 2.55
    _dilation_panel(ax, px, py, pw, ph)
    zoom_link(ax, (5.60, my + 0.62), (9.24, my + 0.62), (px, py), (px + pw, py))

    ax.text(0.45, 5.32, "SparseGate-TCN-RUL", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(10.42, py + ph - 0.28,
            "[[sparse gate]] hanya\nmeneruskan [[channel]]\npaling informatif;\n"
            "model paling ringan,\nsesuai untuk tier [[edge]]",
            ha="left", va="top", fontsize=8, color=MUTED)

    legend_row(ax, 0.45, 0.32, [
        ("input", "[[input]]"), ("gate", "[[gate]]"), ("seq", "[[attention]]"),
        ("proj", "[[dilated convolution]]"), ("out", "[[output]]"),
        ("active", "rekaman yang dijangkau satu [[output]] teratas"),
    ])
    input_strip(ax, 12.0, "sparsegate_tcn_full")
    save(fig, "sparsegate_tcn_full")


# --------------------------------------------------------------------------
# 4 · WDCNN
# --------------------------------------------------------------------------
def wdcnn_full():
    fig, ax = new_fig(12.0, 3.55, strip_pad("wdcnn_full"))
    my = 1.95

    box(ax, 1.05, my, 1.70, 1.10, "[[Raw signal]]\n1 × 2.048", kind="input", sub="[[channel]] [[drive-end]]")
    waveform(ax, 0.40, my - 0.34, 1.30, 0.28, seed=11)
    arrow(ax, (1.92, my), (2.24, my))

    def conv_block(cx, title, conv_label):
        container(ax, cx - 1.06, my - 1.02, 2.12, 2.04, title=title)
        box(ax, cx, my + 0.52, 1.82, 0.5, conv_label, kind="proj", fs=8.5)
        arrow(ax, (cx, my + 0.26), (cx, my + 0.12), shrink=0, lw=0.9)
        box(ax, cx, my - 0.14, 1.82, 0.44, "BatchNorm + ReLU", kind="gate", fs=8.5)
        arrow(ax, (cx, my - 0.37), (cx, my - 0.50), shrink=0, lw=0.9)
        box(ax, cx, my - 0.74, 1.82, 0.44, "MaxPool", kind="util", fs=8.5)

    conv_block(3.42, "Blok 1: [[kernel]] lebar", "Conv1D 64, [[stride]] 16")
    arrow(ax, (4.60, my), (4.94, my))
    conv_block(6.12, "Blok 2 sampai 5 (4×)", "Conv1D [[kernel]] 3")
    arrow(ax, (7.30, my), (7.66, my))
    box(ax, 8.16, my, 0.94, 0.52, "Flatten", kind="util", fs=8.5)
    arrow(ax, (8.65, my), (8.95, my))
    box(ax, 9.70, my, 1.42, 0.72, "Dua [[layer]] FC\n+ [[dropout]]", kind="proj", fs=8.5)
    arrow(ax, (10.43, my), (10.73, my))
    box(ax, 11.28, my, 1.02, 0.72, "Softmax\n10 kelas", kind="out", fs=8.5, bold=True)

    ax.text(0.45, 3.30, "WDCNN ([[Wide First-layer Kernels]])", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(8.95, 3.30, "lima blok mereduksi dimensi temporal",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED, style="italic")

    legend_row(ax, 0.45, 0.40, [
        ("input", "[[input]]"), ("proj", "[[convolution]] / FC"),
        ("gate", "[[normalization]]"), ("util", "[[pooling]] / utilitas"), ("out", "[[output]]"),
    ])
    input_strip(ax, 12.0, "wdcnn_full")
    save(fig, "wdcnn_full")


# --------------------------------------------------------------------------
# 5 · Top-k Sparse Autoencoder
# --------------------------------------------------------------------------
def topk_sae():
    fig, ax = new_fig(12.0, 3.65, strip_pad("topk_sae"))
    my = 1.75

    box(ax, 1.20, my, 1.95, 0.92, "$h \\in \\mathbb{R}^{128}$", kind="input", fs=10,
        sub="[[hidden state]] [[backbone]]\n([[weights]] dibekukan)")
    arrow(ax, (2.20, my), (2.52, my))
    box(ax, 3.10, my, 1.10, 0.56, "[[Pre-encoder]]\n[[bias]]", kind="util", fs=8)
    arrow(ax, (3.67, my), (3.95, my))
    box(ax, 4.60, my, 1.24, 0.62, "[[Encoder]]\nlinear", kind="proj", fs=8.5, sub="$W_{enc}$")
    arrow(ax, (5.24, my), (5.52, my))
    box(ax, 6.16, my, 1.22, 0.62, "ReLU +\nTop-k", kind="gate", fs=8.5, sub="k = 51")
    arrow(ax, (6.79, my), (7.07, my))

    # latent grid: 16 x 8 cells drawn to represent the 1.024 features
    gx, gy = 7.20, my - 0.72
    cell, nx, ny = 0.135, 16, 8
    rng = np.random.default_rng(20)
    lit = set(map(tuple, rng.integers(0, [nx, ny], size=(6, 2))))
    while len(lit) < 6:
        lit.add((int(rng.integers(0, nx)), int(rng.integers(0, ny))))
    for i in range(nx):
        for j in range(ny):
            on = (i, j) in lit
            c = KIND["active"] if on else {"fc": "#E9EDF3", "ec": "#C7CFDC"}
            ax.add_patch(Rectangle((gx + i * cell, gy + j * cell), cell * 0.88,
                                   cell * 0.88, fc=c["fc"], ec=c["ec"], lw=0.6))
    ax.text(gx + nx * cell / 2, gy + ny * cell + 0.16,
            "$z$: 1.024 [[latent feature]] (ekspansi 8×)", ha="center", va="bottom",
            fontsize=FS_BLOCK, color=INK, fontweight="bold")
    ax.text(gx + nx * cell / 2, gy - 0.14, "hanya k = 51 [[feature]] (sekitar 5 %)\naktif per [[sample]]",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED)
    arrow(ax, (gx + nx * cell + 0.06, my), (gx + nx * cell + 0.30, my))
    box(ax, 10.30, my, 1.24, 0.62, "[[Decoder]]\nlinear", kind="proj", fs=8.5, sub="$W_{dec}$")
    arrow(ax, (10.94, my), (11.16, my))
    box(ax, 11.54, my, 0.72, 0.92, "$\\hat{h}$", kind="out", fs=10, sub="MSE\n< 0,001")

    ax.text(0.45, 3.42, "Top-k [[Sparse Autoencoder]] (SAE)", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(0.45, 3.10, "dilatih pasca-hoc pada 20.000 [[hidden state]]; "
                        "satu [[latent feature]] idealnya mengkodekan satu konsep fisika "
                        "(monosemantisitas)",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED)

    legend_row(ax, 0.45, 0.32, [
        ("input", "[[input]]"), ("proj", "[[linear projection]]"), ("gate", "[[Top-k selection]]"),
        ("active", "[[active feature]]"), ("util", "[[inactive feature]] (nol)"), ("out", "[[reconstruction]]"),
    ])
    input_strip(ax, 12.0, "topk_sae")
    save(fig, "topk_sae")


# --------------------------------------------------------------------------
# 6 · SAE -> BPFx procedure
# --------------------------------------------------------------------------
def _panel(ax, x, y, w, h, title, tint="seq"):
    container(ax, x, y, w, h, ec=KIND[tint]["ec"])
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.09",
                                fc=KIND[tint]["fc"], ec="none", alpha=0.20, zorder=0))
    ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center",
            fontsize=FS_TITLE - 0.5, color=INK, fontweight="bold")


def sae_bpfx_pipeline():
    fig, ax = new_fig(12.0, 4.05, strip_pad("sae_bpfx_pipeline"))
    py, ph, pw = 0.32, 3.30, 3.55

    _panel(ax, 0.40, py, pw, ph, "Tahap 1: [[Envelope spectrum]]", tint="seq")
    x1 = 0.40 + pw / 2
    waveform(ax, 0.85, py + 2.42, 2.60, 0.42, seed=5)
    box(ax, x1, py + 1.78, 2.60, 0.42, "Transformasi Hilbert\npada [[raw signal]]", kind="proj", fs=8)
    arrow(ax, (x1, py + 1.55), (x1, py + 1.42), shrink=0, lw=0.9)
    box(ax, x1, py + 1.19, 2.60, 0.42, "[[Envelope]] sinyal analitik", kind="proj", fs=8)
    arrow(ax, (x1, py + 0.96), (x1, py + 0.83), shrink=0, lw=0.9)
    box(ax, x1, py + 0.60, 2.60, 0.42, "FFT [[envelope]]", kind="proj", fs=8)
    arrow(ax, (x1, py + 2.20), (x1, py + 2.01), shrink=1, lw=0.9)

    arrow(ax, (4.02, py + ph / 2), (4.38, py + ph / 2), lw=2.2, style="-|>")

    _panel(ax, 4.42, py, pw, ph, "Tahap 2: Amplitudo BPFx", tint="mem")
    x2, sy = 4.42 + 0.35, py + 1.10
    peaks = [(0.18, 0.85), (0.38, 0.62), (0.58, 0.48), (0.78, 0.35)]
    for (p, _), name in zip(peaks, ("BPFO", "BPFI", "BSF", "FTF")):
        ax.add_patch(Rectangle((x2 + (p - 0.035) * (pw - 0.7), sy), 0.07 * (pw - 0.7),
                               1.15, fc=GOLD_FILL, ec="none", alpha=0.75, zorder=1))
        ax.text(x2 + p * (pw - 0.7), sy - 0.10, name, ha="center", va="top",
                fontsize=FS_TINY, color=INK)
    spectrum(ax, x2, sy, pw - 0.7, 1.05, peaks)
    ax.text(4.42 + pw / 2, py + 0.38, "rata-rata spektrum pada pita ±2 Hz\n"
            "di sekitar tiap frekuensi karakteristik",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    arrow(ax, (8.04, py + ph / 2), (8.40, py + ph / 2), lw=2.2, style="-|>")

    _panel(ax, 8.44, py, pw, ph, "Tahap 3: Korelasi dan [[hit-rate]]", tint="gate")
    x3 = 8.44 + pw / 2
    box(ax, x3, py + 2.28, 3.05, 0.60, "Korelasi Pearson $r$\naktivasi [[feature]] SAE × amplitudo BPFx",
        kind="proj", fs=8)
    arrow(ax, (x3, py + 1.95), (x3, py + 1.80), shrink=0, lw=0.9)
    box(ax, x3, py + 1.44, 3.05, 0.56, "[[Hit-rate]]: proporsi [[feature]]\ndengan |r| ≥ 0,30",
        kind="out", fs=8)
    ax.text(x3, py + 0.62, "korelasi maksimum 0,447 (BPFI, PHM2012)\ndan 0,468 (BPFO, XJTU-SY)",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    ax.text(0.45, 3.84, "Prosedur pemetaan [[feature]] SAE ke frekuensi karakteristik [[bearing]] (BPFx)",
            ha="left", va="center", fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    input_strip(ax, 12.0, "sae_bpfx_pipeline")
    save(fig, "sae_bpfx_pipeline")


# --------------------------------------------------------------------------
# 7 · SHAP -> FSM procedure
# --------------------------------------------------------------------------
def shap_fsm_pipeline():
    fig, ax = new_fig(12.0, 4.6, strip_pad("shap_fsm_pipeline"))
    py, ph, pw, gap = 1.30, 2.55, 2.55, 0.36

    xs = [0.40 + i * (pw + gap) for i in range(4)]

    _panel(ax, xs[0], py, pw, ph, "[[Raw signal]]", tint="input")
    waveform(ax, xs[0] + 0.30, py + 1.45, pw - 0.6, 0.70, seed=13)
    ax.text(xs[0] + pw / 2, py + 0.55, "2.048 titik per segmen\n[[channel]] [[drive-end]] CWRU",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED)

    _panel(ax, xs[1], py, pw, ph, "WDCNN terlatih", tint="proj")
    cx = xs[1] + pw / 2
    for i, lbl in enumerate(("Conv blok 1–5", "FC + [[dropout]]", "Softmax 10 kelas")):
        box(ax, cx, py + 1.62 - i * 0.52, 1.95, 0.42, lbl, kind="proj", fs=8)
        if i < 2:
            arrow(ax, (cx, py + 1.62 - i * 0.52 - 0.22), (cx, py + 1.62 - i * 0.52 - 0.31),
                  shrink=0, lw=0.8)

    _panel(ax, xs[2], py, pw, ph, "SHAP DeepExplainer", tint="gate")
    cx = xs[2] + pw / 2
    rng = np.random.default_rng(4)
    t = np.linspace(0, 1, 90)
    vals = rng.standard_normal(90) * np.exp(-((t - 0.5) ** 2) / 0.05)
    bx, bw2 = xs[2] + 0.30, pw - 0.6
    for i, v in enumerate(vals):
        color = "#C0392B" if v > 0 else "#7FA3C4"
        ax.plot([bx + i / 90 * bw2] * 2, [py + 1.55, py + 1.55 + v * 0.28],
                color=color, lw=0.7, zorder=4)
    ax.plot([bx, bx + bw2], [py + 1.55, py + 1.55], color=MUTED, lw=0.6)
    ax.text(cx, py + 0.55, "[[attribution]] per titik waktu\n300 [[background]] · 500 [[sample]] uji",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED)

    _panel(ax, xs[3], py, pw, ph, "[[Fault Signature Maps]]", tint="mem")
    hx, hy = xs[3] + 0.34, py + 0.98
    rng = np.random.default_rng(9)
    cmap = matplotlib.colormaps["YlGnBu"]
    for r in range(10):
        for ci in range(24):
            v = rng.random() * np.exp(-((ci / 24 - (0.2 + 0.06 * r) % 1) ** 2) / 0.02)
            ax.add_patch(Rectangle((hx + ci * 0.081, hy + r * 0.115), 0.075, 0.105,
                                   fc=cmap(0.15 + 0.8 * min(v, 1)), ec="none"))
    ax.text(xs[3] + pw / 2, py + 0.55, "[[attribution map]] 10 kelas × 2.048 titik\n"
            "3 varian: [[Signed]], [[Absolute]], [[Variance]]",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED)

    for i in range(3):
        arrow(ax, (xs[i] + pw + 0.04, py + ph / 2), (xs[i + 1] - 0.04, py + ph / 2),
              lw=2.0)

    ax.text(0.45, 4.32, "Dari sinyal mentah ke [[Fault Signature Maps]] (FSM)",
            ha="left", va="center", fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(0.45, 3.98, "agregasi [[attribution]] per kelas menghasilkan [[array]] 500 × 2.048 × 10 "
                        "(lebih dari 10 juta nilai SHAP)",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED)

    box(ax, 6.0, 0.62, 11.2, 0.56,
        "Validasi FSM:  diskriminabilitas D = 0,216  ·  stabilitas [[split-half]] 0,940  ·  "
        "monotonisitas keparahan (Ball 17,6 % · IR 13,8 % · OR 8,6 %)",
        kind="util", fs=8.5)
    input_strip(ax, 12.0, "shap_fsm_pipeline")
    save(fig, "shap_fsm_pipeline")


# --------------------------------------------------------------------------
# 8 · Classic ML trio
# --------------------------------------------------------------------------
def classic_ml_trio():
    fig, ax = new_fig(12.0, 4.9, strip_pad("classic_ml_trio"))
    py, ph, pw, gap = 1.30, 2.80, 3.48, 0.38
    xs = [0.40 + i * (pw + gap) for i in range(3)]

    # --- SVM ---------------------------------------------------------------
    _panel(ax, xs[0], py, pw, ph, "SVM dengan [[kernel]] RBF", tint="seq")
    rng = np.random.default_rng(2)
    cxp, cyp = xs[0] + pw / 2, py + 1.45
    a = rng.normal([-0.72, 0.12], 0.30, size=(16, 2))
    b = rng.normal([0.78, -0.10], 0.30, size=(16, 2))
    t = np.linspace(-1.05, 1.05, 120)
    boundary = 0.55 * np.sin(1.8 * t) * 0.4
    ax.plot(cxp + boundary + 0.03 * t, cyp + t * 0.75, color=INK, lw=1.4, zorder=4)
    for off, ls in ((0.22, (0, (4, 3))), (-0.22, (0, (4, 3)))):
        ax.plot(cxp + boundary + off, cyp + t * 0.75, color=MUTED, lw=0.8, ls=ls, zorder=3)
    ax.scatter(cxp + a[:, 0], cyp + a[:, 1], s=14, c=KIND["seq"]["ec"], zorder=5)
    ax.scatter(cxp + b[:, 0], cyp + b[:, 1], s=14, c=GOLD, marker="s", zorder=5)
    for pt in (a[np.argmax(a[:, 0])], b[np.argmin(b[:, 0])]):
        ax.add_patch(Circle((cxp + pt[0], cyp + pt[1]), 0.09, fc="none", ec=INK,
                            lw=1.1, zorder=6))
    ax.text(cxp, py + 0.32, "batas pemisah dengan [[margin]] maksimum;\n[[kernel]] RBF menekuk batas "
            "untuk pola nonlinier", ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    # --- Logistic Regression ----------------------------------------------
    _panel(ax, xs[1], py, pw, ph, "[[Logistic Regression]] ([[one-vs-rest]])", tint="mem")
    gx, gy, gw, gh = xs[1] + 0.55, py + 0.85, pw - 1.1, 1.35
    z = np.linspace(-6, 6, 150)
    sig = 1 / (1 + np.exp(-z))
    ax.plot([gx, gx + gw], [gy, gy], color=MUTED, lw=0.7)
    ax.plot([gx, gx], [gy, gy + gh], color=MUTED, lw=0.7)
    ax.plot(gx + (z + 6) / 12 * gw, gy + sig * gh, color=INK, lw=1.5, zorder=4)
    ax.plot([gx, gx + gw], [gy + gh / 2, gy + gh / 2], color=GOLD, lw=0.9,
            ls=(0, (4, 3)), zorder=3)
    ax.text(gx + gw + 0.06, gy + gh / 2, "0,5", fontsize=FS_TINY, color=GOLD, va="center")
    ax.text(gx + gw / 2, gy - 0.14, "skor linear [[feature]]", ha="center", va="top",
            fontsize=FS_TINY, color=MUTED)
    ax.text(xs[1] + pw / 2, py + 0.32, "[[sigmoid]] memetakan skor ke probabilitas;\n"
            "satu model per kelas (OvR)", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)

    # --- Trees -------------------------------------------------------------
    _panel(ax, xs[2], py, pw, ph, "[[Decision Tree]] dan [[Ensemble]]", tint="gate")

    def tree(cx, cy, s=1.0, lw=1.0, ms=4.0):
        pts = {(0, 0): (cx, cy)}
        for depth in (1, 2):
            for k in range(2 ** depth):
                px_, py_ = cx + (k - (2 ** depth - 1) / 2) * 0.42 * s / depth, cy - 0.34 * s * depth
                pts[(depth, k)] = (px_, py_)
        for (d, k), (x0, y0) in pts.items():
            if d < 2:
                for kk in (2 * k, 2 * k + 1):
                    x1, y1 = pts[(d + 1, kk)]
                    ax.plot([x0, x1], [y0, y1], color=KIND["gate"]["ec"], lw=lw, zorder=3)
        for (d, k), (x0, y0) in pts.items():
            ax.plot([x0], [y0], marker="o", ms=ms, color=INK, zorder=4)

    tree(xs[2] + 0.75, py + 2.15, s=1.0)
    ax.text(xs[2] + 0.75, py + 1.10, "[[Decision Tree]]", ha="center",
            fontsize=FS_TINY, color=INK)
    arrow(ax, (xs[2] + 1.30, py + 1.80), (xs[2] + 1.72, py + 1.80), lw=1.0)
    for i in range(3):
        tree(xs[2] + 2.10 + i * 0.55, py + 2.28, s=0.5, lw=0.8, ms=2.6)
    ax.text(xs[2] + 2.65, py + 1.72, "[[Random Forest]]: [[voting]]\nbanyak [[tree]] ([[bagging]])",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED)
    for i in range(3):
        tree(xs[2] + 2.10 + i * 0.62, py + 1.05, s=0.5, lw=0.8, ms=2.6)
        if i < 2:
            ax.text(xs[2] + 2.41 + i * 0.62, py + 0.88, "+", fontsize=9, color=INK,
                    ha="center")
    ax.text(xs[2] + 2.65, py + 0.34, "XGBoost: [[tree]] berikutnya\nmengoreksi [[error]] ([[boosting]])",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    ax.text(0.45, 4.62, "Tiga model klasik pada [[feature vector]]", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    box(ax, 6.0, 0.62, 11.2, 0.56,
        "[[Output]]: 10 kelas kerusakan CWRU  ·  XAI: SHAP KernelExplainer "
        "(SVM, LR) dan TreeExplainer (DT, RF, XGBoost)",
        kind="util", fs=8.5)
    input_strip(ax, 12.0, "classic_ml_trio")
    save(fig, "classic_ml_trio")


# --------------------------------------------------------------------------
# 9 · Streaming inference engine
# --------------------------------------------------------------------------
def streaming_engine():
    fig, ax = new_fig(12.0, 4.5, strip_pad("streaming_engine"))
    my = 2.72

    steps = [
        (1.30, 1.95, "Akuisisi sinyal", "[[sliding window]]\n64 akuisisi", "input"),
        (3.65, 1.95, "Ekstraksi [[feature]] HI", "[[pipeline]] identik\ndengan pelatihan", "proj"),
        (6.00, 1.95, "Mamba-xLSTM-Net", "di [[server]],\nprotokol WebSocket", "seq"),
        (8.35, 1.95, "Prediksi per akuisisi", "fraksi RUL · status\n[[fusion gate]] · [[attribution]]", "mem"),
        (10.70, 1.95, "[[Dashboard streaming]]", "kurva RUL, [[waveform]],\n[[event log]]", "out"),
    ]
    for cx, w, label, sub, kind in steps:
        box(ax, cx, my, w, 1.10, label, kind=kind, fs=9, sub=sub, bold=kind == "out")
    for (cx1, w1, *_), (cx2, w2, *_) in zip(steps, steps[1:]):
        arrow(ax, (cx1 + w1 / 2 + 0.02, my), (cx2 - w2 / 2 - 0.02, my), lw=1.8)

    ax.text(6.0, my + 0.95, "[[feature attribution]]: [[gradient]]×[[input]] per akuisisi; "
            "Integrated Gradients dipicu otomatis pada transisi status",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED, style="italic")

    # status threshold bar
    bx, bw_, bh, by = 1.50, 9.0, 0.5, 0.92
    ordered = [(0.0, 0.20, "#F3D9D5", "#C0392B", "[[Critical]]\n< 20 %"),
               (0.20, 0.20, "#F3EAD2", GOLD, "[[Degrading]]\n20–40 %"),
               (0.40, 0.60, "#D9EAD9", "#3E8E5B", "[[Healthy]]\n> 40 %")]
    for f0, fw, fc, ec, lbl in ordered:
        ax.add_patch(Rectangle((bx + f0 * bw_, by), fw * bw_, bh, fc=fc, ec=ec, lw=1.2))
        ax.text(bx + (f0 + fw / 2) * bw_, by + bh / 2, lbl, ha="center", va="center",
                fontsize=FS_NOTE, color=INK, linespacing=1.2)
    ax.text(bx, by + bh + 0.12, "Ambang status dari fraksi RUL terprediksi",
            ha="left", va="bottom", fontsize=FS_NOTE, color=INK, fontweight="bold")
    ax.text(bx + bw_ + 0.10, by + bh / 2, "100 %", ha="left", va="center",
            fontsize=FS_TINY, color=MUTED)
    ax.text(bx - 0.10, by + bh / 2, "0 %", ha="right", va="center",
            fontsize=FS_TINY, color=MUTED)

    ax.text(0.45, 4.20, "Mesin inferensi RUL [[streaming]]", ha="left", va="center",
            fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(0.45, 3.86, "satu mesin yang sama untuk dataset benchmark dan rekaman lapangan "
                        "PT SKF Indonesia, sehingga metrik antar dataset sebanding",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED)
    input_strip(ax, 12.0, "streaming_engine")
    save(fig, "streaming_engine")


# --------------------------------------------------------------------------
# 10 · Raw signal to HI feature vector (the machine-learning side of slide 11)
# --------------------------------------------------------------------------
def hi_feature_pipeline():
    """The counterpart of shap_fsm_pipeline: the deep side reads the waveform,
    the classic side reads this vector. Feature names are the ones in
    Lampiran A and mxlstm/data/hi.py; the same extractor serves the RUL path."""
    fig, ax = new_fig(12.0, 4.6, strip_pad("hi_feature_pipeline"))
    py, ph, pw = 1.30, 2.55, 2.55
    xs = [0.40, 3.31, 6.22, 9.13]

    # panel 1: the segment
    _panel(ax, xs[0], py, pw, ph, "[[Raw signal]]", tint="input")
    waveform(ax, xs[0] + 0.30, py + 1.45, pw - 0.6, 0.70, seed=21)
    ax.text(xs[0] + pw / 2, py + 0.55, "segmen 2.048 titik\n[[channel]] [[drive-end]] dan [[fan-end]]",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED)

    # panel 2: the two feature families, per channel
    _panel(ax, xs[1], py, pw, ph, "Ekstraksi [[feature]]", tint="seq")
    c2 = xs[1] + pw / 2
    box(ax, c2, py + 1.68, 2.25, 0.72, "9 domain waktu", kind="proj", fs=8.8,
        sub="RMS, [[peak]], kurtosis, [[skewness]],\n[[crest]], [[shape]], [[impulse]], [[margin]], variansi")
    box(ax, c2, py + 0.80, 2.25, 0.72, "9 domain frekuensi", kind="proj", fs=8.8,
        sub="PSD Welch: sentroid, entropi,\n[[mean]] dan RMS [[frequency]], 5 energi pita")
    ax.text(c2, py + 0.22, "per [[channel]], 18 [[feature]]", ha="center", va="center",
            fontsize=FS_TINY, color=MUTED)

    # panel 3: the vector, drawn as its 18 x 2 cells
    _panel(ax, xs[2], py, pw, ph, "[[Feature vector]] HI", tint="mem")
    c3 = xs[2] + pw / 2
    cell, nx, ny = 0.105, 18, 2
    gx, gy = c3 - nx * cell / 2 + 0.08, py + 1.32
    for r in range(ny):
        for c in range(nx):
            kind = "proj" if c < 9 else "seq"
            ax.add_patch(Rectangle((gx + c * cell, gy + r * cell), cell * 0.9, cell * 0.9,
                                   fc=KIND[kind]["fc"], ec=KIND[kind]["ec"], lw=0.5, zorder=3))
    ax.text(gx - 0.07, gy + cell * 0.45, "FE", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(gx - 0.07, gy + cell * 1.45, "DE", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(gx + 4.5 * cell, gy + 2 * cell + 0.09, "9 waktu", ha="center", va="bottom",
            fontsize=7, color=MUTED)
    ax.text(gx + 13.5 * cell, gy + 2 * cell + 0.09, "9 frekuensi", ha="center", va="bottom",
            fontsize=7, color=MUTED)
    ax.text(c3, py + 0.55, "36-D per segmen\nZ-score [[fit-on-train]]",
            ha="center", va="center", fontsize=FS_NOTE, color=MUTED)

    # panel 4: the models and their explainer
    _panel(ax, xs[3], py, pw, ph, "Model klasik dan SHAP", tint="gate")
    c4 = xs[3] + pw / 2
    box(ax, c4, py + 1.72, 2.25, 0.66, "SVM-RBF · LR\nDT · RF · XGBoost", kind="seq", fs=8.8)
    arrow(ax, (c4, py + 1.37), (c4, py + 1.20), shrink=0, lw=0.9)
    box(ax, c4, py + 0.86, 2.25, 0.62, "SHAP", kind="gate", fs=8.8,
        sub="KernelExplainer · TreeExplainer")
    ax.text(c4, py + 0.25, "[[output]]: 10 kelas dan [[feature ranking]]",
            ha="center", va="center", fontsize=FS_TINY, color=MUTED)

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + pw + 0.04, py + ph / 2), (right - 0.04, py + ph / 2),
              lw=2.2, style="-|>")

    box(ax, 6.0, 0.62, 11.2, 0.56,
        "Ekstraktor yang sama dipakai jalur prognostik: per rekaman, [[channel]] "
        "horizontal dan vertikal, lalu Min–Max dan EMA α = 0,10 (Lampiran A)",
        kind="util", fs=8.5)

    ax.text(0.45, 4.35, "Dari [[raw signal]] ke [[feature vector]] HI", ha="left",
            va="center", fontsize=FS_TITLE + 1, color=INK, fontweight="bold")
    ax.text(0.45, 3.98, "18 [[feature]] per [[channel]] × 2 [[channel]] = 36 dimensi; "
            "model klasik membaca vektor ini, bukan [[waveform]]",
            ha="left", va="center", fontsize=FS_NOTE, color=MUTED)
    input_strip(ax, 12.0, "hi_feature_pipeline")
    save(fig, "hi_feature_pipeline")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Domain chart · four stages of bearing damage on the spectrum
# --------------------------------------------------------------------------
# The draft deck carried a third-party stage chart (a 2001 copyrighted
# drawing reproduced in the SKF training material), which the defence deck
# cannot reuse. This is an original schematic of the same well-known
# progression: four frequency zones across, four damage stages down, and the
# spike-energy (gE / HFD) level in the last column. Amplitudes are
# illustrative, not measured.
def bearing_failure_stages():
    W, H = 6.0, 4.62
    fig, ax = new_fig(W, H)

    zones = [  # (x0, x1, label, tint)
        (1.00, 1.85, "Zona A\nputaran\nporos", "util"),
        (1.85, 3.15, "Zona B\nfrekuensi cacat\n[[bearing]]", "gate"),
        (3.15, 4.35, "Zona C\nfrekuensi natural\nkomponen", "seq"),
        (4.45, 5.05, "Zona D\n[[spike energy]]\ngE / HFD", "active"),
    ]
    y_top, row_h, base_off, plot_h = 3.87, 0.88, 0.78, 0.62
    y_bot = y_top - 4 * row_h

    for x0, x1, label, tint in zones:
        ax.add_patch(Rectangle((x0, y_bot), x1 - x0, y_top - y_bot,
                               fc=KIND[tint]["fc"], ec="none", alpha=0.45, zorder=0))
        ax.text((x0 + x1) / 2, y_top + 0.36, label, ha="center", va="center",
                fontsize=FS_TINY - 0.6, color=INK, fontweight="bold", linespacing=1.15)
    for i in range(5):
        y = y_top - i * row_h
        ax.plot([0.05, 5.95], [y, y], color=MUTED, lw=0.6, ls=(0, (4, 3)), zorder=1)

    # (name, note, gE level, gE note, peaks: (x in inches, amp 0..1, label))
    fn = (3.15 + 4.35) / 2
    stages = [
        ("Tahap 1", "gejala paling awal", 0.35, "gE mulai\nnaik",
         [(1.15, 0.90, "1×"), (1.35, 0.20, "2×")], False),
        ("Tahap 2", "komponen [[ringing]]", 0.55, "gE naik",
         [(1.15, 0.90, "1×"), (1.35, 0.20, "2×"),
          (fn, 0.60, "fₙ"), (fn - 0.16, 0.22, None), (fn + 0.16, 0.22, None)], False),
        ("Tahap 3", "cacat terlihat jelas", 0.80, "gE tinggi",
         [(1.15, 0.90, "1×"), (1.35, 0.25, "2×"),
          (2.15, 0.55, "BPFO"), (2.45, 0.50, "BPFI"), (2.75, 0.32, "2×BPFI"), (3.00, 0.22, None),
          (fn, 0.65, "fₙ"), (fn - 0.16, 0.35, None), (fn + 0.16, 0.35, None)], False),
        ("Tahap 4", "menjelang gagal", 1.00, "gE turun,\nlalu\nmelonjak",
         [(1.15, 0.90, "1×"), (1.35, 0.55, "2×"), (1.55, 0.45, "3×")], True),
    ]

    rng = np.random.default_rng(11)
    for i, (name, note, ge, ge_note, peaks, broadband) in enumerate(stages):
        top = y_top - i * row_h
        base = top - base_off
        # row label
        ax.text(0.50, top - 0.30, name, ha="center", va="center",
                fontsize=FS_BLOCK, color=INK, fontweight="bold")
        ax.text(0.50, top - 0.54, note, ha="center", va="center",
                fontsize=FS_TINY - 1.0, color=MUTED, linespacing=1.1)
        # spectrum across zones A–C
        f = np.linspace(1.00, 4.35, 900)
        s = 0.05 + 0.03 * rng.random(f.size)
        if broadband:
            hump = 0.30 + 0.12 * rng.random(f.size + 8)
            hump = np.convolve(hump, np.ones(9) / 9, mode="valid")
            s = s + np.where(f > 1.85, hump, 0.0) * np.clip((f - 1.85) / 0.25, 0, 1)
        for x, amp, _ in peaks:
            s = s + amp * np.exp(-((f - x) ** 2) / (2 * 0.016**2))
        s = np.clip(s, 0, 1)
        ax.plot(f, base + s * plot_h, color=INK, lw=0.85, zorder=5)
        ax.plot([1.00, 4.35], [base, base], color=MUTED, lw=0.7, zorder=4)
        for x, amp, label in peaks:
            if label:
                ax.text(x, base + amp * plot_h + 0.05, label, ha="center", va="bottom",
                        fontsize=FS_TINY - 1.4, color=INK, zorder=6)
        if i == 1:
            ax.text(fn + 0.27, base + 0.30 * plot_h, "[[sideband]]", ha="left",
                    va="center", fontsize=FS_TINY - 1.4, color=INK, zorder=6)
        if broadband:
            ax.text(3.10, base + 0.50 * plot_h, "getaran acak\nfrekuensi tinggi",
                    ha="center", va="center", fontsize=FS_TINY - 1.2, color=INK,
                    zorder=6, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        # spike-energy bar
        ax.add_patch(Rectangle((4.57, base), 0.36, ge * plot_h,
                               fc=GOLD_FILL, ec=GOLD, lw=1.0, zorder=5))
        ax.plot([4.45, 5.05], [base, base], color=MUTED, lw=0.7, zorder=4)
        ax.text(5.50, top - 0.40, ge_note, ha="center", va="center",
                fontsize=FS_TINY - 1.0, color=INK, linespacing=1.1)

    # axes hints
    arrow(ax, (1.00, y_bot - 0.14), (4.35, y_bot - 0.14), lw=0.9)
    ax.text(2.68, y_bot - 0.26, "frekuensi", ha="center", va="center",
            fontsize=FS_TINY - 0.6, color=MUTED)
    ax.text(4.75, y_bot - 0.20, "energi tinggi", ha="center", va="center",
            fontsize=FS_TINY - 0.6, color=MUTED)
    ax.text(0.50, y_bot - 0.20, "amplitudo skematis", ha="center", va="center",
            fontsize=FS_TINY - 1.2, color=MUTED)
    save(fig, "bearing_failure_stages", dpi=320)  # 6 in canvas: 320 dpi clears the 1.800 px bar


# --------------------------------------------------------------------------
# Building-block charts · one primitive each, shown before the backbones
# --------------------------------------------------------------------------
# The "why these four" slide compares Mamba, xLSTM, N-BEATS, and TCN, so the
# audience meets each primitive on its own first. The Mamba and mLSTM charts
# are the zoom panels of Mamba-xLSTM-Net; the TCN chart is the dilation cone
# of SparseGate-TCN-RUL; the N-BEATS chart is the doubly residual block of
# Oreshkin dkk. (2020) with the three bases V14 Subbab V.1.2 assigns to it.
_BLOCK_W, _BLOCK_H = 8.6, 3.0
_BLOCK_BOX = (0.25, 0.15, 8.10, 2.70)


def block_mamba():
    fig, ax = new_fig(_BLOCK_W, _BLOCK_H)
    _mamba_panel(ax, *_BLOCK_BOX)
    save(fig, "block_mamba", dpi=240)


def block_mlstm():
    fig, ax = new_fig(_BLOCK_W, _BLOCK_H)
    _mlstm_panel(ax, *_BLOCK_BOX)
    save(fig, "block_mlstm", dpi=240)


def block_tcn():
    fig, ax = new_fig(_BLOCK_W, _BLOCK_H)
    _dilation_panel(ax, *_BLOCK_BOX)
    save(fig, "block_tcn", dpi=240)


def block_nbeats():
    fig, ax = new_fig(_BLOCK_W, _BLOCK_H)
    px, py, pw, ph = _BLOCK_BOX
    _panel(ax, px, py, pw, ph, "Blok N-BEATS ([[doubly residual stacking]] dan [[basis expansion]])")
    my = py + 1.16
    ax.plot([px + 0.28], [my], marker="o", ms=4, color=INK)
    ax.text(px + 0.28, my + 0.22, "[[input]]", ha="center", va="bottom", fontsize=FS_TINY, color=MUTED)
    arrow(ax, (px + 0.30, my), (px + 0.72, my))
    container(ax, px + 0.78, my - 0.44, 1.74, 0.88, times="4×")
    box(ax, px + 1.65, my, 1.34, 0.50, "FC + ReLU", kind="proj", fs=8.5)
    arrow(ax, (px + 2.34, my), (px + 2.66, my))
    ax.plot([px + 2.66, px + 2.66], [my - 0.55, my + 0.55], color=INK, lw=1.0)
    arrow(ax, (px + 2.66, my + 0.55), (px + 2.96, my + 0.55), shrink=0)
    arrow(ax, (px + 2.66, my - 0.55), (px + 2.96, my - 0.55), shrink=0)
    box(ax, px + 3.32, my + 0.55, 0.68, 0.44, "$\\theta_b$", kind="util", fs=9)
    box(ax, px + 3.32, my - 0.55, 0.68, 0.44, "$\\theta_f$", kind="util", fs=9)
    arrow(ax, (px + 3.68, my + 0.55), (px + 4.02, my + 0.55))
    arrow(ax, (px + 3.68, my - 0.55), (px + 4.02, my - 0.55))
    box(ax, px + 4.86, my + 0.55, 1.62, 0.50, "[[basis]] $g_b$ → [[backcast]]", kind="seq", fs=8.5)
    box(ax, px + 4.86, my - 0.55, 1.62, 0.50, "[[basis]] $g_f$ → [[forecast]]", kind="mem", fs=8.5)
    # backcast is subtracted from the input before the next block
    ax.plot([px + 5.69, px + 6.10], [my + 0.55, my + 0.55], color=INK, lw=1.0)
    ax.plot([px + 6.10, px + 6.10], [my + 0.55, my + 1.06], color=INK, lw=1.0)
    ax.plot([px + 6.10, px + 0.28], [my + 1.06, my + 1.06], color=INK, lw=1.0)
    arrow(ax, (px + 0.28, my + 1.06), (px + 0.28, my + 0.30), shrink=1)
    gate_glyph(ax, px + 0.28, my + 0.62, "−", kind="gate", r=0.13, fs=9)
    ax.text(px + 3.2, my + 1.06 - 0.05, "[[input]] blok berikutnya = [[input]] − [[backcast]]",
            ha="center", va="top", fontsize=FS_TINY, color=MUTED, style="italic")
    # forecasts of all blocks add up to the estimate
    arrow(ax, (px + 5.69, my - 0.55), (px + 6.14, my - 0.55))
    gate_glyph(ax, px + 6.30, my - 0.55, "+", kind="mem", r=0.16, fs=10)
    arrow(ax, (px + 6.47, my - 0.55), (px + 6.80, my - 0.55), shrink=0)
    box(ax, px + 7.36, my - 0.55, 1.04, 0.58, "Σ [[forecast]]", kind="out", fs=8.5, sub="estimasi RUL")
    ax.text(px + pw / 2, py + 0.14,
            "basis pada penelitian ini: [[trend]] polinomial Bernstein · [[wear]] frekuensi "
            "karakteristik · [[shock]] [[wavelet]] Gabor",
            ha="center", va="bottom", fontsize=FS_TINY, color=MUTED, style="italic")
    save(fig, "block_nbeats", dpi=240)


# Domain and block charts share the palette and the italics hook but are not
# algorithm diagrams: no input contract, no strip.
CHARTS = {
    "bearing_failure_stages": bearing_failure_stages,
    "block_mamba": block_mamba,
    "block_mlstm": block_mlstm,
    "block_nbeats": block_nbeats,
    "block_tcn": block_tcn,
}


DIAGRAMS = {
    "hi_feature_pipeline": hi_feature_pipeline,
    "mamba_xlstm_full": mamba_xlstm_full,
    "nbeats_xlstm_full": nbeats_xlstm_full,
    "sparsegate_tcn_full": sparsegate_tcn_full,
    "wdcnn_full": wdcnn_full,
    "topk_sae": topk_sae,
    "sae_bpfx_pipeline": sae_bpfx_pipeline,
    "shap_fsm_pipeline": shap_fsm_pipeline,
    "classic_ml_trio": classic_ml_trio,
    "streaming_engine": streaming_engine,
}


def save(fig, name: str, dpi: int = 200) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.png"
    fig.savefig(target, dpi=dpi, facecolor="white")
    plt.close(fig)
    print(f"{name:24s} -> {target.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="comma-separated diagram names")
    args = parser.parse_args()
    everything = {**DIAGRAMS, **CHARTS}
    names = args.only.split(",") if args.only else list(everything)
    for name in names:
        if name not in everything:
            print(f"unknown diagram: {name}", file=sys.stderr)
            return 1
        everything[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
