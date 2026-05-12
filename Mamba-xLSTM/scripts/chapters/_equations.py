"""Render LaTeX-style equations to PNG via matplotlib mathtext.

Output PNGs are cached in `Mamba-xLSTM/results/_chapter_assets/equations/`
so repeated builds do not re-render unchanged equations.

Matplotlib's mathtext parser supports a subset of LaTeX. The ``_sanitize_mathtext``
helper rewrites common full-LaTeX commands (``\\bigl``, ``\\dfrac``, ``\\text{...}``)
to their mathtext equivalents so source equations can be authored in normal LaTeX.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ASSET_ROOT = Path(__file__).resolve().parents[2] / "results" / "_chapter_assets" / "equations"


def _key(text: str, dpi: int, fontsize: int) -> str:
    h = hashlib.sha1(f"{text}|{dpi}|{fontsize}".encode("utf-8")).hexdigest()[:14]
    return h


_BIG_OPEN_RE = re.compile(r"\\(?:bigl|Bigl|biggl|Biggl|big|Big|bigg|Bigg)([\(\[\|])")
_BIG_CLOSE_RE = re.compile(r"\\(?:bigr|Bigr|biggr|Biggr|big|Big|bigg|Bigg)([\)\]\|])")


def _sanitize_mathtext(latex: str) -> str:
    """Translate full-LaTeX constructs into mathtext-compatible form.

    Mathtext rejects ``\\bigl(``, ``\\dfrac``, ``\\substack`` and friends; this
    helper rewrites them with the closest supported alternative so dissertation
    equations authored in standard LaTeX render correctly.
    """

    s = latex

    s = _BIG_OPEN_RE.sub(lambda m: r"\left" + m.group(1), s)
    s = _BIG_CLOSE_RE.sub(lambda m: r"\right" + m.group(1), s)

    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")

    s = s.replace(r"\!", "")

    s = re.sub(r"\\le\b", r"\\leq", s)
    s = re.sub(r"\\ge\b", r"\\geq", s)
    s = re.sub(r"\\ne\b", r"\\neq", s)

    return s


def render_equation(latex: str, *, dpi: int = 220, fontsize: int = 16) -> Path:
    """Return Path to PNG for a centered standalone equation written in mathtext.

    ``latex`` should be the math expression *without* enclosing ``$...$``. The
    string is passed through :func:`_sanitize_mathtext` first, so common LaTeX
    constructs such as ``\\bigl(``, ``\\Bigr)`` and ``\\dfrac`` are accepted.
    """

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    out = ASSET_ROOT / f"eq_{_key(latex, dpi, fontsize)}.png"
    if out.exists():
        return out

    text = _sanitize_mathtext(latex)

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(
        0.5,
        0.5,
        f"${text}$",
        ha="center",
        va="center",
        fontsize=fontsize,
        usetex=False,
    )
    fig.savefig(
        out,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.08,
        transparent=False,
        facecolor="white",
    )
    plt.close(fig)
    return out
