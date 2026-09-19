"""Generate Gambar D.1 — inference data-flow schematic for Lampiran D.

Pipeline:
  SKF IMx-8 spindle recording
    → HI 18-D feature extraction
    → SVM-RBF (trained on CWRU)
    → Predicted condition label
    → Qualitative comparison with SKF PdM label

Output:
    dissertation-docx/assets/figures/lampD/lampD_skema_inferensi.png
    writings/disertation/figures/lampD/lampD_skema_inferensi.png
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR_DOCX = Path(__file__).resolve().parents[1] / "assets" / "figures" / "lampD"
OUT_DIR_LATEX = Path(__file__).resolve().parents[2] / "writings" / "disertation" / "figures" / "lampD"
OUT_DIR_DOCX.mkdir(parents=True, exist_ok=True)
OUT_DIR_LATEX.mkdir(parents=True, exist_ok=True)


def draw_box(ax, x, y, w, h, label, sublabel="", color="#1565C0", text_color="white",
             fontsize=11, subfontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.25, sublabel, ha="center", va="center",
                color=text_color, fontsize=subfontsize, style="italic", zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)


def draw_arrow(ax, x1, x2, y, color="#424242"):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(
            arrowstyle="-|>", color=color,
            lw=1.8, mutation_scale=18,
        ),
        zorder=2,
    )


def main():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # Node positions (cx, cy, w, h)
    nodes = [
        (1.3, 2.1, 2.1, 1.6, "Rekaman\nSpindel SKF",  "IMx-8 sensor",   "#0D47A1"),
        (4.0, 2.1, 2.1, 1.6, "Ekstraksi\nFitur HI",   "18 dimensi",     "#1B5E20"),
        (6.7, 2.1, 2.1, 1.6, "SVM-RBF",               "trained CWRU",   "#4A148C"),
        (9.4, 2.1, 2.1, 1.6, "Label\nPrediksi",        "Normal/Warn/Crit","#E65100"),
        (12.0, 2.1, 1.8, 1.6,"Perbandingan\nKualitatif","label PdM SKF", "#B71C1C"),
    ]

    for cx, cy, w, h, label, sub, color in nodes:
        draw_box(ax, cx, cy, w, h, label, sub, color)

    # Arrows between nodes
    arrow_pairs = [
        (1.3 + 1.05, 4.0 - 1.05, 2.1),
        (4.0 + 1.05, 6.7 - 1.05, 2.1),
        (6.7 + 1.05, 9.4 - 1.05, 2.1),
        (9.4 + 1.05, 12.0 - 0.90, 2.1),
    ]
    for x1, x2, y in arrow_pairs:
        draw_arrow(ax, x1, x2, y)

    # Annotations on arrows
    arrow_labels = [
        (2.65, 2.55, "sinyal\nmentah"),
        (5.35, 2.55, "vektor\nfitur"),
        (8.05, 2.55, "skor\nkeputusan"),
        (10.75, 2.55, "label\nkelas"),
    ]
    for x, y, lbl in arrow_labels:
        ax.text(x, y, lbl, ha="center", va="bottom", fontsize=8, color="#616161",
                style="italic")

    # Legend / title area
    ax.text(6.5, 0.4,
            "Gambar D.1  —  Aliran Data Skema Inferensi: PT SKF Indonesia (sanity check eksternal)",
            ha="center", va="center", fontsize=10, color="#212121",
            fontweight="bold",
            bbox=dict(facecolor="#F5F5F5", edgecolor="#BDBDBD", boxstyle="round,pad=0.3"),
            )

    fig.tight_layout(pad=0.3)

    for out_dir in [OUT_DIR_DOCX, OUT_DIR_LATEX]:
        out = out_dir / "lampD_skema_inferensi.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved → {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
