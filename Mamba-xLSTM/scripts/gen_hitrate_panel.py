"""Generate the BPFx hit-rate panel (Gambar in Bab V, fig:bab5_hitrate_panel).

Replaces the previous 4-panel *bootstrap CI* figure (sourced from the
journal_q2/stats pipeline, whose numbers diverged from Tabel V.6) with a
simpler, reproducible 4-panel point hit-rate bar chart whose values match
Tabel V.6 exactly.

Provenance of the point hit-rates:
  * PHM2012, XJTU-SY  -> results/bpfx_mapping/{phm2012,xjtusy}_bpfx_results.json
                         (the canonical SAE post-hoc mapping behind Tabel V.6)
  * IMS, CWRU         -> Tabel V.6 canonical values (no separate JSON saved;
                         traceable to the chapter table / Journal2).

Outputs (PDF for LaTeX + PNG for the DOCX assets):
  writings/disertation/figures/bab5/hitrate_panel.pdf
  writings/disertation/figures/bab5/hitrate_panel.png
  dissertation-docx/assets/figures/bab5/hitrate_panel.png
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]            # Mamba-xLSTM/
_REPO = _ROOT.parent                                   # repo root
_BPFX = _ROOT / "results" / "bpfx_mapping"

# Consistent BPFx order + colours (caption: BPFI biru, BPFO merah, BSF abu-abu)
BPFX_ORDER = ["BPFI", "BPFO", "BSF", "FTF"]
COLORS = {"BPFI": "#1f77b4", "BPFO": "#d62728", "BSF": "#7f7f7f", "FTF": "#9467bd"}


def _load_json_hr(path: Path) -> dict[str, float]:
    """Return hit-rate fraction per BPFx from a bpfx_mapping results JSON."""
    d = json.loads(path.read_text())
    return {k: float(v) for k, v in d["hit_rate"].items()}


def main() -> None:
    # PHM2012 + XJTU-SY straight from the canonical mapping JSONs
    phm = _load_json_hr(_BPFX / "phm2012_bpfx_results.json")
    xjtu = _load_json_hr(_BPFX / "xjtusy_bpfx_results.json")

    # IMS + CWRU: canonical Tabel V.6 values (percent -> fraction)
    ims = {"BPFI": 0.0176, "BPFO": 0.0, "BSF": 0.0049, "FTF": 0.0}
    cwru = {"BPFI": 0.0508, "BPFO": 0.0, "BSF": 0.0, "FTF": 0.0}

    panels = [
        ("PHM2012", phm, "$p < 0{,}001$", False),
        ("XJTU-SY", xjtu, "$p < 0{,}001$", False),
        ("IMS", ims, "$p = 0{,}001$", False),
        ("CWRU", cwru, "$p > 0{,}05$ (underpowered, $n=10$)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    for ax, (name, hr, pval, underpowered) in zip(axes, panels):
        pct = [hr.get(b, 0.0) * 100.0 for b in BPFX_ORDER]
        bars = ax.bar(
            BPFX_ORDER, pct,
            color=[COLORS[b] for b in BPFX_ORDER],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        ymax = max(pct) if max(pct) > 0 else 1.0
        ax.set_ylim(0, ymax * 1.35)
        # annotate the bar values
        for b, v in zip(bars, pct):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.03,
                        f"{v:.2f}".replace(".", ",") + "%",
                        ha="center", va="bottom", fontsize=8.5, zorder=4)
        title_suffix = "  —  underpowered" if underpowered else ""
        ax.set_title(f"{name}{title_suffix}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Hit-rate (%)", fontsize=9.5)
        ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # p-value annotation in the corner
        ax.text(0.97, 0.95, pval, transform=ax.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox=dict(facecolor="#f5f5f5", edgecolor="#cccccc",
                          boxstyle="round,pad=0.25"))

    fig.suptitle(
        "Hit-rate fitur SAE terhadap BPFx per dataset "
        r"($|r| \geq 0{,}30$, $d_{\mathrm{lat}}=1.024$)",
        fontsize=12.5, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_pdf = _REPO / "writings/disertation/figures/bab5/hitrate_panel.pdf"
    out_png = _REPO / "writings/disertation/figures/bab5/hitrate_panel.png"
    out_docx = _REPO / "dissertation-docx/assets/figures/bab5/hitrate_panel.png"
    for p in (out_pdf, out_png, out_docx):
        p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_docx, dpi=150)
    plt.close(fig)

    print("Saved:")
    for p in (out_pdf, out_png, out_docx):
        print(f"  {p}")
    print("\nPanel values (%):")
    for name, hr, pval, _ in panels:
        vals = {b: round(hr.get(b, 0.0) * 100, 2) for b in BPFX_ORDER}
        print(f"  {name:8}: {vals}   {pval}")


if __name__ == "__main__":
    main()
