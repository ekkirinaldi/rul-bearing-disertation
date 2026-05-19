#!/usr/bin/env python3
"""Create minimal placeholder PDF files for missing dissertation figures."""
import subprocess, os, sys

FIGURES = [
    "figures/bab4/shap_kernel_summary.pdf",
    "figures/bab4/shap_tree_summary.pdf",
    "figures/bab4/wdcnn_training_curves.pdf",
    "figures/bab4/shap_signal_ir21.pdf",
    "figures/bab4/fsm_comparison.pdf",
    "figures/bab4/fsm_radar.pdf",
    "figures/bab4/fsm_physics.pdf",
    "figures/bab5/rul_curves_phm2012.pdf",
    "figures/bab5/rul_curves_ims.pdf",
    "figures/bab5/hitrate_panel.pdf",
    "figures/bab5/corr_scatter_phm2012.pdf",
    "figures/bab5/negative_controls.pdf",
    "figures/bab5/sparsity_sweep.pdf",
]

# Minimal 1-page PDF with gray box (Ghostscript PostScript → PDF)
PS_TEMPLATE = """%!PS-Adobe-3.0
%%BoundingBox: 0 0 432 288
%%Pages: 1
%%Page: 1 1
/Helvetica findfont 14 scalefont setfont
0.7 setgray
1 1 430 286 rectstroke
0 setgray
120 140 moveto
({label}) show
showpage
%%EOF
"""

os.makedirs("figures/bab4", exist_ok=True)
os.makedirs("figures/bab5", exist_ok=True)

for f in FIGURES:
    if os.path.exists(f):
        print(f"Exists:  {f}")
        continue
    label = os.path.splitext(os.path.basename(f))[0]
    ps_content = PS_TEMPLATE.format(label=label)
    ps_path = f"/tmp/{label}.ps"
    with open(ps_path, "w") as fh:
        fh.write(ps_content)
    result = subprocess.run(
        ["gs", "-q", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite",
         f"-sOutputFile={f}", ps_path],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"Created: {f}")
    else:
        print(f"FAILED:  {f}\n  {result.stderr}", file=sys.stderr)

print("Done.")
