#!/bin/bash
# Create placeholder PDF figures for missing figure files
set -e

FIGURES=(
  "figures/bab4/shap_kernel_summary.pdf"
  "figures/bab4/shap_tree_summary.pdf"
  "figures/bab4/wdcnn_training_curves.pdf"
  "figures/bab4/shap_signal_ir21.pdf"
  "figures/bab4/fsm_comparison.pdf"
  "figures/bab4/fsm_radar.pdf"
  "figures/bab4/fsm_physics.pdf"
  "figures/bab5/rul_curves_phm2012.pdf"
  "figures/bab5/rul_curves_ims.pdf"
  "figures/bab5/hitrate_panel.pdf"
  "figures/bab5/corr_scatter_phm2012.pdf"
  "figures/bab5/negative_controls.pdf"
  "figures/bab5/sparsity_sweep.pdf"
)

mkdir -p figures/bab4 figures/bab5

for f in "${FIGURES[@]}"; do
  if [ ! -f "$f" ]; then
    label=$(basename "$f" .pdf)
    cat > /tmp/placeholder.tex <<TEXEOF
\documentclass[border=2pt]{standalone}
\usepackage{tikz}
\begin{document}
\begin{tikzpicture}
\draw[gray,thick] (0,0) rectangle (10cm,6cm);
\node[gray,font=\large] at (5cm,3cm) {[Placeholder: $label]};
\end{tikzpicture}
\end{document}
TEXEOF
    pdflatex -interaction=nonstopmode -output-directory=/tmp /tmp/placeholder.tex > /dev/null 2>&1
    cp /tmp/placeholder.pdf "$f"
    echo "Created: $f"
  else
    echo "Exists:  $f"
  fi
done
echo "Done."
