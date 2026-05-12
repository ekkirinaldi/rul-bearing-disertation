# Bearing RUL Prediction — Doctoral Dissertation

**Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time**

Doctoral dissertation by **Toto Suharto** at Institut Teknologi Bandung (ITB).

---

## Overview

This repository contains the research code, experiment results, and LaTeX dissertation source for a doctoral thesis on **Remaining Useful Life (RUL) prediction of rolling element bearings** using deep learning.

The work proposes **Mamba-xLSTM-Net**, a novel hybrid architecture combining:
- **Mamba (Selective State Space Model)** — for capturing long-range degradation trends
- **xLSTM (Extended LSTM with matrix memory)** — for tracking local dynamics and late-life spikes
- **Top-*k* Sparse Autoencoder (SAE)** — for post-hoc interpretability via mapping latent features to bearing characteristic frequencies (BPFO, BPFI, BSF, FTF)

Experiments are conducted on two public run-to-failure datasets: **PHM2012 (FEMTO-PRONOSTIA)** and **XJTU-SY**.

---

## Repository Structure

```
.
├── Mamba-xLSTM/          # Training code, model configs, and experiment results
│   ├── configs/          # Dataset, model, training, and ablation YAML configs
│   ├── scripts/          # Training, comparison, and VPS deployment scripts
│   ├── src/              # Model architectures (Mamba-xLSTM-Net, SAE, baselines)
│   └── results/          # Experiment outputs (reports, chapter assets, BPFx mapping)
│
├── writings/
│   └── disertation/      # LaTeX dissertation source
│       ├── disertasi.tex         # Master file
│       ├── itbdisertasi.cls      # ITB dissertation class
│       ├── chapters/             # Chapter .tex files (Bab I–VI + front matter)
│       ├── lampiran/             # Appendices
│       ├── figures/              # Figures organized by chapter
│       ├── references.bib        # BibLaTeX bibliography
│       └── Makefile              # Build targets (build, clean, check, lint)
│
├── spec/                 # Research plans, algorithm design notes, draft content
├── disertasi/            # Reference materials (ITB template, guidelines)
└── data-bearing/         # Bearing datasets (not committed — download from S3)
```

---

## Compiling the Dissertation

LaTeX is compiled via Docker (requires Docker Desktop):

```bash
cd writings/disertation
docker run --rm -v "$(pwd):/workdir" -w /workdir \
  danteev/texlive \
  latexmk -outdir=build -interaction=nonstopmode disertasi.tex
```

Or using the Makefile if `latexmk` + LuaLaTeX are installed locally:

```bash
make build     # Full compile (LuaLaTeX + Biber)
make clean     # Remove intermediate files
make check     # Run ITB format lint checks
```

Output: `build/disertasi.pdf` (also copied to `disertasi.pdf`)

---

## Training

Training requires a GPU. See [`Mamba-xLSTM/README.md`](Mamba-xLSTM/README.md) for setup.

Quick start (after bootstrapping the GPU environment):

```bash
cd Mamba-xLSTM
source .venv/bin/activate
python scripts/run_algorithm_comparison.py \
  --datasets phm2012 xjtusy \
  --models mamba_xlstm_rul \
  --train configs/train/algorithm_comparison.yaml
```

---

## Dataset

Bearing datasets are **not committed** to this repository due to size.

Download via S3 (public):

```bash
curl -fL -o data-bearing.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip'
unzip -q data-bearing.zip && rm data-bearing.zip
```

Place the `data-bearing/` directory at the repository root alongside `Mamba-xLSTM/`.

---

## Research Contributions

1. **(Technology–Methodology)** Mamba-xLSTM-Net: a hybrid SSM + matrix-memory recurrent architecture with gated fusion, designed for bearing RUL prediction on vibration signals.

2. **(Technology–Methodology)** Top-*k* SAE interpretability framework applied post-hoc to the model's latent representations, with a quantitative procedure mapping learned features to bearing characteristic frequencies (BPFO, BPFI, BSF, FTF).

3. **(Output)** Empirical evidence that deep learning models trained for RUL prediction learn latent representations that correspond to classical vibration diagnosis theory (peak Pearson *r* = 0.507, *p* ≪ 0.05).

---

## Citation

> Suharto, T. (2026). *Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*. Doctoral dissertation, Institut Teknologi Bandung.
