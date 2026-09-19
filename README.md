# Bearing PdM — Doctoral Dissertation

**Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time**

Doctoral dissertation by **Toto Suharto** at Institut Teknologi Bandung (ITB), Program Doktor Teknik dan Manajemen Industri, FTI.

---

## Overview

This repository hosts the dissertation manuscript (Word/DOCX), the sidang deck and supporting artifacts for a doctoral thesis on **interpretable Predictive Maintenance (PdM) of rolling element bearings**. The work spans two parallel tracks plus a synthesis chapter:

- **Track A — Diagnostic** (Bab IV). Multi-model bearing fault classification on CWRU using three algorithm families (kernel: SVM/LR; tree: DT/RF/XGBoost; deep: WDCNN), each with its native SHAP explainer (Kernel / Tree / Deep), culminating in **Fault Signature Maps (FSM)** — signal-level XAI at 2 048-point resolution.
- **Track B — Prognostic** (Bab V). RUL estimation on PHM2012, XJTU-SY, and IMS with three deep-learning backbones (**Mamba-xLSTM-Net**, **N-BEATS-xLSTM-RUL**, **SparseGate-TCN-RUL**), plus a **Top-*k* Sparse Autoencoder** that maps hidden states to bearing characteristic frequencies (BPFO/BPFI/BSF/FTF) under bootstrap CI + permutation test + two negative controls.
- **Convergence** (Bab VI). A **multi-tier PdM blueprint** (Edge IoT → Edge Server → Cloud/GPU) that places each model at the tier matching its parameter count, latency, and XAI type.

The dissertation structure, RQ/Tujuan/Novelti mapping, and per-chapter plan are documented in [`writings/dissertation-outline.md`](writings/dissertation-outline.md) — the **single source of truth** for chapter content.

---

## Repository Layout

```
.
├── dissertation-docx/            # ★ The manuscript (Word)
│   ├── disertasi.docx            # Master document
│   ├── chapters/                 # Bab I–VI
│   ├── lampiran/                 # Lampiran A–G
│   ├── RULES.md                  # Writing + formatting rules for DOCX work
│   ├── tools/                    # insert_docx.py, lint_docx.sh, merge_master.py
│   ├── assets/                   # template.docx, itb-sps.csl, figures/, figure-map.tsv
│   └── Makefile                  # insert, insert-dry, lint, master, clean
│
├── presentation/                 # Sidang deck
│   ├── content/*.yaml            # Slide spec (the deck is written here, not in Word)
│   ├── build.py                  # build / lint / preview / inspect
│   ├── tools/render_diagrams.py  # matplotlib figures used by the slides
│   └── V16 …docx                 # Current manuscript draft circulated for review
│
├── writings/
│   ├── dissertation-outline.md   # ★ Single source of truth for chapter structure
│   ├── journal-q2/               # JETS Q2 paper (separate LaTeX project, not the thesis)
│   ├── Outline_Disertasi_6Bab.pdf  # 6-bab structural target
│   └── SK-Toto.pdf               # Proposal disertasi (Agustus 2025) — Bab I source
│
├── Paper/                        # Four self-papers (primary empirical sources)
│   ├── Conference1_Classification_SVM_LR.pdf       # Kernel family + SHAP KernelExplainer
│   ├── Conference2_Classification_Tree.pdf         # Tree family + SHAP TreeExplainer
│   ├── Journal1_Fault Signature Maps.pdf           # WDCNN + SHAP DeepExplainer + FSM
│   └── Journal2_RUL_Journal.pdf                    # Mamba-xLSTM + Top-k SAE + BPFx
│
├── Notebook/                     # Reproducibility notebooks
│   ├── Conference1_Classification_SVM_LR.ipynb
│   ├── Conference2_Classification_Tree.ipynb
│   └── Journal1_Fault Signature Maps.ipynb         # (Journal 2 notebook pending)
│
├── CLAUDE.md                     # Project + ITB writing-rule instructions for Claude Code
├── new_algorithm.md              # Algorithm brainstorming notes
│
├── Mamba-xLSTM/                  # Python training pipeline (local-only, gitignored)
└── data-bearing/                 # Bearing datasets (local-only, gitignored — see below)
```

**Local-only directories** (excluded via `.gitignore`):

- `Mamba-xLSTM/` — PyTorch/Lightning training pipeline for the three RUL backbones + Top-*k* SAE.
- `data-bearing/` — PHM2012 + XJTU-SY raw data (3 XJTU conditions). Processed parquet cache: `data-bearing/processed/`. Do **not** use legacy `data/cache/`.

---

## Building the Manuscript and the Deck

The manuscript is edited directly in Word; there is no source format to compile it from. The LaTeX tree it was ported from was removed in September 2026 and remains retrievable from git history. What the tooling does now is insert material, lint and merge.

```bash
cd dissertation-docx
make insert-dry        # preview an insertion spec without touching the manuscript
make insert            # apply it (tools/insert_docx.py)
make lint              # ITB format + prose lint; fix every [FATAL] before committing
make master            # merge frontmatter + chapters + lampiran into disertasi.docx
```

After a round of edits in Word, select all and press F9 once in the master so Daftar Isi, Daftar Gambar and Daftar Tabel repopulate.

The sidang deck is generated from a YAML spec, not authored in PowerPoint:

```bash
cd presentation
python3 build.py lint --spec content/sidang-terbuka.yaml              # structure + overflow
python3 build.py build --spec content/sidang-terbuka.yaml --strict    # -> out/*.pptx
python3 build.py preview --spec content/sidang-terbuka.yaml --png     # PDF + slide-NN.png
python3 tools/render_diagrams.py [--only name1,name2]                 # regenerate figures
```

---

## Training (Prognostic Track)

The Python pipeline for the three RUL backbones lives in `Mamba-xLSTM/` (local-only). Requires a CUDA GPU; see `Mamba-xLSTM/README.md` for environment setup.

Quick start:

```bash
cd Mamba-xLSTM
source .venv/bin/activate
python scripts/run_algorithm_comparison.py \
  --datasets phm2012 xjtusy \
  --models mamba_xlstm_rul nbeats_xlstm_rul sparsegate_tcn_rul \
  --train configs/train/algorithm_comparison.yaml
```

For the diagnostic track (CWRU + kernel/tree/deep), reproducibility notebooks in [`Notebook/`](Notebook/) run end-to-end on a CPU and do not require GPU.

---

## Inference Engine (Live RUL Demo)

A local web dashboard (`inference-engine/`) that simulates a live sensor stream from real PHM2012 / XJTU-SY bearing data and runs the dissertation **Mamba-xLSTM-Net** checkpoint on each acquisition. An optional **PT SKF CH-15 OR-1** stream replays the six-month Observer trending export as a transfer demo. Requires `Mamba-xLSTM/.venv` (with checkpoints under `Mamba-xLSTM/results/runs/`) and a populated `data-bearing/` — both local-only.

```bash
# 1. Install web-server deps into the existing Mamba-xLSTM venv
Mamba-xLSTM/.venv/bin/python -m pip install -r inference-engine/requirements.txt

# 2. Launch the dashboard (FastAPI + WebSocket stream on port 8800)
./inference-engine/run.sh
```

Then open **http://localhost:8800**. Headless smoke test:

```bash
cd inference-engine && ../Mamba-xLSTM/.venv/bin/python scripts/smoke_test.py
```

See [`inference-engine/README.md`](inference-engine/README.md) for architecture, dataset/model mapping, and the PT SKF six-month plant stream (`data-bearing/skf-ch15-or1-6m/`).

---

## Datasets

Bearing datasets are **not committed** to this repository (size + licensing).

**Prognostic — PHM2012 + XJTU-SY** (used by `Mamba-xLSTM/`). Place `data-bearing/` at the repository root alongside `Mamba-xLSTM/`.

```bash
# PHM2012 (+ base layout)
curl -fL -o data-bearing.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip'
unzip -q data-bearing.zip && rm -f data-bearing.zip

# XJTU-SY — full three-condition tree (2026-06 repair; required for --datasets xjtusy)
mkdir -p data-bearing
curl -fL -o xtju-sy.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/xtju-sy.zip'
unzip -q -o xtju-sy.zip -d data-bearing/ && rm -f xtju-sy.zip
find data-bearing/xtju-sy -name '*.csv' | wc -l   # expect 9216
```

The legacy `data-bearing.zip` alone does **not** include the repaired XJTU subtree; always add `xtju-sy.zip`. VPS workflow: `.cursor/rules/vps-ssh-key-access.mdc` §6 + §6.3.

**Diagnostic — CWRU** (used by the three notebooks in `Notebook/`): download from the [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). Each notebook documents the exact subset (drive-end + fan-end accelerometer, Load 0–3 HP, 48 kHz, 10 classes).

**Industrial — PT SKF CH-15 OR-1** (used by `inference-engine/` transfer demo only; not for training): place the six-month Observer trending export under `data-bearing/skf-ch15-or1-6m/` (six HTML `.xls` files: A/V/ENV for channels `Ch1-01-…-NDE` and `Ch3-02-…-DE`). This is local plant data, not on S3. See [`inference-engine/README.md`](inference-engine/README.md) for filenames, cadence, and how RUL is computed on this stream.

**Additional prognostic dataset — IMS** (used in Bab V cross-dataset validation): NASA Ames IMS bearing dataset (Rexnord, 2 000 rpm, four bearings).

---

## Research Contributions (N1–N5)

Five novelti anchored in the dissertation. Each is marked **empirical** (quantitatively measurable in Bab IV–V) or **conceptual/synthesis** (blueprint contribution, not a deployment study), per the no-overclaim policy in [CLAUDE.md](CLAUDE.md).

| # | Novelty | Track | Status | Headline evidence |
|---|---------|-------|--------|-------------------|
| **N1** | **Fault Signature Maps (FSM)** — first signal-level XAI for WDCNN at 2 048-point resolution (Signed / Absolute / Variance variants; three validation metrics). | A (Bab IV) | Empirical | Split-half stability 0,940; discriminability 0,216; severity monotonicity Ball 17,6% / IR 13,8% / OR 8,6%. |
| **N2** | **BatchNorm trade-off** — first finding that BatchNorm suppresses FSM discriminability; design guidance between maximum accuracy and maximum interpretability. | A (Bab IV) | Empirical | Discriminability +133% on the BN-free variant, with −3,60 pp accuracy trade-off. |
| **N3** | **SAEBearing** — first adaptation of mechanistic interpretability (Top-*k* SAE, Bricken–Cunningham style) to bearing prognostics. | B (Bab V) | Empirical | Hit-rate BPFI 2,3% (PHM2012) and BPFO 2,2% (XJTU-SY), *p* < 0,001, *r*<sub>max</sub> = 0,507. |
| **N4** | **Universality + statistical rigor** — three backbones × four datasets with bootstrap CI + permutation test + two negative controls (Xavier-init + Gaussian noise). | B (Bab V) | Empirical | BPFx-dominant convergence across architectures; hit-rate collapses to ~0 on both controls. |
| **N5** | **Multi-tier PdM blueprint** — synthesis of FSM (input-attribution) and SAE-BPFx (latent-concept) into a three-tier architecture (Edge IoT → Edge Server → Cloud/GPU) auditable layer-by-layer. | Convergence (Bab VI) | Conceptual / synthesis | Per-tier model placement based on parameter count, latency, and XAI type. **Not** a field-deployment study. |

**Mapping to research questions** (per `writings/dissertation-outline.md` §1.2–§1.3):

- **RQ1** (multi-tier integration for OEE) ↔ Tujuan 1 ↔ **N5** (conceptual)
- **RQ2** (hybrid ML+DL consistency across datasets) ↔ Tujuan 2 ↔ **N4**
- **RQ3** (XAI transparency for industrial adoption) ↔ Tujuan 3 ↔ **N1 + N2 + N3**

---

## Self-Citations

The four papers in [`Paper/`](Paper/) are the primary empirical sources for Bab IV and Bab V. BibTeX keys used throughout the manuscript:

| BibTeX key | Paper | Cited in |
|---|---|---|
| `TotoSuharto2024Conf1SVM` | Conference 1 — SVM/LR + SHAP KernelExplainer on CWRU | Bab IV §IV.2 + §IV.7; Bab II §II.2.3 |
| `TotoSuharto2024Conf2Tree` | Conference 2 — DT/RF/XGBoost + SHAP TreeExplainer on CWRU | Bab IV §IV.3 + §IV.8; Bab II §II.2.3 |
| `TotoSuharto2025Journal1FSM` | Journal 1 — WDCNN + SHAP DeepExplainer + FSM | Bab IV §IV.4–§IV.13; Bab II §II.3.3 |
| `TotoSuharto2025Journal2RUL` | Journal 2 — Mamba-xLSTM + Top-*k* SAE + BPFx mapping | Bab V §V.1–§V.9 throughout |

Citation text is baked into the DOCX rather than generated by a live field, so it is typed by hand and must follow ITB style exactly: `(Penulis, tahun)`, `dkk.` for ≥ 3 authors, `dan` before the last author in the Daftar Pustaka.

---

## Citation

> Suharto, T. (2026). *Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*. Doctoral dissertation, Institut Teknologi Bandung.
