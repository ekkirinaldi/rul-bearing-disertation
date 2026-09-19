# Perawatan Prediktif untuk Sistem Produksi

*Perawatan Prediktif untuk Sistem Produksi dengan Pendekatan Analisis Big Data dan Kecerdasan Buatan Menggunakan Data Kondisi Mesin dan Informasi Kualitas yang Real Time*

Doctoral dissertation of Toto Suharto, Program Doktor Teknik dan Manajemen Industri, Fakultas Teknologi Industri, Institut Teknologi Bandung.

The work is on interpretable predictive maintenance of rolling-element bearings, and it runs in two tracks:

- Diagnostics (Bab IV). Bearing faults are classified on CWRU with kernel models (SVM, LR), tree models (DT, RF, XGBoost) and a WDCNN, each explained with its matching SHAP explainer. It ends with Fault Signature Maps, which attribute the WDCNN decision to the raw 2 048-point signal.
- Prognostics (Bab V). Remaining useful life is estimated on PHM2012, XJTU-SY and IMS with three backbones: Mamba-xLSTM-Net, N-BEATS-xLSTM-RUL and SparseGate-TCN-RUL. A Top-*k* sparse autoencoder is then trained on their hidden states and its features are matched to the bearing defect frequencies (BPFO, BPFI, BSF, FTF). The match is tested with bootstrap intervals, permutation tests and two negative controls.

Bab VI brings the two tracks together as a three-tier maintenance architecture (edge device, edge server, cloud). Each model is placed on the tier that fits its size, latency and kind of explanation.

## Contents

| Directory | |
|---|---|
| [`manuscript/`](manuscript/) | The manuscript (`V16-disertasi.docx`), the ITB template and figures, and the tools that insert material into it and lint it |
| [`presentation/`](presentation/) | The sidang deck, generated from `content/sidang-terbuka.yaml` |
| [`research/`](research/) | Training and analysis code, the notebooks, the published papers, and the results behind every table in Bab IV and Bab V |
| [`docs/`](docs/) | Chapter outline, writing guide, compute setup, and the proposal documents |
| [`archive/`](archive/) | Superseded material kept for traceability |
| `writings/journal-q2/` | A separate journal paper, kept outside git |

## Working on the Manuscript

The manuscript is edited in Word. Rules for prose, numbers, citations, figures and headings are in [docs/writing-guide.md](docs/writing-guide.md). The Word styles and field recipes are in [manuscript/RULES.md](manuscript/RULES.md).

```bash
cd manuscript
make lint          # ITB format and prose checks on V16; fix every [FATAL]
make insert-dry    # preview an insertion spec
make insert        # apply it, writing the next version
```

After editing in Word, select all and press F9 once so the lists of contents, figures and tables are refreshed.

## Building the Deck

```bash
cd presentation
python3 build.py lint --spec content/sidang-terbuka.yaml
python3 build.py build --spec content/sidang-terbuka.yaml --strict   # -> out/*.pptx
python3 build.py preview --spec content/sidang-terbuka.yaml --png
python3 -m pytest tests                                              # includes the check that every number on a slide is in the manuscript
```

## Code and Data

The prognostic pipeline is in `research/Mamba-xLSTM/` and needs a CUDA GPU for full training. The diagnostic notebooks in `research/notebooks/` run on a CPU. The datasets are not in the repository. They are downloaded into `research/data-bearing/` as described in [docs/compute-setup.md](docs/compute-setup.md), except CWRU, which comes from the [Case Western Reserve University Bearing Data Center](https://engineering.case.edu/bearingdatacenter). [research/README.md](research/README.md) maps each chapter to the code and the results it draws on.

The dashboard in `research/inference-engine/` streams recorded PHM2012, XJTU-SY and PT SKF data through the trained Mamba-xLSTM-Net. It appears in Lampiran D.

## Contributions

| | Contribution | Chapter | Evidence |
|---|---|---|---|
| N1 | Fault Signature Maps: signal-level explanation of WDCNN at 2 048-point resolution, in signed, absolute and variance forms | Bab IV | split-half stability 0,940; discriminability 0,216 |
| N2 | BatchNorm lowers the discriminability of the maps; a design choice between accuracy and interpretability | Bab IV | discriminability +133% without BatchNorm, at −3,60 pp accuracy |
| N3 | Top-*k* sparse autoencoder applied to bearing prognostics, with features matched to defect frequencies | Bab V | BPFI hit rate 2,3% (PHM2012), BPFO 2,2% (XJTU-SY), *p* < 0,001 |
| N4 | The same frequency structure appears across three backbones, with bootstrap, permutation and two negative controls | Bab V | hit rate falls to about zero on both controls |
| N5 | Three-tier maintenance architecture combining input attribution and latent concepts | Bab VI | conceptual; not a field deployment |

## Published Papers

The four papers the dissertation builds on are in [`research/papers/`](research/papers/): SVM and LR with SHAP on CWRU; tree models with SHAP on CWRU; WDCNN with Fault Signature Maps; and Mamba-xLSTM with the sparse autoencoder and BPFx mapping.

## Citation

> Suharto, T. (2026): *Perawatan prediktif untuk sistem produksi dengan pendekatan analisis big data dan kecerdasan buatan menggunakan data kondisi mesin dan informasi kualitas yang real time*, Disertasi Doktor, Institut Teknologi Bandung.
