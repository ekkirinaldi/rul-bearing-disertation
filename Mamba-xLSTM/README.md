# Mamba-xLSTM-Net for Bearing RUL Prediction

Implementation of the proposed **Mamba-xLSTM** hybrid (xLSTM branch + bidirectional **Mamba-3** branch + gated/cross-attention fusion) for remaining useful life prediction on bearing run-to-failure datasets, with sparse autoencoder + SHAP interpretability and MC-dropout uncertainty quantification.

> The Mamba branch uses the **Mamba-3** block from Lahoti et al.,
> *Mamba-3: Improved Sequence Modeling using State Space Principles*
> (arXiv:2603.15569, 2026), as released in
> [`state-spaces/mamba`](https://github.com/state-spaces/mamba). It adds
> exponential-trapezoidal discretisation, a complex-valued SSM via a
> data-dependent RoPE on `B`/`C`, and learnable `B`/`C` bias on top of
> the Mamba-2 SSD layout (with the short causal conv removed). On CUDA
> the block is routed to `mamba_ssm.Mamba3`; on CPU/MPS it falls back to
> the bundled pure-PyTorch SISO reference (`_VanillaMamba3`) so the
> project trains end-to-end without any optional dependency.

Specification: [`spec/PROJECT_PLAN.md`](../spec/PROJECT_PLAN.md).

## Layout

```
Mamba-xLSTM/
├── src/mxlstm/
│   ├── compute.py             device + Mamba/xLSTM backend selection
│   ├── data/                  HI extraction, RUL labels, LightningDataModule
│   ├── models/                xLSTM blocks, Mamba blocks, fusion, baseline + proposed nets
│   ├── training/              LightningModule, losses, callbacks
│   ├── eval/                  RMSE/MAE/R²/PHM Score, prediction plots
│   ├── interp/                SHAP, sparse autoencoder, integrated gradients, latent clustering
│   ├── uq/                    MC dropout
│   ├── reporting/             figures + tables + HTML/PDF report builder
│   └── utils/                 seed, config, io, logging, RunLogger
├── configs/                   YAML configs for data, model, training, ablations
├── scripts/                   train.py, run_ablations.sh, run_interpretability.py,
│                              build_report.py, clean_results.py
├── tests/                     unit tests for HI, metrics, model forward shapes, SAE
└── results/                   runs/<id>/, reports/, tables/   (see "Results layout" below)
```

The package is self-contained: PHM2012 and XJTU-SY loaders live in
[`mxlstm/data/adapters.py`](src/mxlstm/data/adapters.py) and read
either the raw CSV folders or the parquet cache under
`data-bearing/processed/`.

**Canonical data layout (do not use legacy `data/cache/` at repo root):**

| Role | Path |
|------|------|
| Raw PHM2012 | `../data-bearing/ieee-phm-2012/` |
| Raw XJTU-SY | `../data-bearing/xtju-sy/` (`35Hz12kN`, `37.5Hz11kN`, `40Hz10kN`) |
| Processed cache | `../data-bearing/processed/{phm2012,xjtusy}/` |
| Dissertation XJTU split | `configs/data/xjtu_sy_available_full.yaml` (9 train / 3 val / 3 test) |

## Setup (Mac, no CUDA)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install this package. Extras are all Mac-friendly:
#   mamba_pure → mambapy (pure-PyTorch Mamba/Mamba-2)
#   xlstm      → NX-AI xLSTM (vanilla backend)
#   interp     → shap, captum, umap-learn, hdbscan
pip install -e ".[mamba_pure,xlstm,interp]"
```

You can also skip every optional extra and the package still trains end-to-end:
the bundled vanilla SSM and vanilla mLSTM/sLSTM implementations make the
pipeline runnable with only `torch + pytorch-lightning + numpy + scipy`.

`mamba_pure` installs [`mambapy`](https://github.com/alxndrTL/mamba.py),
which currently only ships Mamba-1 / Mamba-2; it does **not** implement
Mamba-3, so the block factory transparently downgrades it to the bundled
`_VanillaMamba3` reference (a sequential per-token SISO scan that
implements exponential-trapezoidal discretisation + RoPE-trick complex
SSM + B/C bias) — slow but correct on any device.

On a CUDA box, install `mamba-ssm` from source per the official
[`state-spaces/mamba`](https://github.com/state-spaces/mamba) README
(Mamba-3 is **not** in the published wheels yet):

```bash
MAMBA_FORCE_BUILD=TRUE pip install --no-cache-dir --force-reinstall \
  git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

The Mamba block is then auto-routed to `mamba_ssm.Mamba3` via
`mxlstm.compute._detect_mamba_backend()`. MIMO mode is opt-in via the
`mamba_is_mimo` / `mamba_mimo_rank` config keys.

The NX-AI `xlstm` package builds CUDA kernels by default; on Mac the
config uses the `vanilla` backend (pure PyTorch) for both mLSTM and
sLSTM. Slower than CUDA but functionally equivalent.

## Data layout (already on disk)

```
../data-bearing/
├── ieee-phm-2012/          PHM2012 raw + cached parquet (in ../processed/phm2012/)
└── xtju-sy/                XJTU-SY raw (35Hz12kN, 37.5Hz11kN, 40Hz10kN)
```

Loaders point at these paths automatically via the data configs.

**Download on VPS / fresh machine (S3):**

```bash
cd ..   # parent of Mamba-xLSTM/
curl -fL -o data-bearing.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/data-bearing.zip'
unzip -q data-bearing.zip && rm -f data-bearing.zip
mkdir -p data-bearing
curl -fL -o xtju-sy.zip \
  'https://dataset-bearing-rul.s3.ap-southeast-2.amazonaws.com/data-bearing/xtju-sy.zip'
unzip -q -o xtju-sy.zip -d data-bearing/ && rm -f xtju-sy.zip
```

The repaired XJTU tree (9.216 CSVs, three conditions) ships in **`xtju-sy.zip`** only; the older `data-bearing.zip` does not replace it. See `.cursor/rules/vps-ssh-key-access.mdc` §6.3.

## Quick smoke test

```bash
# Build HI for one PHM2012 bearing and forward through the proposed model
python -m mxlstm.smoke
```

## How to run

From this directory, with the virtual environment activated (see **Setup** above):

```bash
cd Mamba-xLSTM
source .venv/bin/activate
```

Training is `python scripts/train.py --data <yaml> --model <yaml> --train <yaml>` plus optional flags. Configs merge left-to-right; add `--ablation configs/ablation/<name>.yaml` to overlay ablation settings on top of the model config.

### Default training (project configs)

```bash
# Baseline: xLSTM-Transformer (encoder–decoder, paper-style stack; see model docstring)
python scripts/train.py \
    --data configs/data/phm2012.yaml \
    --model configs/model/baseline_xlstm_transformer.yaml \
    --train configs/train/default.yaml \
    --seed 42 --run-id baseline_phm2012_s42

# Proposed: Mamba-xLSTM-Net (xLSTM ‖ bidirectional Mamba → fusion → head)
python scripts/train.py \
    --data configs/data/phm2012.yaml \
    --model configs/model/mamba_xlstm_net.yaml \
    --train configs/train/default.yaml \
    --seed 42 --run-id mamba_xlstm_phm2012_s42
```

XJTU-SY: swap `--data configs/data/xjtu_sy.yaml` and pick a new `--run-id`.

### Paper-fidelity baseline vs proposed (Liu et al., Sensors 2026)

**PHM2012 fair pair:** use the **same** `--data` and `--train` for baseline and Mamba; only change `--model` and `--run-id`. `configs/data/phm2012_paper.yaml` is the Liu PHM2012 track (Table 2 split, horizontal-only, §3.2 HI + ISOMAP, piecewise labels). Use `configs/train/paper_liu2026.yaml` so checkpoints follow `train/loss` (the paper defines no validation split). Requires `PyWavelets` (see `pyproject.toml`). The same data block is available as `configs/data/phm2012_liu2026_strict.yaml` if you prefer that name.

```bash
# PHM 2012 — baseline xLSTM-Transformer
python scripts/train.py \
    --data configs/data/phm2012_paper.yaml \
    --model configs/model/baseline_xlstm_transformer.yaml \
    --train configs/train/paper_liu2026.yaml \
    --seed 42 --run-id baseline_paper_phm_s42

# PHM 2012 — proposed Mamba-xLSTM (identical data + train)
python scripts/train.py \
    --data configs/data/phm2012_paper.yaml \
    --model configs/model/mamba_xlstm_net.yaml \
    --train configs/train/paper_liu2026.yaml \
    --seed 42 --run-id mamba_xlstm_paper_phm_s42
```

For the older **window/batch-only** PHM overlay (dissertation bearing split, 36-D HI, linear labels), use `configs/data/phm2012_window_only.yaml` with `configs/train/paper.yaml`.

**Dissertation XJTU-SY (Tier-S, default HI, linear labels, `L=32`):** use `configs/data/xjtu_sy_available_full.yaml` via `--datasets xjtusy` in `run_algorithm_comparison.py` (all three operating conditions; 9 train / 3 val / 3 test bearings).

**XJTU-SY Liu fair pair (Table 1 + §3.2 ISOMAP + §3.3.1 time index):** use `configs/data/xjtu_sy_paper.yaml` or `configs/data/xjtu_sy_liu2026_strict.yaml` with `configs/train/paper_liu2026.yaml` (12 train / 3 test; separate paper track). Legacy 2-condition split: `configs/data/xjtu_sy.yaml`.

```bash
# XJTU-SY — baseline xLSTM-Transformer (Liu strict)
python scripts/train.py \
    --data configs/data/xjtu_sy_paper.yaml \
    --model configs/model/baseline_xlstm_transformer.yaml \
    --train configs/train/paper_liu2026.yaml \
    --seed 42 --run-id baseline_paper_xjtu_s42

# XJTU-SY — proposed Mamba-xLSTM (identical data + train)
python scripts/train.py \
    --data configs/data/xjtu_sy_paper.yaml \
    --model configs/model/mamba_xlstm_net.yaml \
    --train configs/train/paper_liu2026.yaml \
    --seed 42 --run-id mamba_xlstm_paper_xjtu_s42
```

Metrics: Lightning logs both the project PHM score and `phm_score_paper` (paper Eq. 26–28). Artefacts land under `results/runs/<timestamp>_<run_id>/` (see **Results layout** below).

### Useful flags

| Flag | Purpose |
|------|---------|
| `--ablation PATH` | Merge an ablation YAML on top of data+model |
| `--max-epochs N` | Override `train.max_epochs` without editing YAML |
| `--fast-dev-run` | One train + one val batch; quick wiring check |
| `--no-figures` | Skip post-training plots (faster CI / smoke) |
| `--seed N` / `--run-id ID` | Reproducibility and run folder naming |

## Ablations

```bash
bash scripts/run_ablations.sh phm2012 3   # run all 7 ablations × 3 seeds
```

## Interpretability

```bash
python scripts/run_interpretability.py \
    --checkpoint results/checkpoints/mamba_xlstm_best.ckpt \
    --data configs/data/phm2012.yaml
```

Produces SHAP attributions, SAE latents (with top-activating windows),
UMAP+HDBSCAN latent clusters, and Integrated Gradients heatmaps under
`results/figures/interp/`.

## Results layout

The `results/` tree groups everything per run, with two top-level
folders for cross-run artefacts:

```
results/
├── runs/<timestamp>_<run_id>/      # everything PER RUN (lazily created)
│   ├── config.yaml                 # merged data + model + train config
│   ├── summary.json                # parameters, fit time, test metrics
│   ├── hi_scaler.json
│   ├── test_predictions.npz        # raw per-bearing test predictions
│   ├── checkpoints/                # always; Lightning needs it
│   ├── logs/
│   │   ├── run.log                 # plain-text DEBUG+ log
│   │   ├── events.jsonl            # structured phase/step/metric events
│   │   └── summary.json            # pipeline timing table
│   ├── csv_logs/                   # Lightning CSVLogger metrics.csv
│   ├── figures/                    # per-run plots (see list below)
│   └── interp/                     # only if scripts/run_interpretability.py ran
├── reports/                        # CROSS-RUN HTML + PDF reports
└── tables/                         # CROSS-RUN aggregated tables (md/tex/json)
```

Per-run figures written by `scripts/train.py`:

- `dataset_overview.png` — bearings used (train/val/test) and their lengths
- `hi_traces.png` / `hi_heatmap.png` — HI features for a sample bearing
- `rul_labels.png` — RUL target curves
- `training_curves.png` — loss + metrics over epochs (from the CSV logger)
- `pred_<bearing>.png` — per-bearing test predictions vs ground truth
- `residuals.png` — residual scatter
- `step_timings.png` — pipeline step elapsed seconds

Subfolders (`figures/`, `csv_logs/`, `interp/`) are created **only when a
component actually writes into them**, so failed or `--no-figures` runs
won't leave behind empty stubs.

Each pipeline step also prints progress to stderr
(`▶ Phase: Data preparation`, `· Step: Load + extract HI + fit scaler`)
and the same events are persisted in `events.jsonl`.

### Tidy up `results/`

`scripts/clean_results.py` enforces the layout above: it relocates any
stray `report*.{html,pdf}` from `results/` into `results/reports/`,
prunes empty subfolders inside each run, and (with
`--drop-empty-runs`) removes runs that never produced a `summary.json`.

```bash
python scripts/clean_results.py                      # dry run
python scripts/clean_results.py --apply              # apply
python scripts/clean_results.py --apply --drop-empty-runs
```

## Build a dissertation report (HTML + PDF)

By default reports go to `results/reports/<basename>.html|.pdf`. The
basename is the run id when only one run is supplied, otherwise
`report` (override with `--name`).

```bash
# Report from a single run -> results/reports/<run_id>.html|.pdf
python scripts/build_report.py \
    --runs results/runs/20260421_*_mamba_xlstm_phm2012_s42

# Cross-run / ablation report
python scripts/build_report.py \
    --runs results/runs/* \
    --name ablation_phm2012 \
    --title "Mamba-xLSTM ablations on PHM2012"

# HTML only
python scripts/build_report.py --runs results/runs/<id> --no-pdf

# Custom paths
python scripts/build_report.py --runs results/runs/<id> \
    --out-html /tmp/foo.html --out-pdf /tmp/foo.pdf
```

**Baseline vs Mamba (paper-fidelity) — one PDF per dataset.** After training
both models with the same `phm2012_paper` / `xjtu_sy_paper` data and
`paper.yaml` train config, point `--runs` at the two run directories
(each must contain `summary.json`), then set `--name` so the outputs land in
`results/reports/<name>.{html,pdf}`:

```bash
# PHM 2012
python scripts/build_report.py \
    --runs results/runs/20260421_044040_baseline_paper_phm_s42 \
            results/runs/20260421_132350_mamba_xlstm_paper_phm_s42 \
    --name baseline_vs_mamba_paper_phm \
    --title "PHM 2012: xLSTM-Transformer vs Mamba-xLSTM (paper setup)"

# XJTU-SY (replace with your run timestamps)
python scripts/build_report.py \
    --runs results/runs/<ts>_baseline_paper_xjtu_s42 \
            results/runs/<ts>_mamba_xlstm_paper_xjtu_s42 \
    --name baseline_vs_mamba_paper_xjtu \
    --title "XJTU-SY: xLSTM-Transformer vs Mamba-xLSTM (paper setup)"
```

The HTML is fully self-contained (figures embedded as data URIs). For
PDF export the report tries WeasyPrint first and falls back to
`xhtml2pdf` if WeasyPrint is unavailable. To install WeasyPrint on Mac:

```bash
brew install pango          # one-time system dep
pip install -e ".[report]"
# OR pure-Python fallback (CSS support is more limited):
pip install -e ".[report_pure]"
```

## SSH key access (RunPod / generic VPS)

Key-based login avoids typing the SSH password on every connection. RunPod often uses a **non-default port**; use the host, port, and user from the provider dashboard.

### 1. On the laptop (macOS)

Generate a dedicated key (recommended):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -C "label-for-this-vps"
```

Show the **public** key (single line) and copy it:

```bash
cat ~/.ssh/id_ed25519_runpod.pub
```

### 2. On the VPS (as the SSH user, often `root`)

Ensure `~/.ssh` exists and permissions are correct:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
```

Append the public key with a quoted heredoc so the shell does not expand `$` inside the key line:

```bash
cat >> ~/.ssh/authorized_keys << 'EOF'
ssh-ed25519 AAAA…your-public-key-one-line… your-comment
EOF
```

Paste the full public key line between `<< 'EOF'` and the closing `EOF`. The closing `EOF` must be alone on its own line (no leading spaces). Then:

```bash
chmod 600 ~/.ssh/authorized_keys
```

Minimal container images may lack `nano` / `vim`; the heredoc above is preferred. Alternatively, from the laptop (if password login still works): `ssh-copy-id -i ~/.ssh/id_ed25519_runpod.pub -p <PORT> <USER>@<HOST>`.

### 3. Connect from the laptop

```bash
ssh -i ~/.ssh/id_ed25519_runpod -p <PORT> <USER>@<HOST_OR_IP>
```

Optional `~/.ssh/config` block:

```text
Host my-vps-alias
  HostName <IP_OR_DNS>
  User root
  Port <PORT>
  IdentityFile ~/.ssh/id_ed25519_runpod
  IdentitiesOnly yes
```

Then: `ssh my-vps-alias`.

### 4. If login still fails

- On the server: `~/.ssh` mode `700`, `authorized_keys` mode `600`.
- One key = **one line** in `authorized_keys` (no accidental line breaks).
- Match the **private** key on the laptop (`-i` / `IdentityFile`) to the **public** key on the server.
- Debug: `ssh -v …` and confirm “Offering public key” / success.

### 5. Ephemeral pods (RunPod and similar)

The filesystem under the default user may reset when the pod is **recreated**. Re-add `authorized_keys` or inject keys via the provider UI / startup script if keys must survive redeploys.

Dataset download on a fresh VPS (HTTPS zip beside `Mamba-xLSTM/`) is documented in the repo Cursor rule `vps-ssh-key-access` and in `scripts/rsync_training_bundle_to_vps.sh` comments.
