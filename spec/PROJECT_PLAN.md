# Dissertation Project Plan
## Bearing RUL Prediction with Mamba-xLSTM Hybrid and Sparse Autoencoder Interpretability

---

## 0. Executive Summary

**Baseline paper being replicated:** Liu et al., *"RUL Prediction Based on xLSTM–Transformer Neural Network for Rolling Element Bearings Under Different Working Conditions"* (PMC12987095). It proposes an encoder–decoder with xLSTM + Multi-Head Attention, validated on XJTU-SY and PHM2012, outperforming LSTM and LSTM-Transformer baselines on RMSE, R², and PHM Score.

**Your dissertation contribution (proposed):** A hybrid **Mamba-xLSTM** architecture with **Sparse Autoencoder (SAE) + SHAP** interpretability, specifically tuned for bearing degradation signals. Rationale in §2.

**Novelty check performed:**
- Mamba-SDP (Springer 2025) → uses Mamba + scaled-dot-product attention + FFT (*not* xLSTM)
- Enhanced Mamba (Sci Rep 2025) → Mamba + multi-head attention on aero-engines/batteries (*not* bearings, *not* xLSTM)
- xLSTM-Transformer (your reference paper) → xLSTM + Transformer (*no Mamba*)
- **Mamba + xLSTM for bearing RUL has not been published.** This is a genuine gap.

**Target deliverables:**
1. Faithful replication of xLSTM-Transformer (baseline) on both datasets
2. Proposed model: **Mamba-xLSTM-Net** with explainability
3. Ablation studies isolating each component's contribution
4. Interpretability analysis (SAE latents + SHAP attributions + attention maps)
5. Written dissertation chapters with figures and tables

**Estimated timeline:** 14–20 weeks of full-time effort (see §12).

---

## 1. Understanding the Baseline Paper

### 1.1 Problem Setup
- **Task:** Given a sequence of vibration signal features from a bearing operating under a load, predict the remaining useful life (RUL) at each timestep.
- **Input:** Raw vibration signals sampled at 25.6 kHz → windowed into frames → converted into a Health Indicator (HI) time series.
- **Output:** Normalized RUL ∈ [0, 1], where 1 = brand new, 0 = failure. Sometimes predicted as remaining time-to-failure in minutes.

### 1.2 Data Pipeline (as in paper)
```
Raw Vibration (horizontal + vertical accel.)
    │
    ▼
Segment into samples (e.g., 1 sample = 0.1 s window = 2560 points @ 25.6 kHz for PHM2012)
    │
    ▼
Extract time-domain statistics per sample (RMS, kurtosis, skewness, peak, crest factor, etc.)
    │
    ▼
Extract frequency-domain features (spectral entropy, band energies via FFT)
    │
    ▼
Fuse multi-domain features → Health Indicator (HI) sequence, one value per sample timestep
    │
    ▼
Slide a window of length L over HI → (X_t, y_t) pairs where y_t = normalized RUL
    │
    ▼
Normalize HI with MinMax, smooth with exponential smoothing (α ≈ 0.1–0.3)
```

### 1.3 Baseline Architecture (what we will replicate first)
```
Input HI window [B, L, F]
    │
    ▼
Encoder: Multi-Head Self-Attention (captures global degradation pattern)
    │
    ▼
xLSTM block: mLSTM/sLSTM stack with exponential gating (captures local dynamics)
    │
    ▼
Decoder: Multi-Head Attention over encoder output, cross-attending to xLSTM output
    │
    ▼
Fully Connected regression head → scalar RUL ∈ [0, 1]
```

Key facts about xLSTM (Beck et al., NeurIPS 2024) — you need this for your literature review:
- **sLSTM:** scalar memory with exponential gating + normalization, addresses vanishing gradients near saturation
- **mLSTM:** matrix memory (outer product of keys/values) with parallel processable update, higher capacity
- The exponential input gate `i_t = exp(W_i x_t + b_i)` prevents the late-stage saturation issue that standard LSTM suffers from in slow degradation regions — critical for bearing RUL late-life prediction.

### 1.4 Datasets Used

**XJTU-SY Rolling Element Bearing Accelerated Life Test Dataset**
- 15 bearings across 3 operating conditions (load/speed combinations)
- Condition 1: 2100 rpm, 12 kN
- Condition 2: 2250 rpm, 11 kN
- Condition 3: 2400 rpm, 10 kN
- 25.6 kHz sampling, 1.28 s samples every minute
- Download: https://biaowang.tech/xjtu-sy-bearing-datasets/

**PHM 2012 / PRONOSTIA (FEMTO-ST) Dataset**
- 17 bearings across 3 operating conditions
- 25.6 kHz sampling, 0.1 s samples every 10 s
- Accelerometer readings in horizontal and vertical directions
- IEEE 2012 PHM Challenge dataset. Commonly mirrored at: https://data.nasa.gov or the original FEMTO-ST source.

### 1.5 Evaluation Metrics (must replicate all)

**RMSE (Root Mean Square Error)** — the primary regression metric.

**MAE (Mean Absolute Error)** — robust secondary metric.

**R² (Coefficient of Determination)** — goodness-of-fit.

**PHM Score (asymmetric exponential)** — standard RUL metric that penalizes late predictions more heavily than early ones, because predicting a bearing lasts longer than it does is worse than the reverse.

```
Er_i = (RUL_actual_i - RUL_pred_i) / RUL_actual_i  (percentage error)

A_i = exp(-ln(0.5) * (Er_i / 5))    if Er_i <= 0  (early prediction, lenient)
A_i = exp(+ln(0.5) * (Er_i / 20))   if Er_i > 0   (late prediction, harsh)

Score = (1 / N) * sum(A_i)     [higher is better, max = 1]
```

Implementation will be in `src/evaluation/metrics.py`.

---

## 2. Proposed Novelty: Why Mamba-xLSTM + SAE

### 2.1 Motivation
The baseline xLSTM-Transformer has two weak points worth improving:

1. **Quadratic attention complexity.** Bearing HI sequences can be very long (a PHM2012 bearing run generates 2,000+ samples). Self-attention is O(L²), limiting practical context length. → **Mamba solves this with linear-time selective state-space scanning: O(L) while still modeling long-range dependencies.**

2. **No mechanistic interpretability.** The baseline only reports attention heatmaps, which are known to be unreliable as explanations. → **SAE latents + SHAP attributions** give a principled, post-hoc, industry-relevant story of *which degradation signatures the model relies on*, which is gold for a dissertation viva.

### 2.2 Proposed Architecture: Mamba-xLSTM-Net

```
Input HI window [B, L, F]
    │
    ▼
┌──────────── Feature Projection (Linear → d_model) ────────────┐
│                                                               │
├─── Branch A: xLSTM stack (captures sharp local degradation) ──┤
│       • 2× mLSTM blocks (matrix memory for high-capacity      │
│         feature mixing)                                       │
│       • 1× sLSTM block (scalar with exponential gating for    │
│         late-stage saturation handling)                       │
│                                                               │
├─── Branch B: Bidirectional Mamba stack (global trend) ────────┤
│       • 2× Mamba-2 blocks (selective SSM, bidirectional       │
│         scan, linear complexity)                              │
│                                                               │
├─── Fusion: Cross-Attention (A as Q, B as K/V) ────────────────┤
│       OR Gated sum: f_t = σ(W·[h_A; h_B]) ⊙ h_A + (1-σ) ⊙ h_B │
│                                                               │
├─── Regression head: MLP + Dropout → scalar RUL ───────────────┤
└───────────────────────────────────────────────────────────────┘
```

Why this works:
- **Branch A (xLSTM)** keeps the paper's key insight — exponential gating prevents late-life saturation.
- **Branch B (Bi-Mamba)** replaces the Transformer encoder with a linear-time sequence model that has been shown to outperform Transformers on long time series (MambaTS, ICLR 2025 — see §14).
- **Fusion via gating** lets the network learn to rely on the right branch at each time step — early in life, global trend (Mamba) dominates; late in life, local dynamics (xLSTM) dominate.

### 2.3 Interpretability Stack

Three complementary tools, each answering a different question:

| Question | Tool | What you'll show |
|---|---|---|
| "Which input features drive a prediction at time t?" | **SHAP** (KernelSHAP or DeepSHAP) | Waterfall plots per prediction + global feature importance |
| "What degradation *concepts* has the model internalized?" | **Sparse Autoencoder** on hidden states | SAE latents clustered by degradation phase (healthy / wear / pre-failure) |
| "Which timesteps matter for a prediction?" | **Integrated Gradients** over the time axis | Temporal attribution heatmap |

**SAE specifics** (§8 has full detail):
- Train a top-k SAE on the fused hidden states h_t ∈ ℝ^d_model
- Expansion factor 8× (latent dim = 8·d_model), L1 sparsity penalty
- Inspect top activating examples for each latent — you'll find latents for "rising RMS," "impulsive spikes," "saturation phase," etc.
- This is *mechanistic interpretability*, popular in LLM research (Anthropic, OpenAI), rarely applied to PHM — big novelty point.

### 2.4 Alternative / Fallback Novelties (if Mamba integration is unstable)

If `mamba-ssm` fails to install or train stably on your GPU, here are backup novelty angles, ranked by feasibility:

| Rank | Alternative | Effort | Novelty |
|---|---|---|---|
| 1 | **Bidirectional xLSTM (no Mamba)** + SAE interpretability | Low | Medium |
| 2 | **Wavelet-Mamba** — replace FFT in feature extraction with CWT, feed into Mamba only | Medium | Medium-High |
| 3 | **xLSTM-Transformer + uncertainty quantification (MC Dropout + Conformal Prediction)** | Low | Medium |
| 4 | **Physics-informed loss** (monotonicity + boundary constraints on RUL) added to xLSTM-Transformer | Low | Medium |
| 5 | **Cross-condition domain adaptation** (train on condition 1, test on condition 3) with adversarial alignment | High | High |

Pick #1 or #3 if time is tight. Pick #5 if you want max impact and have time.

---

## 3. Project Structure

Set this up first thing — Cursor works better with clean structure.

```
rul_bearing_dissertation/
├── README.md
├── requirements.txt
├── environment.yml              # conda alternative
├── .gitignore
├── configs/
│   ├── baseline_xlstm_transformer.yaml
│   ├── proposed_mamba_xlstm.yaml
│   ├── ablation_mamba_only.yaml
│   ├── ablation_xlstm_only.yaml
│   └── data_phm2012.yaml / data_xjtu.yaml
├── data/
│   ├── raw/
│   │   ├── phm2012/             # downloaded here, gitignored
│   │   └── xjtu_sy/
│   └── processed/               # HI sequences, cached tensors
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── phm2012_loader.py
│   │   ├── xjtu_loader.py
│   │   ├── hi_extraction.py     # HI construction
│   │   ├── features.py          # time/freq domain features
│   │   └── datamodule.py        # PyTorch Lightning DataModule
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xlstm_blocks.py      # sLSTM, mLSTM
│   │   ├── mamba_blocks.py      # Mamba-2 wrapper, bidirectional
│   │   ├── baseline_xlstm_transformer.py
│   │   ├── mamba_xlstm_net.py   # proposed
│   │   ├── fusion.py            # cross-attn, gated fusion
│   │   └── heads.py             # regression head
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # LightningModule
│   │   ├── losses.py            # MSE, monotonicity loss
│   │   └── callbacks.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # RMSE, R², PHM Score
│   │   └── plots.py             # prediction curves, residuals
│   ├── interpretability/
│   │   ├── __init__.py
│   │   ├── shap_analysis.py
│   │   ├── sparse_autoencoder.py
│   │   ├── integrated_gradients.py
│   │   └── latent_clustering.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
├── scripts/
│   ├── download_data.sh
│   ├── preprocess.py
│   ├── train.py                 # unified entry
│   ├── evaluate.py
│   ├── run_interpretability.py
│   └── run_ablations.sh
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_hi_construction.ipynb
│   ├── 03_baseline_results.ipynb
│   ├── 04_proposed_results.ipynb
│   ├── 05_shap_analysis.ipynb
│   ├── 06_sae_latents.ipynb
│   └── 07_figures_for_thesis.ipynb
├── results/
│   ├── figures/
│   ├── tables/
│   └── checkpoints/
├── tests/
│   ├── test_hi_extraction.py
│   ├── test_metrics.py
│   └── test_models_forward.py
└── thesis/
    ├── chapters/
    ├── references.bib
    └── figures/
```

---

## 4. Environment and Dependencies

### 4.1 Hardware recommendations
- **Minimum:** NVIDIA GPU with ≥ 8 GB VRAM (RTX 3060 / 3070 class). Mamba needs CUDA.
- **Recommended:** RTX 4070 Ti / A40 / A100. 16 GB+ VRAM for comfortable batch sizes.
- **CPU-only fallback:** works but very slow for training; use small models or cloud GPU (Colab Pro, Kaggle, Paperspace).

### 4.2 `requirements.txt`
```
# Core
torch>=2.1.0
torchmetrics>=1.2.0
pytorch-lightning>=2.1.0
numpy>=1.24
scipy>=1.11
pandas>=2.0
scikit-learn>=1.3

# Sequence models
mamba-ssm>=2.0.0         # needs CUDA + causal-conv1d
causal-conv1d>=1.2.0
xlstm>=1.0.0             # NX-AI's official xLSTM implementation

# Signal processing
pywavelets>=1.5          # optional, for wavelet features
librosa>=0.10            # spectral features helpers

# Experiment tracking
hydra-core>=1.3          # config management
wandb>=0.16              # or tensorboard
rich>=13                 # pretty logging

# Interpretability
shap>=0.44
captum>=0.7              # integrated gradients, DeepLIFT
umap-learn>=0.5          # for SAE latent visualization
hdbscan>=0.8

# Viz
matplotlib>=3.8
seaborn>=0.13
plotly>=5.18

# Utilities
tqdm>=4.66
pyyaml>=6.0
```

### 4.3 Setup commands
```bash
# Conda (recommended)
conda create -n rul python=3.11 -y
conda activate rul
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install mamba-ssm (requires CUDA toolkit matching your driver)
pip install causal-conv1d==1.2.2.post1
pip install mamba-ssm==2.2.2

# Install xLSTM
pip install xlstm

# Rest
pip install -r requirements.txt
```

**Common gotchas:**
- `mamba-ssm` wheel must match your CUDA version. If `pip install mamba-ssm` fails, build from source: `pip install mamba-ssm --no-build-isolation`.
- On Windows, Mamba is painful. Use WSL2 with Ubuntu, or a Linux box.
- If you hit `causal-conv1d` build errors, set `TORCH_CUDA_ARCH_LIST=8.0` (or your GPU's compute capability) before the pip command.

---

## 5. Step-by-Step Execution Plan (for Cursor)

Paste these prompts into Cursor one at a time. Each should yield a self-contained file or module you verify before moving on.

### Phase 0 — Scaffolding (½ day)
**Cursor prompt:**
> Create the directory structure from §3 of PROJECT_PLAN.md. Create empty `__init__.py` files. Create `requirements.txt` with the list from §4.2. Create a `.gitignore` for Python + `data/raw/` + `results/checkpoints/`. Initialize git.

### Phase 1 — Data loading (2 days)

**Cursor prompt 1:**
> Implement `src/data/phm2012_loader.py`. It should:
> - Accept a path to the PHM2012 dataset directory.
> - Load horizontal and vertical accelerometer CSVs for each bearing.
> - Return a dict mapping `bearing_id` → numpy array of shape `(num_samples, 2560, 2)` — one sample per 10 s measurement.
> - Also return per-bearing run-to-failure timing (number of samples × 10 s).
> - Include a `__main__` smoke test that loads one bearing and prints its shape.

**Cursor prompt 2:**
> Implement `src/data/xjtu_loader.py` analogously. XJTU-SY files are in `.csv` format with two columns (horizontal/vertical); samples of 32768 points (1.28 s at 25.6 kHz) every 1 minute.

**Cursor prompt 3:**
> Implement `src/data/features.py` with functions:
> - `time_domain_features(x: np.ndarray) -> np.ndarray` returning [RMS, peak, kurtosis, skewness, crest factor, shape factor, impulse factor, margin factor, variance] per sample
> - `frequency_domain_features(x, fs) -> np.ndarray` returning [spectral centroid, spectral entropy, energy in 5 frequency bands, mean frequency, RMS frequency]
> - `combined_features(x, fs)` concatenates both
> Each function takes `x` of shape `(batch, window_len)` and returns `(batch, n_features)`.

**Cursor prompt 4:**
> Implement `src/data/hi_extraction.py`:
> - `extract_hi_sequence(bearing_signals, fs, feature_fn)` applies feature extraction to each sample → yields a sequence `(T, n_features)`.
> - `normalize_hi(hi_seq)` applies MinMax [0,1] scaling per-feature.
> - `smooth_hi(hi_seq, alpha=0.1)` exponential smoothing along time axis.
> - `make_rul_labels(T, normalize=True)` builds linear RUL targets (1.0 → 0.0 linearly from healthy to failure). Add an option for piecewise-linear RUL (cap at 1.0 until degradation point, then linear decay — this is closer to physical reality).

**Cursor prompt 5:**
> Implement `src/data/datamodule.py` as a `pl.LightningDataModule`. It should:
> - Load the chosen dataset (phm2012 or xjtu)
> - Split bearings into train/val/test by bearing_id (NOT by random time — critical for avoiding leakage)
> - Slide windows of length L (configurable) with stride s over each bearing's HI sequence → (x, y) tensors
> - Return `DataLoader`s

Verify phase 1 by running a smoke notebook `notebooks/01_data_exploration.ipynb` that plots the HI of one bearing and its RUL label.

### Phase 2 — Baseline Model: xLSTM-Transformer (3–4 days)

**Cursor prompt 6:**
> Implement `src/models/xlstm_blocks.py`. Use NX-AI's `xlstm` package: import `xLSTMBlockStack` and `xLSTMBlockStackConfig`. Wrap it in a `nn.Module` with a forward signature `(B, L, d_model) → (B, L, d_model)`. Make the stack config (num_blocks, slstm_at, mlstm ratio, num_heads) injectable via a dict.

**Cursor prompt 7:**
> Implement `src/models/baseline_xlstm_transformer.py`:
> - Input projection (Linear: n_features → d_model)
> - Positional encoding (sinusoidal)
> - Multi-head self-attention encoder (`nn.TransformerEncoder` with 2 layers, 4 heads)
> - xLSTM block stack (3 layers, mostly mLSTM with one sLSTM at position 2)
> - Multi-head attention decoder that cross-attends encoder output with xLSTM output as query
> - Regression head: LayerNorm → Linear(d_model → 64) → GELU → Dropout(0.1) → Linear(64 → 1) → Sigmoid
> - Parameter count target: 1–3 M parameters (keep close to paper)
> - Include a `forward(x) -> y_pred` that returns shape `(B,)` — take the last timestep's prediction.

**Cursor prompt 8:**
> Implement `src/training/trainer.py` as a `pl.LightningModule`:
> - Accepts a model instance
> - Loss = MSE (for Sigmoid output) + optional monotonicity penalty (see §7.2)
> - Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
> - Scheduler: CosineAnnealingLR over total epochs
> - Logs train/val RMSE, MAE, R², PHM Score
> - Early stopping on val RMSE (patience=20)

**Cursor prompt 9:**
> Implement `src/evaluation/metrics.py`:
> - `rmse(y_true, y_pred)`, `mae`, `r2`
> - `phm_score(y_true, y_pred, horizon)` implementing the asymmetric exponential score exactly as described in §1.5. Convert normalized RUL back to physical time before computing Er_i (the error must be in the same units as the original paper).

**Cursor prompt 10:**
> Implement `scripts/train.py` using Hydra for config management. Should:
> - Load config from `configs/baseline_xlstm_transformer.yaml`
> - Instantiate datamodule, model, trainer
> - Run `trainer.fit()` then `trainer.test()`
> - Save best checkpoint, metrics to `results/`, prediction curves to `results/figures/`

**Milestone 1:** Run `python scripts/train.py --config-name baseline_xlstm_transformer` on PHM2012. Target RMSE ≤ 0.12 on val (paper reports similar range depending on bearing). Record baseline results in a notebook.

### Phase 3 — Proposed Model: Mamba-xLSTM-Net (4–5 days)

**Cursor prompt 11:**
> Implement `src/models/mamba_blocks.py`. Create a `BidirectionalMamba` class that:
> - Wraps two `mamba_ssm.Mamba2` blocks — one processes forward, one processes the reversed sequence
> - Combines them by concatenation + Linear projection back to d_model
> - Forward signature `(B, L, d_model) → (B, L, d_model)`
> - Configurable d_state (default 64), d_conv (default 4), expand (default 2)

**Cursor prompt 12:**
> Implement `src/models/fusion.py` with two fusion modules:
> - `CrossAttentionFusion(d_model, n_heads)`: Q from branch A, K/V from branch B → Linear
> - `GatedFusion(d_model)`: g = σ(Linear([h_A; h_B])); out = g·h_A + (1-g)·h_B
> Both `(B, L, d) × (B, L, d) → (B, L, d)`.

**Cursor prompt 13:**
> Implement `src/models/mamba_xlstm_net.py`. Architecture per §2.2:
> - Input projection
> - Branch A: xLSTM stack (3 blocks: mLSTM, sLSTM, mLSTM)
> - Branch B: Bidirectional Mamba stack (2 blocks)
> - Fusion module (default GatedFusion, swappable)
> - Regression head as in baseline
> - Forward should also optionally return branch activations and fusion gate values for interpretability.

**Milestone 2:** Train the proposed model on PHM2012. Compare vs baseline. Target: ≥ 10% improvement on PHM Score and RMSE on at least 2/3 conditions.

### Phase 4 — Ablations (2–3 days)

Run these configurations and tabulate results:

| Ablation | What's removed | Hypothesis |
|---|---|---|
| A1 | No Mamba branch (xLSTM only) | Shows Mamba's contribution |
| A2 | No xLSTM branch (Mamba only) | Shows xLSTM's contribution |
| A3 | Unidirectional Mamba | Bi-directionality matters |
| A4 | Concat fusion instead of gated | Gating matters |
| A5 | Standard LSTM instead of xLSTM | Exponential gating matters |
| A6 | No exponential smoothing / HI normalization | Preprocessing matters |
| A7 | Shorter context (L=32) vs (L=128) | Long context helps |

Create `scripts/run_ablations.sh` that launches each config. Put all results into one summary table for the thesis.

### Phase 5 — Interpretability (3–4 days)

**Cursor prompt 14:** SHAP analysis.
> Implement `src/interpretability/shap_analysis.py`:
> - Use `shap.GradientExplainer` or `shap.DeepExplainer` on the trained model
> - Background set: 100 random samples from train set
> - Compute SHAP values per feature per timestep for the test bearings
> - Plot: (a) global feature importance bar chart, (b) per-bearing waterfall at the critical degradation point, (c) SHAP time-heatmap

**Cursor prompt 15:** Sparse Autoencoder on hidden states.
> Implement `src/interpretability/sparse_autoencoder.py` implementing a top-k Sparse Autoencoder:
> - Encoder: Linear(d_model → d_model·8)
> - Activation: keep top-k activations per sample, zero the rest (k ≈ d_model·0.05)
> - Decoder: Linear(d_model·8 → d_model)
> - Loss: reconstruction MSE + L1 penalty on latents
> - Extract hidden states h_t from the trained Mamba-xLSTM-Net's fusion layer on a large unlabeled set → train SAE on these
> - After training, for each latent, find the top-20 activating (bearing, timestep) pairs. Plot their HI signals side-by-side → you'll see interpretable "motifs."

**Cursor prompt 16:** Latent clustering and phase attribution.
> Implement `src/interpretability/latent_clustering.py`:
> - Reduce SAE latent activations with UMAP (2D)
> - Cluster with HDBSCAN
> - Color by degradation phase (healthy / wear / pre-failure) — use the HI threshold from the paper to define phases
> - Expected finding: latents cluster distinctly by phase → the model has learned phase-conditional features.

**Cursor prompt 17:** Integrated Gradients.
> Implement `src/interpretability/integrated_gradients.py` using Captum. Compute IG attributions over the time axis — which timesteps in the input window most influenced the RUL prediction? Plot as heatmap aligned with the HI curve.

### Phase 6 — Uncertainty Quantification (optional, 1–2 days)

Add MC Dropout to the final head. At inference, do 100 forward passes → predictive mean and std. Compute 95 % CI. Check calibration with reliability diagram. (This is low-cost and gives you another dissertation section: "Can we trust the RUL?")

### Phase 7 — XJTU-SY and generalization (1–2 days)

Repeat training on XJTU-SY with the same pipeline. Additional experiment: train on one working condition, test on another (domain shift). Report delta in RMSE.

### Phase 8 — Thesis writeup (3–5 weeks alongside experiments)

Dissertation chapter structure:
1. Introduction and motivation (bearings in industry, cost of failure)
2. Literature review (traditional methods → LSTM → Transformers → xLSTM → Mamba → interpretability for PHM)
3. Background (xLSTM mechanics, Mamba SSM math, SHAP theory, SAE)
4. Proposed method (Mamba-xLSTM-Net)
5. Experimental setup (datasets, preprocessing, hyperparameters, hardware)
6. Results (comparison tables, prediction curves, ablations)
7. Interpretability analysis (SHAP + SAE latents + IG)
8. Discussion and limitations
9. Conclusion and future work

---

## 6. Detailed Algorithm Specifications

### 6.1 Health Indicator extraction — exact recipe

```python
# Pseudocode for HI per sample window
def extract_hi_features(signal, fs):
    # signal: (window_length,) single-channel accelerometer
    
    # Time domain (9 features)
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    kurt = scipy.stats.kurtosis(signal)
    skew = scipy.stats.skew(signal)
    crest = peak / rms
    shape = rms / np.mean(np.abs(signal))
    impulse = peak / np.mean(np.abs(signal))
    margin = peak / np.mean(np.sqrt(np.abs(signal)))**2
    variance = np.var(signal)
    
    # Frequency domain (7 features)
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=min(2048, len(signal)))
    spec_centroid = np.sum(freqs * psd) / np.sum(psd)
    spec_entropy = scipy.stats.entropy(psd / np.sum(psd))
    # Split 0–fs/2 into 5 equal bands, compute energy in each
    band_edges = np.linspace(0, fs/2, 6)
    band_energies = [
        np.sum(psd[(freqs >= band_edges[i]) & (freqs < band_edges[i+1])])
        for i in range(5)
    ]
    
    return np.array([rms, peak, kurt, skew, crest, shape, impulse, margin,
                     variance, spec_centroid, spec_entropy, *band_energies])
```

Post-process:
1. Stack per-bearing HI into `(T, 16)` matrix.
2. Per-feature MinMax normalization using *training bearings only* — fit scaler on train, apply to val/test.
3. Exponential smoothing α=0.1 on each feature's time series.

### 6.2 RUL labels

Two conventions — report both:

**Linear:** `rul_t = (T - t) / T` for t = 0..T-1
**Piecewise linear:** `rul_t = 1.0` for `t < T_degrade`, then linear decay. Uses degradation detection via RMS threshold (Jiang et al. 2023): RMS_t > k·mean(RMS_{0:n-1}) with k ∈ [2.5, 3] marks the onset.

### 6.3 Windowing for training

```
Window length L = 32 (for XJTU) or L = 64 (for PHM2012)
Stride s = 1 during training, s = L//2 for evaluation
For each window [t-L+1 : t+1], target = rul_t
```

### 6.4 Mamba-2 block inner loop (conceptual — library handles this)

```
For each sequence position i:
    x_i = input[:, i, :]
    # Selective parameters: Δ, B, C depend on x_i
    Δ_i, B_i, C_i = linear_projections(x_i)
    A_discrete = exp(Δ_i * A)
    B_discrete = Δ_i * B_i
    # State update (linear recurrence)
    h_i = A_discrete * h_{i-1} + B_discrete * x_i
    y_i = C_i * h_i
```

In practice, use `from mamba_ssm import Mamba2; block = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2)`.

### 6.5 xLSTM (mLSTM) block inner loop

```
For each sequence position i:
    q_i = W_q x_i
    k_i = W_k x_i / sqrt(d_head)
    v_i = W_v x_i
    i_i = exp(W_i x_i + b_i)        # exponential input gate
    f_i = σ(W_f x_i + b_f)          # forget gate
    o_i = σ(W_o x_i + b_o)
    # Matrix memory update
    C_i = f_i * C_{i-1} + i_i * (v_i * k_i.T)   # outer product
    n_i = f_i * n_{i-1} + i_i * k_i             # normalizer
    h_i = o_i * (C_i @ q_i) / max(|n_i.T @ q_i|, 1)
```

In practice, use the `xlstm` package — do not reimplement.

---

## 7. Training Configuration

### 7.1 Suggested starting hyperparameters

```yaml
# configs/proposed_mamba_xlstm.yaml
data:
  dataset: phm2012         # or xjtu_sy
  window_length: 64
  stride_train: 1
  stride_eval: 32
  batch_size: 128
  n_features: 16
  train_bearings: [1_1, 1_2, 1_4, 2_1, 2_3, 2_5, 3_1]
  val_bearings: [1_5, 2_2]
  test_bearings: [1_3, 1_6, 1_7, 2_4, 2_6, 2_7, 3_2, 3_3]

model:
  d_model: 128
  xlstm:
    num_blocks: 3
    slstm_positions: [1]
    num_heads: 4
    dropout: 0.1
  mamba:
    num_blocks: 2
    d_state: 64
    d_conv: 4
    expand: 2
    bidirectional: true
  fusion: gated         # or cross_attention
  head_hidden: 64
  dropout: 0.1

training:
  max_epochs: 200
  optimizer: adamw
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine
  warmup_epochs: 5
  gradient_clip_val: 1.0
  early_stopping_patience: 20
  monotonicity_weight: 0.01
  precision: 16-mixed    # bf16-mixed if A100
  seed: 42
```

### 7.2 Loss function

```
L_total = MSE(y_pred, y_true) + λ_mono · L_mono
L_mono = mean(max(0, y_pred[t+1] - y_pred[t]))   # penalize non-decreasing RUL predictions
```

The monotonicity penalty is light regularization — RUL should decrease over time. Use λ_mono = 0.01 to 0.1.

### 7.3 Reproducibility

- Set all seeds (Python, NumPy, PyTorch, CUDA).
- Log every hyperparameter to W&B.
- Use `torch.use_deterministic_algorithms(True)` for final runs.
- Run each config with 3 different seeds, report mean ± std in tables. This is non-negotiable for dissertation — single-run numbers are scientifically weak.

---

## 8. Interpretability: Deep Dive

### 8.1 SHAP implementation plan

```python
import shap

# Wrap model for SHAP (it expects a callable numpy → numpy)
def model_fn(x_np):
    x = torch.from_numpy(x_np).float().to(device)
    with torch.no_grad():
        return model(x).cpu().numpy()

# Background set: 100 random windows from train set
background = train_windows[np.random.choice(len(train_windows), 100, replace=False)]

explainer = shap.KernelExplainer(model_fn, background)

# Explain a specific test window (near the degradation point)
shap_values = explainer.shap_values(test_windows[critical_idx:critical_idx+1])
# shap_values has shape (1, L, F) — attribution per timestep per feature

# Visualizations
shap.summary_plot(shap_values.reshape(-1, n_features), 
                  test_windows[critical_idx:critical_idx+1].reshape(-1, n_features),
                  feature_names=FEATURE_NAMES)
```

Key plots for the thesis:
- Global feature importance (aggregate over test set)
- Attribution heatmap (time × feature) for the prediction at the onset of severe degradation
- Waterfall for a specific prediction

### 8.2 Sparse Autoencoder — careful details

Training SAEs is finicky. Follow this recipe (derived from Anthropic's "Towards Monosemanticity" methodology, adapted for smaller scale):

```python
class TopKSparseAutoencoder(nn.Module):
    def __init__(self, d_model, expansion=8, k=None):
        super().__init__()
        d_latent = d_model * expansion
        self.k = k or int(d_latent * 0.05)
        self.encoder = nn.Linear(d_model, d_latent)
        self.decoder = nn.Linear(d_latent, d_model, bias=False)
        # Tie decoder bias to "pre-encoder bias"
        self.pre_bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        # Subtract pre-bias
        x_centered = x - self.pre_bias
        # Encode
        z = self.encoder(x_centered)
        # Top-k
        topk_vals, topk_idx = z.topk(self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals.relu())
        # Decode + re-add bias
        x_hat = self.decoder(z_sparse) + self.pre_bias
        return x_hat, z_sparse

# Training
loss = F.mse_loss(x_hat, x)  # pure reconstruction for top-k variant
# Optional: "auxiliary loss" = reconstruct from dead features to revive them
```

**Post-training analysis workflow:**
1. Collect fused hidden states from the trained Mamba-xLSTM-Net across all bearings: `H ∈ (total_timesteps, d_model)`.
2. Train SAE on H for 50–100 epochs.
3. For each of the d_latent features, compute its activation on every `(bearing, timestep)` pair.
4. For each latent, rank pairs by activation → take top 20.
5. Plot the corresponding HI signal windows side by side. Look for common structure (e.g., "all these windows show a step-change in RMS") — this IS the latent's interpretable meaning.
6. Label interesting latents. Expected findings on bearings:
   - A latent for "low RMS, high kurtosis" (early incipient fault)
   - A latent for "rising energy in high-frequency bands" (bearing raceway damage)
   - A latent for "saturation plateau" (late-life)
   - A latent for each operating condition

This alone is a paper-worthy contribution. Frame it as *"Discovering interpretable degradation features in a deep RUL model via sparse dictionary learning."*

### 8.3 Attention maps (sanity check)

Extract attention weights from the fusion cross-attention (if you pick that fusion variant). Plot as heatmap — does the model attend to the pre-failure phase when predicting RUL near the end? Sanity check only, not a primary explanation method.

### 8.4 Concept activation vectors (stretch)

If time permits: implement TCAV-style concept activation. Define concepts like "high kurtosis phase" via simple rules, train linear probes on hidden states, and check whether directional derivatives of the prediction along concept vectors make physical sense.

---

## 9. Expected Results (targets)

Based on the paper's reported numbers and room for improvement:

| Model | RMSE (↓) | MAE (↓) | R² (↑) | PHM Score (↑) |
|---|---|---|---|---|
| LSTM (paper baseline) | ~0.15 | ~0.11 | ~0.78 | ~0.42 |
| xLSTM (paper ablation) | ~0.13 | ~0.10 | ~0.82 | ~0.48 |
| xLSTM-Transformer (paper) | ~0.10 | ~0.075 | ~0.88 | ~0.55 |
| **Mamba-xLSTM (proposed, target)** | **< 0.09** | **< 0.07** | **> 0.90** | **> 0.60** |

Exact numbers vary by bearing and condition — always report per-bearing and aggregated.

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `mamba-ssm` fails to install | Med | High | Use fallback novelty #1 (Bi-xLSTM) or Colab Pro |
| xLSTM training unstable | Med | Med | Lower lr, gradient clip, warmup, use `xlstm` package defaults first |
| PHM2012 or XJTU-SY data access blocked | Low | High | Use alternatives: CWRU, IMS Bearing, or Paderborn. Pipeline is dataset-agnostic |
| Model overfits (few bearings) | High | Med | Data augmentation (Gaussian noise, time-masking), cross-bearing CV |
| Results don't beat baseline | Med | High | Try fusion variants; try pretrain-finetune; add UQ; reframe as "comparable with interpretability gain" |
| SAE latents are meaningless | Med | Low | Tune k and expansion factor; try gated SAE variant; fallback to SHAP only |
| Supervisor wants more novelty | Low | Med | Have alternatives from §2.4 ready |

---

## 11. Evaluation Protocol (for fair, publishable results)

1. **Train/val/test split by bearing ID, never by time.**
2. Three random seeds per configuration. Report mean ± std.
3. Statistical significance: paired t-test or Wilcoxon signed-rank on per-bearing RMSE between baseline and proposed.
4. Include prediction plots for at least 4 test bearings per dataset.
5. Report number of trainable parameters and wall-clock inference time per sample — inference speed is a selling point for Mamba.
6. Do not cherry-pick bearings. Report the hard cases too.
7. All hyperparameters, seeds, and configs committed to git.

---

## 12. Timeline (14–20 weeks, part-time)

| Week | Focus |
|---|---|
| 1 | Setup, environment, download data, exploration notebook |
| 2 | Feature extraction, HI construction, data module |
| 3–4 | Baseline model implementation and training |
| 5 | Baseline results on PHM2012 and XJTU-SY |
| 6–7 | Proposed Mamba-xLSTM-Net implementation |
| 8 | Proposed model training, hyperparameter tuning |
| 9 | Ablation studies |
| 10 | SHAP analysis |
| 11–12 | SAE training and latent interpretation |
| 13 | Uncertainty quantification (optional) + generalization experiments |
| 14 | Consolidate all results, make final figures |
| 15–18 | Thesis writing |
| 19 | Supervisor review, revisions |
| 20 | Final submission |

---

## 13. Key References (start your `references.bib`)

### Baseline paper
- Liu et al. (2025). "RUL Prediction Based on xLSTM–Transformer Neural Network for Rolling Element Bearings Under Different Working Conditions." Sensors. PMC12987095.

### Core methods
- Beck, M. et al. (2024). "xLSTM: Extended Long Short-Term Memory." NeurIPS 2024. arXiv:2405.04517
- Gu, A., Dao, T. (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv:2312.00752
- Dao, T., Gu, A. (2024). "Transformers are SSMs: Generalized Models and Efficient Algorithms for Sequence Modeling Through Structured State-Space Duality (Mamba-2)." ICML 2024
- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS 2017

### Mamba for time series / RUL
- Wang, Z. et al. (2025). "Is Mamba effective for time series forecasting?" Neurocomputing.
- MambaTS (ICLR 2025).
- Liu, F. et al. (2025). "Enhanced Mamba model with multi-head attention mechanism and learnable scaling parameters for remaining useful life prediction." Sci Rep 15:7178.
- Mamba-SDP for bearing RUL (Journal of Mechanical Science and Technology, 2025).

### Datasets
- Wang, B. et al. (2020). "XJTU-SY Rolling Element Bearing Accelerated Life Test Datasets: A Tutorial." J. Mech. Eng.
- Nectoux, P. et al. (2012). "PRONOSTIA: An experimental platform for bearings accelerated degradation tests." IEEE PHM.

### Interpretability
- Lundberg, S., Lee, S.-I. (2017). "A Unified Approach to Interpreting Model Predictions (SHAP)." NeurIPS 2017.
- Sundararajan, M. et al. (2017). "Axiomatic Attribution for Deep Networks (Integrated Gradients)." ICML 2017.
- Bricken et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic.
- Cunningham, H. et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv.

### Bearing RUL reviews
- Lei, Y. et al. (2018). "Machinery health prognostics: A systematic review from data acquisition to RUL prediction." MSSP 104: 799–834.

---

## 14. What to tell Cursor at the very start

Paste this into Cursor as your first system prompt / project README:

> We are implementing a research project on Remaining Useful Life (RUL) prediction for rolling element bearings. The plan is in PROJECT_PLAN.md. Follow it phase by phase. Before each file creation, read the relevant section of PROJECT_PLAN.md to understand the spec. Write type-hinted, tested, well-commented PyTorch code. Use PyTorch Lightning and Hydra. Never train on the test bearings. Always seed everything. When in doubt, ask me a clarifying question rather than hallucinate a dataset path or hyperparameter.

---

## 15. Final Sanity Checklist Before Submitting

- [ ] Reproduced baseline paper numbers within ±10%
- [ ] Proposed model beats baseline on all 4 metrics on both datasets
- [ ] All ablations run with 3 seeds, reported mean ± std
- [ ] SHAP plots included for at least 2 test bearings per dataset
- [ ] SAE latents have at least 5 interpretable examples documented
- [ ] Per-bearing prediction curves shown in the thesis
- [ ] Inference latency measured and reported
- [ ] All code in git, all hyperparams in YAML, all results reproducible from a single `python scripts/train.py` command per config
- [ ] References cross-checked for accuracy (titles, authors, years)
- [ ] Limitations section honest about failure modes
- [ ] Future work section non-generic

Good luck.
