# Cursor Prompt Spec — Angle B
## Conformal Prediction + Unsupervised Domain Adaptation for Calibrated Cross-Condition Bearing RUL

This document is a modular implementation roadmap. Feed each **PHASE** to Cursor as a separate prompt (in order). Each phase is self-contained and produces working, tested code. The meta-goal: a dissertation-grade research codebase that is reproducible, configurable, and publication-ready for RESS/MSSP/IEEE-TII.

---

## 0. META-INSTRUCTIONS TO CURSOR (prepend this to every phase)

> You are helping me build a PhD dissertation research codebase on bearing Remaining Useful Life (RUL) prediction. The scientific contribution is **combining Unsupervised Domain Adaptation (UDA) with Conformal Prediction (CP) for calibrated cross-condition/cross-dataset RUL**, evaluated bidirectionally on PHM2012 (FEMTO-ST PRONOSTIA) and XJTU-SY.
>
> Engineering requirements:
> - Python 3.11+, PyTorch 2.x, type hints, Google-style docstrings
> - Config-driven (Hydra or OmegaConf) — no hardcoded paths/hyperparameters
> - Deterministic seeding everywhere; one `set_seed(seed)` utility used throughout
> - Logging via `loguru`; experiment tracking via `wandb` (optional flag)
> - Unit tests with `pytest` for every module that has non-trivial logic
> - Modular: every component (data, feature, model, UDA, CP, eval) is a separate package under `src/`
> - Use `pyproject.toml` (not `setup.py`), `uv` or `pip` for dependency mgmt
> - Never write monolithic scripts; everything is importable library code + thin CLI entrypoints
>
> Scientific requirements:
> - Every experiment must be reproducible from a single config file
> - Save all intermediate artifacts (features, predictions, calibration scores) to disk so downstream analysis never re-runs training
> - Report RMSE, MAE, PHM2012 asymmetric Score, empirical coverage, mean prediction interval width (MPIW), and calibration (reliability diagram, expected calibration error)
> - Use paired statistical tests (Wilcoxon signed-rank) when comparing methods across bearings

---

## PHASE 1 — Project Scaffold & Environment

**Prompt to Cursor:**

> Create a research project scaffold with the following structure. Initialize `pyproject.toml` with the listed dependencies, write a `README.md` with setup instructions, `.gitignore` for Python + data artifacts, and a `Makefile` with `install`, `test`, `lint`, `format`, `clean` targets.

```
bearing-rul-cpda/
├── pyproject.toml
├── README.md
├── Makefile
├── .gitignore
├── .pre-commit-config.yaml          # black, isort, ruff, mypy
├── configs/
│   ├── config.yaml                  # top-level Hydra config
│   ├── data/
│   │   ├── phm2012.yaml
│   │   └── xjtusy.yaml
│   ├── model/
│   │   ├── cnn_lstm.yaml
│   │   ├── tcn.yaml
│   │   └── transformer.yaml
│   ├── uda/
│   │   ├── none.yaml                # source-only baseline
│   │   ├── dann.yaml
│   │   ├── mmd.yaml
│   │   ├── deep_coral.yaml
│   │   └── cdan.yaml
│   ├── cp/
│   │   ├── split.yaml
│   │   ├── weighted.yaml            # Barber et al. non-exchangeable
│   │   ├── aci.yaml                 # adaptive CP
│   │   └── cqr.yaml                 # conformalized quantile regression
│   └── experiment/
│       ├── phm2012_cross_cond.yaml
│       ├── xjtusy_cross_cond.yaml
│       ├── phm2012_to_xjtusy.yaml
│       └── xjtusy_to_phm2012.yaml
├── src/
│   └── brul/                        # package
│       ├── __init__.py
│       ├── data/
│       │   ├── phm2012.py
│       │   ├── xjtusy.py
│       │   ├── base.py              # abstract BearingDataset
│       │   ├── windowing.py
│       │   └── labels.py            # RUL label construction
│       ├── features/
│       │   ├── time_domain.py
│       │   ├── freq_domain.py
│       │   ├── time_freq.py         # wavelet, STFT
│       │   └── health_indicator.py
│       ├── models/
│       │   ├── backbones.py         # CNN, TCN, LSTM, Transformer
│       │   ├── heads.py             # regression, quantile, domain disc
│       │   └── grl.py               # gradient reversal layer
│       ├── uda/
│       │   ├── base.py              # UDATrainer interface
│       │   ├── dann.py
│       │   ├── mmd.py
│       │   ├── coral.py
│       │   └── cdan.py
│       ├── cp/
│       │   ├── base.py              # Conformalizer interface
│       │   ├── split_cp.py
│       │   ├── weighted_cp.py       # non-exchangeable
│       │   ├── aci.py               # adaptive conformal inference
│       │   ├── cqr.py               # conformalized quantile regression
│       │   └── metrics.py           # coverage, MPIW, SSC, WSC
│       ├── training/
│       │   ├── trainer.py
│       │   ├── losses.py
│       │   └── schedulers.py
│       ├── eval/
│       │   ├── rul_metrics.py       # RMSE, MAE, Score
│       │   ├── calibration.py
│       │   └── stat_tests.py
│       ├── viz/
│       │   ├── signals.py
│       │   ├── degradation.py
│       │   └── calibration_plots.py
│       └── utils/
│           ├── seed.py
│           ├── io.py
│           ├── logging.py
│           └── paths.py
├── scripts/
│   ├── download_data.sh
│   ├── extract_features.py
│   ├── train_baseline.py
│   ├── train_uda.py
│   ├── apply_cp.py
│   └── run_experiment.py            # end-to-end: config -> results.json
├── notebooks/
│   ├── 01_eda_phm2012.ipynb
│   ├── 02_eda_xjtusy.ipynb
│   ├── 03_domain_shift_analysis.ipynb
│   ├── 04_baseline_results.ipynb
│   ├── 05_uda_results.ipynb
│   ├── 06_cp_coverage.ipynb
│   └── 07_final_dissertation_figures.ipynb
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   ├── test_uda.py
│   ├── test_cp.py
│   └── test_metrics.py
└── data/
    ├── raw/                          # raw downloaded datasets
    ├── processed/                    # windowed tensors
    └── features/                     # extracted handcrafted features
```

**Dependencies (in `pyproject.toml`):**

```toml
[project]
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "pandas>=2.2",
    "scipy>=1.12",
    "scikit-learn>=1.4",
    "pywavelets>=1.5",
    "hydra-core>=1.3",
    "omegaconf>=2.3",
    "loguru>=0.7",
    "tqdm>=4.66",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "pyarrow>=15",
    "mapie>=0.9",            # conformal prediction baselines
    "einops>=0.7",
]
[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov", "black", "isort", "ruff", "mypy", "pre-commit"]
track = ["wandb>=0.16"]
```

**Deliverables:** scaffold runs `make install && make test` cleanly (tests are empty placeholders at this point but collect).

---

## PHASE 2 — Dataset Download & Raw Data Parsers

**Prompt to Cursor:**

> Implement dataset parsers for PHM2012 and XJTU-SY that produce a unified `BearingRun` object. Each run = one bearing's full life cycle.

### 2.1 Dataset facts to encode

**PHM2012 (FEMTO-ST PRONOSTIA):**
- 3 operating conditions: (1) 1800 rpm / 4000 N, (2) 1650 rpm / 4200 N, (3) 1500 rpm / 5000 N
- 17 bearings total: Condition 1 = Bearings 1_1..1_7, Condition 2 = 2_1..2_7, Condition 3 = 3_1..3_3
- Sampling: 25.6 kHz, 0.1 s per acquisition, every 10 s
- Channels: horizontal accel (col 5) + vertical accel (col 6) + temperature (separate files)
- Labels: Bearings 1_1, 1_2, 2_1, 2_2, 3_1, 3_2 have full life; 1_3..1_7, 2_3..2_7, 3_3 are truncated test sets (RUL known only at end)
- Dataset source: https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset or https://biaowang.tech/

**XJTU-SY:**
- 3 conditions: (1) 2100 rpm / 12 kN, (2) 2250 rpm / 11 kN, (3) 2400 rpm / 10 kN
- 15 bearings: 5 per condition, named `Bearing1_1`..`Bearing3_5`
- Sampling: 25.6 kHz, 1.28 s per acquisition, every 1 min (32768 samples per file)
- Channels: horizontal + vertical accelerometer
- Files: `1.csv`, `2.csv`, ... numbered chronologically
- All bearings run to failure (vibration threshold-based stop)
- Dataset source: https://biaowang.tech/xjtu-sy-bearing-datasets/

### 2.2 Required classes

```python
# src/brul/data/base.py
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class BearingRun:
    """One bearing's full life cycle."""
    dataset: str                       # "phm2012" | "xjtusy"
    condition: int                     # 1, 2, or 3
    bearing_id: str                    # e.g. "1_1"
    signal: np.ndarray                 # shape (T, C): T timesteps × C channels (H, V[, Temp])
    channel_names: list[str]
    fs: int                            # sampling frequency (Hz)
    window_length_samples: int         # samples per acquisition window
    acquisition_interval_s: float      # seconds between acquisitions
    rpm: float
    load_N: float
    full_life: bool                    # True if run to failure
    eol_index: int | None              # index of End Of Life acquisition (if known)
    metadata: dict                     # anything else (FPT if annotated, etc.)

    def duration_s(self) -> float: ...
    def time_axis(self) -> np.ndarray: ...
    def rul_seconds(self) -> np.ndarray: ...   # RUL in seconds per acquisition
    def rul_normalized(self) -> np.ndarray: ... # linear RUL in [0,1], 1 at start
```

### 2.3 Implementation requirements

- `load_phm2012(root: Path, condition: int | None = None) -> list[BearingRun]`
- `load_xjtusy(root: Path, condition: int | None = None) -> list[BearingRun]`
- Both parsers cache parsed runs as `.parquet` in `data/processed/` with hash of raw files → avoid re-parsing
- Handle file ordering carefully: both datasets use numeric filenames; sort with `int()` conversion not lexicographic
- For PHM2012, join temperature files on timestamp; impute missing temperature with interpolation
- For XJTU-SY, truncate the last N acquisitions only if vibration exceeds 20 g (to remove post-failure noise) — make configurable

### 2.4 Tests

- `test_phm2012_loads_all_17_bearings()`
- `test_xjtusy_loads_all_15_bearings()`
- `test_sampling_frequency_25600()`
- `test_rul_monotonically_decreasing_for_full_life_bearings()`

---

## PHASE 3 — RUL Labeling + Windowing

**Prompt to Cursor:**

> Implement RUL label construction and windowing. Several labeling schemes exist in the literature — implement all and make selectable via config.

### 3.1 Labeling schemes

1. **Linear RUL**: RUL(t) = (T_EOL − t) / T_EOL ∈ [0, 1]. Simple, dominant in XJTU-SY papers. **Default.**
2. **Piecewise-linear with FPT** (Babu et al. style): RUL = 1 before First Prediction Time (FPT), then linear decay. FPT detected via 3σ rule on smoothed RMS (configurable).
3. **Exponential**: RUL(t) = exp(-α·(t/T_EOL)), for those who want explicit degradation acceleration.

Implement each as `LabelFunction` protocol. FPT detector is a separate `src/brul/data/fpt.py` with the 3σ-RMS method + smoothing window configurable.

### 3.2 Windowing

- Each acquisition (2560 samples at PHM2012, 32768 at XJTU-SY) is one "frame"
- Model input = stack of `W` consecutive frames (configurable, default W=10)
- Stride configurable (default = 1 for training, = 1 for test/calibration)
- For very long raw signals inside a frame, optionally sub-window (e.g. 1024-sample sub-windows with 50% overlap) — exposed as preprocessing option
- Label for a window = RUL at the last frame in the window

### 3.3 Output format

```python
# After windowing:
X: np.ndarray           # (N, W, C, L_sub)  or  (N, W, C*L_sub) for feature-based
y: np.ndarray           # (N,)   normalized RUL
meta: pd.DataFrame      # (N, ...)  bearing_id, condition, timestamp, etc.
```

Save as `.npz` per bearing for memory efficiency; lazy-load in DataLoader.

### 3.4 PyTorch Dataset

```python
class BearingWindowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        runs: list[BearingRun],
        label_fn: LabelFunction,
        window_length: int = 10,
        stride: int = 1,
        mode: Literal["raw", "features"] = "raw",
        feature_extractor: FeatureExtractor | None = None,
        transform: Callable | None = None,
    ): ...
```

### 3.5 Tests

- `test_linear_label_is_1_at_start_and_0_at_eol()`
- `test_fpt_detection_on_synthetic_signal()` — inject step change at known point, assert FPT within ±5 frames
- `test_windowing_produces_correct_shapes()`

---

## PHASE 4 — Exploratory Data Analysis (EDA)

**Prompt to Cursor:**

> Generate a full EDA notebook `notebooks/01_eda_phm2012.ipynb`, `02_eda_xjtusy.ipynb`, and a cross-domain shift notebook `03_domain_shift_analysis.ipynb`. Every plot must be saved to `figures/eda/` as PDF + PNG. Use a consistent colour scheme across notebooks.

### 4.1 Per-dataset EDA (notebooks 01 and 02)

1. **Inventory table**: bearing_id × condition × duration × #acquisitions × full_life flag
2. **Raw signal inspection**: for each full-life bearing, plot horizontal + vertical channels at (a) start, (b) 50% life, (c) 80% life, (d) end. Side-by-side subplots.
3. **Degradation trend plots**: RMS and kurtosis over full life cycle for all bearings, with log scale option
4. **Envelope spectrum at EOL**: FFT of envelope (Hilbert transform of bandpass-filtered signal) at last 5 acquisitions — look for BPFO, BPFI, BSF, FTF peaks if bearing geometry is known
5. **Distribution of run durations per condition**: violin plot, shows expected run length variability
6. **First Prediction Time (FPT) histogram**: apply the FPT detector, plot distribution per condition
7. **Temperature analysis (PHM2012 only)**: temperature trajectory vs vibration RMS — are they correlated? Is temperature an independent degradation signal?
8. **Non-monotonicity detection**: find bearings where smoothed RMS decreases over any ≥5-acquisition window (possible self-healing) — list them; these are evaluation edge cases later

### 4.2 Cross-domain shift analysis (notebook 03) — **critical for the dissertation narrative**

This notebook is what motivates the need for UDA + non-exchangeable CP. Every finding here becomes a sentence in the paper's introduction.

1. **Feature distribution comparison across conditions**: extract 16 time-domain + 8 frequency-domain features per acquisition; plot KDE of each feature by condition (PHM2012) and by condition (XJTU-SY). Quantify shift using:
   - **Maximum Mean Discrepancy (MMD)** with RBF kernel (bootstrap p-value)
   - **Wasserstein-1 distance** per feature
   - **Kolmogorov-Smirnov statistic** per feature
2. **Cross-dataset feature shift (PHM2012 vs XJTU-SY)**: same features, plot overlaid. Which features transfer best? (Often envelope-spectrum band energies; often NOT raw RMS.)
3. **t-SNE / UMAP visualisation**: project raw-window features into 2D, colour by (a) condition, (b) dataset, (c) RUL bucket. Visually confirms shift.
4. **Correlation of features with RUL**: Spearman ρ of each feature with normalised RUL, per condition. Features that correlate consistently across conditions are robust features.
5. **Label drift**: distribution of run durations — if Condition 1 bearings live 3× longer than Condition 3, raw-time RUL labels are on different scales. This is a motivation for normalised RUL.
6. **Degradation-stage alignment**: for each bearing, align by percentile-of-life and plot mean RMS per percentile bin. Mismatch between conditions = visual evidence of shift.

### 4.3 Deliverable

One-page "EDA summary" markdown listing top 5 findings, with references to specific figures. This becomes Section 3.1 of the dissertation.

---

## PHASE 5 — Handcrafted Feature Extraction

**Prompt to Cursor:**

> Implement an extensible feature extraction library. We want both handcrafted features (for interpretability + a strong baseline) and the option to feed raw windows to deep models.

### 5.1 Time-domain features (per frame)

RMS, peak, peak-to-peak, crest factor, kurtosis, skewness, shape factor, impulse factor, clearance factor, margin factor, variance, standard deviation, mean absolute deviation, root sum of squares, energy. → 15 features × 2 channels (H, V) = 30.

### 5.2 Frequency-domain features (per frame)

FFT → compute: spectral centroid, spectral spread, spectral skewness, spectral kurtosis, spectral entropy, dominant frequency, amplitude at dominant frequency, energy in 5 bands (divided log-spaced across 0–Nyquist). → 12 × 2 = 24.

### 5.3 Envelope spectrum features

Bandpass 500–10000 Hz → Hilbert → FFT of envelope → amplitude at known bearing characteristic frequencies if geometry known; otherwise top-5 peak amplitudes + their frequencies. (For PHM2012 NSK 6804 and XJTU-SY LDK UER204, default geometries are well-documented.)

### 5.4 Time-frequency features

- **Wavelet packet energies**: 4-level decomposition → 16 sub-band energies (db4 default, configurable)
- **STFT-based spectral flux** (frame-to-frame spectral change)

### 5.5 Health Indicator (HI) construction

Implement three HI types as they will be important for UDA feature-space work:
1. **PCA-HI**: first principal component of standardised feature matrix (Guo et al., 2017)
2. **Autoencoder-HI**: reconstruction error of an AE trained only on the healthy region (< FPT) of source bearings
3. **Monotonicity-optimised HI**: linear combination of features maximising Spearman monotonicity + condition-invariance (formulated as a constrained optimisation)

Expose as `HealthIndicator` classes with `.fit(healthy_features)` and `.transform(features)` methods.

### 5.6 Interface

```python
class FeatureExtractor(Protocol):
    feature_names: list[str]
    def extract(self, raw_frame: np.ndarray) -> np.ndarray: ...  # (F,) feature vector
```

### 5.7 Tests

- Time-domain features on known signals (pure sine → expected RMS = amplitude/√2)
- Wavelet energies sum to total energy (Parseval)
- Regression test: features extracted on a fixed synthetic signal match committed reference values

---

## PHASE 6 — Baseline Model (Source-Only, No Adaptation)

**Prompt to Cursor:**

> Build the baseline RUL regressor. This is the "source-only" performance floor we compare UDA against.

### 6.1 Architecture options (all configurable)

All backbones take input `(B, W, C, L)` (batch, window, channels, raw-length) OR `(B, W, F)` (feature-based) and output `(B, D_feat)`:

1. **CNN-LSTM** (Zhu et al. 2019 style): 1D CNN blocks with batchnorm + ReLU → temporal pooling → LSTM → FC
2. **TCN** (Bai et al. 2018): dilated causal convolutions
3. **Transformer encoder**: sinusoidal positional encoding + multi-head self-attention stack
4. **MLP on handcrafted features**: for the "classical" baseline

A regression head produces scalar RUL ∈ [0, 1]. A **quantile head** produces $(\hat{q}_{0.05}, \hat{q}_{0.5}, \hat{q}_{0.95})$ for CQR-based CP later.

### 6.2 Training

- Optimiser: AdamW, lr 1e-3, weight decay 1e-4 (config)
- Scheduler: cosine with warmup (config)
- Loss: MSE (default); Pinball loss for the quantile head
- Early stopping on validation RMSE
- Gradient clipping at 1.0
- Mixed precision (AMP) optional

### 6.3 Training/validation/calibration splits

**This is critical and dissertation-specific.** Implement a rigorous splitting helper:

```python
def source_target_split(
    all_runs: list[BearingRun],
    source_condition: int,
    target_condition: int,
    train_source_ratio: float = 0.7,   # of source bearings
    calib_fraction_from_target: float = 0.0,  # = 0 for pure UDA; > 0 only if allowed
    seed: int = 42,
) -> SplitResult: ...
```

Rules:
- **Bearing-level splits only** — never split by time within a bearing (leakage)
- For Angle B, default is **fully label-free target** → target has NO calibration access; CP on source only
- Optional "semi-supervised CP" variant: tiny fraction of target bearings used for calibration — reported separately

### 6.4 Experiments to run

Baseline table with 4 backbones × 4 train/test condition combinations (PHM2012: train C1 test C2, train C1 test C3, train C2 test C1, train C2 test C3; same for XJTU-SY; same for cross-dataset). Report RMSE, MAE, Score for each.

### 6.5 Tests

- `test_model_forward_shape()` for every backbone
- `test_training_loop_decreases_loss_on_toy_data()`
- `test_bearing_level_split_has_no_overlap()`

---

## PHASE 7 — Unsupervised Domain Adaptation Methods

**Prompt to Cursor:**

> Implement four UDA methods that all share a common trainer interface. Each method is a thin wrapper around the baseline backbone + an adaptation loss term.

### 7.1 Common interface

```python
class UDATrainer(ABC):
    def __init__(self, backbone: nn.Module, rul_head: nn.Module, config: DictConfig): ...
    @abstractmethod
    def adaptation_loss(
        self, feat_s: Tensor, feat_t: Tensor, y_s: Tensor, preds_s: Tensor, epoch: int
    ) -> dict[str, Tensor]: ...
    def training_step(self, batch_source, batch_target, epoch) -> dict[str, Tensor]: ...
```

### 7.2 Methods

**1. DANN (Ganin & Lempitsky 2015)**
- Domain discriminator: 2-layer MLP on features, binary cross-entropy source vs target
- Gradient Reversal Layer (GRL) between feature extractor and discriminator
- λ schedule: `λ(p) = 2 / (1 + exp(-γ·p)) − 1`, `p = epoch / total_epochs`, γ=10 (standard)

**2. MMD (multi-kernel Maximum Mean Discrepancy)**
- MK-MMD with RBF kernels, bandwidths = median heuristic × {0.25, 0.5, 1, 2, 4}
- Loss: `L_task + β · MMD²(feat_s, feat_t)`, β schedule linear warmup over 10 epochs

**3. Deep CORAL (Sun & Saenko 2016)**
- Covariance alignment: `‖Cov(feat_s) − Cov(feat_t)‖²_F / (4·d²)`
- Minimal tuning, often strong baseline

**4. CDAN (Long et al. 2018) — adapted for regression**
- Conditional adversarial via outer product of feature and predicted RUL (discretise RUL into K bins for conditioning tensor, K=10 default)
- Entropy-weighted version (CDAN+E)

### 7.3 Important implementation details

- **Batch construction**: alternating source/target batches, each seeing one source batch (labelled) and one target batch (unlabelled) per step
- **Feature extractor**: shared across source and target (by construction, it's the same model)
- **Quantile head** must also be adapted — the RUL head produces 3 outputs (q05, q50, q95), domain adaptation on the shared feature is what matters
- **Numerical stability**: MMD can blow up with small batches — require minimum batch size 32 per domain
- **Bearing-aware batching**: do NOT shuffle bearings away — within a batch, include multiple bearings per domain to avoid degenerate feature alignment on a single trajectory

### 7.4 Tests

- `test_grl_reverses_gradient()`
- `test_mmd_zero_on_identical_distributions()`
- `test_coral_matches_closed_form_on_gaussian()`

### 7.5 Ablation experiments

For each UDA method, run:
- (a) source-only baseline (same backbone, no adaptation)
- (b) UDA with no target data (sanity check — should ≈ baseline)
- (c) full UDA
- (d) "oracle": train on target labels directly (upper bound)

Report RMSE gap closure: `(oracle − source_only) / (oracle − UDA) × 100%`.

---

## PHASE 8 — Conformal Prediction Module (The Novel Contribution)

**Prompt to Cursor:**

> This is the main novelty of the dissertation. Implement four CP variants and rigorously evaluate their coverage under (i) in-distribution (ii) cross-condition (iii) cross-dataset settings. The novelty is **applying non-exchangeable CP methods to the UDA setting** — which the literature has not done for bearing RUL.

### 8.1 Background (encode as docstrings)

Standard split conformal assumes exchangeability between calibration and test data. Under domain shift this assumption is violated. Three ways to handle:
- **Weighted conformal prediction** (Tibshirani et al. 2019; Barber et al. 2023): reweight calibration scores by importance weights that reflect the test distribution
- **Adaptive conformal inference (ACI)** (Gibbs & Candès 2021): online updates of the target miscoverage level based on observed coverage gaps
- **Conformalized quantile regression (CQR)** (Romano et al. 2019): combine a quantile regressor with conformalisation for adaptive interval width

### 8.2 Common interface

```python
class Conformalizer(ABC):
    def __init__(self, alpha: float = 0.1): ...        # target miscoverage
    @abstractmethod
    def calibrate(self, preds_calib: Preds, y_calib: np.ndarray, **kwargs) -> None: ...
    @abstractmethod
    def predict(self, preds_test: Preds) -> Intervals: ...

@dataclass
class Preds:
    point: np.ndarray                                   # (N,) point predictions
    lower_quantile: np.ndarray | None                   # (N,) q_{alpha/2} if CQR
    upper_quantile: np.ndarray | None                   # (N,) q_{1-alpha/2} if CQR
    features: np.ndarray | None                         # (N, D) for weighted CP

@dataclass
class Intervals:
    lower: np.ndarray
    upper: np.ndarray
    def coverage(self, y_true): ...
    def width(self): ...
```

### 8.3 Methods

**1. Split CP (baseline)**
- Nonconformity score: `s_i = |y_i − ŷ_i|` (absolute residual)
- Quantile: `q̂ = ⌈(n+1)(1−α)⌉/n`-th empirical quantile of calibration scores
- Interval: `[ŷ − q̂, ŷ + q̂]`

**2. Normalised/locally-adaptive Split CP**
- `s_i = |y_i − ŷ_i| / σ̂(x_i)` where σ̂ is a learned heteroscedastic estimator or MC-dropout std
- Interval: `[ŷ − q̂·σ̂(x), ŷ + q̂·σ̂(x)]` — adapts width to instance difficulty

**3. Conformalized Quantile Regression (CQR)**
- Backbone predicts `(q̂_{α/2}, q̂_{1−α/2})`; nonconformity: `s_i = max(q̂_{α/2}(x_i) − y_i, y_i − q̂_{1−α/2}(x_i))`
- Interval: `[q̂_{α/2}(x) − q̂, q̂_{1−α/2}(x) + q̂]`

**4. Weighted Conformal Prediction (non-exchangeable)** — **THE KEY METHOD**
- Weights `w_i` reflect test distribution likelihood under source:
  - (a) Kernel density ratio estimation with a domain-discriminator (already trained during DANN!) → reuse discriminator outputs as likelihood ratio
  - (b) Nearest-neighbour weighting in feature space (Guan 2022 style)
- Weighted quantile of calibration scores: `q̂ = inf{ q : Σ w_i·𝟙{s_i ≤ q} ≥ (1−α)·Σ w_i }`

**5. Adaptive Conformal Inference (ACI)**
- Maintain running α_t; update `α_{t+1} = α_t + η·(α − err_t)` where `err_t = 𝟙{y_t ∉ interval_t}`
- Online flavour natural for deployment; also viable "offline" by sweeping over the target bearing's acquisitions in chronological order

### 8.4 Calibration regimes to benchmark

| Regime | Calibration data | Notes |
|---|---|---|
| A. In-distribution | Held-out source bearings | Classical CP, should attain nominal coverage |
| B. Naive cross-condition | Source calibration bearings, target test bearings | Expected to UNDER-COVER; baseline for novelty |
| C. UDA + naive CP | Same as B but using UDA-trained model | Demonstrates coverage gap persists even with UDA point accuracy gain |
| D. UDA + weighted CP | Importance-weighted via discriminator | **Novel contribution: expect coverage restoration** |
| E. UDA + ACI | Online recalibration on target stream | **Novel contribution: expect coverage + narrow intervals** |
| F. UDA + CQR + weighted CP | Quantile head + weighting | **Best expected performance** |

### 8.5 Metrics

- Empirical coverage: `mean(lower ≤ y ≤ upper)` — target 1 − α
- Mean Prediction Interval Width (MPIW)
- **Size-Stratified Coverage (SSC)**: coverage within quintiles of interval width
- **Worst-Slab Coverage (WSC)** across bearings/conditions
- Calibration gap: `|coverage − (1 − α)|`
- Reliability diagram at multiple α ∈ {0.05, 0.1, 0.2}

### 8.6 Tests

- `test_split_cp_coverage_on_gaussian_regression()` — synthetic, assert coverage within ±2% of 1−α at n=1000
- `test_weighted_cp_reduces_to_split_cp_when_weights_uniform()`
- `test_aci_coverage_converges_to_target()`

---

## PHASE 9 — Experimental Protocol & Full Experiment Runner

**Prompt to Cursor:**

> Build `scripts/run_experiment.py` that takes a single Hydra config and produces a `results.json` with everything needed to populate the dissertation tables.

### 9.1 Experiment matrix

**Cross-condition within dataset (main result):**

| Source | Target | Dataset |
|---|---|---|
| C1 | C2, C3 | PHM2012 |
| C2 | C1, C3 | PHM2012 |
| C3 | C1, C2 | PHM2012 |
| C1 | C2, C3 | XJTU-SY |
| C2 | C1, C3 | XJTU-SY |
| C3 | C1, C2 | XJTU-SY |

**Cross-dataset (stress test):**

| Source | Target | Notes |
|---|---|---|
| PHM2012 (all C) | XJTU-SY (all C) | Harder; different sensor mounting |
| XJTU-SY (all C) | PHM2012 (all C) | Same |

For each source→target pair, run:
- 4 backbones × 5 UDA methods (incl. source-only) × 6 CP regimes = 120 configurations per pair × 16 pairs = **1920 runs**

This is too many. In practice:
1. Fix the best backbone per-dataset after Phase 6
2. Run 5 UDA × 6 CP = 30 configurations per pair → 480 total
3. Use 3 seeds each → 1440 trainings (achievable; ≈ 3–4 weeks on one A100)

**Reduce further for dissertation timeline**: pick 3 UDA methods (source-only, DANN, MMD) × 4 CP (split, CQR, weighted, ACI) × 2 datasets × 6 cond pairs × 3 seeds = 432 runs. Each run ≈ 30 min on A100 = 200 GPU-hours ≈ 9 days.

### 9.2 Output schema

`results.json` per run:

```json
{
  "run_id": "phm2012_c1_to_c2_tcn_dann_weightedcp_seed42",
  "config_hash": "...",
  "metrics": {
    "point": {"rmse": ..., "mae": ..., "score_phm": ...},
    "interval": {"coverage_0.05": ..., "mpiw_0.05": ..., "ssc_0.05": [...], "wsc_0.05": ...,
                 "coverage_0.1": ..., "mpiw_0.1": ..., ...},
    "per_bearing": [{"bearing_id": "2_1", "rmse": ..., "coverage": ...}, ...]
  },
  "artifacts": {"preds_path": "...", "intervals_path": "...", "model_path": "..."}
}
```

### 9.3 Aggregation script

`scripts/aggregate_results.py` reads all `results.json`, joins into a DataFrame, produces:
- Main results table (LaTeX, Markdown)
- Per-bearing boxplots
- Coverage-vs-MPIW scatter (Pareto front)
- Statistical significance: Wilcoxon signed-rank test (paired over bearings) comparing weighted CP vs naive split CP

---

## PHASE 10 — Analysis & Dissertation Figures

**Prompt to Cursor:**

> Generate publication-quality analysis and figures.

### 10.1 Required figures

1. **Fig. 1**: Overview diagram — source data → UDA feature extractor → RUL + quantile heads → weighted/ACI conformalisation → intervals. (Use TikZ or matplotlib + `patch`; save as PDF.)
2. **Fig. 2**: Example RUL trajectory on one target bearing showing (a) true RUL, (b) naive CP intervals (under-covering), (c) weighted CP intervals (covering). Annotate coverage rates.
3. **Fig. 3**: Coverage vs 1−α sweep for each CP method on target bearings (reliability diagram). Target line at y=x.
4. **Fig. 4**: MPIW distributions per method (violin plot) across target bearings.
5. **Fig. 5**: Coverage gap heatmap: rows = source→target pairs, columns = CP methods, cells = |coverage − (1−α)|.
6. **Fig. 6**: Size-stratified coverage per method (bar plot).
7. **Fig. 7**: Ablation — coverage vs UDA strength (MMD β sweep).
8. **Fig. 8**: Per-bearing scatter of CP width vs error — each point a bearing-acquisition, colour by domain, showing adaptivity of weighted CP to harder target instances.

### 10.2 Required tables

- **Table 1**: Dataset inventory
- **Table 2**: Cross-condition point-accuracy (RMSE/MAE/Score) — rows are UDA methods, columns are source→target pairs
- **Table 3**: Cross-condition interval metrics (coverage, MPIW) at α=0.1
- **Table 4**: Cross-dataset results (PHM2012↔XJTU-SY)
- **Table 5**: Ablation — effect of each UDA component on CP coverage
- **Table 6**: Computational cost — training time, inference latency, memory

### 10.3 Discussion artefacts

Markdown `analysis/discussion.md` auto-generated with numerical claims pulled from `results.json`:
- "Weighted CP restores coverage from X.XX to Y.YY% across 12 of 12 cross-condition pairs at α=0.1"
- "The average MPIW of weighted CP is Z% larger than naive CP — the price of calibration"
- Any bearing where ANY method fails to cover → flagged for case study

---

## PHASE 11 — Reproducibility, Documentation, Publication

**Prompt to Cursor:**

> Finalise the project for submission.

### 11.1 Reproducibility

- `make reproduce` runs the full pipeline end-to-end given raw data and a fixed seed list, produces the final `results/` directory
- Docker image (CUDA 12.1 base) + `docker-compose.yml`
- Pin all dependencies in `requirements.lock` via `uv lock`
- Include a `CITATION.cff` and `zenodo.json`

### 11.2 Documentation

- `docs/` built with MkDocs Material, includes: installation, data setup, running experiments, API reference (auto-generated from docstrings via `mkdocstrings`)
- Tutorial notebook: "Reproduce Table 3 in 30 minutes"

### 11.3 Paper & dissertation artefacts

- Auto-generate LaTeX tables from `results.json` into `paper/tables/`
- Auto-save all figures to `paper/figures/`
- Chapter 5 of dissertation: results chapter drafted as markdown that pulls numbers from `results/` at build time (no manual copy-paste of numbers)

### 11.4 Release

- GitHub repo with MIT or Apache-2.0 license
- Zenodo DOI for archived snapshot
- Pre-print on arXiv (category `eess.SP` or `cs.LG`)

---

## APPENDIX A — Common Pitfalls to Avoid (tell Cursor to double-check these)

1. **Data leakage by within-bearing splits** — never mix windows of the same bearing across train/val/test
2. **RUL normalisation inconsistency** — either everyone uses normalised RUL or everyone uses absolute; never mix in one table
3. **PHM2012 Score function** — use the original asymmetric exponential score, not a symmetric variant; cite the 2012 challenge description
4. **Exchangeability violation silently ignored** — running split CP on target data calibrated on source IS the failure case; this must be explicitly reported, not worked around
5. **FPT-based labelling leakage** — if using piecewise-linear RUL, FPT must be detected only on source bearings; target FPT is unknown at test time
6. **Domain discriminator as importance weight** — the discriminator gives p(domain=target | x); convert to likelihood ratio via Bayes: `w(x) = p(target|x) / p(source|x) × p(source) / p(target)`; include the prior ratio
7. **MMD with tiny batches** — MMD is O(B²) and biased for small B; require B ≥ 32 and use the unbiased estimator
8. **CP calibration size** — weighted CP's effective sample size `n_eff = (Σ w)² / Σ w²` must be reported; small n_eff → wide intervals
9. **Coverage measured per-bearing, not globally** — a method that covers 100% on long bearings and 0% on short bearings has average ≈ 90% which is misleading
10. **Don't report cherry-picked target conditions** — all cross-condition pairs go in the main table

---

## APPENDIX B — Suggested Timeline for a 12-Month Dissertation Phase

| Month | Phase | Deliverable |
|---|---|---|
| 1 | 1–2 | Scaffold + parsers + data sanity checks |
| 2 | 3–4 | Labelling + windowing + EDA notebooks (Chapter 3 draft) |
| 3 | 5–6 | Features + baseline models (Chapter 4 draft) |
| 4–5 | 7 | UDA implementation + experiments (Chapter 5 part 1) |
| 6–7 | 8 | CP implementation + synthetic validation (Chapter 5 part 2) |
| 8–9 | 9 | Full experimental matrix, 3-seed runs |
| 10 | 10 | Analysis + figures + discussion |
| 11 | 11 | Documentation, reproducibility, preprint submission |
| 12 | — | Dissertation writing, defence prep, journal paper revisions |

---

## APPENDIX C — Stretch Goals (Only If Main Work Is Ahead of Schedule)

- **C.1** Compare against CURA (source-free DA, MSSP 2025) as an additional baseline
- **C.2** Add Bayesian CP (Fong & Holmes 2021) as a 5th CP method
- **C.3** Test on a third dataset (IMS Cincinnati or FEMTO-ST BTR) to demonstrate generality
- **C.4** Edge deployment study: INT8-quantise the best model, measure latency on Jetson Nano, verify coverage survives quantisation
- **C.5** Extend to multivariate RUL (horizontal + vertical + temperature for PHM2012) and test whether coverage behaves differently with multimodal input

---

## HOW TO USE THIS DOCUMENT IN CURSOR

1. Open Cursor, create the repo folder, and open this doc in a tab
2. For each phase, copy the **"Prompt to Cursor"** text plus the relevant sub-sections into Cursor's Composer
3. Tell Cursor: *"Implement Phase X exactly as specified in the attached document. Include tests. Stop and ask before touching code outside this phase's scope."*
4. After each phase, review diffs, run `make test`, and commit with a semantic message (`feat(data): implement PHM2012 parser`)
5. Track deviations from spec in `DECISIONS.md` with justifications — this is dissertation evidence

Good luck — and remember: the novelty is in **Phase 8 (Weighted + ACI CP under UDA)**. Everything else is infrastructure supporting that claim.
