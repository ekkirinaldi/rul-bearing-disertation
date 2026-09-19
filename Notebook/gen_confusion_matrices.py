"""Generate SVM-RBF and LR confusion matrices for Lampiran E.

Reproduces the core classification experiment from
Conference1_Classification_SVM_LR.ipynb using the CWRU 48kHz data.

Outputs:
    Notebook/output/lampE_cm_svm.png
    Notebook/output/lampE_cm_lr.png
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

CWRU_DIR = Path(__file__).resolve().parents[1] / "data-bearing" / "cwru"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SEG_LEN = 2048          # samples per segment (48 kHz, ~0.042 s)
N_SAMPLES_PER_CLASS = 230

LOAD = "1"              # load index for all classes

FILE_MAP = {
    "Ball_007_1":    f"B007_{LOAD}_123.mat",
    "Ball_014_1":    f"B014_{LOAD}_190.mat",
    "Ball_021_1":    f"B021_{LOAD}_227.mat",
    "IR_007_1":      f"IR007_{LOAD}_110.mat",
    "IR_014_1":      f"IR014_{LOAD}_175.mat",
    "IR_021_1":      f"IR021_{LOAD}_214.mat",
    "Normal_1":      f"Time_Normal_{LOAD}_098.mat",
    "OR_007_6_1":    f"OR007_6_{LOAD}_136.mat",
    "OR_014_6_1":    f"OR014_6_{LOAD}_202.mat",
    "OR_021_6_1":    f"OR021_6_{LOAD}_239.mat",
}


def extract_time_features(seg: np.ndarray) -> dict:
    """9 time-domain statistical features used in Conference1."""
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.max(np.abs(seg)))
    return {
        "max":       float(np.max(seg)),
        "min":       float(np.min(seg)),
        "mean":      float(np.mean(seg)),
        "sd":        float(np.std(seg)),
        "rms":       rms,
        "skewness":  float(stats.skew(seg)),
        "kurtosis":  float(stats.kurtosis(seg, fisher=True)),
        "crest":     peak / rms if rms > 1e-10 else 0.0,
        "form":      rms / (np.mean(np.abs(seg)) + 1e-10),
    }


def load_class(mat_path: Path, label: str, n: int) -> pd.DataFrame:
    mat = sio.loadmat(str(mat_path))
    # DE channel key pattern: X<number>_DE_time
    de_key = next(k for k in mat if k.endswith("_DE_time"))
    signal = mat[de_key].flatten()
    rows = []
    for start in range(0, len(signal) - SEG_LEN + 1, SEG_LEN):
        seg = signal[start: start + SEG_LEN]
        rows.append(extract_time_features(seg))
        if len(rows) >= n:
            break
    df = pd.DataFrame(rows)
    df["fault"] = label
    return df


def main() -> None:
    print("Loading CWRU data …")
    frames = []
    for label, fname in FILE_MAP.items():
        fpath = CWRU_DIR / fname
        if not fpath.exists():
            print(f"  WARN: {fpath.name} not found, skipping {label}")
            continue
        df = load_class(fpath, label, N_SAMPLES_PER_CLASS)
        print(f"  {label}: {len(df)} samples")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    print(f"\nDataset shape: {data.shape}")
    print("Classes:", sorted(data["fault"].unique()))

    fault_type = np.array(sorted(data["fault"].unique()))

    # Train/test split — 750 test samples, stratified
    train_data, test_data = train_test_split(
        data, test_size=750, stratify=data["fault"], random_state=RANDOM_SEED
    )
    feature_cols = [c for c in data.columns if c != "fault"]
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data[feature_cols])
    test_scaled = scaler.transform(test_data[feature_cols])

    # ---------------------------------------------------------------
    # SVM-RBF with GridSearch (limited to keep runtime reasonable)
    # ---------------------------------------------------------------
    print("\nTraining SVM-RBF (GridSearchCV) …")
    params = {"C": [0.1, 1, 10, 100], "kernel": ["rbf"], "gamma": ["scale", "auto"]}
    gs = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        params, cv=5, n_jobs=-1, verbose=0,
    )
    gs.fit(train_scaled, train_data["fault"])
    best_svm = gs.best_estimator_
    print(f"  Best params: {gs.best_params_}")

    test_pred_svm = best_svm.predict(test_scaled)
    acc_svm = accuracy_score(test_data["fault"], test_pred_svm)
    print(f"  Test accuracy SVM-RBF: {acc_svm*100:.1f}%")
    cm_svm = confusion_matrix(test_data["fault"], test_pred_svm, labels=fault_type)

    # ---------------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------------
    print("\nTraining Logistic Regression …")
    lr_model = LogisticRegression(
        solver="lbfgs", max_iter=1000,
        random_state=RANDOM_SEED, C=1.0,
    )
    lr_model.fit(train_scaled, train_data["fault"])
    test_pred_lr = lr_model.predict(test_scaled)
    acc_lr = accuracy_score(test_data["fault"], test_pred_lr)
    print(f"  Test accuracy LR: {acc_lr*100:.1f}%")
    cm_lr = confusion_matrix(test_data["fault"], test_pred_lr, labels=fault_type)

    # Short labels for axis ticks
    short = [
        "Ball007", "Ball014", "Ball021",
        "IR007",   "IR014",   "IR021",
        "Normal",
        "OR007",   "OR014",   "OR021",
    ]
    if len(fault_type) != len(short):
        short = fault_type.tolist()

    # ---------------------------------------------------------------
    # Plot: SVM confusion matrix (test set only — dissertation figure)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_svm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short, yticklabels=short,
        linewidths=0.4, linecolor="#cccccc",
        ax=ax,
    )
    ax.set_title(
        f"Matriks Konfusi SVM-RBF — Set Uji CWRU (10 kelas)\n"
        f"Akurasi {acc_svm*100:.1f}% ({int(np.diag(cm_svm).sum())}/{len(test_data)} sampel)",
        fontsize=12,
    )
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Kelas Nyata", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    svm_path = OUT_DIR / "lampE_cm_svm.png"
    fig.savefig(svm_path, dpi=150)
    plt.close(fig)
    print(f"\n  SVM CM → {svm_path}")

    # ---------------------------------------------------------------
    # Plot: LR confusion matrix
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_lr, annot=True, fmt="d", cmap="Oranges",
        xticklabels=short, yticklabels=short,
        linewidths=0.4, linecolor="#cccccc",
        ax=ax,
    )
    ax.set_title(
        f"Matriks Konfusi Logistic Regression — Set Uji CWRU (10 kelas)\n"
        f"Akurasi {acc_lr*100:.1f}% ({int(np.diag(cm_lr).sum())}/{len(test_data)} sampel)",
        fontsize=12,
    )
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Kelas Nyata", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    lr_path = OUT_DIR / "lampE_cm_lr.png"
    fig.savefig(lr_path, dpi=150)
    plt.close(fig)
    print(f"  LR CM → {lr_path}")

    print(f"\nDone.  SVM acc={acc_svm*100:.1f}%  LR acc={acc_lr*100:.1f}%")


if __name__ == "__main__":
    main()
