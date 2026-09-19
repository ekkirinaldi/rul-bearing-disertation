"""Generate DT / RF / XGBoost confusion matrices for Lampiran F.

Reproduces the core classification experiment from
Conference2_Classification_Tree.ipynb using the CWRU 48kHz data.

Outputs:
    Notebook/output/lampF_cm_dt.png
    Notebook/output/lampF_cm_rf.png
    Notebook/output/lampF_cm_xgb.png
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CWRU_DIR = Path(__file__).resolve().parents[1] / "data-bearing" / "cwru"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SEG_LEN = 2048
N_SAMPLES_PER_CLASS = 230
LOAD = "1"

FILE_MAP = {
    "Ball_007_1":  f"B007_{LOAD}_123.mat",
    "Ball_014_1":  f"B014_{LOAD}_190.mat",
    "Ball_021_1":  f"B021_{LOAD}_227.mat",
    "IR_007_1":    f"IR007_{LOAD}_110.mat",
    "IR_014_1":    f"IR014_{LOAD}_175.mat",
    "IR_021_1":    f"IR021_{LOAD}_214.mat",
    "Normal_1":    f"Time_Normal_{LOAD}_098.mat",
    "OR_007_6_1":  f"OR007_6_{LOAD}_136.mat",
    "OR_014_6_1":  f"OR014_6_{LOAD}_202.mat",
    "OR_021_6_1":  f"OR021_6_{LOAD}_239.mat",
}


def extract_time_features(seg: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(seg ** 2)))
    peak = float(np.max(np.abs(seg)))
    return {
        "max":      float(np.max(seg)),
        "min":      float(np.min(seg)),
        "mean":     float(np.mean(seg)),
        "sd":       float(np.std(seg)),
        "rms":      rms,
        "skewness": float(stats.skew(seg)),
        "kurtosis": float(stats.kurtosis(seg, fisher=True)),
        "crest":    peak / rms if rms > 1e-10 else 0.0,
        "form":     rms / (np.mean(np.abs(seg)) + 1e-10),
    }


def load_class(mat_path: Path, label: str, n: int) -> pd.DataFrame:
    mat = sio.loadmat(str(mat_path))
    de_key = next(k for k in mat if k.endswith("_DE_time"))
    signal = mat[de_key].flatten()
    rows = []
    for start in range(0, len(signal) - SEG_LEN + 1, SEG_LEN):
        rows.append(extract_time_features(signal[start: start + SEG_LEN]))
        if len(rows) >= n:
            break
    df = pd.DataFrame(rows)
    df["fault"] = label
    return df


def plot_cm(cm, labels, title, cmap, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    short = [l.replace("_1", "").replace("_6", "") for l in labels]
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=short, yticklabels=short,
        linewidths=0.4, linecolor="#cccccc", ax=ax,
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Prediksi", fontsize=11)
    ax.set_ylabel("Kelas Nyata", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  → {path}")


def main() -> None:
    print("Loading CWRU data …")
    frames = []
    for label, fname in FILE_MAP.items():
        fpath = CWRU_DIR / fname
        if not fpath.exists():
            print(f"  WARN: {fname} not found")
            continue
        df = load_class(fpath, label, N_SAMPLES_PER_CLASS)
        print(f"  {label}: {len(df)} samples")
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    fault_type = np.array(sorted(data["fault"].unique()))
    print(f"\nDataset: {data.shape}  classes: {len(fault_type)}")

    train_data, test_data = train_test_split(
        data, test_size=750, stratify=data["fault"], random_state=RANDOM_SEED
    )

    # Tree models don't need scaling, but use unscaled features
    X_train = train_data.iloc[:, :-1].values
    X_test = test_data.iloc[:, :-1].values
    le = LabelEncoder()
    y_train = le.fit_transform(train_data["fault"].values)
    y_test = le.transform(test_data["fault"].values)

    # ---------------------------------------------------------------
    # Decision Tree
    # ---------------------------------------------------------------
    print("\nTraining Decision Tree (GridSearchCV) …")
    dt_params = {
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_SEED), dt_params,
        cv=5, n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_
    dt_pred = best_dt.predict(X_test)
    acc_dt = accuracy_score(y_test, dt_pred)
    print(f"  DT accuracy: {acc_dt*100:.1f}%  params: {dt_grid.best_params_}")
    cm_dt = confusion_matrix(y_test, dt_pred)

    # ---------------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------------
    print("\nTraining Random Forest (GridSearchCV) …")
    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED), rf_params,
        cv=5, n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    rf_pred = best_rf.predict(X_test)
    acc_rf = accuracy_score(y_test, rf_pred)
    print(f"  RF accuracy: {acc_rf*100:.1f}%  params: {rf_grid.best_params_}")
    cm_rf = confusion_matrix(y_test, rf_pred)

    # ---------------------------------------------------------------
    # XGBoost
    # ---------------------------------------------------------------
    print("\nTraining XGBoost (GridSearchCV) …")
    xgb_params = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.1, 0.3],
    }
    xgb_grid = GridSearchCV(
        xgb.XGBClassifier(random_state=RANDOM_SEED, eval_metric="mlogloss",
                          use_label_encoder=False),
        xgb_params, cv=5, n_jobs=-1,
    )
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_
    xgb_pred = best_xgb.predict(X_test)
    acc_xgb = accuracy_score(y_test, xgb_pred)
    print(f"  XGB accuracy: {acc_xgb*100:.1f}%  params: {xgb_grid.best_params_}")
    cm_xgb = confusion_matrix(y_test, xgb_pred)

    labels = le.classes_.tolist()

    # ---------------------------------------------------------------
    # Save figures
    # ---------------------------------------------------------------
    print("\nSaving confusion matrix figures …")
    plot_cm(
        cm_dt, labels,
        f"Matriks Konfusi Decision Tree — Set Uji CWRU (10 kelas)\n"
        f"Akurasi {acc_dt*100:.1f}% ({int(np.diag(cm_dt).sum())}/750 sampel)",
        "Blues", OUT_DIR / "lampF_cm_dt.png",
    )
    plot_cm(
        cm_rf, labels,
        f"Matriks Konfusi Random Forest — Set Uji CWRU (10 kelas)\n"
        f"Akurasi {acc_rf*100:.1f}% ({int(np.diag(cm_rf).sum())}/750 sampel)",
        "Greens", OUT_DIR / "lampF_cm_rf.png",
    )
    plot_cm(
        cm_xgb, labels,
        f"Matriks Konfusi XGBoost — Set Uji CWRU (10 kelas)\n"
        f"Akurasi {acc_xgb*100:.1f}% ({int(np.diag(cm_xgb).sum())}/750 sampel)",
        "Purples", OUT_DIR / "lampF_cm_xgb.png",
    )

    print(f"\nDone.  DT={acc_dt*100:.1f}%  RF={acc_rf*100:.1f}%  XGB={acc_xgb*100:.1f}%")


if __name__ == "__main__":
    main()
