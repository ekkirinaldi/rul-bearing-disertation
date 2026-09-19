"""Generate FSM full-resolution heatmaps for Lampiran G.

Trains WDCNN variant A (with BatchNorm) and variant C (without BatchNorm)
on CWRU data, then computes SHAP DeepExplainer values and plots the
10 × 2048 FSM heatmaps.

Outputs:
    research/notebooks/output/lampG_fsm_varA.png   (G.1 — with BatchNorm)
    research/notebooks/output/lampG_fsm_varC.png   (G.2 — without BatchNorm)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import shap

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

CWRU_DIR = Path(__file__).resolve().parents[1] / "data-bearing" / "cwru"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

SEG_LEN = 2048
N_PER_CLASS = 230
LOAD = "1"
EPOCHS = 40
BATCH = 64
LR = 0.001

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

CLASS_NAMES = [
    "Ball_007_1", "Ball_014_1", "Ball_021_1",
    "IR_007_1", "IR_014_1", "IR_021_1",
    "Normal_1",
    "OR_007_6_1", "OR_014_6_1", "OR_021_6_1",
]


# ── WDCNN model ────────────────────────────────────────────────────────────

class WDCNN(nn.Module):
    """WDCNN from Zhang et al. (2017), Sensors."""
    def __init__(self, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn

        def _conv_block(in_ch, out_ch, kernel, stride, padding):
            layers: list[nn.Module] = [nn.Conv1d(in_ch, out_ch, kernel, stride, padding)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers += [nn.ReLU(), nn.MaxPool1d(2, 2)]
            return nn.Sequential(*layers)

        self.conv1 = _conv_block(1, 16, 64, 16, 24)
        self.conv2 = _conv_block(16, 32, 3, 1, 1)
        self.conv3 = _conv_block(32, 64, 3, 1, 1)
        self.conv4 = _conv_block(64, 64, 3, 1, 1)
        self.conv5 = _conv_block(64, 64, 3, 1, 1)

        # Compute fc input size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, SEG_LEN)
            for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
                dummy = layer(dummy)
            fc_in = dummy.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes),
        )

    def forward(self, x):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            x = layer(x)
        return self.fc(x)


# ── Data loading ───────────────────────────────────────────────────────────

def load_dataset():
    X_list, y_list = [], []
    for label in CLASS_NAMES:
        fname = FILE_MAP[label]
        mat = sio.loadmat(str(CWRU_DIR / fname))
        de_key = next(k for k in mat if k.endswith("_DE_time"))
        sig = mat[de_key].flatten()
        segs = []
        for s in range(0, len(sig) - SEG_LEN + 1, SEG_LEN):
            segs.append(sig[s: s + SEG_LEN])
            if len(segs) >= N_PER_CLASS:
                break
        X_list.append(np.stack(segs).astype(np.float32))
        y_list.extend([label] * len(segs))
        print(f"  {label}: {len(segs)} segments")
    X = np.concatenate(X_list, axis=0)  # (N, 2048)
    y = np.array(y_list)
    return X, y


def make_loaders(X, y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=750, stratify=y_enc, random_state=RANDOM_SEED
    )
    # Normalise
    mu = X_train.mean()
    sd = X_train.std() + 1e-8
    X_train = (X_train - mu) / sd
    X_test = (X_test - mu) / sd

    # Add channel dim → (N, 1, 2048)
    X_tr_t = torch.from_numpy(X_train[:, None, :])
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.from_numpy(X_test[:, None, :])
    y_te_t = torch.tensor(y_test, dtype=torch.long)

    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=BATCH)
    return tr_loader, te_loader, X_tr_t, X_te_t, y_te_t, le


# ── Training ───────────────────────────────────────────────────────────────

def train_model(model, tr_loader, te_loader, device):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS):
        model.train()
        for bx, by in tr_loader:
            opt.zero_grad()
            loss = criterion(model(bx.to(device)), by.to(device))
            loss.backward()
            opt.step()
        sched.step()
        # eval
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for bx, by in te_loader:
                preds = model(bx.to(device)).argmax(1).cpu()
                correct += (preds == by).sum().item()
                tot += len(by)
        acc = correct / tot
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}  acc={acc*100:.1f}%")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"    Best acc: {best_acc*100:.2f}%")
    return model


# ── FSM computation ────────────────────────────────────────────────────────

def compute_fsm(model, X_bg, X_te, y_te, class_names, device, n_bg_per_class=20, n_te=100):
    """Compute mean |SHAP| FSM for each class."""
    model_cpu = model.to("cpu").eval()
    nc = len(class_names)

    # Background: stratified sample
    bg_idx = []
    for c in range(nc):
        mask = (y_te.numpy() == c)
        pool = np.where(mask)[0]
        sel = np.random.choice(pool, size=min(n_bg_per_class, len(pool)), replace=False)
        bg_idx.extend(sel.tolist())
    background = X_te[bg_idx].cpu()

    # Test sample: n_te evenly across classes
    te_idx = []
    for c in range(nc):
        mask = (y_te.numpy() == c)
        pool = np.where(mask)[0]
        n = min(n_te // nc, len(pool))
        te_idx.extend(np.random.choice(pool, size=n, replace=False).tolist())
    test_samp = X_te[te_idx].cpu()
    y_te_samp = y_te[te_idx].numpy()

    print(f"    SHAP background: {len(bg_idx)}  test samples: {len(te_idx)}")

    explainer = shap.DeepExplainer(model_cpu, background)
    sv_raw = explainer.shap_values(test_samp)

    # sv_raw: list of (N, 1, 2048) or tensor
    if isinstance(sv_raw, list):
        sv = [np.array(s).squeeze(1) for s in sv_raw]   # list of (N, 2048)
    else:
        sv = np.array(sv_raw)
        if sv.ndim == 4:  # (N, 1, 2048, nc) or similar
            sv = [sv[:, 0, :, c] for c in range(sv.shape[-1])]
        elif sv.ndim == 3 and sv.shape[-1] == nc:
            sv = [sv[:, :, c] for c in range(nc)]
        else:
            sv = [sv[:, 0, :]]

    # FSM: mean |SHAP| per class using samples OF that class
    fsm = np.zeros((nc, 2048))
    for c in range(nc):
        sv_c = sv[c] if c < len(sv) else sv[0]   # (N, 2048)
        mask = y_te_samp == c
        if mask.sum() > 0:
            fsm[c] = np.mean(np.abs(sv_c[mask]), axis=0)
        else:
            fsm[c] = np.mean(np.abs(sv_c), axis=0)
    return fsm


def plot_fsm(fsm, class_names, title, path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(12, 6))
    # Normalise each row independently for visibility
    fsm_norm = fsm / (fsm.max(axis=1, keepdims=True) + 1e-10)
    im = ax.imshow(fsm_norm, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_yticks(range(len(class_names)))
    labels = [n.replace("_1", "").replace("_6", "") for n in class_names]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Posisi sinyal (indeks sampel)", fontsize=11)
    ax.set_ylabel("Jenis kerusakan", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="Mean |SHAP| ternormalisasi (per kelas)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cpu")
    print("Loading CWRU data …")
    X, y = load_dataset()
    print(f"Dataset: {X.shape}")

    tr_loader, te_loader, X_tr_t, X_te_t, y_te_t, le = make_loaders(X, y)
    class_names_enc = [CLASS_NAMES[i] for i in range(10)]  # already sorted by LabelEncoder

    # ---- Variant A (with BatchNorm) ----
    print("\nTraining WDCNN variant A (with BatchNorm) …")
    model_A = WDCNN(use_bn=True)
    model_A = train_model(model_A, tr_loader, te_loader, device)
    print("  Computing FSM variant A …")
    fsm_A = compute_fsm(model_A, X_tr_t, X_te_t, y_te_t, class_names_enc, device)
    plot_fsm(
        fsm_A, class_names_enc,
        "FSM Absolute Full-Resolution — Varian A (dengan BatchNorm)\nMean |SHAP DeepExplainer| per kelas, PHM2012 CWRU (2048 posisi/kelas)",
        OUT_DIR / "lampG_fsm_varA.png",
    )

    # ---- Variant C (without BatchNorm) ----
    print("\nTraining WDCNN variant C (without BatchNorm) …")
    model_C = WDCNN(use_bn=False)
    model_C = train_model(model_C, tr_loader, te_loader, device)
    print("  Computing FSM variant C …")
    fsm_C = compute_fsm(model_C, X_tr_t, X_te_t, y_te_t, class_names_enc, device)
    plot_fsm(
        fsm_C, class_names_enc,
        "FSM Absolute Full-Resolution — Varian C (tanpa BatchNorm)\nMean |SHAP DeepExplainer| per kelas, PHM2012 CWRU (2048 posisi/kelas)",
        OUT_DIR / "lampG_fsm_varC.png",
        cmap="plasma",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
