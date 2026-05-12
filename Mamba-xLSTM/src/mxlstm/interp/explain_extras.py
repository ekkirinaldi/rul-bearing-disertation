"""Model-specific explanations (figures) written beside SHAP/Captum artefacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mxlstm.compute import get_device
from mxlstm.data.datamodule import RULDataModule
from mxlstm.models.baseline_xlstm_transformer import XLSTMTransformer
from mxlstm.models.diffusion_rul import DiffusionRUL
from mxlstm.models.liquid_wave_rul import LiquidWaveRUL
from mxlstm.models.mamba_xlstm_net import MambaXLSTMNet
from mxlstm.models.nbeats_rul import NBeatsRUL
from mxlstm.models.physics_nbeats_rul import PhysicsNBeatsRUL
from mxlstm.training.lit_module import RULLitModule
from omegaconf import OmegaConf


def resolve_checkpoint(run_dir: Path) -> Path:
    summ_path = run_dir / "summary.json"
    best: str | None = None
    if summ_path.exists():
        try:
            best = json.loads(summ_path.read_text()).get("best_checkpoint") or ""
        except json.JSONDecodeError:
            best = ""
    if best:
        p = Path(str(best))
        if p.is_file():
            return p
    cks = list((run_dir / "checkpoints").glob("*.ckpt"))
    if not cks:
        raise FileNotFoundError(f"No checkpoints under {run_dir / 'checkpoints'}")
    cks_sorted = sorted(cks, key=lambda x: x.stat().st_mtime, reverse=True)
    return cks_sorted[0]


def _load_build_model():
    train_path = Path(__file__).resolve().parents[3] / "scripts" / "train.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("mxlstm_train_script_build", train_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_model


def _build_dm(cfg) -> RULDataModule:
    dm = RULDataModule(
        dataset=cfg.data.dataset,
        root=cfg.data.root,
        train_bearings=list(cfg.data.train_bearings),
        val_bearings=list(cfg.data.val_bearings),
        test_bearings=list(cfg.data.test_bearings),
        window_length=int(cfg.data.window_length),
        stride_train=int(cfg.data.stride_train),
        stride_eval=int(cfg.data.stride_eval),
        label_scheme=str(cfg.data.label_scheme),
        smoothing_alpha=float(cfg.data.smoothing_alpha),
        n_bands=int(cfg.data.n_bands),
        hi_pipeline=str(cfg.data.get("hi_pipeline", "default")),
        phm_horizontal_only=bool(cfg.data.get("phm_horizontal_only", False)),
        allow_train_val_overlap=bool(cfg.data.get("allow_train_val_overlap", False)),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.get("num_workers", 0)),
        cache_dir=cfg.data.get("cache_dir", None),
    )
    dm.setup()
    return dm


def write_extra_explanation_figures(run_dir: Path) -> Path:
    """Load ``run_dir/config.yaml``, best checkpoint, and write PNGs under ``explain/``.

    Intended to run after ``run_interpretability`` (same directory).
    """
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    device = get_device()
    ckpt = resolve_checkpoint(run_dir)
    out_dir = run_dir / "explain"
    out_dir.mkdir(parents=True, exist_ok=True)

    dm = _build_dm(cfg)
    build_model = _load_build_model()
    backbone = build_model(cfg, n_features=dm.n_features, context_length=int(cfg.data.window_length))
    lit = RULLitModule.load_from_checkpoint(str(ckpt), model=backbone, map_location=device)
    lit.eval()
    model = lit.model

    loader = dm.test_dataloader()
    batch = next(iter(loader))
    n_show = min(24, batch[0].size(0))
    x = batch[0][:n_show].to(device)
    metas = batch[3][:n_show]
    cond_ids = torch.tensor(
        [int(m.get("condition", 1)) for m in metas],
        device=device,
        dtype=torch.long,
    )

    if isinstance(model, XLSTMTransformer):
        attn = model.encoder_first_layer_self_attention_mean(x).float().cpu().numpy()
        plt.figure(figsize=(5, 4))
        plt.imshow(attn, cmap="viridis", aspect="auto")
        plt.colorbar(label="mean attention")
        plt.xlabel("Key position")
        plt.ylabel("Query position")
        plt.title("Encoder L0 self-attention (batch mean)")
        plt.tight_layout()
        plt.savefig(out_dir / "encoder_self_attn_l0_mean.png", dpi=150)
        plt.close()

    if isinstance(model, MambaXLSTMNet):
        with torch.no_grad():
            _, extra = model(x, return_hidden=True)
        gate = extra.get("gate")
        if gate is not None:
            g = gate.abs().detach().mean(dim=0).float().cpu().numpy()
            plt.figure(figsize=(6, 3))
            plt.imshow(g.T, cmap="magma", aspect="auto")
            plt.colorbar(label="|gate| mean over batch")
            plt.xlabel("Time")
            plt.ylabel("Channel index")
            plt.title("Fusion gate magnitude (mean over batch)")
            plt.tight_layout()
            plt.savefig(out_dir / "mamba_gated_fusion_gate_mean.png", dpi=150)
            plt.close()

    if isinstance(model, LiquidWaveRUL):
        ex = model.explain(x)
        bw = ex["band_weights"].float().mean(dim=(0, 1)).numpy()
        plt.figure(figsize=(5, 3))
        plt.bar(range(len(bw)), bw)
        plt.xlabel("Pseudo-band index")
        plt.ylabel("Mean attention mass")
        plt.title("LiquidWave band attention")
        plt.tight_layout()
        plt.savefig(out_dir / "liquidwave_band_weights_mean.png", dpi=150)
        plt.close()

    if isinstance(model, NBeatsRUL):
        ex = model.explain(x)
        comps = torch.stack(
            [
                ex["trend_contribution"].abs().mean(),
                ex["wear_contribution"].abs().mean(),
                ex["shock_contribution"].abs().mean(),
                ex["residual_correction"].abs().mean(),
            ],
        ).numpy()
        names = ["Trend", "Wear", "Shock", "Residual"]
        plt.figure(figsize=(5, 3))
        plt.bar(names, comps)
        plt.ylabel("Mean |contribution| (test minibatch)")
        plt.title("N-BEATS stack magnitudes")
        plt.tight_layout()
        plt.savefig(out_dir / "nbeats_component_magnitudes.png", dpi=150)
        plt.close()

    if isinstance(model, PhysicsNBeatsRUL):
        ex = model.explain(x, condition_ids=cond_ids)
        comps = torch.stack(
            [
                ex["trend_contribution"].abs().mean(),
                ex["wear_contribution"].abs().mean(),
                ex["shock_contribution"].abs().mean(),
                ex["residual_correction"].abs().mean(),
            ],
        ).numpy()
        names = ["Trend", "Wear", "Shock", "Residual"]
        plt.figure(figsize=(5, 3))
        plt.bar(names, comps)
        plt.ylabel("Mean |contribution| (test minibatch)")
        plt.title("Physics-N-BEATS stack magnitudes")
        plt.tight_layout()
        plt.savefig(out_dir / "physics_nbeats_component_magnitudes.png", dpi=150)
        plt.close()
        wear_keys = sorted(k for k in ex if k.startswith("wear_") and k.endswith("_mag"))
        if wear_keys:
            vals = torch.stack([ex[k].float().mean() for k in wear_keys]).numpy()
            labels = [k.replace("wear_", "").replace("_mag", "") for k in wear_keys]
            plt.figure(figsize=(max(6, len(labels) * 0.35), 3))
            plt.bar(labels, vals)
            plt.ylabel("Mean |theta| (batch)")
            plt.title("Physics-N-BEATS wear-theta (fault-frequency channels)")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(out_dir / "physics_nbeats_wear_theta_profile.png", dpi=150)
            plt.close()

    if isinstance(model, DiffusionRUL):
        ex = model.explain(x)
        fs0 = ex["feature_score"][0].float().cpu().numpy()
        if fs0.ndim == 2:
            vec = fs0.mean(axis=0)
        else:
            vec = fs0.flatten()
        plt.figure(figsize=(max(6, len(vec) * 0.2), 3))
        plt.bar(range(len(vec)), vec)
        plt.xlabel("HI feature index")
        plt.ylabel("Mean |score| across time")
        plt.title("Diffusion feature score profile (first window)")
        plt.tight_layout()
        plt.savefig(out_dir / "diffusion_feature_score_profile.png", dpi=150)
        plt.close()

    meta = {"checkpoint": str(ckpt)}
    (out_dir / "extras_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Architecture-specific explanations for a trained run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    rd = Path(args.run_dir)
    if not (rd / "config.yaml").exists():
        print("[explain_extras] missing config.yaml", file=sys.stderr)
        sys.exit(1)
    p = write_extra_explanation_figures(rd)
    print(f"[explain_extras] wrote figures under {p}")


if __name__ == "__main__":
    main()
