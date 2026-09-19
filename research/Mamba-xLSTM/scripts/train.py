"""Unified training entry point with per-step logging and figure export.

Each phase / step of the pipeline is announced (and timed) through
``mxlstm.utils.run_logger.RunLogger``. The same logger persists everything
to ``<run_dir>/logs/run.log`` and ``<run_dir>/logs/events.jsonl``. After
training, dataset / HI / training-curve / per-bearing prediction figures
are written under ``<run_dir>/figures/`` so the dissertation report can
embed them.

Example::

    python scripts/train.py \
        --data configs/data/phm2012.yaml \
        --model configs/model/mamba_xlstm_net.yaml \
        --train configs/train/default.yaml \
        --seed 42 --run-id mamba_xlstm_phm2012_s42
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import CSVLogger

_PKG = Path(__file__).resolve().parents[1] / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mxlstm.compute import get_accelerator, get_compute_profile
from mxlstm.data.datamodule import RULDataModule
from mxlstm.models.baseline_xlstm_transformer import XLSTMTransformer
from mxlstm.models.vanilla_xlstm_rul import VanillaXLSTMRUL
from mxlstm.models.diffusion_rul import DiffusionRUL
from mxlstm.models.liquid_wave_rul import LiquidWaveRUL
from mxlstm.models.mamba_blocks import MambaBackendChoice
from mxlstm.models.mamba_rul import MambaRUL
from mxlstm.models.mamba_xlstm_net import MambaXLSTMConfig, MambaXLSTMNet
from mxlstm.models.patch_tst_rul import PatchTSTRUL
from mxlstm.models.nbeats_rul import NBeatsRUL
from mxlstm.models.physics_nbeats_rul import NBeatsXLSTMRUL, PhysicsNBeatsRUL
from mxlstm.models.phase_moe_xlstm_rul import PhaseMoExLSTMRUL
from mxlstm.models.sparse_gate_tcn_rul import SparseGateTCNRUL
from mxlstm.reporting.figures import (
    plot_dataset_overview,
    plot_hi_heatmap,
    plot_hi_trace,
    plot_per_bearing_predictions,
    plot_per_bearing_residuals,
    plot_residual_histogram,
    plot_residuals_overall,
    plot_rul_labels_for_bearings,
    plot_step_timings,
    plot_training_curves,
)
from mxlstm.training.callbacks import make_callbacks
from mxlstm.training.lit_module import RULLitModule
from mxlstm.utils.config import load_configs, save_yaml
from mxlstm.utils.io import make_run_dir
from mxlstm.utils.run_logger import RunLogger
from mxlstm.utils.seed import seed_everything


def _run_id_from_run_dir(run_dir: Path, fallback: str) -> str:
    """Parse ``YYYYMMDD_HHMMSS_<run_id>`` folder names from ``make_run_dir``."""
    stem = run_dir.name
    parts = stem.split("_", 2)
    if len(parts) == 3 and len(parts[0]) == 8 and parts[0].isdigit() and len(parts[1]) == 6 and parts[1].isdigit():
        return parts[2]
    return fallback


def _checkpoint_epoch(ckpt_path: Path) -> int | None:
    """Best-effort epoch index from a Lightning checkpoint (for user hints)."""
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if isinstance(obj, dict):
        if "epoch" in obj:
            return int(obj["epoch"])
        loops = obj.get("loops") or {}
        fit_loop = loops.get("fit_loop") or {}
        ep = (fit_loop.get("epoch_progress") or {}).get("current") or {}
        if "completed" in ep:
            return int(ep["completed"])
    return None


def _build_model(cfg: DictConfig, n_features: int, context_length: int) -> torch.nn.Module:
    name = str(cfg.model.name).lower()
    base = str(cfg.model.get("base", "")).lower() or name

    if "xlstm_transformer" in (name, base):
        return XLSTMTransformer(
            n_features=n_features,
            d_model=int(cfg.model.get("d_model", 32)),
            n_heads=int(cfg.model.get("n_heads", 4)),
            encoder_layers=int(cfg.model.get("encoder_layers", 1)),
            decoder_layers=int(cfg.model.get("decoder_layers", 1)),
            xlstm_blocks=int(cfg.model.get("xlstm_blocks", 2)),
            slstm_positions=list(cfg.model.get("slstm_positions", [1])),
            head_hidden=int(cfg.model.get("head_hidden", 32)),
            head_pool=str(cfg.model.get("head_pool", "flatten")),
            context_length=int(context_length),
            dropout=float(cfg.model.get("dropout", 0.1)),
            ff_dim=int(cfg.model.get("ff_dim", 64)),
            force_fallback=bool(cfg.model.get("xlstm_force_fallback", False)),
        )

    if "vanilla_xlstm_rul" in (name, base):
        return VanillaXLSTMRUL(
            n_features=n_features,
            context_length=int(context_length),
            d_model=int(cfg.model.get("d_model", 64)),
            num_blocks=int(cfg.model.get("num_blocks", 3)),
            slstm_positions=list(cfg.model.get("slstm_positions", [1])),
            n_heads=int(cfg.model.get("n_heads", 4)),
            head_hidden=int(cfg.model.get("head_hidden", 64)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            xlstm_force_fallback=bool(cfg.model.get("xlstm_force_fallback", False)),
        )

    if "mamba_xlstm_net" in (name, base):
        mxc = MambaXLSTMConfig(
            n_features=n_features,
            d_model=int(cfg.model.get("d_model", 128)),
            context_length=int(context_length),
            xlstm_blocks=int(cfg.model.get("xlstm_blocks", 3)),
            slstm_positions=list(cfg.model.get("slstm_positions", [1])),
            xlstm_heads=int(cfg.model.get("xlstm_heads", 4)),
            xlstm_force_fallback=bool(cfg.model.get("xlstm_force_fallback", False)),
            use_vanilla_lstm=bool(cfg.model.get("use_vanilla_lstm", False)),
            mamba_blocks=int(cfg.model.get("mamba_blocks", 2)),
            mamba_d_state=int(cfg.model.get("mamba_d_state", 128)),
            mamba_d_conv=int(cfg.model.get("mamba_d_conv", 4)),
            mamba_expand=int(cfg.model.get("mamba_expand", 2)),
            mamba_headdim=int(cfg.model.get("mamba_headdim", 64)),
            mamba_n_groups=int(cfg.model.get("mamba_n_groups", 1)),
            mamba_rope_fraction=float(cfg.model.get("mamba_rope_fraction", 0.5)),
            mamba_is_mimo=bool(cfg.model.get("mamba_is_mimo", False)),
            mamba_mimo_rank=int(cfg.model.get("mamba_mimo_rank", 4)),
            mamba_chunk_size=int(cfg.model.get("mamba_chunk_size", 64)),
            mamba_is_outproj_norm=bool(cfg.model.get("mamba_is_outproj_norm", False)),
            mamba_bidirectional=bool(cfg.model.get("mamba_bidirectional", True)),
            mamba_backend=str(cfg.model.get("mamba_backend", "auto")),
            fusion=str(cfg.model.get("fusion", "gated")),
            head_hidden=int(cfg.model.get("head_hidden", 64)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )
        return MambaXLSTMNet(mxc)

    if "liquid_wave_rul" in (name, base) or "liquidwave_rul" in (name, base):
        return LiquidWaveRUL(
            n_features=n_features,
            n_bands=int(cfg.model.get("n_bands", 6)),
            n_band_feats=int(cfg.model.get("n_band_feats", 4)),
            hidden_dim=int(cfg.model.get("hidden_dim", 32)),
            attn_heads=int(cfg.model.get("attn_heads", 4)),
            ltc_unfolds=int(cfg.model.get("ltc_unfolds", 4)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )

    if "nbeats_rul" in (name, base) or "n_beats_rul" in (name, base):
        return NBeatsRUL(
            context_length=int(context_length),
            n_features=n_features,
            hidden_dim=int(cfg.model.get("hidden_dim", 128)),
            trend_blocks=int(cfg.model.get("trend_blocks", 2)),
            wear_blocks=int(cfg.model.get("wear_blocks", 2)),
            shock_blocks=int(cfg.model.get("shock_blocks", 2)),
            poly_degree=int(cfg.model.get("poly_degree", 3)),
            n_harmonics=int(cfg.model.get("n_harmonics", 6)),
            n_shock_basis=int(cfg.model.get("n_shock_basis", 16)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )

    if "physics_nbeats_rul" in name or "physics_nbeats" in base:
        _ek = cfg.model.get("encoder_kernel_size", None)
        _encoder_kernel = None if _ek is None else int(_ek)
        return PhysicsNBeatsRUL(
            context_length=int(context_length),
            n_features=n_features,
            dataset=str(cfg.data.dataset),
            hidden_dim=int(cfg.model.get("hidden_dim", 96)),
            trend_blocks=int(cfg.model.get("trend_blocks", 2)),
            wear_blocks=int(cfg.model.get("wear_blocks", 2)),
            shock_blocks=int(cfg.model.get("shock_blocks", 2)),
            poly_degree=int(cfg.model.get("poly_degree", 4)),
            n_shock_basis=int(cfg.model.get("n_shock_basis", 14)),
            film_num_embeddings=int(cfg.model.get("film_num_embeddings", 4)),
            kurt_index=int(cfg.model.get("kurt_index", 8)),
            dropout=float(cfg.model.get("dropout", 0.15)),
            lambda_wear_sparse=float(cfg.model.get("lambda_wear_sparse", 1e-3)),
            use_xlstm_front=False,
            encoder_kernel_size=_encoder_kernel,
            monotone_trend=bool(cfg.model.get("monotone_trend", True)),
            hybrid_learned_shock=bool(cfg.model.get("hybrid_learned_shock", False)),
            n_learned_shock_basis=int(cfg.model.get("n_learned_shock_basis", 8)),
        )

    if "nbeats_xlstm_rul" in (name, base):
        _ek = cfg.model.get("encoder_kernel_size", None)
        _encoder_kernel = None if _ek is None else int(_ek)
        return NBeatsXLSTMRUL(
            context_length=int(context_length),
            n_features=n_features,
            dataset=str(cfg.data.dataset),
            hidden_dim=int(cfg.model.get("hidden_dim", 96)),
            trend_blocks=int(cfg.model.get("trend_blocks", 2)),
            wear_blocks=int(cfg.model.get("wear_blocks", 2)),
            shock_blocks=int(cfg.model.get("shock_blocks", 2)),
            poly_degree=int(cfg.model.get("poly_degree", 4)),
            n_shock_basis=int(cfg.model.get("n_shock_basis", 14)),
            film_num_embeddings=int(cfg.model.get("film_num_embeddings", 4)),
            kurt_index=int(cfg.model.get("kurt_index", 8)),
            dropout=float(cfg.model.get("dropout", 0.15)),
            lambda_wear_sparse=float(cfg.model.get("lambda_wear_sparse", 1e-3)),
            xlstm_d_model=int(cfg.model.get("xlstm_d_model", 64)),
            xlstm_heads=int(cfg.model.get("xlstm_heads", 4)),
            xlstm_num_blocks=int(cfg.model.get("xlstm_num_blocks", 2)),
            xlstm_inter_dropout=float(cfg.model.get("xlstm_inter_dropout", 0.0)),
            encoder_kernel_size=_encoder_kernel,
            monotone_trend=bool(cfg.model.get("monotone_trend", True)),
            hybrid_learned_shock=bool(cfg.model.get("hybrid_learned_shock", False)),
            n_learned_shock_basis=int(cfg.model.get("n_learned_shock_basis", 8)),
        )

    if "diffusion_rul" in (name, base):
        return DiffusionRUL(
            n_features=n_features,
            context_length=int(context_length),
            d_model=int(cfg.model.get("d_model", 96)),
            n_heads=int(cfg.model.get("n_heads", 4)),
            n_layers=int(cfg.model.get("n_layers", 2)),
            n_noise_levels=int(cfg.model.get("n_noise_levels", 4)),
            max_noise_std=float(cfg.model.get("max_noise_std", 0.35)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            inference_noise_scale=float(cfg.model.get("inference_noise_scale", 0.5)),
            n_inference_samples=int(cfg.model.get("n_inference_samples", 4)),
            lambda_denoise=float(cfg.model.get("lambda_denoise", 0.1)),
            film_num_embeddings=int(cfg.model.get("film_num_embeddings", 0)),
            pool=str(cfg.model.get("pool", "mean_last")),
        )

    if "sparse_gate_tcn_rul" in (name, base) or "sparse_gate_tcn" in name or "sparse_gate_tcn" in base:
        ch = cfg.model.get("tcn_channels", [64, 64, 128, 128])
        tcn_channels = tuple(int(c) for c in list(ch))
        return SparseGateTCNRUL(
            n_features=n_features,
            tcn_channels=tcn_channels,
            tcn_kernel=int(cfg.model.get("tcn_kernel", 3)),
            gate_hidden=int(cfg.model.get("gate_hidden", 32)),
            gate_context=int(cfg.model.get("gate_context", 5)),
            attn_d_model=int(cfg.model.get("attn_d_model", 32)),
            attn_heads=int(cfg.model.get("attn_heads", 4)),
            head_hidden=int(cfg.model.get("head_hidden", 64)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            lambda_sparse=float(cfg.model.get("lambda_sparse", 1e-3)),
            lambda_entropy=float(cfg.model.get("lambda_entropy", 1e-3)),
        )

    if "phase_moe_xlstm_rul" in (name, base) or "phase_moe" in name or "phase_moe" in base:
        return PhaseMoExLSTMRUL(
            n_features=n_features,
            d_model=int(cfg.model.get("d_model", 128)),
            n_phases=int(cfg.model.get("n_phases", 3)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            hi_index=int(cfg.model.get("hi_index", 0)),
            kurt_index=int(cfg.model.get("kurt_index", 2)),
            healthy_window=int(cfg.model.get("healthy_window", 8)),
            lambda_mono=float(cfg.model.get("lambda_mono", 0.1)),
            lambda_paris=float(cfg.model.get("lambda_paris", 0.05)),
            lambda_phase=float(cfg.model.get("lambda_phase", 0.1)),
            healthy_threshold=float(cfg.model.get("healthy_threshold", 0.7)),
            prefailure_threshold=float(cfg.model.get("prefailure_threshold", 0.3)),
        )

    if "patch_tst_rul" in (name, base) or "patchtst_rul" in (name, base):
        _ff = cfg.model.get("ffn_dim", None)
        ffn_dim = int(_ff) if _ff is not None else None
        return PatchTSTRUL(
            n_features=n_features,
            context_length=int(context_length),
            d_model=int(cfg.model.get("d_model", 96)),
            n_heads=int(cfg.model.get("n_heads", 4)),
            n_encoder_layers=int(cfg.model.get("n_encoder_layers", 2)),
            patch_len=int(cfg.model.get("patch_len", 16)),
            stride=int(cfg.model.get("stride", 8)),
            dropout=float(cfg.model.get("dropout", 0.1)),
            ffn_dim=ffn_dim,
        )

    if "mamba_rul" in (name, base):
        _mb = str(cfg.model.get("mamba_backend", "auto")).lower()
        backend: MambaBackendChoice = (
            _mb if _mb in ("auto", "mamba_ssm", "mambapy", "vanilla") else "auto"
        )
        return MambaRUL(
            n_features=n_features,
            context_length=int(context_length),
            d_model=int(cfg.model.get("d_model", 96)),
            mamba_blocks=int(cfg.model.get("mamba_blocks", 3)),
            mamba_d_state=int(cfg.model.get("mamba_d_state", 64)),
            mamba_d_conv=int(cfg.model.get("mamba_d_conv", 4)),
            mamba_expand=int(cfg.model.get("mamba_expand", 2)),
            mamba_headdim=int(cfg.model.get("mamba_headdim", 32)),
            mamba_n_groups=int(cfg.model.get("mamba_n_groups", 1)),
            mamba_rope_fraction=float(cfg.model.get("mamba_rope_fraction", 0.5)),
            mamba_is_mimo=bool(cfg.model.get("mamba_is_mimo", False)),
            mamba_mimo_rank=int(cfg.model.get("mamba_mimo_rank", 4)),
            mamba_chunk_size=int(cfg.model.get("mamba_chunk_size", 64)),
            mamba_is_outproj_norm=bool(cfg.model.get("mamba_is_outproj_norm", False)),
            mamba_bidirectional=bool(cfg.model.get("mamba_bidirectional", True)),
            mamba_backend=backend,
            head_hidden=int(cfg.model.get("head_hidden", 64)),
            dropout=float(cfg.model.get("dropout", 0.1)),
        )

    raise ValueError(f"Unknown model name: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--ablation", type=Path, default=None,
                   help="Optional ablation YAML overlaid on top of model+data configs")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument(
        "--ckpt-path",
        type=Path,
        default=None,
        help="Resume Lightning training from this checkpoint (e.g. checkpoints/last.ckpt). "
        "Set --max-epochs to the new total epoch budget (must exceed the epoch stored in the ckpt). "
        "If --run-dir is omitted, it defaults to the run folder parent of checkpoints/.",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use this existing results/runs/<timestamp>_<run_id>/ directory (logs, checkpoints). "
        "Required when resuming without --ckpt-path if you need a fixed folder.",
    )
    p.add_argument("--fast-dev-run", action="store_true",
                   help="Lightning fast_dev_run: 1 batch each phase, no checkpoint.")
    p.add_argument("--limit-train-batches", type=float, default=None)
    p.add_argument("--limit-val-batches", type=float, default=None)
    p.add_argument("--no-figures", action="store_true",
                   help="Skip post-training figure generation (faster smoke runs).")
    p.add_argument(
        "--config-overlay",
        type=Path,
        default=None,
        help="Optional YAML merged last (after ablation): tune train/model keys from Optuna outputs.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Figure helpers (kept here to keep the script self-contained at orchestration
# level; actual plotting lives in mxlstm.reporting.figures).
# ---------------------------------------------------------------------------


def _make_dataset_figure(dm: RULDataModule, run_dir: Path, rl: RunLogger) -> None:
    rows = []
    for split_name, bearings in [("train", dm.train_bearings_data),
                                  ("val", dm.val_bearings_data),
                                  ("test", dm.test_bearings_data)]:
        for b in bearings:
            rows.append({
                "bearing_id": b.bearing_id,
                "split": split_name,
                "n_acquisitions": int(b.hi.shape[0]),
                "condition": int(b.condition),
            })
    if not rows:
        rl.warning("No bearings to plot in dataset overview.")
        return
    p = plot_dataset_overview(rows, run_dir / "figures" / "dataset_overview.png",
                              title=f"{dm.dataset_name.upper()} bearings used in this run")
    rl.artefact("dataset_overview", p)


def _make_hi_figures(dm: RULDataModule, run_dir: Path, rl: RunLogger) -> None:
    if not dm.train_bearings_data:
        return
    fig_dir = run_dir / "figures"
    sample = dm.train_bearings_data[0]
    feat_names = sample.feature_names or [f"f{i}" for i in range(sample.hi.shape[1])]
    hi_trace_path = plot_hi_trace(sample.bearing_id, sample.hi, feat_names,
                                  fig_dir / "hi_trace.png")
    rl.artefact("hi_trace", hi_trace_path)
    hi_heatmap_path = plot_hi_heatmap(sample.bearing_id, sample.hi, feat_names,
                                      fig_dir / "hi_heatmap.png")
    rl.artefact("hi_heatmap", hi_heatmap_path)

    items = [(b.bearing_id, b.rul) for b in dm.train_bearings_data[:6]]
    label_paths = plot_rul_labels_for_bearings(items, fig_dir)
    for p in label_paths:
        rl.artefact(f"rul_label_{p.stem.replace('rul_label_', '')}", p)
    rl.metric("n_rul_label_plots", len(label_paths))


def _make_training_curves(csv_log_dir: Path, run_dir: Path, rl: RunLogger) -> None:
    metrics_csv = csv_log_dir / "metrics.csv"
    if not metrics_csv.exists():
        rl.warning(f"No metrics.csv found at {metrics_csv}; skipping training curves.")
        return
    paths = plot_training_curves(metrics_csv, run_dir / "figures")
    if not paths:
        rl.warning("metrics.csv had no plottable training-curve keys.")
        return
    for p in paths:
        rl.artefact(p.stem, p)
    rl.metric("n_training_curve_plots", len(paths))


def _make_prediction_figures(per_bearing: dict, run_dir: Path, rl: RunLogger) -> None:
    if not per_bearing:
        rl.warning("No per-bearing predictions captured.")
        return
    fig_dir = run_dir / "figures"
    pred_paths = plot_per_bearing_predictions(per_bearing, fig_dir)
    rl.metric("n_prediction_plots", len(pred_paths))
    residual_paths = plot_per_bearing_residuals(per_bearing, fig_dir)
    rl.metric("n_residual_plots", len(residual_paths))
    overall = plot_residuals_overall(per_bearing, fig_dir / "residuals_overall.png")
    rl.artefact("residuals_overall", overall)
    hist = plot_residual_histogram(per_bearing, fig_dir / "residuals_histogram.png")
    rl.artefact("residuals_histogram", hist)


def _persist_test_predictions(per_bearing: dict, run_dir: Path, rl: RunLogger) -> None:
    if not per_bearing:
        rl.warning("No per-bearing predictions to persist.")
        return

    path = run_dir / "test_predictions.npz"
    np.savez_compressed(
        path,
        **{f"{bid}_t": d["t"] for bid, d in per_bearing.items()},
        **{f"{bid}_pred": d["pred"] for bid, d in per_bearing.items()},
        **{f"{bid}_y": d["y"] for bid, d in per_bearing.items()},
    )
    rl.artefact("test_predictions", path)


def main() -> None:
    args = parse_args()

    # ----- 1. Configuration ------------------------------------------------
    cfg_paths: list[Path] = [args.data, args.model, args.train]
    if args.ablation is not None:
        cfg_paths.append(args.ablation)
    if args.config_overlay is not None:
        cfg_paths.append(args.config_overlay)
    cfg = load_configs(cfg_paths)

    seed = int(args.seed if args.seed is not None else cfg.train.get("seed", 42))
    if args.max_epochs is not None:
        cfg.train.max_epochs = int(args.max_epochs)

    if args.fast_dev_run and args.ckpt_path is not None:
        raise SystemExit("--fast-dev-run and --ckpt-path are mutually exclusive.")

    ckpt_path = args.ckpt_path.resolve() if args.ckpt_path is not None else None
    if ckpt_path is not None and not ckpt_path.is_file():
        raise SystemExit(f"--ckpt-path does not exist or is not a file: {ckpt_path}")

    # ----- 2. Run directory + logger --------------------------------------
    run_id = args.run_id or f"{cfg.model.name}_{cfg.data.dataset}_s{seed}"
    run_dir: Path
    if args.run_dir is not None:
        run_dir = args.run_dir.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--run-dir is not a directory: {run_dir}")
        run_id = _run_id_from_run_dir(run_dir, run_id)
    elif ckpt_path is not None:
        if ckpt_path.parent.name != "checkpoints":
            raise SystemExit(
                f"--ckpt-path must live under .../checkpoints/<file>.ckpt (got {ckpt_path})"
            )
        run_dir = ckpt_path.parent.parent.resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"Derived run directory from ckpt is not a directory: {run_dir}")
        run_id = _run_id_from_run_dir(run_dir, run_id)
    else:
        run_dir = make_run_dir(cfg.train.results_root, run_id=run_id)

    if ckpt_path is not None:
        ck_ep = _checkpoint_epoch(ckpt_path)
        if ck_ep is not None and int(cfg.train.max_epochs) <= ck_ep:
            raise SystemExit(
                f"--max-epochs ({cfg.train.max_epochs}) must be greater than checkpoint epoch "
                f"({ck_ep}) so training can advance (Lightning uses max_epochs as total budget)."
            )

    with RunLogger(run_dir, run_id=run_id) as rl:
        rl.info(f"Run id: {run_id}")
        rl.info(f"Configs: {[str(p) for p in cfg_paths]}")
        save_yaml(cfg, run_dir / "config.yaml")
        rl.artefact("config", run_dir / "config.yaml")
        if ckpt_path is not None:
            rl.info(f"Resuming from checkpoint: {ckpt_path}")

        with rl.phase("Setup"):
            with rl.step("Seed everything"):
                seed_everything(seed, deterministic=False)
                rl.metric("seed", seed)
            with rl.step("Detect compute profile"):
                profile = get_compute_profile()
                rl.metric("device", profile.device)
                rl.metric("accelerator", profile.accelerator)
                rl.metric("mamba_backend", profile.mamba_backend)
                rl.metric("precision", profile.precision)
                if profile.accelerator == "cuda":
                    torch.set_float32_matmul_precision("high")
                    rl.metric("float32_matmul_precision", "high")

        with rl.phase("Data preparation"):
            with rl.step("Build DataModule"):
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
                    num_workers=int(cfg.data.num_workers),
                    cache_dir=cfg.data.get("cache_dir", None),
                )
                rl.metric("dataset", cfg.data.dataset)
                rl.metric("train_bearings", list(cfg.data.train_bearings))
                rl.metric("val_bearings", list(cfg.data.val_bearings))
                rl.metric("test_bearings", list(cfg.data.test_bearings))

            with rl.step("Load + extract HI + fit scaler"):
                dm.setup()
                n_features = dm.n_features
                rl.metric("n_features", int(n_features))
                rl.metric("train_windows", len(dm._train_ds))
                rl.metric("val_windows", len(dm._val_ds))
                rl.metric("test_windows", len(dm._test_ds))

            with rl.step("Save HI scaler"):
                dm.save_scaler(run_dir / "hi_scaler.json")
                rl.artefact("hi_scaler", run_dir / "hi_scaler.json")

            if not args.no_figures:
                with rl.step("Plot dataset / HI / RUL figures"):
                    _make_dataset_figure(dm, run_dir, rl)
                    _make_hi_figures(dm, run_dir, rl)

        with rl.phase("Model construction"):
            with rl.step(f"Build model ({cfg.model.name})"):
                model = _build_model(cfg, n_features=n_features,
                                     context_length=int(cfg.data.window_length))
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                rl.metric("n_params", int(n_params))

            with rl.step("Wrap in LightningModule"):
                lit = RULLitModule(
                    model=model,
                    lr=float(cfg.train.lr),
                    weight_decay=float(cfg.train.weight_decay),
                    warmup_epochs=int(cfg.train.warmup_epochs),
                    max_epochs=int(cfg.train.max_epochs),
                    monotonicity_weight=float(cfg.train.monotonicity_weight),
                    scheduler=str(cfg.train.scheduler),
                    model_specific_loss=bool(cfg.train.get("model_specific_loss", False))
                    or bool(cfg.model.get("model_specific_loss", False)),
                    xlstm_lr_mult=float(cfg.train.get("xlstm_lr_mult", 1.0)),
                    freeze_xlstm_epochs=int(cfg.train.get("freeze_xlstm_epochs", 0)),
                )

        with rl.phase("Training"):
            with rl.step("Build trainer + callbacks"):
                callbacks = make_callbacks(
                    checkpoint_dir=run_dir / "checkpoints",
                    monitor=str(cfg.train.get("checkpoint_monitor", "val/rmse")),
                    patience=int(cfg.train.early_stopping_patience),
                )
                accel = get_accelerator()
                precision = str(cfg.train.get("precision", profile.precision))
                csv_logger = CSVLogger(save_dir=str(run_dir), name="csv_logs")
                trainer = pl.Trainer(
                    max_epochs=int(cfg.train.max_epochs),
                    accelerator=accel,
                    devices=1,
                    precision=precision if accel != "mps" else "32-true",
                    gradient_clip_val=float(cfg.train.gradient_clip_val),
                    callbacks=callbacks,
                    log_every_n_steps=10,
                    default_root_dir=str(run_dir),
                    fast_dev_run=bool(args.fast_dev_run),
                    limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
                    limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
                    logger=csv_logger,
                )
                rl.metric("max_epochs", int(cfg.train.max_epochs))
                rl.metric("batch_size", int(cfg.data.batch_size))

            with rl.step("Fit"):
                t0 = time.time()
                if ckpt_path is not None:
                    trainer.fit(lit, datamodule=dm, ckpt_path=str(ckpt_path))
                else:
                    trainer.fit(lit, datamodule=dm)
                fit_seconds = time.time() - t0
                rl.metric("fit_seconds", round(fit_seconds, 2))
                best_ckpt = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
                if best_ckpt:
                    rl.artefact("best_checkpoint", best_ckpt)

        with rl.phase("Evaluation"):
            with rl.step("Test on held-out bearings"):
                if args.fast_dev_run:
                    test_out = trainer.test(lit, datamodule=dm, verbose=False)
                else:
                    test_out = trainer.test(lit, datamodule=dm, ckpt_path="best", verbose=False)
                test_metrics = test_out[0] if test_out else {}
                for k, v in test_metrics.items():
                    rl.metric(k, float(v) if isinstance(v, (int, float)) else v)

            with rl.step("Persist per-bearing test predictions"):
                per_bearing = lit._gather_per_bearing(lit._test_outputs)
                _persist_test_predictions(per_bearing, run_dir, rl)

            if not args.no_figures:
                with rl.step("Plot training curves"):
                    csv_log_dir = Path(csv_logger.log_dir)
                    _make_training_curves(csv_log_dir, run_dir, rl)

                with rl.step("Plot per-bearing test predictions"):
                    _make_prediction_figures(per_bearing, run_dir, rl)

        with rl.phase("Reporting artefacts"):
            with rl.step("Write summary.json"):
                summary = {
                    "run_id": run_id,
                    "seed": seed,
                    "n_params": int(n_params),
                    "fit_seconds": float(fit_seconds),
                    "test_metrics": {k: float(v) if isinstance(v, (int, float)) else v
                                      for k, v in test_metrics.items()},
                    "best_checkpoint": best_ckpt,
                    "dataset": cfg.data.dataset,
                    "model_name": cfg.model.name,
                }
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
                rl.artefact("summary", run_dir / "summary.json")

        rl.info(f"Run complete. Artefacts in {run_dir}")

    # ----- after RunLogger closes (so logs/summary.json exists) ----------
    timings_summary_path = run_dir / "logs" / "summary.json"
    if not args.no_figures and timings_summary_path.exists():
        try:
            payload = json.loads(timings_summary_path.read_text() or "{}")
            if payload.get("timings"):
                plot_step_timings(payload["timings"], run_dir / "figures" / "step_timings.png")
        except json.JSONDecodeError:
            pass


if __name__ == "__main__":
    main()
