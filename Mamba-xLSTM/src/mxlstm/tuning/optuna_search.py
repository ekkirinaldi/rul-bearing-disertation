"""Optuna search for train + model knobs (per dataset and architecture).

Optimizes minimal validation RMSE over a capped training horizon. Studies are
SQLite-backed under ``results/_tuning/optuna_<dataset>_<model>.db``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

import optuna
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback

from mxlstm.compute import get_accelerator, get_compute_profile
from mxlstm.data.datamodule import RULDataModule
from mxlstm.training.callbacks import make_callbacks
from mxlstm.training.lit_module import RULLitModule


class BestValRmseRecorder(pl.Callback):
    """Track the best ``val/rmse`` seen across fit epochs."""

    def __init__(self) -> None:
        super().__init__()
        self.best_rmse: float = float("inf")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        del pl_module  # unused
        if getattr(trainer, "sanity_checking", False):
            return
        m = trainer.callback_metrics.get("val/rmse")
        if m is None:
            return
        v = float(m.detach().item()) if hasattr(m, "detach") else float(m)
        self.best_rmse = min(self.best_rmse, v)


def _clone_cfg(base: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(base, resolve=True))


def apply_trial_hparams(trial: optuna.Trial, model_key: str, cfg: DictConfig) -> None:
    cfg.train.lr = float(trial.suggest_float("train_lr", 3e-4, 1.5e-3, log=True))
    cfg.train.weight_decay = float(trial.suggest_float("train_weight_decay", 1e-4, 1e-3, log=True))
    cfg.train.monotonicity_weight = float(trial.suggest_float("train_monotonicity_weight", 0.0, 0.1))

    if model_key == "xlstm_transformer":
        cfg.model.d_model = int(trial.suggest_categorical("model_d_model", [32, 48, 64]))
        cand_heads = [h for h in (4, 8) if int(cfg.model.d_model) % h == 0]
        cfg.model.n_heads = int(trial.suggest_categorical("model_n_heads", cand_heads))
        cfg.model.xlstm_blocks = int(trial.suggest_categorical("model_xlstm_blocks", [2, 3]))
        cfg.model.dropout = float(trial.suggest_float("model_dropout", 0.05, 0.25))
        cfg.model.head_pool = str(trial.suggest_categorical("model_head_pool", ["flatten", "last"]))

    elif model_key == "mamba_xlstm_net":
        cfg.model.d_model = int(trial.suggest_categorical("model_d_model", [96, 128]))
        cfg.model.xlstm_blocks = int(trial.suggest_categorical("model_xlstm_blocks", [2, 3]))
        cfg.model.mamba_blocks = int(trial.suggest_categorical("model_mamba_blocks", [1, 2, 3]))
        cfg.model.mamba_d_state = int(trial.suggest_categorical("model_mamba_d_state", [64, 128]))
        cfg.model.dropout = float(trial.suggest_float("model_dropout", 0.05, 0.25))

    elif model_key == "liquid_wave_rul":
        cfg.model.n_bands = int(trial.suggest_categorical("model_n_bands", [4, 6, 8]))
        cfg.model.n_band_feats = int(trial.suggest_categorical("model_n_band_feats", [6, 8, 12]))
        hd = int(trial.suggest_categorical("model_hidden_dim", [48, 64, 96]))
        cfg.model.hidden_dim = hd
        cand_ah = [h for h in (2, 4) if hd % h == 0]
        cfg.model.attn_heads = int(trial.suggest_categorical("model_attn_heads", cand_ah))
        cfg.model.ltc_unfolds = int(trial.suggest_categorical("model_ltc_unfolds", [2, 3, 4]))
        cfg.model.dropout = float(trial.suggest_float("model_dropout", 0.05, 0.25))

    elif model_key == "nbeats_rul":
        cfg.model.hidden_dim = int(trial.suggest_categorical("model_hidden_dim", [64, 96, 128]))
        cfg.model.trend_blocks = int(trial.suggest_categorical("model_trend_blocks", [1, 2, 3]))
        cfg.model.wear_blocks = int(trial.suggest_categorical("model_wear_blocks", [1, 2, 3]))
        cfg.model.shock_blocks = int(trial.suggest_categorical("model_shock_blocks", [1, 2, 3]))
        cfg.model.poly_degree = int(trial.suggest_categorical("model_poly_degree", [3, 4, 5]))
        cfg.model.n_harmonics = int(trial.suggest_categorical("model_n_harmonics", [3, 5, 7]))
        cfg.model.n_shock_basis = int(trial.suggest_categorical("model_n_shock_basis", [8, 14, 20]))
        cfg.model.dropout = float(trial.suggest_float("model_dropout", 0.05, 0.3))

    elif model_key == "diffusion_rul":
        d_model = int(trial.suggest_categorical("model_d_model", [64, 72, 96]))
        cfg.model.d_model = d_model
        cand_nh = [h for h in (4, 6, 8) if d_model % h == 0]
        cfg.model.n_heads = int(trial.suggest_categorical("model_n_heads", cand_nh))
        cfg.model.n_layers = int(trial.suggest_categorical("model_n_layers", [2, 3]))
        cfg.model.n_noise_levels = int(trial.suggest_categorical("model_n_noise_levels", [3, 4, 6]))
        cfg.model.max_noise_std = float(trial.suggest_float("model_max_noise_std", 0.15, 0.35))
        cfg.model.dropout = float(trial.suggest_float("model_dropout", 0.1, 0.3))

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def _build_dm_and_model(
    cfg: DictConfig,
    build_model: Callable[..., torch.nn.Module],
) -> tuple[RULDataModule, torch.nn.Module]:
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
    dm.setup()
    n_features = dm.n_features
    L = int(cfg.data.window_length)
    model = build_model(cfg, n_features=n_features, context_length=L)
    return dm, model


def objective(
    trial: optuna.Trial,
    *,
    base_cfg: DictConfig,
    model_key: str,
    max_epochs: int,
    build_model: Callable[..., torch.nn.Module],
) -> float:
    cfg = _clone_cfg(base_cfg)
    apply_trial_hparams(trial, model_key, cfg)
    cfg.train.max_epochs = int(max_epochs)

    try:
        dm, model = _build_dm_and_model(cfg, build_model)
    except Exception:
        raise optuna.TrialPruned() from None

    profile = get_compute_profile()
    if profile.accelerator == "cuda":
        torch.set_float32_matmul_precision("high")

    lit = RULLitModule(
        model=model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        warmup_epochs=int(cfg.train.warmup_epochs),
        max_epochs=int(cfg.train.max_epochs),
        monotonicity_weight=float(cfg.train.monotonicity_weight),
        scheduler=str(cfg.train.scheduler),
    )

    with tempfile.TemporaryDirectory(prefix=f"mxlstm_optuna_{trial.number}_") as tmp:
        ckpt_dir = Path(tmp) / "ckpt"
        callbacks = make_callbacks(
            checkpoint_dir=ckpt_dir,
            monitor=str(cfg.train.get("checkpoint_monitor", "val/rmse")),
            patience=int(cfg.train.early_stopping_patience),
            use_rich_progress=False,
            epoch_progress_lines=False,
            checkpointing=False,
        )
        rmse_cb = BestValRmseRecorder()
        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/rmse"))
        callbacks.append(rmse_cb)

        accel = get_accelerator()
        precision = str(cfg.train.get("precision", profile.precision))
        trainer = pl.Trainer(
            max_epochs=int(max_epochs),
            accelerator=accel,
            devices=1,
            precision=precision if accel != "mps" else "32-true",
            gradient_clip_val=float(cfg.train.gradient_clip_val),
            callbacks=callbacks,
            log_every_n_steps=999_999,
            enable_checkpointing=False,
            logger=pl.loggers.CSVLogger(save_dir=str(tmp), name="logs"),
        )
        try:
            trainer.fit(lit, datamodule=dm)
        except Exception:
            raise optuna.TrialPruned() from None

    score = rmse_cb.best_rmse if rmse_cb.best_rmse != float("inf") else float("inf")
    if score != score:  # NaN
        return float("inf")
    return score


def _best_params_to_overlay(best_params: dict[str, Any]) -> DictConfig:
    overlay = OmegaConf.create({"train": OmegaConf.create({}), "model": OmegaConf.create({})})
    for key, val in best_params.items():
        if key.startswith("train_"):
            subkey = key.removeprefix("train_")
            overlay.train[subkey] = val
        elif key.startswith("model_"):
            subkey = key.removeprefix("model_")
            overlay.model[subkey] = val
    return overlay


def run_study_and_save_overlay(
    *,
    study_name: str,
    storage_path: Path,
    base_cfg: DictConfig,
    model_key: str,
    n_trials: int,
    epochs_per_trial: int,
    seed: int,
    out_yaml: Path,
    build_model: Callable[..., torch.nn.Module],
) -> DictConfig:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4),
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def wrapped(trial: optuna.Trial) -> float:
        return objective(
            trial,
            base_cfg=base_cfg,
            model_key=model_key,
            max_epochs=int(epochs_per_trial),
            build_model=build_model,
        )

    study.optimize(wrapped, n_trials=n_trials, gc_after_trial=True)
    overlay = _best_params_to_overlay(study.best_params)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(overlay, out_yaml)
    return overlay
