"""Run Optuna hyperparameter search per (dataset, model); write overlays to configs/_tuned/."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
_PKG = _ROOT / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mxlstm.tuning.optuna_search import run_study_and_save_overlay  # noqa: E402
from mxlstm.utils.config import load_configs  # noqa: E402


def _load_build_model():
    train_path = _ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("mxlstm_train_script", train_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._build_model


_DATA_CFG = {
    "phm2012": _ROOT / "configs" / "data" / "phm2012.yaml",
    "xjtusy": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
    "xjtu_sy": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
    "xjtu_legacy": _ROOT / "configs" / "data" / "xjtu_sy.yaml",
    "xjtu_available": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
}
_MODEL_CFG = {
    "xlstm_transformer": _ROOT / "configs" / "model" / "baseline_xlstm_transformer.yaml",
    "mamba_xlstm_net": _ROOT / "configs" / "model" / "mamba_xlstm_net.yaml",
    "liquid_wave_rul": _ROOT / "configs" / "model" / "liquid_wave_rul.yaml",
    "nbeats_rul": _ROOT / "configs" / "model" / "nbeats_rul.yaml",
    "diffusion_rul": _ROOT / "configs" / "model" / "diffusion_rul.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=sorted(_DATA_CFG), required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(_MODEL_CFG), required=True)
    parser.add_argument(
        "--train",
        type=Path,
        default=_ROOT / "configs" / "train" / "algorithm_comparison.yaml",
    )
    parser.add_argument("--ablation", type=Path, default=None)
    parser.add_argument("--data-batch-size", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--epochs-per-trial", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=_ROOT / "results" / "_tuning",
        help="SQLite studies live here as optuna_<dataset>_<model>.db",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "configs" / "_tuned",
        help="Best-overlap YAML paths: out_dir/<dataset>_<model>.yaml",
    )
    return parser.parse_args()


def _data_yaml(ds: str, batch_size: int | None) -> Path:
    data_path = _DATA_CFG[ds]
    if batch_size is None:
        return data_path
    cfg = OmegaConf.load(data_path)
    cfg.data.batch_size = int(batch_size)
    tmp_dir = _ROOT / "results" / "_comparison_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"tune_{ds}_batch{batch_size}.yaml"
    OmegaConf.save(cfg, tmp_path)
    return tmp_path


def main() -> None:
    args = parse_args()
    build_model = _load_build_model()

    for ds in args.datasets:
        data_path = _data_yaml(ds, args.data_batch_size)
        for mk in args.models:
            cfg_paths_mk = [data_path, _MODEL_CFG[mk], args.train]
            if args.ablation is not None:
                cfg_paths_mk.append(args.ablation)
            base_cfg = load_configs(cfg_paths_mk)
            db = args.storage_dir / f"optuna_{ds}_{mk}.db"
            out_yaml = args.out_dir / f"{ds}_{mk}.yaml"
            print(f"[optuna] study={db.name} dataset={ds} model={mk} -> {out_yaml}", flush=True)
            run_study_and_save_overlay(
                study_name=f"{ds}_{mk}",
                storage_path=db,
                base_cfg=base_cfg,
                model_key=mk,
                n_trials=int(args.n_trials),
                epochs_per_trial=int(args.epochs_per_trial),
                seed=int(args.seed),
                out_yaml=out_yaml,
                build_model=build_model,
            )


if __name__ == "__main__":
    main()
