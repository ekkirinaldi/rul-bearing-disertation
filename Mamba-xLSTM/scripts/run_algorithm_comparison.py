"""Run all RUL algorithms on the selected datasets and build comparison reports."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]

_DATA_CONFIGS = {
    "phm2012": _ROOT / "configs" / "data" / "phm2012.yaml",
    # Default dissertation XJTU-SY comparison: dual-condition val/test incl. test [1_5, 2_3].
    # Legacy single-test split lives in configs/data/xjtu_sy.yaml (historical artifact).
    "xjtusy": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
    "xjtu_sy": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
    "xjtu_legacy": _ROOT / "configs" / "data" / "xjtu_sy.yaml",
    # Dense eval + dual-condition val/test split (same as xjtusy default above).
    "xjtu_available": _ROOT / "configs" / "data" / "xjtu_sy_available_full.yaml",
}

_MODEL_CONFIGS = {
    "xlstm_transformer": _ROOT / "configs" / "model" / "baseline_xlstm_transformer.yaml",
    "vanilla_xlstm_rul": _ROOT / "configs" / "model" / "vanilla_xlstm_rul.yaml",
    "mamba_xlstm_net": _ROOT / "configs" / "model" / "mamba_xlstm_net.yaml",
    "mamba_rul": _ROOT / "configs" / "model" / "mamba_rul.yaml",
    "patch_tst_rul": _ROOT / "configs" / "model" / "patch_tst_rul.yaml",
    # liquid_wave_rul: EXCLUDED from default runs — worst-performing on both PHM2012
    # (rank 11/12, RMSE 0.2697, R²=0.071) and XJTU-SY (rank 9/12, RMSE 0.2580).
    # Still registered for --models opt-in if explicitly needed.
    "liquid_wave_rul": _ROOT / "configs" / "model" / "liquid_wave_rul.yaml",
    "nbeats_rul": _ROOT / "configs" / "model" / "nbeats_rul.yaml",
    # physics_nbeats_rul: EXCLUDED from default runs — consistently poor on both datasets
    # (PHM2012 rank 9/12 RMSE 0.2539, R²=0.177; XJTU-SY rank 11/12 RMSE 0.2597, R²=-0.004).
    "physics_nbeats_rul": _ROOT / "configs" / "model" / "physics_nbeats_rul.yaml",
    "nbeats_xlstm_rul": _ROOT / "configs" / "model" / "nbeats_xlstm_rul.yaml",
    "diffusion_rul": _ROOT / "configs" / "model" / "diffusion_rul.yaml",
    # phase_moe_xlstm_rul: EXCLUDED from default runs — worst on both datasets
    # (PHM2012 rank 12/12 RMSE 0.2802, R²=-0.003; XJTU-SY rank 12/12 RMSE 0.2679, R²=-0.069).
    # R² negative on both datasets at 30 epochs; may need significantly more epochs to converge.
    "phase_moe_xlstm_rul": _ROOT / "configs" / "model" / "phase_moe_xlstm_rul.yaml",
    "sparse_gate_tcn_rul": _ROOT / "configs" / "model" / "sparse_gate_tcn_rul.yaml",
}

# Models excluded from the default comparison set due to poor performance in cloud30 runs
# (30 ep, bf16-mixed, seed=42). Still available via --models for targeted experiments.
_EXCLUDED_FROM_DEFAULT = {"liquid_wave_rul", "physics_nbeats_rul", "phase_moe_xlstm_rul"}

_MODEL_REPORT_LABEL = {
    "xlstm_transformer": "Baseline (xLSTM–Transformer)",
    "vanilla_xlstm_rul": "Vanilla xLSTM-RUL",
    "mamba_xlstm_net": "Mamba-xLSTM-Net",
    "mamba_rul": "Mamba-RUL (SSM-only)",
    "patch_tst_rul": "PatchTST-RUL",
    "liquid_wave_rul": "LiquidWave-RUL",
    "nbeats_rul": "N-BEATS-RUL",
    "physics_nbeats_rul": "Physics-N-BEATS-RUL",
    "nbeats_xlstm_rul": "N-BEATS-xLSTM-RUL",
    "diffusion_rul": "Diffusion-RUL",
    "phase_moe_xlstm_rul": "PhaseMoE-xLSTM-RUL",
    "sparse_gate_tcn_rul": "SparseGate-TCN-RUL",
}

_TUNED_DIR = _ROOT / "configs" / "_tuned"
_SNAPSHOT_SUBDIR = _ROOT / "results" / "_comparison_configs"

# Standard training modes — always run to max_epochs (patience=9999, no early stop).
# quick : 30 ep,  fp32-true  → local validation on any machine
# full  : 200 ep, fp32-true  → dissertation budget, works on CPU/MPS/CUDA
# cloud : 75 ep, bf16-mixed → default GPU/RunPod preset (cloud_full_75.yaml)
_MODE_TO_TRAIN: dict[str, Path] = {
    "quick": _ROOT / "configs" / "train" / "quick_run.yaml",
    "full":  _ROOT / "configs" / "train" / "full_run.yaml",
    "cloud": _ROOT / "configs" / "train" / "cloud_full_75.yaml",
}


def _merge_and_save(paths: list[Path], dest: Path) -> None:
    merged = OmegaConf.create({})
    for p in paths:
        merged = OmegaConf.merge(merged, OmegaConf.load(p))
    dest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(merged, dest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["phm2012", "xjtusy"],
        choices=sorted(_DATA_CONFIGS),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[m for m in _MODEL_CONFIGS if m not in _EXCLUDED_FROM_DEFAULT],
        choices=sorted(_MODEL_CONFIGS),
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "cloud"],
        default="quick",
        help=(
            "Training budget shortcut (default: quick). "
            "quick = 30 ep, fp32, any machine. "
            "full  = 200 ep, fp32, any machine (dissertation budget). "
            "cloud = 75 ep, bf16-mixed, GPU/RunPod (default cloud preset). "
            "Overridden by --train if both are supplied."
        ),
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=None,
        help=(
            "Explicit train config YAML (max_epochs, scheduler, …). "
            "When set, overrides --mode. "
            "Use --mode quick/full/cloud for standard presets."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ablation",
        type=Path,
        default=None,
        help="Optional YAML forwarded to scripts/train.py (e.g. configs/ablation/gpu_throughput.yaml).",
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--data-batch-size", type=int, default=None)
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument(
        "--report-name",
        type=str,
        default="algorithm_comparison",
        help="Basename for HTML/PDF under results/reports/. Use a distinct "
        "value per model mix (e.g. diffusion_nbeats_s42) so a later run with "
        "different --models does not overwrite this file.",
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default=None,
        help="Cover tagline under the title (default: “Bearing remaining-useful-life prediction”). "
             "Algorithms are rendered as separate bullets.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Train only — do not run build_report. Use scripts/build_report.py with "
        "chosen run dirs and a NEW --name to avoid overwriting an existing comparison HTML.",
    )
    parser.add_argument(
        "--use-tuned",
        action="store_true",
        help="If configs/_tuned/<dataset>_<model>.yaml exists, pass it via train.py "
        "--config-overlay (after ablation merge). Missing files log a warning and are skipped.",
    )
    parser.add_argument(
        "--with-explainability",
        action="store_true",
        help="After each training run: SHAP/IG/SAE(UMAP) plus architecture-specific PNGs "
        "under results/runs/<run_id>/explain/. Requires optional deps (pip install '.[interp]').",
    )
    parser.add_argument(
        "--parallel-models",
        action="store_true",
        help="For each dataset, launch one subprocess per model in parallel (same GPU). "
        "Faster wall-clock when VRAM allows; reduce --data-batch-size if two jobs OOM.",
    )
    return parser.parse_args()


def _latest_run_dir(results_root: Path, run_id: str) -> Path:
    candidates = sorted(
        (results_root / "runs").glob(f"*_{run_id}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(f"Could not find run directory for run_id={run_id}")
    return candidates[0]


def _data_config_for_run(args: argparse.Namespace, dataset_key: str) -> Path:
    data_path = _DATA_CONFIGS[dataset_key]
    if args.data_batch_size is None:
        return data_path
    cfg = OmegaConf.load(data_path)
    cfg.data.batch_size = int(args.data_batch_size)
    tmp_dir = _ROOT / "results" / "_comparison_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{dataset_key}_batch{args.data_batch_size}.yaml"
    OmegaConf.save(cfg, tmp_path)
    return tmp_path


def _tuned_yaml(dataset_key: str, model_key: str) -> Path:
    return _TUNED_DIR / f"{dataset_key}_{model_key}.yaml"


def _run_followup_explainability(run_dir: Path) -> None:
    src = str(_ROOT / "src")
    pp_env_orig = dict(os.environ)
    existing = pp_env_orig.get("PYTHONPATH", "")
    if existing:
        pp_env = dict(os.environ, PYTHONPATH=os.pathsep.join([src, existing]))
    else:
        pp_env = dict(os.environ, PYTHONPATH=src)

    interp_cmd = [
        sys.executable,
        "-u",
        str(_ROOT / "scripts" / "run_interpretability.py"),
        "--from-run",
        str(run_dir),
        "--out-dir",
        str(run_dir / "explain"),
    ]
    subprocess.check_call(interp_cmd, cwd=_ROOT, env=pp_env)

    extras_cmd = [
        sys.executable,
        "-u",
        "-m",
        "mxlstm.interp.explain_extras",
        "--run-dir",
        str(run_dir),
    ]
    subprocess.check_call(extras_cmd, cwd=_ROOT, env=pp_env)


def _build_train_cmd(
    args: argparse.Namespace, dataset_key: str, model_key: str, progress_prefix: str
) -> tuple[list[str], dict[str, str], str]:
    run_id = f"algorithm_comparison_{dataset_key}_{model_key}_s{args.seed}"
    data_config = _data_config_for_run(args, dataset_key)
    env = os.environ.copy()
    env["MXLSTM_PROGRESS_PREFIX"] = progress_prefix

    merge_sources: list[Path] = [data_config, _MODEL_CONFIGS[model_key], args.train]
    cmd = [
        sys.executable,
        "-u",
        "scripts/train.py",
        "--data",
        str(data_config),
        "--model",
        str(_MODEL_CONFIGS[model_key]),
        "--train",
        str(args.train),
        "--seed",
        str(args.seed),
        "--run-id",
        run_id,
    ]
    tuned_path = _tuned_yaml(dataset_key, model_key)
    if args.ablation is not None:
        cmd.extend(["--ablation", str(args.ablation)])
        merge_sources.append(args.ablation)
    if getattr(args, "use_tuned", False):
        if tuned_path.is_file():
            cmd.extend(["--config-overlay", str(tuned_path)])
            merge_sources.append(tuned_path)
        else:
            print(
                f"[comparison] WARN --use-tuned but missing overlay {tuned_path} "
                f"({dataset_key}/{model_key}) — training base hyperparameters.",
                flush=True,
            )

    snapshot = _SNAPSHOT_SUBDIR / f"{dataset_key}_{model_key}_s{args.seed}_merged.yaml"
    _merge_and_save(merge_sources, snapshot)

    if args.max_epochs is not None:
        cmd.extend(["--max-epochs", str(args.max_epochs)])
    if args.limit_train_batches is not None:
        cmd.extend(["--limit-train-batches", str(args.limit_train_batches)])
    if args.limit_val_batches is not None:
        cmd.extend(["--limit-val-batches", str(args.limit_val_batches)])
    if args.fast_dev_run:
        cmd.append("--fast-dev-run")
    if args.no_figures:
        cmd.append("--no-figures")
    return cmd, env, run_id


def _run_train(args: argparse.Namespace, dataset_key: str, model_key: str, run_idx: int, run_total: int) -> Path:
    label = _MODEL_REPORT_LABEL.get(model_key, model_key)
    prefix = f"[{run_idx}/{run_total}] [{dataset_key} | {label}] "
    cmd, env, run_id = _build_train_cmd(args, dataset_key, model_key, prefix)
    subprocess.check_call(cmd, cwd=_ROOT, env=env)
    run_dir = _latest_run_dir(_ROOT / "results", run_id)
    print(f"[comparison] DONE [{run_idx}/{run_total}] {dataset_key}/{model_key} → {run_dir}", flush=True)
    print(f"[comparison]      summary: {run_dir / 'summary.json'}", flush=True)

    if getattr(args, "with_explainability", False):
        try:
            print(f"[comparison] explainability → {run_dir / 'explain'}", flush=True)
            _run_followup_explainability(run_dir)
        except subprocess.CalledProcessError as exc:
            print(f"[comparison] WARN explainability pipeline failed ({exc})", flush=True)

    return run_dir


def _run_train_parallel_for_dataset(
    args: argparse.Namespace,
    dataset_key: str,
    model_keys: list[str],
    run_idx_start: int,
    run_total: int,
) -> list[Path]:
    """Launch one training subprocess per model in parallel; return run_dirs in ``model_keys`` order."""
    procs: list[subprocess.Popen] = []
    meta: list[tuple[str, str]] = []
    for j, model_key in enumerate(model_keys):
        label = _MODEL_REPORT_LABEL.get(model_key, model_key)
        idx = run_idx_start + j
        prefix = f"[PAR {j + 1}/{len(model_keys)}] [{idx}/{run_total}] [{dataset_key} | {label}] "
        cmd, env, run_id = _build_train_cmd(args, dataset_key, model_key, prefix)
        print(f"[comparison] POPEN parallel {dataset_key}/{model_key} run_id={run_id}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(_ROOT), env=env))
        meta.append((model_key, run_id))
    failures: list[tuple[str, int]] = []
    for proc, (model_key, _rid) in zip(procs, meta, strict=True):
        rc = int(proc.wait())
        if rc != 0:
            failures.append((model_key, rc))
    if failures:
        detail = "; ".join(f"{m} exit={c}" for m, c in failures)
        raise RuntimeError(f"parallel training failed: {detail}")

    run_dirs: list[Path] = []
    for model_key in model_keys:
        run_id = f"algorithm_comparison_{dataset_key}_{model_key}_s{args.seed}"
        run_dir = _latest_run_dir(_ROOT / "results", run_id)
        run_dirs.append(run_dir)
        print(f"[comparison] DONE (parallel) {dataset_key}/{model_key} → {run_dir}", flush=True)
        print(f"[comparison]      summary: {run_dir / 'summary.json'}", flush=True)
        if getattr(args, "with_explainability", False):
            try:
                print(f"[comparison] explainability → {run_dir / 'explain'}", flush=True)
                _run_followup_explainability(run_dir)
            except subprocess.CalledProcessError as exc:
                print(f"[comparison] WARN explainability pipeline failed ({exc})", flush=True)
    return run_dirs


def _build_reports(args: argparse.Namespace, run_dirs: list[Path]) -> None:
    reports_dir = _ROOT / "results" / "reports"
    tagline = args.subtitle or "Bearing remaining-useful-life prediction"
    cmd = [
        sys.executable,
        "scripts/build_report.py",
        "--runs",
        *[str(p) for p in run_dirs],
        "--name",
        args.report_name,
        "--title",
        "RUL algorithm comparison",
        "--subtitle",
        tagline,
        "--reports-dir",
        str(reports_dir),
    ]
    for m in args.models:
        cmd.extend(["--subtitle-line", _MODEL_REPORT_LABEL.get(m, m)])
    if args.no_pdf:
        cmd.append("--no-pdf")
    subprocess.check_call(cmd, cwd=_ROOT)


def main() -> None:
    args = parse_args()

    # Resolve training config: explicit --train overrides --mode.
    if args.train is None:
        args.train = _MODE_TO_TRAIN[args.mode]
    if not args.train.exists():
        raise FileNotFoundError(f"Train config not found: {args.train}")

    run_dirs: list[Path] = []
    total = len(args.datasets) * len(args.models)
    idx = 0
    train_cfg = OmegaConf.load(args.train)
    planned_epochs = (
        int(args.max_epochs) if args.max_epochs is not None else int(train_cfg.train.max_epochs)
    )
    mode_label = f"--mode {args.mode}" if args.train == _MODE_TO_TRAIN.get(args.mode) else f"--train {args.train.name}"
    print(
        f"[comparison] Planned {total} training run(s): "
        f"datasets={list(args.datasets)} models={list(args.models)} seed={args.seed} "
        f"{mode_label} max_epochs={planned_epochs} "
        f"parallel_models={getattr(args, 'parallel_models', False)}",
        flush=True,
    )
    if args.use_tuned:
        print(
            "[comparison] --use-tuned: overlays from configs/_tuned/<dataset>_<model>.yaml when present",
            flush=True,
        )
    if getattr(args, "parallel_models", False) and len(args.models) > 1:
        print(
            "[comparison] --parallel-models: per-dataset, all models run as separate subprocesses in parallel.",
            flush=True,
        )

    for dataset_key in args.datasets:
        if getattr(args, "parallel_models", False) and len(args.models) > 1:
            banner = "=" * 72
            print(f"\n{banner}", flush=True)
            print(
                f"[comparison] START (parallel) dataset={dataset_key} models={list(args.models)}",
                flush=True,
            )
            print(f"{banner}\n", flush=True)
            batch = _run_train_parallel_for_dataset(
                args, dataset_key, list(args.models), idx + 1, total
            )
            run_dirs.extend(batch)
            idx += len(args.models)
        else:
            for model_key in args.models:
                idx += 1
                label = _MODEL_REPORT_LABEL.get(model_key, model_key)
                banner = "=" * 72
                print(f"\n{banner}", flush=True)
                print(f"[comparison] START [{idx}/{total}] dataset={dataset_key} model={label}", flush=True)
                print(f"{banner}\n", flush=True)
                run_dirs.append(_run_train(args, dataset_key, model_key, idx, total))

    print(f"[comparison] building report from {len(run_dirs)} runs", flush=True)
    if args.skip_report:
        print(
            "[comparison] --skip-report: skipped build_report. Example merge for two models on both datasets "
            '(latest timestamp wins if you paste globs):\n'
            '  scripts/build_report.py --reports-dir results/reports --name my_comparison_s42 \\\n'
            '    --title "Model A vs Model B" \\\n'
            '    --subtitle "PHM2012 + XJTU-SY — full runs" \\\n'
            '    --runs \\\n'
            '      "$(ls -dt results/runs/*_algorithm_comparison_phm2012_diffusion_rul_s42 | head -1)" \\\n'
            '      "$(ls -dt results/runs/*_algorithm_comparison_phm2012_nbeats_rul_s42 | head -1)" \\\n'
            '      "$(ls -dt results/runs/*_algorithm_comparison_xjtusy_diffusion_rul_s42 | head -1)" \\\n'
            '      "$(ls -dt results/runs/*_algorithm_comparison_xjtusy_nbeats_rul_s42 | head -1)"',
            flush=True,
        )
        return
    _build_reports(args, run_dirs)


if __name__ == "__main__":
    main()
