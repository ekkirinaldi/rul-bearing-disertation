"""End-to-end interpretability run: SHAP + SAE + UMAP + IG.

Each step is logged through ``RunLogger`` and figures are written into
``<run_dir>/<out-dir-name>/``.

Example::

    python scripts/run_interpretability.py \
        --checkpoint results/runs/<id>/checkpoints/last.ckpt \
        --data configs/data/phm2012.yaml \
        --model configs/model/mamba_xlstm_net.yaml \
        --train configs/train/default.yaml \
        --out-dir results/runs/<id>/interp

Or reuse a merged ``config.yaml`` from a finished training run::

    python scripts/run_interpretability.py \\
        --from-run results/runs/<id> \\
        --out-dir results/runs/<id>/explain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1] / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

from mxlstm.compute import get_device  # noqa: E402
from mxlstm.data.datamodule import RULDataModule  # noqa: E402
from mxlstm.interp.integrated_gradients import (  # noqa: E402
    integrated_gradients,
    plot_time_attribution,
)
from mxlstm.interp.latent_clustering import cluster_hdbscan, plot_latent_scatter, reduce_umap  # noqa: E402
from mxlstm.interp.explain_extras import resolve_checkpoint  # noqa: E402
from mxlstm.interp.sae import (  # noqa: E402
    SAEConfig,
    TopKSparseAutoencoder,
    collect_hidden_states,
    save_sae,
    train_sae,
)
from mxlstm.interp.shap_explain import (  # noqa: E402
    aggregate_feature_importance,
    compute_shap_values,
    plot_global_importance,
    plot_time_feature_heatmap,
)
from mxlstm.training.lit_module import RULLitModule  # noqa: E402
from mxlstm.utils.config import load_configs  # noqa: E402
from mxlstm.utils.run_logger import RunLogger  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--from-run",
        type=Path,
        default=None,
        help="Run directory that contains merged config.yaml (and typically summary.json).",
    )
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--train", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--n-shap-samples", type=int, default=64)
    p.add_argument("--sae-epochs", type=int, default=50)
    p.add_argument("--sae-expansion", type=int, default=8)
    return p.parse_args()


def _load_cfg(args: argparse.Namespace) -> DictConfig:
    if args.from_run is not None:
        cfg_path = args.from_run / "config.yaml"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing {cfg_path}")
        return OmegaConf.load(cfg_path)
    if args.data is None or args.model is None or args.train is None:
        raise ValueError("Provide --from-run or all of --data, --model, --train.")
    return load_configs([args.data, args.model, args.train])


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args)
    if args.from_run is not None:
        run_dir = args.from_run.resolve()
        args.out_dir = args.out_dir or (run_dir / "explain")
        args.checkpoint = resolve_checkpoint(run_dir) if args.checkpoint is None else args.checkpoint
    if args.out_dir is None:
        raise ValueError("--out-dir is required when --from-run is not set.")
    if args.checkpoint is None:
        raise ValueError("Missing --checkpoint (and could not infer from run directory).")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Use the parent of out_dir (the run_dir) as logger root if it looks like one,
    # otherwise log inside out_dir itself.
    log_root = args.out_dir.parent if (args.out_dir.parent / "summary.json").exists() else args.out_dir
    with RunLogger(log_root, run_id=f"interp_{log_root.name}") as rl:
        rl.info(f"Interpretability run for checkpoint {args.checkpoint}")

        with rl.phase("Setup"):
            with rl.step("Load configs"):
                rl.metric("from_run", str(args.from_run) if args.from_run else "")

            with rl.step("Detect device"):
                device = get_device()
                rl.metric("device", str(device))

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
                    num_workers=int(cfg.data.get("num_workers", 0)),
                    cache_dir=cfg.data.get("cache_dir", None),
                )
                dm.setup()
                feature_names = (
                    dm.pipeline.feature_names
                    if dm.pipeline is not None
                    else (dm.train_bearings_data[0].feature_names if dm.train_bearings_data else [])
                )
                rl.metric("n_features", len(feature_names))

            with rl.step("Collect test tensors for attribution"):
                test_loader = dm.test_dataloader()
                test_x_chunks = []
                test_y_chunks = []
                for batch in test_loader:
                    test_x_chunks.append(batch[0].numpy())
                    test_y_chunks.append(batch[1].numpy())
                    if sum(b.shape[0] for b in test_x_chunks) >= args.n_shap_samples:
                        break
                test_x = np.concatenate(test_x_chunks, axis=0)[: args.n_shap_samples]
                test_y = np.concatenate(test_y_chunks, axis=0)[: args.n_shap_samples]
                rl.metric("n_interp_windows", int(test_x.shape[0]))

            with rl.step("Load checkpoint"):
                train_path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
                import importlib.util

                spec = importlib.util.spec_from_file_location("mxlstm_train_script_build", train_path)
                assert spec is not None and spec.loader is not None
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                model = mod._build_model(cfg, n_features=dm.n_features,
                                         context_length=int(cfg.data.window_length))
                lit = RULLitModule.load_from_checkpoint(
                    str(args.checkpoint), model=model, map_location=device)
                lit = lit.to(device).eval()

        # ---- SHAP ----
        with rl.phase("SHAP attribution"):
            try:
                with rl.step("Collect background"):
                    train_loader = dm.train_dataloader()
                    bg_batch = next(iter(train_loader))[0].numpy()

                with rl.step("Compute SHAP values"):
                    shap_values = compute_shap_values(lit.model, bg_batch[:64], test_x, device=device)
                    importances = aggregate_feature_importance(shap_values, feature_names)
                    p = plot_global_importance(importances, args.out_dir / "shap_global.png")
                    if p is not None:
                        rl.artefact("shap_global", p)
                    idx = int(np.argmin(np.abs(test_y - 0.5))) if len(test_y) > 0 else 0
                    p2 = plot_time_feature_heatmap(
                        shap_values[idx], feature_names, args.out_dir / f"shap_heatmap_{idx}.png")
                    if p2 is not None:
                        rl.artefact("shap_heatmap", p2)
                    (args.out_dir / "shap_global.json").write_text(json.dumps(importances, indent=2))
            except ImportError as e:
                rl.warning(f"SHAP skipped: {e}")

        # ---- Integrated Gradients ----
        with rl.phase("Integrated Gradients"):
            try:
                with rl.step("Compute IG attributions"):
                    ig_inputs = torch.from_numpy(test_x[:8]).float()
                    ig_attr = integrated_gradients(lit.model, ig_inputs, device=device)
                    for i in range(min(4, ig_attr.shape[0])):
                        p = plot_time_attribution(
                            ig_attr[i],
                            args.out_dir / f"ig_{i}.png",
                            feature_names=feature_names)
                        if p is not None:
                            rl.artefact(f"ig_{i}", p)
            except (ImportError, NameError, RuntimeError) as e:
                rl.warning(f"Integrated Gradients skipped: {e}")

        # ---- SAE on fused hidden states ----
        if hasattr(lit.model, "cfg") and hasattr(lit.model.cfg, "d_model"):
            with rl.phase("Sparse autoencoder + UMAP/HDBSCAN"):
                try:
                    with rl.step("Collect hidden states"):
                        hidden = collect_hidden_states(
                            lit.model, dm.train_dataloader(), device=device,
                            layer="fused", max_samples=20_000,
                        )
                        rl.metric("hidden_shape", list(hidden.shape))

                    with rl.step(f"Train SAE ({args.sae_epochs} epochs)"):
                        sae = TopKSparseAutoencoder(SAEConfig(
                            d_model=hidden.shape[1], expansion=args.sae_expansion))
                        history = train_sae(sae, hidden, epochs=args.sae_epochs, device=device)
                        save_sae(sae, args.out_dir / "sae.pt")
                        # train_sae returns list[dict] with keys "epoch" and "recon"
                        (args.out_dir / "sae_history.json").write_text(json.dumps(history, indent=2))
                        rl.artefact("sae", args.out_dir / "sae.pt")
                        if history:
                            last = history[-1]
                            final_loss = float(last.get("recon", last.get("loss", 0.0)))
                            rl.metric("sae_final_loss", final_loss)

                    with rl.step("UMAP + HDBSCAN clustering"):
                        sae_device = next(sae.parameters()).device
                        with torch.no_grad():
                            latents = (
                                sae.encode(
                                    torch.from_numpy(hidden[:5000].astype(np.float32)).to(sae_device)
                                )
                                .cpu().numpy()
                            )
                        embedding = reduce_umap(latents)
                        cluster_labels = cluster_hdbscan(embedding)
                        p = plot_latent_scatter(
                            embedding,
                            cluster_labels,
                            args.out_dir / "sae_umap_clusters.png",
                            title="SAE latent UMAP (HDBSCAN clusters)")
                        if p is not None:
                            rl.artefact("sae_umap_clusters", p)
                except ImportError as e:
                    rl.warning(f"SAE/UMAP skipped: {e}")

        rl.info(f"Interpretability artefacts in {args.out_dir}")


if __name__ == "__main__":
    main()
