"""Dissertation-derived context for the dashboard.

Surfaces the parts of the prognostic study (Bab V) that the live streaming
view alone does not show:

* **RUL accuracy metrics** (RMSE / MAE / R² / PHM Score) of the trained
  backbone, read from ``run_dir/summary.json``.
* **Layer-2 mechanistic interpretability** — the Top-k SAE → BPFx mapping
  (Bab V §V.2–§V.6, novelti N3/N4): bearing characteristic frequencies and
  the hit-rate of SAE latent concepts against them, read from the precomputed
  ``results/bpfx_mapping/{key}_bpfx_results.json``.
* **Backbone identity** and **multi-tier PdM placement** (Bab VI) as static
  descriptors.

This is read-only reference data; nothing here runs the model. The live
forward/backward path stays in :mod:`app.engine` / :mod:`app.explain`.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.model_registry import load_dataset_spec

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BPFX_DIR = _REPO_ROOT / "Mamba-xLSTM" / "results" / "bpfx_mapping"

# Plain-language meaning of each bearing characteristic frequency (BPFx).
BPFX_FAULT: dict[str, dict[str, str]] = {
    "BPFO": {"full": "Ball Pass Frequency, Outer race", "fault": "outer-race defect"},
    "BPFI": {"full": "Ball Pass Frequency, Inner race", "fault": "inner-race defect"},
    "BSF": {"full": "Ball Spin Frequency", "fault": "rolling-element (ball) defect"},
    "FTF": {"full": "Fundamental Train Frequency", "fault": "cage defect"},
}

# Static backbone + multi-tier descriptors (Bab V §V.1, Bab VI Tabel VI.1).
_BACKBONE = {
    "name": "Mamba-xLSTM-Net",
    "desc": "BiMamba-3 selective SSM + xLSTM memory (mLSTM×2, sLSTM), gated fusion, sigmoid RUL head",
    "input": "Health-Indicator sequence, window L=32",
    # The dissertation evaluates three RUL backbones (Bab V §V.1); the live engine
    # serves Mamba-xLSTM-Net. Winner differs by dataset — see §V.5.
    "note": (
        "One of three Bab V RUL backbones (Mamba-xLSTM-Net, N-BEATS-xLSTM-RUL, "
        "SparseGate-TCN-RUL). This engine runs Mamba-xLSTM-Net — the reported "
        "winner on XJTU-SY and IMS; on PHM2012 the reported best is "
        "SparseGate-TCN-RUL."
    ),
}

# Top-k Sparse Autoencoder operating point (Bab V §V.2, §V.8).
_SAE = {
    "d_latent": 1024,
    "k": 51,
    "sparsity_pct": 5.0,
    "note": (
        "1,024-dim latent dictionary with k=51 active neurons per sample "
        "(~5% sparsity) — a conservative operating point, not the hit-rate peak. "
        "Top-k enforces exact sparsity with no L1 shrinkage artifact."
    ),
}

# Three-tier PdM architecture (Bab VI Tabel VI.1). This dashboard is Tier 3.
_TIERS = [
    {
        "n": 1,
        "label": "Tier 1 · Edge IoT",
        "location": "Sensor / SKF IMx-8",
        "model": "SVM/LR on 36-D HI",
        "xai": "SHAP KernelExplainer",
        "output": "Anomaly triage (yes/no)",
        "current": False,
    },
    {
        "n": 2,
        "label": "Tier 2 · Edge Server",
        "location": "Plant gateway",
        "model": "WDCNN + Fault Signature Maps",
        "xai": "SHAP DeepExplainer + FSM (Layer 1: raw-signal attribution)",
        "output": "Fault type & severity (99.87% acc.)",
        "current": False,
    },
    {
        "n": 3,
        "label": "Tier 3 · Cloud / GPU",
        "location": "AWS / on-premise",
        "model": "RUL backbone + Top-k SAE→BPFx",
        "xai": "SAE→BPFx mapping (Layer 2: latent concept); bootstrap CI; permutation test",
        "output": "RUL estimate + latent–physics audit",
        "current": True,
    },
]
# Backwards-compatible single-tier descriptor (the current tier).
_TIER = next(t for t in _TIERS if t["current"])


def _best_epoch(best_checkpoint: str) -> int | None:
    """Parse ``.../071.ckpt`` → 71."""
    stem = Path(str(best_checkpoint)).stem
    return int(stem) if stem.isdigit() else None


def _load_metrics(run_dir: Path) -> dict | None:
    summ = run_dir / "summary.json"
    if not summ.is_file():
        return None
    try:
        data = json.loads(summ.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    tm = data.get("test_metrics", {})
    return {
        "rmse": tm.get("test/rmse"),
        "mae": tm.get("test/mae"),
        "r2": tm.get("test/r2"),
        "phm_score": tm.get("test/phm_score"),
        "n_params": data.get("n_params"),
        "best_epoch": _best_epoch(data.get("best_checkpoint", "")),
        "source_dataset": data.get("dataset"),
    }


def _load_bpfx(model_key: str) -> dict | None:
    """Assemble the SAE→BPFx mapping for a benchmark dataset, or None."""
    path = _BPFX_DIR / f"{model_key}_bpfx_results.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    hit_rate = raw.get("hit_rate", {})
    freqs = raw.get("characteristic_frequencies_hz", {})
    top5 = raw.get("top5_features", {})

    # Dominant BPFx = highest hit-rate.
    dominant = max(hit_rate, key=lambda k: hit_rate.get(k, 0.0)) if hit_rate else None

    # Strongest single SAE↔BPFx correlation across all bands.
    best_feat = None
    for bp, feats in top5.items():
        for f in feats:
            r = f.get("r", 0.0)
            if best_feat is None or abs(r) > abs(best_feat["r"]):
                best_feat = {"bpfx": bp, "feature_idx": f.get("feature_idx"), "r": r}

    bands = []
    for bp in ("BPFO", "BPFI", "BSF", "FTF"):
        if bp not in hit_rate and bp not in freqs:
            continue
        meta = BPFX_FAULT.get(bp, {})
        bands.append(
            {
                "bpfx": bp,
                "full": meta.get("full", bp),
                "fault": meta.get("fault", bp),
                "freq_hz": freqs.get(bp),
                "hit_rate": hit_rate.get(bp, 0.0),
                "dominant": bp == dominant,
            }
        )

    # Grounded interpretation copy (Bab V §V.4–§V.8). The dashboard must not
    # overclaim: hit-rates are single-digit %, a weak but statistically detected
    # correspondence — an emergent property, never "rediscovered physics".
    dom_meta = BPFX_FAULT.get(dominant, {}) if dominant else {}
    dom_hit = hit_rate.get(dominant, 0.0) if dominant else 0.0
    if dominant and dom_hit > 0.0:
        fault = dom_meta.get("fault", "unknown defect")
        art = "an" if fault[:1].lower() in "aeiou" else "a"
        verdict = (
            f"The strongest learned concepts correlate with {dominant} "
            f"({dom_meta.get('full', dominant)}, indicating {art} {fault}) "
            f"— a weak but statistically detected correspondence (hit-rate "
            f"{dom_hit * 100:.2f}%, p<0.001 on PHM2012/XJTU-SY). This is an emergent "
            f"property of representations trained on degradation data, not an "
            f"architectural artifact."
        )
    else:
        verdict = (
            "No BPFx shows a non-zero hit-rate in this configuration — no "
            "latent–physics correspondence is detected for this dataset."
        )
    caveat = (
        "Hit-rates are single-digit % and scale with the sparsity budget k; what "
        "is robust across k is the rank order of dominant BPFx, not the magnitude. "
        "Cross-architecture universality is partial (clean on PHM2012/CWRU, mixed on "
        "XJTU-SY/IMS). Validated with bootstrap 95% CI, a permutation test, and two "
        "negative controls (Xavier-init, Gaussian hidden states); CWRU is underpowered "
        "(n=10) and excluded from formal claims."
    )

    return {
        "d_latent": raw.get("d_latent"),
        "n_recordings": raw.get("n_recordings"),
        "corr_threshold": raw.get("corr_threshold"),
        "bands": bands,
        "dominant": dominant,
        "best_feature": best_feat,
        "verdict": verdict,
        "caveat": caveat,
    }


def dissertation_context(key: str) -> dict:
    """Read-only dissertation context for dataset ``key``.

    For the SKF plant stream the model is a PHM2012 transfer, so metrics are
    flagged as the transfer source and the SAE→BPFx mapping (validated only on
    benchmarks) is reported as not computed for the plant transfer.
    """
    spec = load_dataset_spec(key)
    metrics = _load_metrics(spec.run_dir)
    is_transfer = spec.inference_model_key != spec.key
    bpfx = None if is_transfer else _load_bpfx(spec.inference_model_key)

    return {
        "key": key,
        "backbone": _BACKBONE,
        "tier": _TIER,
        "tiers": _TIERS,
        "sae": _SAE,
        "metrics": metrics,
        "metrics_source": spec.inference_model_key,
        "is_transfer": is_transfer,
        "bpfx": bpfx,
        "bpfx_note": (
            "SAE→BPFx mechanistic mapping is validated on the public run-to-failure "
            "benchmarks (PHM2012, XJTU-SY). It is not computed for the SKF plant "
            "transfer stream, which has no labelled failure population for correlation."
            if bpfx is None and is_transfer
            else None
        ),
    }
