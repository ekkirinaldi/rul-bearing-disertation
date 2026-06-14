"""Explainability helpers: live branch-gate + saliency, on-demand Integrated Gradients.

Two tiers, both operating on the Mamba-xLSTM-Net backbone (``lit.model``):

* :func:`live_explanation` — one forward+backward per acquisition. Returns the
  prediction, the fusion **gate** balance (xLSTM vs Mamba), and the top driving
  HI features via single-pass ``|grad x input|`` saliency. Cheap enough to run
  every frame.
* :func:`integrated_gradients_explanation` — on-demand, Captum-backed Integrated
  Gradients over the current buffered window. Heavier (``n_steps`` model
  evaluations); run in a threadpool from the server.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from mxlstm.interp.integrated_gradients import integrated_gradients


def _feature_name(names: list[str], i: int) -> str:
    if i < len(names):
        return names[i]
    return f"f{i}"


def _unwrap_pred(out: Any) -> tuple[torch.Tensor, dict | None]:
    """Normalize a backbone forward result into ``(pred, hidden|None)``."""
    if isinstance(out, tuple):
        pred = out[0]
        hidden = out[1] if len(out) > 1 else None
        if isinstance(hidden, dict):
            return pred, hidden
        return pred, None
    return out, None


def _branch_gate(hidden: dict | None) -> dict[str, float] | None:
    """Fusion gate balance: ``out = g·xLSTM + (1-g)·Mamba`` ⇒ mean(g) = xLSTM share."""
    gate = hidden.get("gate") if hidden else None
    if gate is None:
        return None
    xlstm = min(max(float(gate.detach().float().mean().cpu().item()), 0.0), 1.0)
    return {"xlstm": xlstm, "mamba": 1.0 - xlstm}


def live_explanation(
    lit: Any,
    window: np.ndarray,
    device: torch.device | str,
    feature_names: list[str],
    *,
    with_saliency: bool = True,
    top_k: int = 5,
) -> dict[str, Any]:
    """Prediction + branch gate, and (when ``with_saliency``) top driving features.

    ``window`` is ``(L, F)`` (one acquisition window, already scaled/smoothed).
    Returns ``{"pred": float, "branch_gate": {...}|None, "top_drivers": [...]|None}``.

    The prediction + gate path runs under ``no_grad`` (cheap, no autograd graph
    retained — this is what keeps memory flat). The single-pass ``|grad·input|``
    saliency requires a backward pass, so callers should throttle it (the top
    drivers change slowly) rather than running it every acquisition.
    """
    base = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0).to(device)

    if not with_saliency:
        with torch.no_grad():
            out = lit.model(base, return_hidden=True)
            pred_t, hidden = _unwrap_pred(out)
            pred_val = float(pred_t.reshape(-1)[0].cpu().item())
            gate = _branch_gate(hidden)
        del base, out, pred_t, hidden
        return {"pred": pred_val, "branch_gate": gate, "top_drivers": None}

    x = base.requires_grad_(True)
    with torch.enable_grad():
        out = lit.model(x, return_hidden=True)
        pred_t, hidden = _unwrap_pred(out)
        pred_scalar = pred_t.reshape(-1)[0]
        pred_scalar.backward()

    pred_val = float(pred_scalar.detach().cpu().item())
    branch_gate = _branch_gate(hidden)

    top_drivers = None
    if x.grad is not None:
        grad = x.grad.detach()[0]
        inp = x.detach()[0]
        sal = (grad * inp).abs().sum(dim=0).cpu().numpy()
        direction = grad.mean(dim=0).cpu().numpy()
        total = float(sal.sum()) + 1e-12
        weights = sal / total
        order = np.argsort(weights)[::-1][:top_k]
        top_drivers = [
            {
                "name": _feature_name(feature_names, int(i)),
                "weight": float(weights[i]),
                "dir": "up" if float(direction[i]) >= 0.0 else "down",
            }
            for i in order
        ]

    del x, out, pred_t, hidden, pred_scalar
    return {"pred": pred_val, "branch_gate": branch_gate, "top_drivers": top_drivers}


def integrated_gradients_explanation(
    lit: Any,
    window: np.ndarray,
    device: torch.device | str,
    feature_names: list[str],
    *,
    n_steps: int = 32,
    top_k: int = 10,
    time_bins: int = 32,
) -> dict[str, Any]:
    """Captum Integrated Gradients over ``window`` ``(L, F)``.

    Returns a per-feature importance bar (top_k) and a downsampled
    time×feature heatmap for the most important features.
    """
    x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0)
    attr = integrated_gradients(lit.model, x, n_steps=n_steps, device=device)[0]
    per_feat = np.abs(attr).sum(axis=0)
    order = np.argsort(per_feat)[::-1][:max(top_k, 1)]
    features = [
        {"name": _feature_name(feature_names, int(i)), "value": float(per_feat[i])}
        for i in order
    ]

    L = attr.shape[0]
    bins = min(time_bins, L)
    edges = np.linspace(0, L, bins + 1, dtype=int)
    heat = np.empty((len(order), bins), dtype=np.float64)
    for col in range(bins):
        a, b = edges[col], max(edges[col + 1], edges[col] + 1)
        heat[:, col] = attr[a:b, order].mean(axis=0)

    return {
        "features": features,
        "heatmap": {
            "feature_names": [_feature_name(feature_names, int(i)) for i in order],
            "values": heat.tolist(),
            "n_time": int(L),
            "bins": int(bins),
        },
        "n_steps": int(n_steps),
    }
