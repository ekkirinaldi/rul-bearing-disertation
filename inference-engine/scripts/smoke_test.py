#!/usr/bin/env python3
"""Headless smoke test for the streaming inference engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "Mamba-xLSTM" / "src"))

from app.engine import StreamSession, load_model  # noqa: E402
from app.skf_loader import load_skf_stream  # noqa: E402


def _assert_rul_in_range(pred: float, ctx: str) -> None:
    assert 0.0 <= pred <= 1.0, f"{ctx}: pred_rul={pred} out of [0,1]"


def _assert_explainability(frame: dict, ctx: str) -> None:
    """Validate the new live explainability + time-to-failure fields."""
    rem = frame["pred_remaining_s"]
    if frame["ttf_capped"]:
        assert rem is None, f"{ctx}: capped TTF should have no finite remaining"
    else:
        assert rem is not None and rem >= 0.0, f"{ctx}: pred_remaining_s={rem} invalid"
        assert frame["pred_eol_iso"], f"{ctx}: missing predicted EOL timestamp"

    gate = frame["branch_gate"]
    if gate is not None:
        assert abs((gate["xlstm"] + gate["mamba"]) - 1.0) < 1e-5, f"{ctx}: gate must sum to 1"

    drivers = frame["top_drivers"]
    assert drivers, f"{ctx}: expected non-empty top_drivers"
    for d in drivers:
        assert {"name", "weight", "dir"} <= d.keys(), f"{ctx}: malformed driver {d}"
        assert d["dir"] in ("up", "down"), f"{ctx}: bad dir {d['dir']}"


def test_phm2012_bearing_1_3() -> None:
    print("=== PHM2012 bearing 1_3 ===")
    load_model("phm2012")
    session = StreamSession(dataset_key="phm2012", bearing_id="1_3")
    preds: list[float] = []
    remaining: list[float] = []
    first_pred_t: int | None = None
    first_pred_val: float | None = None
    checked_explain = False

    run_dir = session.spec.run_dir
    ref = np.load(run_dir / "test_predictions.npz", allow_pickle=True)
    ref_t = ref["1_3_t"]
    ref_pred = ref["1_3_pred"]

    while True:
        frame = session.step()
        if frame is None:
            break
        if not frame["warmup"] and frame["pred_rul"] is not None:
            _assert_rul_in_range(frame["pred_rul"], f"t={frame['t']}")
            preds.append(frame["pred_rul"])
            if frame["pred_remaining_s"] is not None:
                remaining.append(frame["pred_remaining_s"])
            if not checked_explain:
                _assert_explainability(frame, f"t={frame['t']}")
                checked_explain = True
            if first_pred_t is None:
                first_pred_t = frame["t"]
                first_pred_val = frame["pred_rul"]

    assert len(preds) > 10, "expected many predictions after warmup"
    assert preds[-1] < preds[0], f"RUL should trend down: first={preds[0]:.4f} last={preds[-1]:.4f}"
    assert checked_explain, "never saw a post-warmup frame to validate explainability"
    assert len(remaining) > 10, "expected finite useful-time-left estimates"
    assert remaining[-1] < remaining[0], (
        f"useful time left should trend down: first={remaining[0]:.0f}s last={remaining[-1]:.0f}s"
    )
    print(f"  predictions: {len(preds)}, first={preds[0]:.4f}, last={preds[-1]:.4f}")
    print(f"  useful time left: first={remaining[0]:.0f}s last={remaining[-1]:.0f}s")

    if first_pred_t is not None and first_pred_val is not None:
        idx = np.where(ref_t == first_pred_t)[0]
        if idx.size:
            ref_val = float(ref_pred[idx[0]])
            diff = abs(first_pred_val - ref_val)
            print(f"  spot-check t={first_pred_t}: stream={first_pred_val:.4f} ref={ref_val:.4f} diff={diff:.4f}")
            assert diff < 0.15, f"prediction drift too large at t={first_pred_t}"
    print("  OK")


def test_integrated_gradients() -> None:
    print("=== Integrated Gradients (PHM2012 1_3) ===")
    session = StreamSession(dataset_key="phm2012", bearing_id="1_3")
    # Advance just past warm-up so a full window is buffered.
    for _ in range(session.window_length + 2):
        if session.step() is None:
            break
    expl = session.explain_current()
    assert expl is not None, "explain_current returned None after warm-up"
    assert expl["features"], "IG returned no per-feature importances"
    hm = expl["heatmap"]
    assert len(hm["values"]) == len(hm["feature_names"]), "heatmap rows != feature count"
    assert all(len(row) == hm["bins"] for row in hm["values"]), "ragged heatmap rows"
    print(f"  top feature: {expl['features'][0]['name']}, heatmap {len(hm['values'])}x{hm['bins']}")
    print("  OK")


def test_xjtusy_bearing_1_5() -> None:
    print("=== XJTU-SY bearing 1_5 ===")
    load_model("xjtusy")
    session = StreamSession(dataset_key="xjtusy", bearing_id="1_5")
    preds: list[float] = []

    while True:
        frame = session.step()
        if frame is None:
            break
        if not frame["warmup"] and frame["pred_rul"] is not None:
            _assert_rul_in_range(frame["pred_rul"], f"t={frame['t']}")
            preds.append(frame["pred_rul"])

    assert len(preds) > 10
    assert preds[-1] <= preds[0] + 0.05, f"RUL should not increase much: first={preds[0]:.4f} last={preds[-1]:.4f}"
    print(f"  predictions: {len(preds)}, first={preds[0]:.4f}, last={preds[-1]:.4f}")
    print("  OK")


def test_skf_ch15_or1_ch1_01_nde() -> None:
    print("=== SKF CH-15 OR-1 ch1_01_nde (transfer) ===")
    root = ROOT.parent / "data-bearing" / "skf-ch15-or1"
    run = load_skf_stream(root, "ch1_01_nde")
    assert run.n_acquisitions > 50, f"expected trending points, got {run.n_acquisitions}"
    print(f"  aligned trending points: {run.n_acquisitions}, interval ~{run.acquisition_interval_s:.0f}s")

    session = StreamSession(dataset_key="skf_ch15_or1", bearing_id="ch1_01_nde")
    preds: list[float] = []
    while True:
        frame = session.step()
        if frame is None:
            break
        assert frame["gt_rul"] is None
        assert frame["has_gt_rul"] is False
        if not frame["warmup"] and frame["pred_rul"] is not None:
            _assert_rul_in_range(frame["pred_rul"], f"t={frame['t']}")
            preds.append(frame["pred_rul"])

    assert len(preds) > 10
    print(f"  transfer RUL predictions: {len(preds)}, first={preds[0]:.4f}, last={preds[-1]:.4f}")
    print("  OK")


def main() -> None:
    test_phm2012_bearing_1_3()
    test_integrated_gradients()
    test_xjtusy_bearing_1_5()
    test_skf_ch15_or1_ch1_01_nde()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
