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

    # Streaming IG (reduced n_steps, runs under the model lock from the server).
    stream_expl = session.compute_ig_stream()
    assert stream_expl is not None, "compute_ig_stream returned None after warm-up"
    assert stream_expl["features"], "streaming IG returned no per-feature importances"
    assert stream_expl["n_steps"] <= expl["n_steps"], "streaming IG should use no more steps"
    print(f"  streaming IG: {stream_expl['n_steps']} steps, top {stream_expl['features'][0]['name']}")
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
    print("=== SKF CH-15 OR-1 6m ch1_01_nde (transfer) ===")
    root = ROOT.parent / "data-bearing" / "skf-ch15-or1-6m"
    run = load_skf_stream(root, "ch1_01_nde")
    # The 6-month export is a full healthy → failure run, not the short tail.
    assert run.n_acquisitions > 200, f"expected the long run, got {run.n_acquisitions}"
    span_days = (run.points[-1].timestamp - run.points[0].timestamp).total_seconds() / 86400.0
    assert span_days > 90.0, f"expected multi-month span, got {span_days:.1f}d"
    assert run.failure_time is not None and 0 < run.eol_index < run.n_acquisitions
    print(
        f"  aligned points: {run.n_acquisitions}, span={span_days:.1f}d, "
        f"EOL idx={run.eol_index} @ {run.failure_time:%Y-%m-%d %H:%M}, "
        f"median dt~{run.acquisition_interval_s/3600:.1f}h"
    )

    session = StreamSession(dataset_key="skf_ch15_or1", bearing_id="ch1_01_nde")
    preds: list[float] = []
    remaining: list[float] = []
    elapsed_seen: list[float] = []
    rms_v_seen: list[float] = []
    while True:
        frame = session.step()
        if frame is None:
            break
        # Plant data has NO labelled RUL ground truth — chart shows a single
        # causal predicted curve, no truth overlay.
        assert frame["has_gt_rul"] is False, "SKF must not expose ground-truth RUL"
        assert frame["gt_rul"] is None, "SKF gt_rul must be None"
        assert frame["gt_remaining_s"] is None, "SKF gt_remaining_s must be None"
        elapsed_seen.append(frame["elapsed_s"])
        # Raw HI scalars must be physically sane: the synthetic waveform RMS is
        # normalised to accel/envelope (≲ ~10 gE), so a value in the hundreds
        # means the rms_v feature index is wrong (regression guard).
        if frame["hi"] is not None:
            rms_v_seen.append(frame["hi"]["rms_v"])
        if not frame["warmup"] and frame["pred_rul"] is not None:
            _assert_rul_in_range(frame["pred_rul"], f"t={frame['t']}")
            preds.append(frame["pred_rul"])
            if frame["pred_remaining_s"] is not None:
                remaining.append(frame["pred_remaining_s"])

    assert len(preds) > 10
    # Elapsed time must be timestamp-driven (monotonic, non-uniform), not a
    # fixed grid: total elapsed should match the multi-month wall-clock span.
    assert elapsed_seen == sorted(elapsed_seen), "elapsed_s must be monotonic"
    assert elapsed_seen[-1] / 86400.0 > 90.0, "elapsed_s must reflect the months-long run"
    assert max(rms_v_seen) < 50.0, (
        f"rms_v out of physical range (max={max(rms_v_seen):.1f}); feature index likely wrong"
    )
    assert preds[-1] < preds[0], (
        f"causal envelope RUL should trend down: first={preds[0]:.4f} last={preds[-1]:.4f}"
    )
    print(f"  causal envelope RUL: {len(preds)} preds, first={preds[0]:.4f}, last={preds[-1]:.4f}")
    print(f"  rms_v range: max={max(rms_v_seen):.3f} (sane, < 50)")
    if remaining:
        print(f"  predicted time-left: first={remaining[0]/86400:.1f}d last={remaining[-1]/3600:.1f}h")
    print("  OK")


def test_record_replay() -> None:
    """Record a full run, locate + explain its drops, persist, reload, delete."""
    from app import replay_store as rs

    print("=== Record / replay (XJTU-SY 1_5) ===")
    prog: list[tuple[int, int]] = []
    run = rs.record_run("xjtusy", "1_5", progress_cb=lambda t, n: prog.append((t, n)))
    rid = run["run_id"]
    try:
        m, s = run["meta"], run["summary"]
        assert m["n_frames"] > 10, "expected a full run of frames"
        assert prog and prog[-1][0] == prog[-1][1], "progress must reach 100%"
        # Persisted frames must be replay-sufficient and size-bounded.
        mid = run["frames"][m["n_frames"] // 2]
        assert mid["waveform"]["horizontal"], "frame missing waveform for replay"
        assert len(mid["waveform"]["horizontal"]) <= rs._WAVE_PTS + 1

        # Drops should be explained with the existing interpretability tooling.
        for d in run["drops"]:
            assert 0.0 < d["magnitude"] <= 1.0, f"bad drop magnitude {d['magnitude']}"
            assert d["from_t"] < d["to_t"], "drop must span forward in time"
            assert d.get("hi_delta"), "drop missing raw-HI delta explanation"
            if d.get("ig") is not None:
                assert d["ig"]["features"], "drop IG present but empty"
        print(f"  frames={m['n_frames']} drops={s['n_drops']} biggest={s['biggest_drop']:.3f}")

        # Store roundtrip.
        listed = {r["run_id"] for r in rs.list_runs()}
        assert rid in listed, "saved run missing from index"
        reloaded = rs.load_run(rid)
        assert reloaded and len(reloaded["frames"]) == m["n_frames"], "reload mismatch"
        print(f"  persisted + reloaded {rid}")
    finally:
        assert rs.delete_run(rid), "delete_run failed"
        assert rs.load_run(rid) is None, "run not deleted"
    print("  OK")


def main() -> None:
    test_phm2012_bearing_1_3()
    test_integrated_gradients()
    test_xjtusy_bearing_1_5()
    test_skf_ch15_or1_ch1_01_nde()
    test_record_replay()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
