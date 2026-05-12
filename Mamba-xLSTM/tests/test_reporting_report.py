import json
from argparse import Namespace
from pathlib import Path

import numpy as np

from mxlstm.reporting.report import build_report
from scripts.build_report import _resolve_paths


def _write_minimal_run(
    root: Path,
    run_id: str,
    *,
    include_predictions: bool = True,
    rmse: float = 0.1,
) -> Path:
    run_dir = root / run_id
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "figures").mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": 42,
                "n_params": 123,
                "fit_seconds": 1.2,
                "test_metrics": {
                    "test/rmse": rmse,
                    "test/mae": rmse / 2,
                    "test/phm_score": 0.9,
                    "test/phm_score_paper": 10.0 + rmse,
                },
                "dataset": "phm2012",
                "model_name": "mamba_xlstm_net",
            }
        )
    )
    (run_dir / "config.yaml").write_text("data:\n  dataset: phm2012\nmodel:\n  name: mamba_xlstm_net\n")
    (run_dir / "logs" / "summary.json").write_text(
        json.dumps({"timings": [{"kind": "step", "phase": "Evaluation", "name": "Test", "elapsed_s": 0.25}]})
    )
    (run_dir / "logs" / "events.jsonl").write_text(
        json.dumps({"ts": 1.0, "level": "INFO", "kind": "step_start", "message": "Test"}) + "\n"
    )

    if include_predictions:
        t = np.arange(8, dtype=np.float32)
        y = np.linspace(1.0, 0.0, 8, dtype=np.float32)
        payload = {}
        for idx, bid in enumerate(["1_3", "2_3"]):
            payload[f"{bid}_t"] = t
            payload[f"{bid}_y"] = y
            payload[f"{bid}_pred"] = np.clip(y + (idx + 1) * 0.02, 0.0, 1.0)
        np.savez_compressed(run_dir / "test_predictions.npz", **payload)

    return run_dir


def test_build_report_regenerates_prediction_figures_and_honors_gallery_limit(tmp_path):
    run_dir = _write_minimal_run(tmp_path, "mamba_xlstm_phm_s42")
    out_html = tmp_path / "reports" / "paper.html"
    stale_dir = tmp_path / "reports" / "paper" / "figures"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.png").write_text("old")

    result = build_report([run_dir], out_html=out_html, out_pdf=None, gallery_limit=1)

    html = out_html.read_text()
    figures_dir = Path(result["figures_dir"])
    assert figures_dir.exists()
    assert not (figures_dir / "stale.png").exists()
    assert result["n_figures"] >= 5
    assert "data:image/png;base64," in html
    assert "1 additional prediction figure(s) omitted by the gallery limit." in html
    assert "1 additional residual figure(s) omitted by the gallery limit." in html
    assert (figures_dir / "mamba_xlstm_phm_s42" / "pred_1_3.png").exists()


def test_build_report_handles_runs_without_prediction_artifacts(tmp_path):
    run_dir = _write_minimal_run(tmp_path, "metrics_only_s42", include_predictions=False)
    out_html = tmp_path / "reports" / "metrics_only.html"

    result = build_report([run_dir], out_html=out_html, out_pdf=None)

    html = out_html.read_text()
    assert result["n_figures"] == 1  # step timing still regenerates from logs/summary.json
    assert "test/rmse" in html
    assert "Per-bearing test predictions" not in html


def test_ablation_section_includes_paper_phm_score_metric(tmp_path):
    run_a = _write_minimal_run(tmp_path, "mamba_xlstm_phm_s42", include_predictions=False, rmse=0.1)
    run_b = _write_minimal_run(tmp_path, "mamba_xlstm_phm_s43", include_predictions=False, rmse=0.2)
    out_html = tmp_path / "reports" / "ablation.html"

    result = build_report([run_a, run_b], out_html=out_html, out_pdf=None)

    html = out_html.read_text()
    assert result["n_figures"] >= 1
    assert "test/phm_score_paper" in html
    assert (Path(result["figures_dir"]) / "ablation").exists()


def test_cli_default_paths_stay_under_reports_dir(tmp_path):
    run_dir = tmp_path / "runs" / "20260101_mamba_xlstm_phm_s42"
    reports_dir = tmp_path / "reports"
    args = Namespace(
        name=None,
        out_html=None,
        out_pdf=None,
        no_pdf=False,
        reports_dir=reports_dir,
    )

    out_html, out_pdf = _resolve_paths(args, [run_dir])

    assert out_html == reports_dir / "20260101_mamba_xlstm_phm_s42.html"
    assert out_pdf == reports_dir / "20260101_mamba_xlstm_phm_s42.pdf"


def test_ablation_rows_use_friendly_labels_for_algorithm_comparison_runs(tmp_path):
    run_d = _write_minimal_run(
        tmp_path, "algorithm_comparison_phm2012_diffusion_rul_s42", include_predictions=False, rmse=0.2
    )
    summary = json.loads((run_d / "summary.json").read_text())
    summary["run_id"] = "algorithm_comparison_phm2012_diffusion_rul_s42"
    summary["dataset"] = "phm2012"
    summary["model_name"] = "diffusion_rul"
    (run_d / "summary.json").write_text(json.dumps(summary))

    run_n = _write_minimal_run(
        tmp_path, "algorithm_comparison_phm2012_nbeats_rul_s42", include_predictions=False, rmse=0.15
    )
    summary_n = json.loads((run_n / "summary.json").read_text())
    summary_n["run_id"] = "algorithm_comparison_phm2012_nbeats_rul_s42"
    summary_n["dataset"] = "phm2012"
    summary_n["model_name"] = "nbeats_rul"
    (run_n / "summary.json").write_text(json.dumps(summary_n))

    out_html = tmp_path / "reports" / "algo_compare.html"
    build_report([run_d, run_n], out_html=out_html, out_pdf=None)
    text = out_html.read_text()
    assert "PHM2012 \u00b7 Diffusion-RUL" in text
    assert "PHM2012 \u00b7 N-BEATS-RUL" in text


def test_ablation_rows_use_cond12_label_for_xjtu_available_comparison_runs(tmp_path):
    run_a = _write_minimal_run(
        tmp_path,
        "algorithm_comparison_xjtu_available_nbeats_rul_s42",
        include_predictions=False,
        rmse=0.12,
    )
    summary_a = json.loads((run_a / "summary.json").read_text())
    summary_a["run_id"] = "algorithm_comparison_xjtu_available_nbeats_rul_s42"
    summary_a["dataset"] = "xjtu_available"
    summary_a["model_name"] = "nbeats_rul"
    (run_a / "summary.json").write_text(json.dumps(summary_a))

    run_b = _write_minimal_run(
        tmp_path,
        "algorithm_comparison_xjtu_available_diffusion_rul_s43",
        include_predictions=False,
        rmse=0.09,
    )
    summary_b = json.loads((run_b / "summary.json").read_text())
    summary_b["run_id"] = "algorithm_comparison_xjtu_available_diffusion_rul_s43"
    summary_b["dataset"] = "xjtu_available"
    summary_b["model_name"] = "diffusion_rul"
    (run_b / "summary.json").write_text(json.dumps(summary_b))

    out_html = tmp_path / "reports" / "algo_compare_xjtu_avail.html"
    build_report([run_a, run_b], out_html=out_html, out_pdf=None)
    text = out_html.read_text()
    assert "XJTU-SY (cond. 1\u20132) \u00b7 N-BEATS-RUL" in text
    assert "XJTU-SY (cond. 1\u20132) \u00b7 Diffusion-RUL" in text
