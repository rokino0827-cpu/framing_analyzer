import sys
from pathlib import Path

import numpy as np
from pytest import approx

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from framing_analyzer.sv2000_evaluator import SV2000Evaluator


def _build_frame_data():
    frame_scores = {
        "conflict": np.array([0.1, 0.5, 0.9]),
        "human": np.array([0.2, 0.6, 0.8]),
        "econ": np.array([0.3, 0.5, 0.7]),
        "moral": np.array([0.0, 0.4, 1.0]),
        "resp": np.array([0.25, 0.55, 0.85]),
    }
    predictions = {f"sv_{k}_pred": v for k, v in frame_scores.items()}
    predictions["sv_frame_avg_pred"] = np.mean(list(frame_scores.values()), axis=0)
    ground_truth = {f"y_{k}": v for k, v in frame_scores.items()}
    return predictions, ground_truth


def _build_fusion_data(frame_avg):
    fusion_results = []
    fusion_ground_truth = []

    for idx, avg_score in enumerate(frame_avg):
        bias_score = 0.15 + 0.3 * idx
        omission_score = bias_score * 0.5
        relative_score = 0.1 * (idx + 1)
        quote_score = 0.05
        final_intensity = (
            0.5 * avg_score
            + 0.2 * bias_score
            + 0.15 * omission_score
            + 0.1 * relative_score
            + 0.05 * quote_score
        )
        fusion_results.append(
            {
                "final_intensity": final_intensity,
                "sv_frame_avg_pred": avg_score,
                "sv_frame_avg": avg_score,
                "bias_score": bias_score,
                "omission_score": omission_score,
                "relative_score": relative_score,
                "quote_score": quote_score,
            }
        )
        fusion_ground_truth.append({"ground_truth_intensity": final_intensity})

    return fusion_results, fusion_ground_truth


def test_sv2000_alignment_metrics_cover_all_frames():
    evaluator = SV2000Evaluator()
    predictions, ground_truth = _build_frame_data()

    results = evaluator.evaluate_frame_alignment(predictions, ground_truth)

    assert results["overall_alignment_score"] == approx(1.0, rel=1e-4)

    for frame, metrics in results["frame_correlations"].items():
        assert metrics["pearson_r"] == approx(1.0)
        assert metrics["spearman_r"] == approx(1.0)
        assert metrics["mae"] == approx(0.0)
        assert metrics["n_samples"] == 3
        assert frame in {"conflict", "human", "econ", "moral", "resp"}

    avg_metrics = results["frame_average_alignment"]
    assert avg_metrics["pearson_r"] == approx(1.0)
    assert avg_metrics["mae"] == approx(0.0)
    assert avg_metrics["r2_score"] == approx(1.0)

    for metrics in results["frame_presence_detection"].values():
        assert metrics["auc_roc"] == approx(1.0)
        assert metrics["auc_pr"] == approx(1.0)
        assert 0.0 <= metrics["positive_rate"] <= 1.0


def test_sv2000_fusion_performance_and_component_contributions(tmp_path):
    evaluator = SV2000Evaluator()
    predictions, ground_truth = _build_frame_data()
    fusion_results, fusion_ground_truth = _build_fusion_data(predictions["sv_frame_avg_pred"])

    fusion_eval = evaluator.evaluate_fusion_performance(fusion_results, fusion_ground_truth)
    fusion_metrics = fusion_eval["performance_comparison"]["fusion"]

    assert fusion_metrics["pearson_r"] == approx(1.0, rel=1e-6)
    assert fusion_metrics["mae"] == approx(0.0)

    component_analysis = fusion_eval["component_analysis"]
    assert set(component_analysis.keys()) == {
        "sv_frame_avg_pred",
        "bias_score",
        "omission_score",
        "relative_score",
        "quote_score",
    }
    assert any(value["correlation_with_final"] > 0.8 for value in component_analysis.values())

    improvements = fusion_eval["performance_comparison"].get("improvements")
    assert improvements is not None
    assert improvements["fusion_vs_sv2000"] >= 0.0
    assert improvements["fusion_vs_bias"] >= 0.0

    alignment_results = evaluator.evaluate_frame_alignment(predictions, ground_truth)
    combined_results = {**alignment_results, **fusion_eval}
    report_path = tmp_path / "report.md"
    report_text = evaluator.generate_evaluation_report(combined_results, output_path=str(report_path))

    assert report_text.startswith("# SV2000 Framing Alignment Evaluation Report")
    assert "SV2000 Frame Alignment" in report_text
    assert "Fusion Performance" in report_text
    assert "Performance Comparison" in report_text
    assert "Improvements" in report_text
    assert report_path.exists()
