"""SV2000集成测试：使用真实数据与完整评估流程。"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from pytest import approx

# 使用无GUI后端，避免测试环境缺少显示器
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from framing_analyzer.sv2000_evaluator import SV2000Evaluator


DATA_PATH = ROOT / "data/stratified_validation_sample_by_frame_avg.csv"
RESULT_JSON = ROOT / "results/test_sample/SV2000_1.json"
FRAME_COLUMNS = {
    "conflict": [
        "sv_conflict_q1_reflects_disagreement",
        "sv_conflict_q2_refers_to_two_sides",
        "sv_conflict_q3_refers_to_winners_losers_optional",
        "sv_conflict_q4_reproach_between_sides",
    ],
    "human": [
        "sv_human_q1_human_example_or_face",
        "sv_human_q2_adjectives_personal_vignettes",
        "sv_human_q3_feelings_empathy",
        "sv_human_q4_how_people_affected",
        "sv_human_q5_visual_information_optional",
    ],
    "econ": [
        "sv_econ_q1_financial_losses_gains",
        "sv_econ_q2_costs_degree_of_expense",
        "sv_econ_q3_economic_consequences_pursue_or_not",
    ],
    "moral": [
        "sv_moral_q1_moral_message",
        "sv_moral_q2_morality_god_religious_tenets",
        "sv_moral_q3_social_prescriptions",
    ],
    "resp": [
        "sv_resp_q1_government_ability_solve",
        "sv_resp_q2_individual_group_responsible",
        "sv_resp_q3_government_responsible",
        "sv_resp_q4_solution_proposed",
        "sv_resp_q5_urgent_action_required_optional",
    ],
}


def _gpu_usage_summary() -> str:
    """返回GPU占用概览，若不可用则给出说明。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return "GPU: 不可用，使用CPU执行"

        summaries = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024**3)
            summaries.append(f"GPU[{idx}] {props.name} 总显存≈{total_gb:.2f}GB")
        return "GPU: 可用；" + "; ".join(summaries)
    except Exception as exc:  # pragma: no cover - 仅用于健壮性
        return f"GPU: 状态检查失败 ({exc.__class__.__name__})"


def _build_integration_inputs(sample_size: int = 20):
    df = pd.read_csv(DATA_PATH, nrows=sample_size)
    frame_avg = df["sv_frame_avg"].to_numpy(dtype=float)
    scale = np.linspace(0.05, 0.95, sample_size)
    noise = np.linspace(0.0, 0.02, sample_size)

    frame_scores = {}
    for frame, columns in FRAME_COLUMNS.items():
        frame_scores[frame] = df[columns].mean(axis=1).to_numpy(dtype=float)

    predictions = {f"sv_{k}_pred": v for k, v in frame_scores.items()}
    predictions["sv_frame_avg_pred"] = np.mean(list(frame_scores.values()), axis=0)

    ground_truth = {
        f"y_{k}": np.clip(v + noise, 0, 1) for k, v in frame_scores.items()
    }

    fusion_results = []
    fusion_ground_truth = []
    for idx, avg_score in enumerate(predictions["sv_frame_avg_pred"]):
        bias_score = float(np.clip(frame_avg[idx] + 0.1 * scale[idx], 0, 1))
        omission_score = float(np.clip(bias_score * 0.75, 0, 1))
        relative_score = float(0.08 + 0.01 * idx)  # 随着样本增加略微上升
        quote_score = float(0.04 + 0.005 * (idx % 4))
        final_intensity = float(
            0.5 * avg_score
            + 0.2 * bias_score
            + 0.15 * omission_score
            + 0.1 * relative_score
            + 0.05 * quote_score
        )

        fusion_results.append(
            {
                "final_intensity": final_intensity,
                "sv_frame_avg_pred": float(avg_score),
                "sv_frame_avg": float(avg_score),
                "bias_score": bias_score,
                "omission_score": omission_score,
                "relative_score": relative_score,
                "quote_score": quote_score,
            }
        )
        fusion_ground_truth.append(
            {"ground_truth_intensity": float(np.clip(final_intensity + noise[idx], 0, 1))}
        )

    return predictions, ground_truth, fusion_results, fusion_ground_truth


def test_sv2000_end_to_end_pipeline(tmp_path):
    print(_gpu_usage_summary())
    evaluator = SV2000Evaluator()
    predictions, ground_truth, fusion_results, fusion_ground_truth = _build_integration_inputs()

    alignment_results = evaluator.evaluate_frame_alignment(predictions, ground_truth)
    assert alignment_results["overall_alignment_score"] >= 0.6
    assert set(alignment_results["frame_correlations"].keys()) == {
        "conflict",
        "human",
        "econ",
        "moral",
        "resp",
    }

    fusion_eval = evaluator.evaluate_fusion_performance(fusion_results, fusion_ground_truth)
    fusion_metrics = fusion_eval["performance_comparison"]["fusion"]
    assert fusion_metrics["pearson_r"] == approx(1.0, rel=1e-3)
    assert fusion_metrics["mae"] < 0.02
    assert "improvements" in fusion_eval["performance_comparison"]

    combined_results = {**alignment_results, **fusion_eval}
    report_path = tmp_path / "integration_report.md"
    report_text = evaluator.generate_evaluation_report(combined_results, output_path=str(report_path))

    assert report_text.startswith("# SV2000 Framing Alignment Evaluation Report")
    assert "Fusion Performance" in report_text
    assert report_path.exists()

    plot_dir = tmp_path / "plots"
    evaluator.create_visualization(combined_results, output_dir=str(plot_dir))
    assert (plot_dir / "frame_correlations.png").exists()
    assert (plot_dir / "component_contributions.png").exists()

    RESULT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, ensure_ascii=False, indent=2)

    assert RESULT_JSON.exists()
