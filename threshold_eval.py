from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from framing_analyzer.config import create_sv2000_config
from framing_analyzer.sv2000_evaluator import SV2000Evaluator
from framing_analyzer.sv2000_trainer import SV2000TrainingPipeline

DEFAULT_DATA_PATH = "/root/autodl-tmp/framing_analyzer/data/filtered_labels_with_average.csv"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/framing_analyzer/outputs/threshold_eval"
DEFAULT_MORAL_THRESHOLD = 0.65
DEFAULT_RESP_THRESHOLD = 0.65
DEFAULT_MIN_PRECISION = 0.2
DEFAULT_MIN_RECALL = 0.2
DEFAULT_F_BETA = 0.5
DEFAULT_BASE_THRESHOLD = 0.5
DEFAULT_LABEL_THRESHOLD = 0.5


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_dir / path


def run_eval(
    *,
    tag: str,
    pipeline: SV2000TrainingPipeline,
    output_dir: Path,
    presence_thresholds: Dict[str, float] | None = None,
    label_threshold: float = DEFAULT_LABEL_THRESHOLD,
    apply_thresholds: bool | None = None,
) -> Dict[str, Any]:
    """Run SV2000 evaluation and save report/metrics for a tag."""
    texts, targets = pipeline.data_loader.get_training_data(
        mode=pipeline.trainer.config.training_mode, split="test"
    )
    metrics, preds, targets_for_metrics = pipeline.trainer.evaluate(
        texts, targets, return_outputs=True, apply_thresholds=apply_thresholds
    )

    evaluator = SV2000Evaluator()
    gt = pipeline._targets_to_frame_dict(targets_for_metrics)
    results = evaluator.evaluate_frame_alignment(
        preds,
        gt,
        presence_thresholds=presence_thresholds,
        label_threshold=label_threshold,
    )

    report_path = output_dir / tag / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.generate_evaluation_report(results, str(report_path))

    metrics_path = output_dir / tag / "metrics.json"
    metrics_path.write_text(
        json.dumps({"metrics": metrics, "alignment": results}, ensure_ascii=False, indent=2)
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SV2000 evaluation with and without presence thresholds."
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to filtered labels CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write evaluation reports.",
    )
    parser.add_argument(
        "--moral-threshold",
        type=float,
        default=DEFAULT_MORAL_THRESHOLD,
        help="Presence threshold for moral frame.",
    )
    parser.add_argument(
        "--resp-threshold",
        type=float,
        default=DEFAULT_RESP_THRESHOLD,
        help="Presence threshold for responsibility frame.",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=DEFAULT_MIN_PRECISION,
        help="Minimum precision target when tuning thresholds.",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=DEFAULT_MIN_RECALL,
        help="Minimum recall target when tuning thresholds.",
    )
    parser.add_argument(
        "--f-beta",
        type=float,
        default=DEFAULT_F_BETA,
        help="F-beta used for threshold tuning (beta < 1 favors precision).",
    )
    parser.add_argument(
        "--base-threshold",
        type=float,
        default=DEFAULT_BASE_THRESHOLD,
        help="Baseline threshold used to compute precision/recall constraints.",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Apply temperature calibration on validation set before evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = resolve_path(args.data_path, base_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    output_dir = resolve_path(args.output_dir, base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = create_sv2000_config()
    config.sv_framing.apply_presence_thresholds = False
    config.sv_framing.frame_presence_thresholds = None

    pipeline = SV2000TrainingPipeline(config, str(data_path))
    pipeline.data_loader.create_train_val_split(val_ratio=0.2, test_ratio=0.1)
    pipeline.trainer.load_model(config.sv_framing.pretrained_model_path)

    evaluator = SV2000Evaluator()
    val_texts, val_targets = pipeline.data_loader.get_training_data(
        mode=pipeline.trainer.config.training_mode, split="val"
    )
    tuned_thresholds = {}
    tuning_details = {}
    baseline_thresholds = {name: args.base_threshold for name in evaluator.frame_names}
    if val_texts:
        if args.calibrate:
            pipeline.trainer.calibrate_with_temperature(val_texts, val_targets)
        _, val_preds, val_targets_for_metrics = pipeline.trainer.evaluate(
            val_texts,
            val_targets,
            return_outputs=True,
            apply_thresholds=False,
        )
        val_gt = pipeline._targets_to_frame_dict(val_targets_for_metrics)
        baseline_presence = evaluator.evaluate_frame_alignment(
            val_preds,
            val_gt,
            presence_thresholds=baseline_thresholds,
            label_threshold=DEFAULT_LABEL_THRESHOLD,
        ).get("frame_presence_detection", {})
        min_precision = {}
        min_recall = {}
        for frame in ["moral", "resp"]:
            frame_metrics = baseline_presence.get(frame, {})
            min_precision[frame] = frame_metrics.get("precision", args.min_precision)
            min_recall[frame] = frame_metrics.get("recall", args.min_recall)
        tuned_thresholds, tuning_details = evaluator.tune_presence_thresholds(
            val_preds,
            val_gt,
            frames=["moral", "resp"],
            beta=args.f_beta,
            min_precision=min_precision,
            min_recall=min_recall,
        )

    baseline = run_eval(
        tag="baseline",
        pipeline=pipeline,
        output_dir=output_dir,
        presence_thresholds=baseline_thresholds,
        label_threshold=DEFAULT_LABEL_THRESHOLD,
        apply_thresholds=False,
    )

    if not tuned_thresholds:
        tuned_thresholds = {
            "moral": args.moral_threshold,
            "resp": args.resp_threshold,
        }

    tuned = run_eval(
        tag="with_thresholds",
        pipeline=pipeline,
        output_dir=output_dir,
        presence_thresholds={**baseline_thresholds, **tuned_thresholds},
        label_threshold=DEFAULT_LABEL_THRESHOLD,
        apply_thresholds=False,
    )

    summary_path = output_dir / "comparison_presence.json"
    summary_path.write_text(
        json.dumps(
            {
                "baseline": baseline.get("frame_presence_detection", {}),
                "with_thresholds": tuned.get("frame_presence_detection", {}),
                "tuned_thresholds": tuned_thresholds,
                "tuning_details": tuning_details,
                "baseline_thresholds": baseline_thresholds,
                "label_threshold": DEFAULT_LABEL_THRESHOLD,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"完成，对比报告目录: {output_dir}")


if __name__ == "__main__":
    main()
