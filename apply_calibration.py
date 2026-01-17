"""
将训练报告中的温度标定与logit偏置应用到SV2000模型，并输出带标定的新权重。
"""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from framing_analyzer.config import create_sv2000_config

try:
    from framing_analyzer.sv_framing_head import SVFramingHead
    _IMPORT_ERROR = None
except ImportError as exc:  # 捕获未安装依赖的情况，给出清晰提示
    SVFramingHead = None
    _IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="从训练报告读取标定参数，固化到SV2000模型权重。"
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("outputs/run4_bias_calib/training_report.json"),
        help="训练报告路径，需包含 calibration.temperature 与 calibration.bias 字段。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("outputs/run4_bias_calib/best_sv2000_model.pt"),
        help="原始SV2000模型权重路径。",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/run4_bias_calib/best_sv2000_model_calibrated.pt"),
        help="输出的带标定权重路径。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if _IMPORT_ERROR is not None:
        print(
            "缺少依赖，加载 SVFramingHead 失败，请先安装 sentence-transformers (或确保本地离线包可用)：\n"
            f"ImportError: {_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 1

    if not torch.cuda.is_available():
        print("CUDA 不可用，已停止；请在有 GPU 的环境运行该脚本。", file=sys.stderr)
        return 1

    if not args.report_path.is_file():
        print(f"训练报告不存在: {args.report_path}", file=sys.stderr)
        return 1

    if not args.model_path.is_file():
        print(f"模型权重不存在: {args.model_path}", file=sys.stderr)
        return 1

    report = json.loads(args.report_path.read_text())
    calibration = report.get("calibration") or {}
    temperature = calibration.get("temperature")
    bias = calibration.get("bias")

    cfg = create_sv2000_config().sv_framing
    model = SVFramingHead(cfg)
    model.load_model(str(args.model_path))

    if temperature is not None:
        temperature_tensor = torch.as_tensor(temperature, dtype=torch.float32)
        if temperature_tensor.numel() != 5:
            raise ValueError("temperature 向量长度必须为 5")
        model.set_temperature(temperature_tensor)

    if bias is not None:
        bias_tensor = torch.as_tensor(bias, dtype=torch.float32)
        if bias_tensor.numel() != 5:
            raise ValueError("bias 向量长度必须为 5")
        model.set_logit_bias(bias_tensor)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_path))
    print(f"标定模型已保存: {args.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
