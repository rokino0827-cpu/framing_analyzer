#!/usr/bin/env python3
"""
SV2000模型训练脚本
用于训练自定义的SV2000框架预测模型
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

# 添加项目路径：指向仓库上级目录以启用包名 `framing_analyzer`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from framing_analyzer.sv2000_trainer import SV2000TrainingPipeline
from framing_analyzer.sv2000_data_loader import SV2000DataLoader
from framing_analyzer.config import create_sv2000_config, AnalyzerConfig, SVFramingConfig
from framing_analyzer.sv2000_evaluator import SV2000Evaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NpEncoder(json.JSONEncoder):
    """JSON编码器，安全处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def create_training_config(args) -> AnalyzerConfig:
    """创建训练配置"""
    config = create_sv2000_config()
    
    # 更新配置参数
    config.sv_framing.learning_rate = args.learning_rate
    config.sv_framing.encoder_learning_rate = args.encoder_learning_rate or args.learning_rate
    # 恢复run3的头部学习率策略：未显式指定时使用max(learning_rate, 1e-3)
    config.sv_framing.head_learning_rate = (
        args.head_learning_rate if args.head_learning_rate is not None
        else max(args.learning_rate, 1e-3)
    )
    config.sv_framing.batch_size = args.batch_size
    config.sv_framing.dropout_rate = args.dropout_rate
    config.sv_framing.training_mode = args.training_mode
    config.sv_framing.device = args.device
    config.sv_framing.model_save_path = args.output_dir
    config.sv_framing.max_length = args.max_length
    config.sv_framing.fine_tune_encoder = args.fine_tune_encoder
    config.sv_framing.trainable_encoder_layers = args.trainable_encoder_layers
    config.sv_framing.max_grad_norm = args.max_grad_norm
    
    # 编码器配置
    if args.encoder_name:
        config.sv_framing.encoder_name = args.encoder_name
    if args.encoder_local_path:
        config.sv_framing.encoder_local_path = args.encoder_local_path
    
    # 融合配置
    config.fusion.use_ridge_optimization = args.optimize_weights
    config.fusion.ridge_alpha = args.ridge_alpha
    config.fusion.cross_validation_folds = args.cv_folds

    # 分组切分配置
    config.sv_framing.use_group_split = not args.disable_group_split
    if args.group_column:
        config.sv_framing.group_column = args.group_column
    if args.publication_column:
        config.sv_framing.publication_column = args.publication_column
    if args.time_column:
        config.sv_framing.time_column = args.time_column
    if args.time_freq:
        config.sv_framing.time_freq = args.time_freq
    # 采样与损失
    config.sv_framing.balance_batches = not args.no_balance_batches
    config.sv_framing.loss_type = args.loss_type
    config.sv_framing.focal_gamma = args.focal_gamma
    config.sv_framing.dynamic_frame_reweight = not args.disable_dynamic_reweight
    config.sv_framing.item_consistency_weight = args.item_consistency_weight
    config.sv_framing.return_item_predictions = args.return_item_predictions
    
    # 框架loss权重（JSON字符串）
    if args.frame_loss_weights:
        try:
            config.sv_framing.frame_loss_weights = json.loads(args.frame_loss_weights)
        except json.JSONDecodeError as exc:
            raise ValueError("frame_loss_weights 需为JSON格式，例如 '{\"moral\":1.6,\"resp\":1.4}'") from exc
    
    return config

def validate_data(data_path: str, config: AnalyzerConfig) -> Dict[str, Any]:
    """验证训练数据"""
    logger.info(f"验证训练数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"训练数据文件不存在: {data_path}")
    
    # 加载数据验证器
    data_loader = SV2000DataLoader(data_path, config)
    
    # 验证数据格式
    validation_results = data_loader.validate_annotation_format()

    logger.info("数据验证结果:")
    logger.info(f"  总样本数: {validation_results['total_samples']}")
    logger.info(f"  有效样本数: {validation_results['valid_samples']}")
    missing_fields = validation_results.get('missing_fields', [])
    logger.info(f"  缺失字段: {missing_fields}")
    
    if validation_results['valid_samples'] == 0:
        raise ValueError("没有有效的训练样本")
    
    if validation_results['valid_samples'] < 50:
        logger.warning("训练样本数量较少，可能影响模型性能")
    
    return validation_results

def train_model(args):
    """训练SV2000模型"""
    logger.info("开始SV2000模型训练")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建配置
    config = create_training_config(args)
    
    # 验证数据
    validation_results = validate_data(args.data_path, config)
    
    # 初始化训练管道
    logger.info("初始化训练管道...")
    trainer = SV2000TrainingPipeline(config, args.data_path)
    
    # 运行训练
    logger.info(f"开始训练 {args.epochs} 轮...")
    training_report = trainer.run_full_training(
        num_epochs=args.epochs,
        validation_split=args.validation_split,
        test_split=args.test_split,
        early_stopping_patience=args.early_stopping_patience,
        save_best_model=True
    )
    
    # 保存训练报告
    report_path = os.path.join(args.output_dir, "training_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(training_report, f, indent=2, ensure_ascii=False, cls=NpEncoder)
    
    logger.info(f"训练报告已保存: {report_path}")
    
    # 显示训练结果
    logger.info("训练完成！")
    logger.info("最终结果:")
    
    final_metrics = training_report.get('final_metrics', {})
    val_metrics = final_metrics.get('val', {}) if isinstance(final_metrics, dict) else {}
    logger.info(f"  验证损失: {val_metrics.get('loss', 'N/A')}")
    logger.info(f"  平均相关性: {val_metrics.get('avg_correlation', 'N/A')}")
    best_epoch = training_report.get('best_epoch') or training_report.get('training_history', {}).get('best_epoch')
    logger.info(f"  最佳轮次: {best_epoch if best_epoch is not None else 'N/A'}")
    
    if 'optimized_weights' in training_report:
        logger.info("优化后的融合权重:")
        for component, weight in training_report['optimized_weights'].items():
            logger.info(f"  {component}: {weight:.4f}")
    
    # 模型文件路径
    model_path = os.path.join(args.output_dir, "best_sv2000_model.pt")
    if os.path.exists(model_path):
        logger.info(f"最佳模型已保存: {model_path}")
    calibrated_path = training_report.get('calibrated_model_path')
    if calibrated_path:
        logger.info(f"标定后模型已保存: {calibrated_path}")
    
    return training_report

def evaluate_model(args, training_report: Dict[str, Any]):
    """评估训练好的模型"""
    logger.info("评估训练好的模型...")
    
    # 加载评估器
    evaluator = SV2000Evaluator()

    def _normalize_ground_truth(ground_truth: Any) -> Dict[str, Any]:
        """统一将ground truth转换为评估器期望的字典格式"""
        if isinstance(ground_truth, dict):
            return ground_truth
        arr = np.asarray(ground_truth)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return {}
        frame_names = ["conflict", "human", "econ", "moral", "resp"]
        gt = {}
        for idx, name in enumerate(frame_names):
            if idx < arr.shape[1]:
                gt[f"y_{name}"] = arr[:, idx]
        gt["frame_avg"] = arr.mean(axis=1)
        return gt
    
    # 从训练报告中获取预测结果
    if 'validation_predictions' in training_report:
        predictions = training_report.get('validation_predictions')
        if predictions is None:
            logger.warning("训练报告未包含验证集预测，跳过评估报告生成")
            return
        ground_truth = _normalize_ground_truth(training_report.get('validation_ground_truth'))
        if not ground_truth:
            logger.warning("验证集ground truth 缺失或为空，跳过评估报告生成")
            return
        
        # 评估框架对齐
        alignment_results = evaluator.evaluate_frame_alignment(predictions, ground_truth)
        
        logger.info("框架对齐评估结果:")
        logger.info(f"  整体对齐分数: {alignment_results['overall_alignment_score']:.4f}")
        
        for frame, metrics in alignment_results['frame_correlations'].items():
            logger.info(f"  {frame}: Pearson={metrics['pearson_r']:.4f}, MAE={metrics['mae']:.4f}")
        
        # 保存评估报告
        eval_report_path = os.path.join(args.output_dir, "evaluation_report.md")
        report_content = evaluator.generate_evaluation_report(alignment_results, eval_report_path)
        
        logger.info(f"评估报告已保存: {eval_report_path}")
    
    else:
        logger.warning("训练报告中没有验证预测结果，跳过评估")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SV2000模型训练脚本")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, default="data/filtered_labels_with_average.csv",
                       help="训练数据CSV文件路径，默认使用本仓库10k样本集")
    parser.add_argument("--output_dir", type=str, default="./sv2000_models",
                       help="模型输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批处理大小")
    # 恢复run3默认：头部学习率可达1e-3，主学习率默认1e-3以匹配最佳实验
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                       help="学习率（冻结编码器时主要作用于分类头）")
    parser.add_argument("--encoder_learning_rate", type=float, default=1e-5,
                       help="编码器学习率（启用微调时生效，默认更低以保证稳定微调）")
    parser.add_argument("--head_learning_rate", type=float, default=None,
                       help="分类头学习率（默认None时回退到max(lr,1e-3)以匹配run3表现）")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Dropout率")
    parser.add_argument("--validation_split", type=float, default=0.2,
                       help="验证集比例")
    parser.add_argument("--test_split", type=float, default=0.1,
                       help="测试集比例")
    parser.add_argument("--early_stopping_patience", type=int, default=3,
                       help="早停耐心值")
    parser.add_argument("--max_length", type=int, default=512,
                       help="编码器最大序列长度，控制截断")
    
    # 模型参数
    parser.add_argument("--encoder_name", type=str, 
                       default="bge_m3",
                       help="编码器模型名称")
    parser.add_argument("--encoder_local_path", type=str,
                       help="本地编码器路径")
    parser.add_argument("--trainable_encoder_layers", type=int, default=2,
                       help="轻量微调：解冻编码器末尾的层数，0表示全冻结（默认2层便于温和提升）")
    parser.add_argument("--training_mode", type=str, default="frame_level",
                       choices=["frame_level", "item_level"],
                       help="训练模式")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备 (auto/cpu/cuda)")
    parser.add_argument("--fine_tune_encoder", action="store_true",
                       help="是否微调编码器（默认冻结）")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪阈值，0或负数表示不裁剪")
    parser.add_argument("--frame_loss_weights", type=str, default=None,
                       help="JSON 字典自定义框架loss权重，如 '{\"moral\":1.6,\"resp\":1.4}'")
    parser.add_argument("--loss_type", type=str, default="focal", choices=["bce", "focal"],
                       help="损失函数类型，默认Focal缓解易样本主导")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal loss gamma")
    parser.add_argument("--item_consistency_weight", type=float, default=0.3,
                       help="题项→框架一致性loss权重，0可关闭")
    parser.add_argument("--no_balance_batches", action="store_true",
                       help="禁用基于标签稀缺度的平衡采样")
    parser.add_argument("--disable_dynamic_reweight", action="store_true",
                       help="禁用基于验证误差的动态框架加权")
    parser.add_argument("--return_item_predictions", action="store_true",
                       help="推理时返回题项概率/得分，便于误差分析")

    # 分组切分
    parser.add_argument("--disable_group_split", action="store_true",
                       help="禁用按事件/媒体时间窗口的Group split")
    parser.add_argument("--group_column", type=str,
                       help="事件簇分组列名，默认 event_cluster/cluster_id 自动检测")
    parser.add_argument("--publication_column", type=str,
                       help="媒体列名，用于 publication+时间 窗口分组")
    parser.add_argument("--time_column", type=str,
                       help="时间列名（ISO时间或日期字符串）")
    parser.add_argument("--time_freq", type=str, default="W",
                       help="时间窗口频率，默认为周W，可设为M/D")
    
    # 融合优化参数
    parser.add_argument("--optimize_weights", action="store_true",
                       help="是否优化融合权重")
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                       help="Ridge回归正则化参数")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="交叉验证折数")
    
    # 其他参数
    parser.add_argument("--evaluate", action="store_true",
                       help="训练后进行评估")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 训练模型
        training_report = train_model(args)
        
        # 评估模型（如果指定）
        if args.evaluate:
            evaluate_model(args, training_report)
        
        logger.info("训练流程完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
