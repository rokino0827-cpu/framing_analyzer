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

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

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
    config.sv_framing.batch_size = args.batch_size
    config.sv_framing.dropout_rate = args.dropout_rate
    config.sv_framing.training_mode = args.training_mode
    config.sv_framing.device = args.device
    config.sv_framing.model_save_path = args.output_dir
    config.sv_framing.max_length = args.max_length
    
    # 编码器配置
    if args.encoder_name:
        config.sv_framing.encoder_name = args.encoder_name
    if args.encoder_local_path:
        config.sv_framing.encoder_local_path = args.encoder_local_path
    
    # 融合配置
    config.fusion.use_ridge_optimization = args.optimize_weights
    config.fusion.ridge_alpha = args.ridge_alpha
    config.fusion.cross_validation_folds = args.cv_folds
    
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
    logger.info(f"  验证损失: {final_metrics.get('val_loss', 'N/A')}")
    logger.info(f"  平均相关性: {final_metrics.get('avg_correlation', 'N/A')}")
    logger.info(f"  最佳轮次: {final_metrics.get('best_epoch', 'N/A')}")
    
    if 'optimized_weights' in training_report:
        logger.info("优化后的融合权重:")
        for component, weight in training_report['optimized_weights'].items():
            logger.info(f"  {component}: {weight:.4f}")
    
    # 模型文件路径
    model_path = os.path.join(args.output_dir, "best_sv2000_model.pt")
    if os.path.exists(model_path):
        logger.info(f"最佳模型已保存: {model_path}")
    
    return training_report

def evaluate_model(args, training_report: Dict[str, Any]):
    """评估训练好的模型"""
    logger.info("评估训练好的模型...")
    
    # 加载评估器
    evaluator = SV2000Evaluator()
    
    # 从训练报告中获取预测结果
    if 'validation_predictions' in training_report:
        predictions = training_report['validation_predictions']
        ground_truth = training_report['validation_ground_truth']
        
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
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="./sv2000_models",
                       help="模型输出目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=10,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
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
                       default="sentence-transformers/all-MiniLM-L6-v2",
                       help="编码器模型名称")
    parser.add_argument("--encoder_local_path", type=str,
                       help="本地编码器路径")
    parser.add_argument("--training_mode", type=str, default="frame_level",
                       choices=["frame_level", "item_level"],
                       help="训练模式")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备 (auto/cpu/cuda)")
    
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
