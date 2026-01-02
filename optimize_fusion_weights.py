#!/usr/bin/env python3
"""
融合权重优化脚本
用于优化SV2000多组件融合的权重配置
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from framing_analyzer.fusion_scorer import FusionWeightOptimizer, ComponentAnalyzer
from framing_analyzer.config import create_sv2000_config, FusionConfig
from framing_analyzer.sv2000_data_loader import SV2000DataLoader
from framing_analyzer import FramingAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_fusion_data(data_path: str, config) -> List[Dict[str, float]]:
    """加载用于融合优化的数据"""
    logger.info(f"加载融合优化数据: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载数据
    data_loader = SV2000DataLoader(data_path, config)
    train_data, val_data = data_loader.load_training_data(validation_split=0.2)
    
    # 模拟分析器获取各组件分数
    analyzer = FramingAnalyzer(config)
    
    fusion_data = []
    
    logger.info("分析样本以获取组件分数...")
    for i, sample in enumerate(train_data):
        if i % 50 == 0:
            logger.info(f"处理进度: {i}/{len(train_data)}")
        
        try:
            # 分析文章获取组件分数
            result = analyzer.analyze_article(
                sample['content'], 
                title=sample.get('title', '')
            )
            
            # 构建融合数据点
            fusion_point = {
                # 组件分数
                'sv_frame_avg_pred': getattr(result, 'sv_frame_avg', 0.0) or 0.0,
                'bias_score': result.components.get('headline', 0.0) * 0.3 + 
                             result.components.get('lede', 0.0) * 0.4 + 
                             result.components.get('narration', 0.0) * 0.3,
                'omission_score': getattr(result, 'omission_score', 0.0) or 0.0,
                'relative_score': 0.0,  # 暂时设为0
                'quote_score': result.components.get('quotes', 0.0),
                
                # 真实标签（计算平均框架强度作为目标）
                'ground_truth_intensity': np.mean([
                    sample.get('y_conflict', 0.0),
                    sample.get('y_human', 0.0),
                    sample.get('y_econ', 0.0),
                    sample.get('y_moral', 0.0),
                    sample.get('y_resp', 0.0)
                ])
            }
            
            fusion_data.append(fusion_point)
            
        except Exception as e:
            logger.warning(f"处理样本 {i} 时出错: {e}")
            continue
    
    logger.info(f"成功处理 {len(fusion_data)} 个样本")
    return fusion_data

def generate_weight_candidates() -> List[Dict[str, float]]:
    """生成权重候选配置"""
    candidates = []
    
    # 默认配置
    candidates.append({
        'sv_frame_avg_pred': 0.5,
        'bias_score': 0.2,
        'omission_score': 0.15,
        'relative_score': 0.1,
        'quote_score': 0.05
    })
    
    # SV2000主导配置
    candidates.append({
        'sv_frame_avg_pred': 0.7,
        'bias_score': 0.15,
        'omission_score': 0.1,
        'relative_score': 0.03,
        'quote_score': 0.02
    })
    
    # 平衡配置
    candidates.append({
        'sv_frame_avg_pred': 0.4,
        'bias_score': 0.25,
        'omission_score': 0.2,
        'relative_score': 0.1,
        'quote_score': 0.05
    })
    
    # 偏见检测主导配置
    candidates.append({
        'sv_frame_avg_pred': 0.3,
        'bias_score': 0.4,
        'omission_score': 0.15,
        'relative_score': 0.1,
        'quote_score': 0.05
    })
    
    # 省略检测增强配置
    candidates.append({
        'sv_frame_avg_pred': 0.45,
        'bias_score': 0.15,
        'omission_score': 0.3,
        'relative_score': 0.05,
        'quote_score': 0.05
    })
    
    return candidates

def optimize_weights_ridge(fusion_data: List[Dict], config: FusionConfig) -> Dict[str, float]:
    """使用Ridge回归优化权重"""
    logger.info("使用Ridge回归优化融合权重...")
    
    optimizer = FusionWeightOptimizer(config)
    optimized_weights = optimizer.optimize_weights(fusion_data)
    
    logger.info("Ridge回归优化结果:")
    for component, weight in optimized_weights.items():
        logger.info(f"  {component}: {weight:.4f}")
    
    return optimized_weights

def cross_validate_weights(fusion_data: List[Dict], config: FusionConfig) -> Dict[str, Any]:
    """交叉验证不同权重配置"""
    logger.info("交叉验证权重配置...")
    
    optimizer = FusionWeightOptimizer(config)
    weight_candidates = generate_weight_candidates()
    
    cv_results = optimizer.cross_validate_weights(fusion_data, weight_candidates)
    
    logger.info("交叉验证结果:")
    for config_name, results in cv_results.items():
        logger.info(f"  {config_name}:")
        logger.info(f"    相关性: {results['correlation']:.4f}")
        logger.info(f"    MAE: {results['mae']:.4f}")
        logger.info(f"    综合评分: {results['score']:.4f}")
    
    # 找到最佳配置
    best_config = max(cv_results.items(), key=lambda x: x[1]['score'])
    logger.info(f"最佳配置: {best_config[0]} (评分: {best_config[1]['score']:.4f})")
    
    return cv_results

def analyze_component_importance(fusion_data: List[Dict]) -> Dict[str, Any]:
    """分析组件重要性"""
    logger.info("分析组件重要性...")
    
    # 模拟融合结果
    fusion_results = []
    for data_point in fusion_data:
        fusion_result = {
            'final_intensity': data_point['ground_truth_intensity'],
            'sv_frame_avg_pred': data_point['sv_frame_avg_pred'],
            'bias_score': data_point['bias_score'],
            'omission_score': data_point['omission_score'],
            'relative_score': data_point['relative_score'],
            'quote_score': data_point['quote_score']
        }
        fusion_results.append(fusion_result)
    
    # 分析组件相关性
    correlations = ComponentAnalyzer.analyze_component_correlations(fusion_results)
    
    logger.info("组件重要性分析:")
    for component, metrics in correlations.items():
        logger.info(f"  {component}:")
        logger.info(f"    与最终分数相关性: {metrics['correlation_with_final']:.4f}")
        logger.info(f"    平均分数: {metrics['mean_score']:.4f}")
        logger.info(f"    贡献强度: {metrics['contribution_strength']:.4f}")
    
    return correlations

def save_optimization_results(results: Dict[str, Any], output_path: str):
    """保存优化结果"""
    logger.info(f"保存优化结果到: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存JSON格式结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 生成可读报告
    report_path = output_path.replace('.json', '_report.md')
    generate_optimization_report(results, report_path)

def generate_optimization_report(results: Dict[str, Any], report_path: str):
    """生成优化报告"""
    report_lines = [
        "# 融合权重优化报告",
        "",
        f"生成时间: {results.get('timestamp', 'N/A')}",
        f"数据样本数: {results.get('sample_count', 'N/A')}",
        "",
        "## Ridge回归优化结果",
        ""
    ]
    
    if 'ridge_optimized_weights' in results:
        report_lines.append("### 优化后权重")
        for component, weight in results['ridge_optimized_weights'].items():
            report_lines.append(f"- **{component}**: {weight:.4f}")
        report_lines.append("")
    
    if 'cross_validation_results' in results:
        report_lines.extend([
            "## 交叉验证结果",
            "",
            "| 配置 | 相关性 | MAE | 综合评分 |",
            "|------|--------|-----|----------|"
        ])
        
        for config_name, cv_result in results['cross_validation_results'].items():
            corr = cv_result.get('correlation', 0)
            mae = cv_result.get('mae', 0)
            score = cv_result.get('score', 0)
            report_lines.append(f"| {config_name} | {corr:.4f} | {mae:.4f} | {score:.4f} |")
        
        report_lines.append("")
    
    if 'component_importance' in results:
        report_lines.extend([
            "## 组件重要性分析",
            "",
            "| 组件 | 相关性 | 平均分数 | 贡献强度 |",
            "|------|--------|----------|----------|"
        ])
        
        for component, metrics in results['component_importance'].items():
            corr = metrics.get('correlation_with_final', 0)
            mean_score = metrics.get('mean_score', 0)
            contrib = metrics.get('contribution_strength', 0)
            report_lines.append(f"| {component} | {corr:.4f} | {mean_score:.4f} | {contrib:.4f} |")
        
        report_lines.append("")
    
    report_lines.extend([
        "## 建议",
        "",
        "1. 使用Ridge回归优化的权重作为基准配置",
        "2. 根据具体应用场景调整权重",
        "3. 定期重新优化权重以适应新数据",
        "4. 关注高贡献强度的组件",
        ""
    ])
    
    # 写入报告文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"优化报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="融合权重优化脚本")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="训练数据CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="./optimization_results",
                       help="优化结果输出目录")
    
    # 优化参数
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
                       help="Ridge回归正则化参数")
    parser.add_argument("--cv_folds", type=int, default=5,
                       help="交叉验证折数")
    parser.add_argument("--max_samples", type=int, default=1000,
                       help="最大样本数（用于加速）")
    
    # 分析选项
    parser.add_argument("--skip_ridge", action="store_true",
                       help="跳过Ridge回归优化")
    parser.add_argument("--skip_cv", action="store_true",
                       help="跳过交叉验证")
    parser.add_argument("--skip_analysis", action="store_true",
                       help="跳过组件重要性分析")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建配置
        config = create_sv2000_config()
        config.fusion.ridge_alpha = args.ridge_alpha
        config.fusion.cross_validation_folds = args.cv_folds
        
        # 加载数据
        fusion_data = load_fusion_data(args.data_path, config)
        
        # 限制样本数量（如果指定）
        if args.max_samples and len(fusion_data) > args.max_samples:
            logger.info(f"限制样本数量为 {args.max_samples}")
            fusion_data = fusion_data[:args.max_samples]
        
        # 初始化结果字典
        results = {
            'timestamp': str(np.datetime64('now')),
            'sample_count': len(fusion_data),
            'parameters': {
                'ridge_alpha': args.ridge_alpha,
                'cv_folds': args.cv_folds,
                'max_samples': args.max_samples
            }
        }
        
        # Ridge回归优化
        if not args.skip_ridge:
            ridge_weights = optimize_weights_ridge(fusion_data, config.fusion)
            results['ridge_optimized_weights'] = ridge_weights
        
        # 交叉验证
        if not args.skip_cv:
            cv_results = cross_validate_weights(fusion_data, config.fusion)
            results['cross_validation_results'] = cv_results
        
        # 组件重要性分析
        if not args.skip_analysis:
            component_importance = analyze_component_importance(fusion_data)
            results['component_importance'] = component_importance
        
        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, "fusion_optimization_results.json")
        save_optimization_results(results, output_path)
        
        logger.info("融合权重优化完成！")
        
        # 显示推荐配置
        if 'ridge_optimized_weights' in results:
            logger.info("推荐的融合权重配置:")
            for component, weight in results['ridge_optimized_weights'].items():
                logger.info(f"  config.fusion.{component.replace('_pred', '').replace('_score', '')}: {weight:.4f}")
        
    except Exception as e:
        logger.error(f"优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()