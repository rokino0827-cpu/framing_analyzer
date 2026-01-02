#!/usr/bin/env python3
"""
SV2000框架分析快速开始脚本
提供一键式的训练、优化和测试流程
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_data_preparation(args):
    """运行数据准备"""
    logger.info("步骤 1: 数据准备")
    
    from framing_analyzer.prepare_training_data import main as prepare_main
    
    # 构建参数
    prepare_args = [
        "--generate_sample",
        "--num_samples", str(args.num_samples),
        "--output_dir", args.work_dir,
        "--output_name", "sv2000_sample"
    ]
    
    # 临时替换sys.argv
    original_argv = sys.argv
    sys.argv = ["prepare_training_data.py"] + prepare_args
    
    try:
        prepare_main()
        sample_data_path = os.path.join(args.work_dir, "sv2000_sample_sample.csv")
        logger.info(f"示例数据已生成: {sample_data_path}")
        return sample_data_path
    finally:
        sys.argv = original_argv

def run_model_training(args, data_path):
    """运行模型训练"""
    logger.info("步骤 2: 模型训练")
    
    from framing_analyzer.train_sv2000_model import main as train_main
    
    # 构建参数
    train_args = [
        "--data_path", data_path,
        "--output_dir", os.path.join(args.work_dir, "models"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--device", args.device,
        "--evaluate"
    ]
    
    if args.optimize_weights:
        train_args.append("--optimize_weights")
    
    # 临时替换sys.argv
    original_argv = sys.argv
    sys.argv = ["train_sv2000_model.py"] + train_args
    
    try:
        train_main()
        model_path = os.path.join(args.work_dir, "models", "best_sv2000_model.pt")
        logger.info(f"模型训练完成: {model_path}")
        return model_path
    finally:
        sys.argv = original_argv

def run_weight_optimization(args, data_path):
    """运行权重优化"""
    logger.info("步骤 3: 权重优化")
    
    from framing_analyzer.optimize_fusion_weights import main as optimize_main
    
    # 构建参数
    optimize_args = [
        "--data_path", data_path,
        "--output_dir", os.path.join(args.work_dir, "optimization"),
        "--max_samples", str(min(args.num_samples, 200))  # 限制样本数以加速
    ]
    
    # 临时替换sys.argv
    original_argv = sys.argv
    sys.argv = ["optimize_fusion_weights.py"] + optimize_args
    
    try:
        optimize_main()
        results_path = os.path.join(args.work_dir, "optimization", "fusion_optimization_results.json")
        logger.info(f"权重优化完成: {results_path}")
        return results_path
    finally:
        sys.argv = original_argv

def run_system_test(args, model_path):
    """运行系统测试"""
    logger.info("步骤 4: 系统测试")
    
    from framing_analyzer import FramingAnalyzer
    from framing_analyzer.config import create_sv2000_config
    
    # 创建配置
    config = create_sv2000_config()
    
    # 如果有训练好的模型，配置模型路径
    if model_path and os.path.exists(model_path):
        config.sv_framing.pretrained_model_path = model_path
        logger.info(f"使用训练好的模型: {model_path}")
    else:
        logger.info("使用默认配置进行测试")
    
    # 测试文章
    test_articles = [
        {
            'content': '''
            The ongoing territorial dispute has led to increased military tensions 
            between the two nations. Both sides have mobilized troops along the 
            border, raising concerns about potential armed conflict. International 
            observers are calling for immediate diplomatic intervention.
            ''',
            'title': 'Military Tensions Rise Over Territorial Dispute',
            'id': 'test_1'
        },
        {
            'content': '''
            Local volunteers have organized a massive relief effort for families 
            affected by the recent flooding. Community centers are serving as 
            temporary shelters, while donations of food and clothing continue 
            to pour in from neighboring towns.
            ''',
            'title': 'Community Rallies to Help Flood Victims',
            'id': 'test_2'
        }
    ]
    
    try:
        # 初始化分析器
        analyzer = FramingAnalyzer(config)
        
        # 测试分析
        logger.info("运行测试分析...")
        results = analyzer.analyze_batch(test_articles)
        
        # 显示结果
        logger.info("测试结果:")
        for result in results['results']:
            if not result.get('error'):
                logger.info(f"  文章: {result['title']}")
                logger.info(f"    强度: {result['framing_intensity']:.3f}")
                logger.info(f"    标签: {result['pseudo_label']}")
                
                if 'sv_frame_avg' in result:
                    logger.info(f"    SV2000平均: {result['sv_frame_avg']:.3f}")
                    logger.info(f"    冲突: {result.get('sv_conflict', 0):.3f}")
                    logger.info(f"    人情: {result.get('sv_human', 0):.3f}")
                    logger.info(f"    经济: {result.get('sv_econ', 0):.3f}")
                    logger.info(f"    道德: {result.get('sv_moral', 0):.3f}")
                    logger.info(f"    责任: {result.get('sv_resp', 0):.3f}")
        
        logger.info("系统测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"系统测试失败: {e}")
        return False

def generate_summary_report(args):
    """生成总结报告"""
    logger.info("生成总结报告...")
    
    report_lines = [
        "# SV2000框架分析快速开始报告",
        "",
        f"工作目录: {args.work_dir}",
        f"生成时间: {pd.Timestamp.now()}",
        "",
        "## 执行步骤",
        "",
        "1. ✅ 数据准备 - 生成示例训练数据",
        "2. ✅ 模型训练 - 训练SV2000框架预测模型",
        "3. ✅ 权重优化 - 优化多组件融合权重",
        "4. ✅ 系统测试 - 验证完整系统功能",
        "",
        "## 生成的文件",
        "",
        "### 数据文件",
        f"- `{args.work_dir}/sv2000_sample_sample.csv` - 示例训练数据",
        "",
        "### 模型文件",
        f"- `{args.work_dir}/models/best_sv2000_model.pt` - 训练好的SV2000模型",
        f"- `{args.work_dir}/models/training_report.json` - 训练报告",
        "",
        "### 优化结果",
        f"- `{args.work_dir}/optimization/fusion_optimization_results.json` - 权重优化结果",
        f"- `{args.work_dir}/optimization/fusion_optimization_results_report.md` - 优化报告",
        "",
        "## 下一步建议",
        "",
        "1. **使用真实数据**: 替换示例数据为真实的SV2000标注数据",
        "2. **调整参数**: 根据数据特点调整训练参数",
        "3. **扩展功能**: 启用省略检测、相对框架等高级功能",
        "4. **部署应用**: 将训练好的模型部署到生产环境",
        "",
        "## 使用训练好的模型",
        "",
        "```python",
        "from framing_analyzer import FramingAnalyzer",
        "from framing_analyzer.config import create_sv2000_config",
        "",
        "# 创建配置",
        "config = create_sv2000_config()",
        f"config.sv_framing.pretrained_model_path = '{args.work_dir}/models/best_sv2000_model.pt'",
        "",
        "# 初始化分析器",
        "analyzer = FramingAnalyzer(config)",
        "",
        "# 分析文章",
        "result = analyzer.analyze_article('Your article content here...')",
        "print(f'框架强度: {result.framing_intensity:.3f}')",
        "```",
        "",
        "详细使用说明请参考 `SV2000_USAGE_GUIDE.md`"
    ]
    
    # 保存报告
    report_path = os.path.join(args.work_dir, "quick_start_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"总结报告已保存: {report_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="SV2000框架分析快速开始")
    
    # 基础参数
    parser.add_argument("--work_dir", type=str, default="./sv2000_quickstart",
                       help="工作目录")
    
    # 数据参数
    parser.add_argument("--num_samples", type=int, default=200,
                       help="生成的示例数据样本数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=5,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--device", type=str, default="cpu",
                       help="计算设备")
    
    # 功能选项
    parser.add_argument("--skip_training", action="store_true",
                       help="跳过模型训练")
    parser.add_argument("--skip_optimization", action="store_true",
                       help="跳过权重优化")
    parser.add_argument("--optimize_weights", action="store_true",
                       help="在训练中优化权重")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("SV2000框架分析快速开始")
    logger.info("=" * 60)
    logger.info(f"工作目录: {args.work_dir}")
    
    try:
        # 导入pandas用于报告生成
        import pandas as pd
        
        # 步骤1: 数据准备
        data_path = run_data_preparation(args)
        
        # 步骤2: 模型训练
        model_path = None
        if not args.skip_training:
            model_path = run_model_training(args, data_path)
        
        # 步骤3: 权重优化
        if not args.skip_optimization:
            run_weight_optimization(args, data_path)
        
        # 步骤4: 系统测试
        test_success = run_system_test(args, model_path)
        
        # 生成总结报告
        if test_success:
            generate_summary_report(args)
        
        logger.info("=" * 60)
        logger.info("快速开始流程完成！")
        logger.info("=" * 60)
        
        if test_success:
            logger.info("✅ 所有步骤都成功完成")
            logger.info(f"📁 查看工作目录: {args.work_dir}")
            logger.info(f"📄 查看报告: {args.work_dir}/quick_start_report.md")
        else:
            logger.warning("⚠️  部分步骤可能未完全成功，请查看日志")
        
    except Exception as e:
        logger.error(f"快速开始过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()