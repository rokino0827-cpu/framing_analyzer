#!/usr/bin/env python3
"""
使用现有机器标注数据训练SV2000模型
专门针对用户的stratified_validation_sample_by_frame_avg.csv数据
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

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用现有数据训练SV2000模型")
    
    # 数据路径
    parser.add_argument("--data_path", type=str, 
                       default="data/filtered_labels_with_average.csv",
                       help="数据文件路径")
    parser.add_argument("--work_dir", type=str, default="./sv2000_training_results",
                       help="工作目录")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=15,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="批处理大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="学习率")
    parser.add_argument("--device", type=str, default="auto",
                       help="计算设备")
    
    # 流程控制
    parser.add_argument("--skip_validation", action="store_true",
                       help="跳过数据验证")
    parser.add_argument("--skip_training", action="store_true",
                       help="跳过模型训练")
    parser.add_argument("--skip_optimization", action="store_true",
                       help="跳过权重优化")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 创建工作目录
    os.makedirs(args.work_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("使用现有机器标注数据训练SV2000模型")
    logger.info("=" * 80)
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"工作目录: {args.work_dir}")
    
    try:
        adapted_data_path = args.data_path
        
        # 步骤1: 数据验证和适配
        if not args.skip_validation:
            logger.info("\n步骤 1: 数据验证和适配")
            logger.info("-" * 50)
            
            from framing_analyzer.validate_existing_data import main as validate_main
            
            # 首先检查数据结构
            validate_args = [
                "--input_path", args.data_path,
                "--inspect_only"
            ]
            
            if args.verbose:
                validate_args.append("--verbose")
            
            # 临时替换sys.argv进行检查
            original_argv = sys.argv
            sys.argv = ["validate_existing_data.py"] + validate_args
            
            try:
                validate_main()
            except SystemExit:
                pass  # 忽略正常退出
            finally:
                sys.argv = original_argv
            
            # 询问用户是否需要适配数据
            print("\n" + "=" * 60)
            print("数据检查完成！")
            print("请查看上面的检查报告，确认数据格式是否符合SV2000训练要求。")
            
            # 检查是否是用户的机器标注格式
            user_format_detected = False
            try:
                df_check = pd.read_csv(args.data_path)
                user_format_detected = ('sv_frame_avg' in df_check.columns and 
                                      any('sv_' in col and '_q' in col for col in df_check.columns))
            except:
                pass
            
            if user_format_detected:
                print("\n✅ 检测到您的机器标注数据格式！")
                print("系统将自动处理您的详细问题级别列并计算框架分数。")
                need_adaptation = 'n'  # 不需要额外适配
            else:
                print("\n如果数据格式不符合要求，我们可以自动适配数据格式。")
                need_adaptation = input("是否需要适配数据格式？(y/N): ").strip().lower()
            
            if need_adaptation in ['y', 'yes', '是']:
                logger.info("开始自动适配数据格式...")
                
                adapted_data_path = os.path.join(args.work_dir, "adapted_training_data.csv")
                
                adapt_args = [
                    "--input_path", args.data_path,
                    "--output_path", adapted_data_path,
                    "--auto_adapt"
                ]
                
                if args.verbose:
                    adapt_args.append("--verbose")
                
                # 临时替换sys.argv进行适配
                sys.argv = ["validate_existing_data.py"] + adapt_args
                
                try:
                    validate_main()
                    logger.info(f"数据适配完成: {adapted_data_path}")
                except SystemExit:
                    pass
                finally:
                    sys.argv = original_argv
            else:
                logger.info("使用原始数据格式进行训练")
        
        # 步骤2: 模型训练
        if not args.skip_training:
            logger.info("\n步骤 2: SV2000模型训练")
            logger.info("-" * 50)
            
            from framing_analyzer.train_sv2000_model import main as train_main
            
            model_output_dir = os.path.join(args.work_dir, "models")
            
            train_args = [
                "--data_path", adapted_data_path,
                "--output_dir", model_output_dir,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--learning_rate", str(args.learning_rate),
                "--device", args.device,
                "--optimize_weights",
                "--evaluate"
            ]
            
            if args.verbose:
                train_args.append("--verbose")
            
            # 临时替换sys.argv进行训练
            sys.argv = ["train_sv2000_model.py"] + train_args
            
            try:
                train_main()
                logger.info("模型训练完成！")
            except SystemExit:
                pass
            finally:
                sys.argv = original_argv
        
        # 步骤3: 权重优化
        if not args.skip_optimization:
            logger.info("\n步骤 3: 融合权重优化")
            logger.info("-" * 50)
            
            from framing_analyzer.optimize_fusion_weights import main as optimize_main
            
            optimization_output_dir = os.path.join(args.work_dir, "optimization")
            
            optimize_args = [
                "--data_path", adapted_data_path,
                "--output_dir", optimization_output_dir,
                "--max_samples", "1000"  # 限制样本数以加速
            ]
            
            if args.verbose:
                optimize_args.append("--verbose")
            
            # 临时替换sys.argv进行优化
            sys.argv = ["optimize_fusion_weights.py"] + optimize_args
            
            try:
                optimize_main()
                logger.info("权重优化完成！")
            except SystemExit:
                pass
            finally:
                sys.argv = original_argv
        
        # 步骤4: 系统测试
        logger.info("\n步骤 4: 系统测试")
        logger.info("-" * 50)
        
        from framing_analyzer import FramingAnalyzer
        from framing_analyzer.config import create_sv2000_config
        
        # 创建配置
        config = create_sv2000_config()
        
        # 配置训练好的模型路径
        model_path = os.path.join(args.work_dir, "models", "best_sv2000_model.pt")
        if os.path.exists(model_path):
            config.sv_framing.pretrained_model_path = model_path
            logger.info(f"使用训练好的模型: {model_path}")
        else:
            logger.info("使用默认配置进行测试")
        
        # 测试文章
        test_articles = [
            {
                'content': '''
                The escalating trade war between the two economic superpowers has 
                created uncertainty in global markets. Economists warn that the 
                ongoing tariff disputes could lead to a significant slowdown in 
                international commerce and affect millions of jobs worldwide.
                ''',
                'title': 'Trade War Threatens Global Economic Stability',
                'id': 'test_1'
            },
            {
                'content': '''
                Local residents have come together to rebuild their community center 
                after it was destroyed in last month\'s storm. Volunteers from 
                neighboring towns have donated materials and labor, showing the 
                power of human solidarity in times of crisis.
                ''',
                'title': 'Community Rebuilds After Storm Damage',
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
                        logger.info(f"    SV2000框架分数:")
                        logger.info(f"      平均: {result['sv_frame_avg']:.3f}")
                        logger.info(f"      冲突: {result.get('sv_conflict', 0):.3f}")
                        logger.info(f"      人情: {result.get('sv_human', 0):.3f}")
                        logger.info(f"      经济: {result.get('sv_econ', 0):.3f}")
                        logger.info(f"      道德: {result.get('sv_moral', 0):.3f}")
                        logger.info(f"      责任: {result.get('sv_resp', 0):.3f}")
                    
                    if 'fusion_weights' in result:
                        logger.info(f"    融合权重: {result['fusion_weights']}")
            
            logger.info("系统测试完成！")
            
        except Exception as e:
            logger.error(f"系统测试失败: {e}")
        
        # 生成使用指南
        logger.info("\n步骤 5: 生成使用指南")
        logger.info("-" * 50)
        
        usage_guide_path = os.path.join(args.work_dir, "USAGE_GUIDE.md")
        
        guide_content = f"""# 训练结果使用指南

## 训练信息

- **数据源**: {args.data_path}
- **训练时间**: {pd.Timestamp.now()}
- **训练参数**: 
  - 轮数: {args.epochs}
  - 批大小: {args.batch_size}
  - 学习率: {args.learning_rate}
  - 设备: {args.device}

## 生成的文件

### 模型文件
- `models/best_sv2000_model.pt` - 训练好的SV2000模型
- `models/training_report.json` - 详细训练报告

### 优化结果
- `optimization/fusion_optimization_results.json` - 权重优化结果
- `optimization/fusion_optimization_results_report.md` - 优化报告

### 数据文件
- `adapted_training_data.csv` - 适配后的训练数据（如果进行了适配）

## 使用训练好的模型

```python
from framing_analyzer import FramingAnalyzer
from framing_analyzer.config import create_sv2000_config

# 创建配置
config = create_sv2000_config()

# 加载训练好的模型
config.sv_framing.pretrained_model_path = "{model_path}"

# 初始化分析器
analyzer = FramingAnalyzer(config)

# 分析文章
article_text = "Your news article content here..."
result = analyzer.analyze_article(article_text, title="Article Title")

# 查看结果
print(f"框架强度: {{result.framing_intensity:.3f}}")
print(f"SV2000分数: {{result.sv_frame_avg:.3f}}")
print(f"各框架分数:")
print(f"  冲突: {{result.sv_conflict:.3f}}")
print(f"  人情: {{result.sv_human:.3f}}")
print(f"  经济: {{result.sv_econ:.3f}}")
print(f"  道德: {{result.sv_moral:.3f}}")
print(f"  责任: {{result.sv_resp:.3f}}")
```

## 下一步建议

1. **评估模型性能**: 查看训练报告中的验证指标
2. **调整融合权重**: 根据优化结果调整权重配置
3. **扩展数据集**: 使用更多数据继续训练
4. **部署应用**: 将模型集成到生产环境

## 故障排除

如果遇到问题，请：
1. 检查训练报告中的错误信息
2. 验证数据格式是否正确
3. 确认模型文件是否完整
4. 参考 `TRAINING_GUIDE.md` 获取详细帮助

---
*训练完成时间: {pd.Timestamp.now()}*
"""
        
        with open(usage_guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        logger.info(f"使用指南已生成: {usage_guide_path}")
        
        # 最终总结
        logger.info("\n" + "=" * 80)
        logger.info("训练流程完成！")
        logger.info("=" * 80)
        logger.info(f"📁 工作目录: {args.work_dir}")
        logger.info(f"📄 使用指南: {usage_guide_path}")
        
        if os.path.exists(model_path):
            logger.info(f"🤖 训练模型: {model_path}")
        
        optimization_results = os.path.join(args.work_dir, "optimization", "fusion_optimization_results.json")
        if os.path.exists(optimization_results):
            logger.info(f"⚖️  权重优化: {optimization_results}")
        
        logger.info("\n✅ 现在可以使用训练好的SV2000模型进行新闻框架分析！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # 导入pandas用于时间戳
    import pandas as pd
    main()
