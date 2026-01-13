"""
SV2000框架对齐示例脚本
演示如何使用新的SV2000功能进行新闻框架分析
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from framing_analyzer import FramingAnalyzer
from framing_analyzer.config import create_sv2000_config, AnalyzerConfig, SVFramingConfig, FusionConfig
from framing_analyzer.sv2000_evaluator import SV2000Evaluator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("SV2000框架对齐 - 基础使用示例")
    print("=" * 60)
    
    # 创建SV2000配置
    config = create_sv2000_config()
    
    # 示例文章
    sample_articles = [
        {
            'content': '''
            The ongoing conflict between the two nations has escalated dramatically, 
            with both sides reporting significant casualties. Military officials 
            confirmed that the dispute centers around territorial claims that have 
            been contested for decades. Citizens on both sides are calling for 
            their governments to find a peaceful resolution to avoid further bloodshed.
            ''',
            'title': 'Territorial Conflict Escalates Between Nations',
            'id': 'article_1'
        },
        {
            'content': '''
            Local community members came together yesterday to support a family 
            whose home was destroyed in the recent floods. Volunteers organized 
            food drives and temporary shelter, demonstrating the power of human 
            compassion in times of crisis. The family expressed deep gratitude 
            for the overwhelming support from their neighbors.
            ''',
            'title': 'Community Rallies to Help Flood Victims',
            'id': 'article_2'
        },
        {
            'content': '''
            The new economic policy is expected to create thousands of jobs 
            while reducing the national deficit by 15% over the next three years. 
            However, critics argue that the proposed tax increases will burden 
            middle-class families and small businesses. The finance minister 
            defended the plan, citing long-term economic stability as the priority.
            ''',
            'title': 'New Economic Policy Sparks Debate',
            'id': 'article_3'
        }
    ]
    
    try:
        # 初始化分析器
        print("初始化SV2000分析器...")
        analyzer = FramingAnalyzer(config)
        
        # 分析单篇文章
        print("\n--- 单篇文章分析 ---")
        result = analyzer.analyze_article(
            sample_articles[0]['content'], 
            title=sample_articles[0]['title']
        )
        
        print(f"文章: {sample_articles[0]['title']}")
        print(f"SV2000框架分数:")
        if hasattr(result, 'sv_conflict') and result.sv_conflict is not None:
            print(f"  冲突框架: {result.sv_conflict:.3f}")
            print(f"  人情框架: {result.sv_human:.3f}")
            print(f"  经济框架: {result.sv_econ:.3f}")
            print(f"  道德框架: {result.sv_moral:.3f}")
            print(f"  责任框架: {result.sv_resp:.3f}")
            print(f"  框架平均: {result.sv_frame_avg:.3f}")
        else:
            print("  SV2000模式未启用或模型未加载")
        
        print(f"最终强度: {result.framing_intensity:.3f}")
        print(f"伪标签: {result.pseudo_label}")
        
        # 批量分析
        print("\n--- 批量文章分析 ---")
        batch_results = analyzer.analyze_batch(sample_articles)
        
        print(f"成功分析: {batch_results['metadata']['successful_analyses']}")
        print(f"失败分析: {batch_results['metadata']['failed_analyses']}")
        
        print("\n批量结果摘要:")
        for result in batch_results['results']:
            if not result.get('error'):
                print(f"  {result['title'][:30]}... -> 强度: {result['framing_intensity']:.3f}")
                if 'sv_frame_avg' in result:
                    print(f"    SV2000平均: {result['sv_frame_avg']:.3f}")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        print("注意: SV2000功能需要相应的模型文件，如果没有预训练模型，将使用传统模式")

def example_custom_configuration():
    """自定义配置示例"""
    print("\n" + "=" * 60)
    print("SV2000框架对齐 - 自定义配置示例")
    print("=" * 60)
    
    # 创建自定义配置
    config = AnalyzerConfig()
    
    # 配置SV2000组件
    config.sv_framing = SVFramingConfig(
        enabled=True,
        encoder_name="bge_m3",
        hidden_size=1024,
        dropout_rate=0.1,
        batch_size=8,  # 较小的批处理大小
        device="cpu",  # 强制使用CPU
        training_mode="frame_level"
    )
    
    # 配置融合权重
    config.fusion = FusionConfig(
        alpha=0.6,    # 增加SV2000权重
        beta=0.15,    # 减少偏见检测权重
        gamma=0.1,    # 省略检测权重
        delta=0.1,    # 相对框架权重
        epsilon=0.05, # 引用分析权重
        enforce_positive_weights=True,
        normalize_weights=True
    )
    
    # 配置输出选项
    config.output.include_components = True
    config.output.include_evidence = True
    config.output.include_statistics = True
    
    print("自定义配置:")
    print(f"  SV2000启用: {config.sv_framing.enabled}")
    print(f"  编码器: {config.sv_framing.encoder_name}")
    print(f"  设备: {config.sv_framing.device}")
    print(f"  融合权重: α={config.fusion.alpha}, β={config.fusion.beta}")
    
    # 测试配置
    sample_text = "This is a test article about economic policy changes."
    
    try:
        analyzer = FramingAnalyzer(config)
        result = analyzer.analyze_article(sample_text, title="Test Article")
        
        print(f"\n测试结果:")
        print(f"  强度: {result.framing_intensity:.3f}")
        print(f"  统计信息: {len(result.statistics)} 项")
        
        if hasattr(result, 'fusion_weights') and result.fusion_weights:
            print(f"  融合权重: {result.fusion_weights}")
        
    except Exception as e:
        print(f"配置测试失败: {e}")

def example_evaluation():
    """评估示例"""
    print("\n" + "=" * 60)
    print("SV2000框架对齐 - 评估示例")
    print("=" * 60)
    
    # 模拟预测和真实数据
    predictions = {
        'sv_conflict_pred': [0.3, 0.7, 0.2, 0.8, 0.1],
        'sv_human_pred': [0.5, 0.4, 0.8, 0.2, 0.9],
        'sv_econ_pred': [0.2, 0.9, 0.1, 0.7, 0.3],
        'sv_moral_pred': [0.6, 0.3, 0.7, 0.4, 0.8],
        'sv_resp_pred': [0.4, 0.6, 0.5, 0.3, 0.7],
        'sv_frame_avg_pred': [0.4, 0.58, 0.46, 0.48, 0.56]
    }
    
    ground_truth = {
        'y_conflict': [0.2, 0.8, 0.1, 0.9, 0.0],
        'y_human': [0.6, 0.3, 0.9, 0.1, 0.8],
        'y_econ': [0.1, 0.9, 0.2, 0.8, 0.2],
        'y_moral': [0.7, 0.2, 0.8, 0.3, 0.9],
        'y_resp': [0.3, 0.7, 0.4, 0.2, 0.6]
    }
    
    # 初始化评估器
    evaluator = SV2000Evaluator()
    
    # 评估框架对齐
    print("评估SV2000框架对齐...")
    alignment_results = evaluator.evaluate_frame_alignment(predictions, ground_truth)
    
    print(f"整体对齐分数: {alignment_results['overall_alignment_score']:.3f}")
    
    print("\n逐框架相关性:")
    for frame, metrics in alignment_results['frame_correlations'].items():
        print(f"  {frame.capitalize()}: Pearson={metrics['pearson_r']:.3f}, MAE={metrics['mae']:.3f}")
    
    if 'frame_average_alignment' in alignment_results:
        fa = alignment_results['frame_average_alignment']
        if 'pearson_r' in fa:
            print(f"\n框架平均对齐: Pearson={fa['pearson_r']:.3f}, MAE={fa['mae']:.3f}")
    
    # 模拟融合性能评估
    print("\n评估融合性能...")
    fusion_results = [
        {
            'final_intensity': 0.6,
            'sv_frame_avg_pred': 0.5,
            'bias_score': 0.4,
            'omission_score': 0.2,
            'relative_score': 0.1,
            'quote_score': 0.3
        },
        {
            'final_intensity': 0.7,
            'sv_frame_avg_pred': 0.6,
            'bias_score': 0.5,
            'omission_score': 0.3,
            'relative_score': 0.2,
            'quote_score': 0.4
        },
        {
            'final_intensity': 0.4,
            'sv_frame_avg_pred': 0.3,
            'bias_score': 0.2,
            'omission_score': 0.1,
            'relative_score': 0.0,
            'quote_score': 0.2
        }
    ]
    
    ground_truth_intensity = [0.65, 0.75, 0.35]
    
    fusion_performance = evaluator.evaluate_fusion_performance(fusion_results, ground_truth_intensity)
    
    if 'component_analysis' in fusion_performance:
        print("\n组件贡献分析:")
        for component, metrics in fusion_performance['component_analysis'].items():
            print(f"  {component}: 相关性={metrics['correlation_with_final']:.3f}")
    
    if 'performance_comparison' in fusion_performance:
        print("\n性能比较:")
        pc = fusion_performance['performance_comparison']
        for method in ['fusion', 'sv2000_only', 'bias_only']:
            if method in pc and isinstance(pc[method], dict):
                metrics = pc[method]
                if 'pearson_r' in metrics:
                    print(f"  {method}: 相关性={metrics['pearson_r']:.3f}, MAE={metrics['mae']:.3f}")

def example_mode_comparison():
    """模式对比示例"""
    print("\n" + "=" * 60)
    print("SV2000框架对齐 - 模式对比示例")
    print("=" * 60)
    
    sample_text = '''
    The economic crisis has led to widespread unemployment and social unrest. 
    Government officials are under pressure to implement immediate reforms 
    to address the growing inequality and restore public confidence in 
    the financial system.
    '''
    
    # 传统模式
    print("--- 传统模式分析 ---")
    legacy_config = AnalyzerConfig()
    legacy_config.disable_sv2000_mode()
    
    try:
        legacy_analyzer = FramingAnalyzer(legacy_config)
        legacy_result = legacy_analyzer.analyze_article(sample_text, title="Economic Crisis Report")
        
        print(f"传统模式结果:")
        print(f"  强度: {legacy_result.framing_intensity:.3f}")
        print(f"  伪标签: {legacy_result.pseudo_label}")
        print(f"  组件分数: {legacy_result.components}")
        
    except Exception as e:
        print(f"传统模式分析失败: {e}")
    
    # SV2000模式
    print("\n--- SV2000模式分析 ---")
    sv2000_config = create_sv2000_config()
    
    try:
        sv2000_analyzer = FramingAnalyzer(sv2000_config)
        sv2000_result = sv2000_analyzer.analyze_article(sample_text, title="Economic Crisis Report")
        
        print(f"SV2000模式结果:")
        print(f"  强度: {sv2000_result.framing_intensity:.3f}")
        print(f"  伪标签: {sv2000_result.pseudo_label}")
        
        if hasattr(sv2000_result, 'sv_conflict') and sv2000_result.sv_conflict is not None:
            print(f"  SV2000框架分数:")
            print(f"    冲突: {sv2000_result.sv_conflict:.3f}")
            print(f"    人情: {sv2000_result.sv_human:.3f}")
            print(f"    经济: {sv2000_result.sv_econ:.3f}")
            print(f"    道德: {sv2000_result.sv_moral:.3f}")
            print(f"    责任: {sv2000_result.sv_resp:.3f}")
            print(f"    平均: {sv2000_result.sv_frame_avg:.3f}")
        else:
            print("  SV2000功能未完全加载")
        
        if hasattr(sv2000_result, 'fusion_weights') and sv2000_result.fusion_weights:
            print(f"  融合权重: {sv2000_result.fusion_weights}")
        
    except Exception as e:
        print(f"SV2000模式分析失败: {e}")
        print("注意: 需要预训练的SV2000模型才能完全运行")

def main():
    """主函数"""
    print("SV2000框架对齐功能演示")
    print("注意: 某些功能需要预训练模型，如果模型不存在将显示相应提示")
    
    try:
        # 运行各个示例
        example_basic_usage()
        example_custom_configuration()
        example_evaluation()
        example_mode_comparison()
        
        print("\n" + "=" * 60)
        print("演示完成！")
        print("=" * 60)
        print("\n要使用完整的SV2000功能，请:")
        print("1. 准备SV2000标注数据")
        print("2. 使用sv2000_trainer.py训练模型")
        print("3. 配置模型路径并重新运行")
        print("\n详细使用说明请参考 SV2000_USAGE_GUIDE.md")
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
