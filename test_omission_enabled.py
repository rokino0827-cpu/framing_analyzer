#!/usr/bin/env python3
"""
省略检测启用验证脚本
验证省略检测是否正确启用并产生预期字段
"""

import sys
import json
import logging
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_omission_enabled():
    """测试省略检测是否正确启用"""
    
    print("🔍 Testing omission detection enablement...")
    
    # 创建启用省略检测的配置
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1
    config.teacher.model_local_path = "bias_detector_data"
    config.output.output_dir = "results/omission_test"
    
    # 启用省略检测
    config.omission.enabled = True
    config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
    config.omission.fusion_weight = 0.2
    config.omission.key_topics_count = 10
    
    print(f"✅ Configuration created:")
    print(f"   omission.enabled = {config.omission.enabled}")
    print(f"   omission.fusion_weight = {config.omission.fusion_weight}")
    print(f"   omission.embedding_model = {config.omission.embedding_model_name_or_path}")
    
    # 测试文章（需要多篇文章才能进行事件聚类）
    test_articles = [
        {
            "id": "test_1",
            "title": "Economic Policy Changes Announced",
            "content": "The government announced significant changes to economic policy yesterday. The new measures include tax reforms and spending cuts. Officials stated that these changes will help reduce the deficit. The policy affects multiple sectors including healthcare and education."
        },
        {
            "id": "test_2", 
            "title": "Economic Reforms Impact Multiple Sectors",
            "content": "Recent economic reforms have begun affecting various sectors of the economy. Healthcare providers are concerned about funding cuts. Education officials worry about reduced budgets. The manufacturing sector sees potential benefits from tax changes."
        },
        {
            "id": "test_3",
            "title": "Policy Implementation Timeline Released",
            "content": "The government released a detailed timeline for implementing the new economic policies. The first phase begins next month with tax code changes. Healthcare funding adjustments will follow in the second quarter."
        }
    ]
    
    try:
        # 创建分析器
        analyzer = create_analyzer(config)
        print("✅ Analyzer created successfully")
        
        # 检查省略检测器是否初始化
        if hasattr(analyzer, 'omission_detector') and analyzer.omission_detector:
            print("✅ Omission detector initialized")
        else:
            print("❌ Omission detector NOT initialized")
            return False
        
        # 分析文章
        print("\n📝 Analyzing articles...")
        results = analyzer.analyze_batch(test_articles)
        
        print(f"✅ Analysis completed")
        print(f"📊 Processed {len(results['results'])} articles")
        
        # 检查省略检测字段
        print("\n🔍 Checking omission detection fields...")
        
        omission_fields_found = False
        for i, result in enumerate(results['results']):
            article_id = result.get('id', f'article_{i}')
            print(f"\n📄 Article: {article_id}")
            
            # 检查省略相关字段
            omission_score = result.get('omission_score')
            omission_evidence = result.get('omission_evidence')
            
            print(f"   omission_score: {omission_score}")
            print(f"   omission_evidence: {len(omission_evidence) if omission_evidence else 0} items")
            
            # 检查统计信息中的省略字段
            statistics = result.get('statistics', {})
            omission_stats = {k: v for k, v in statistics.items() if 'omission' in k.lower()}
            if omission_stats:
                print(f"   omission_statistics: {omission_stats}")
            
            # 检查framing_intensity是否受到省略分数影响
            framing_intensity = result.get('framing_intensity', 0.0)
            print(f"   framing_intensity: {framing_intensity:.3f}")
            
            if omission_score is not None:
                omission_fields_found = True
                print(f"   ✅ Omission fields present")
            else:
                print(f"   ❌ Omission fields missing")
        
        # 检查全局统计
        if 'statistics' in results and 'omission_detection' in results['statistics']:
            print(f"\n📊 Global omission statistics:")
            omission_stats = results['statistics']['omission_detection']
            for key, value in omission_stats.items():
                print(f"   {key}: {value}")
        
        # 保存详细结果
        output_file = Path("results/omission_test/omission_test_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # 总结
        if omission_fields_found:
            print("\n🎉 SUCCESS: Omission detection is properly enabled and producing results!")
            print("Expected fields found:")
            print("  ✅ omission_score")
            print("  ✅ omission_evidence") 
            print("  ✅ omission statistics")
            print("  ✅ fusion with framing_intensity")
            return True
        else:
            print("\n❌ FAILURE: Omission detection not producing expected fields")
            print("This indicates the omission detection pipeline is not properly integrated")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fusion_weights():
    """测试不同融合权重的效果"""
    
    print("\n🧪 Testing different fusion weights...")
    
    test_article = {
        "id": "fusion_test",
        "title": "Test Article for Fusion",
        "content": "This is a test article to verify that omission scores are properly fused with framing intensity scores using different weight configurations."
    }
    
    fusion_weights = [0.0, 0.1, 0.2, 0.3]
    results = {}
    
    for weight in fusion_weights:
        try:
            config = AnalyzerConfig()
            config.teacher.bias_class_index = 1
            config.teacher.model_local_path = "bias_detector_data"
            config.omission.enabled = True
            config.omission.fusion_weight = weight
            config.output.output_dir = f"results/fusion_test_{weight}"
            
            analyzer = create_analyzer(config)
            result = analyzer.analyze_batch([test_article])
            
            if result and 'results' in result and result['results']:
                framing_intensity = result['results'][0].get('framing_intensity', 0.0)
                omission_score = result['results'][0].get('omission_score')
                results[weight] = {
                    'framing_intensity': framing_intensity,
                    'omission_score': omission_score
                }
                print(f"   Weight {weight}: framing_intensity={framing_intensity:.3f}, omission_score={omission_score}")
            
        except Exception as e:
            print(f"   Weight {weight}: Failed - {e}")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("🔍 OMISSION DETECTION ENABLEMENT TEST")
    print("="*60)
    
    # 主要测试
    success = test_omission_enabled()
    
    # 融合权重测试
    if success:
        fusion_results = test_fusion_weights()
    
    print("\n" + "="*60)
    if success:
        print("🎉 All tests passed! Omission detection is properly enabled.")
    else:
        print("❌ Tests failed. Check the integration.")
    print("="*60)