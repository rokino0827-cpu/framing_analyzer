#!/usr/bin/env python3
"""
配置bias_class_index的示例代码
消除 "Could not determine bias class index" 警告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framing_analyzer import AnalyzerConfig, create_analyzer

def create_configured_analyzer():
    """创建已配置bias_class_index的分析器"""
    
    # 方法1：直接设置bias_class_index（推荐）
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1  # 根据验证结果设置，通常是1
    
    # 可选：同时设置其他teacher配置
    config.teacher.model_name = "himel7/bias-detector"
    config.teacher.model_local_path = "bias_detector_data"
    config.teacher.batch_size = 16
    
    analyzer = create_analyzer(config)
    return analyzer

def create_analyzer_with_omission():
    """创建启用省略检测的分析器示例"""
    
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1
    config.teacher.model_local_path = "bias_detector_data"
    
    # 启用省略检测
    config.omission.enabled = True
    config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
    config.omission.fusion_weight = 0.2
    
    analyzer = create_analyzer(config)
    return analyzer

def test_omission_detection():
    """测试省略检测功能"""
    
    print("🔍 测试省略检测功能...")
    analyzer = create_analyzer_with_omission()
    
    test_articles = [
        {
            "id": "test_omission_1",
            "title": "Economic Policy Changes",
            "content": "The government announced new economic policies. Tax reforms will be implemented next year."
        },
        {
            "id": "test_omission_2", 
            "title": "Economic Reforms Impact",
            "content": "The new economic policies affect healthcare and education sectors. Budget cuts are expected in multiple areas."
        }
    ]
    
    results = analyzer.analyze_batch(test_articles)
    
    for result in results['results']:
        article_id = result.get('id')
        omission_score = result.get('omission_score')
        framing_intensity = result.get('framing_intensity')
        
        print(f"📄 {article_id}:")
        print(f"   Framing Intensity: {framing_intensity:.3f}")
        print(f"   Omission Score: {omission_score}")
    
    return results

if __name__ == "__main__":
    # 运行基础测试
    print("1️⃣  测试基础配置...")
    result = create_configured_analyzer()
    
    # 运行省略检测测试
    print("\n2️⃣  测试省略检测...")
    omission_results = test_omission_detection()
    
    print("\n💡 配置说明:")
    print("1. 首先运行 verify_bias_class.py 确定正确的索引")
    print("2. 在代码中设置 config.teacher.bias_class_index = <验证得到的索引>")
    print("3. 可选：启用省略检测 config.omission.enabled = True")
    print("4. 重新运行，警告应该消失")
    
    print("\n🔍 如果仍有警告，请检查:")
    print("- bias_class_index 是否设置正确")
    print("- 配置是否传递给了分析器")
    print("- 模型是否正确加载")