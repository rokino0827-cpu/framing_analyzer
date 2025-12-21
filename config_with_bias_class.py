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

def create_analyzer_with_bias_class_name():
    """使用bias_class_name的方式（如果模型有明确标签名）"""
    
    config = AnalyzerConfig()
    # 如果模型的标签是有意义的名称，可以用这种方式
    # config.teacher.bias_class_name = "BIASED"  # 示例
    
    # 但对于LABEL_0/LABEL_1这种通用标签，还是用index更可靠
    config.teacher.bias_class_index = 1
    
    analyzer = create_analyzer(config)
    return analyzer

def test_configured_analyzer():
    """测试配置后的分析器"""
    
    print("🔧 创建已配置的分析器...")
    analyzer = create_configured_analyzer()
    
    print("📝 测试分析...")
    test_text = """
    The government announced new economic policies yesterday. 
    These measures are expected to impact various sectors of the economy.
    Officials stated that the implementation will begin next quarter.
    """
    
    result = analyzer.analyze_article(test_text, "Economic Policy Update")
    
    print("✅ 分析完成！")
    print(f"框架偏见分数: {result.framing_score:.3f}")
    print(f"偏见强度: {result.bias_intensity}")
    
    return result

if __name__ == "__main__":
    # 运行测试
    result = test_configured_analyzer()
    
    print("\n💡 配置说明:")
    print("1. 首先运行 verify_bias_class.py 确定正确的索引")
    print("2. 在代码中设置 config.teacher.bias_class_index = <验证得到的索引>")
    print("3. 重新运行，警告应该消失")
    
    print("\n🔍 如果仍有警告，请检查:")
    print("- bias_class_index 是否设置正确")
    print("- 配置是否传递给了分析器")
    print("- 模型是否正确加载")