#!/usr/bin/env python3
"""
测试省略检测功能集成的简单脚本
"""

import sys
import os
from pathlib import Path

# 将仓库根目录加入sys.path，确保可以导入framing_analyzer包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_imports():
    """测试所有省略检测相关的导入"""
    print("Testing imports...")
    
    try:
        from framing_analyzer import (
            OmissionDetector, OmissionResult, OmissionGraph, 
            GraphNode, GraphEdge, OmissionAwareGraphBuilder,
            OmissionConfig, create_omission_enabled_config
        )
        print("✓ All omission-related imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_creation():
    """测试配置创建"""
    print("Testing configuration creation...")
    
    try:
        from framing_analyzer import create_omission_enabled_config, AnalyzerConfig
        
        # 测试默认配置
        config = AnalyzerConfig()
        print(f"✓ Default config created, has omission: {hasattr(config, 'omission')}")
        
        # 测试省略启用配置
        omission_config = create_omission_enabled_config()
        if hasattr(omission_config, 'omission'):
            print(f"✓ Omission-enabled config created, enabled: {omission_config.omission.enabled}")
        else:
            print("✗ Omission config not found in omission-enabled config")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_analyzer_creation():
    """测试分析器创建"""
    print("Testing analyzer creation...")
    
    try:
        from framing_analyzer import create_analyzer
        
        # 测试默认分析器（期望omission默认关闭）
        analyzer = create_analyzer()
        omission_enabled = analyzer.omission_detector is not None
        print(f"✓ Default analyzer created, omission detector enabled: {omission_enabled}")
        
        # 测试启用省略检测的分析器
        omission_analyzer = create_analyzer(enable_omission=True)
        omission_enabled = omission_analyzer.omission_detector is not None
        print(f"✓ Omission-enabled analyzer created, omission detector enabled: {omission_enabled}")
        if not omission_enabled:
            print("✗ Expected omission detector to be enabled")
            return False
        
        # 测试analyze_article方法签名
        import inspect
        sig = inspect.signature(analyzer.analyze_article)
        params = list(sig.parameters.keys())
        if 'article_id' in params and 'event_cluster' in params:
            print("✓ analyze_article method has correct signature with article_id and event_cluster parameters")
        else:
            print(f"✗ analyze_article method signature incorrect: {params}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Analyzer creation error: {e}")
        return False

def test_omission_components():
    """测试省略检测组件"""
    print("Testing omission detection components...")
    
    try:
        from framing_analyzer.omission_detector import OmissionDetector
        from framing_analyzer.omission_graph import OmissionAwareGraphBuilder
        from framing_analyzer.config import OmissionConfig
        
        # 创建配置
        config = OmissionConfig()
        print(f"✓ OmissionConfig created with similarity_threshold: {config.similarity_threshold}")
        
        # 测试图构建器
        graph_builder = OmissionAwareGraphBuilder(config)
        print("✓ OmissionAwareGraphBuilder created")
        
        # 测试聚类方法存在
        from framing_analyzer.config import AnalyzerConfig
        full_config = AnalyzerConfig()
        full_config.omission.enabled = True
        
        # 注意：不实际创建OmissionDetector，因为它需要spacy模型加载
        # 但可以检查方法是否存在
        import inspect
        detector_methods = [method for method in dir(OmissionDetector) if not method.startswith('_')]
        if 'cluster_articles_by_event' in detector_methods:
            print("✓ cluster_articles_by_event method found")
        else:
            print("✗ cluster_articles_by_event method missing")
            return False
        
        if 'detect_omissions' in detector_methods:
            print("✓ detect_omissions method found")
        else:
            print("✗ detect_omissions method missing")
            return False
        
        print("✓ Omission components structure validated")
        
        return True
    except Exception as e:
        print(f"✗ Omission components error: {e}")
        return False

def main():
    """运行所有测试"""
    print("=== Omission Detection Integration Test ===\n")
    
    tests = [
        test_imports,
        test_config_creation,
        test_analyzer_creation,
        test_omission_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test failed with exception: {e}\n")
    
    print(f"=== Test Results: {passed}/{total} passed ===")
    
    if passed == total:
        print("🎉 All tests passed! Omission detection integration is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
