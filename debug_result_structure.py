#!/usr/bin/env python3
"""
调试脚本：检查分析器返回值结构
"""

import sys
import json
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

def debug_result_structure():
    """调试返回值结构"""
    
    print("🔍 Debugging analyzer result structure...")
    
    # 创建配置
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1
    config.teacher.model_local_path = "bias_detector_data"
    config.output.output_dir = "results/debug"
    
    # 测试文章
    test_articles = [
        {
            "id": "debug_test_1",
            "title": "Test Article",
            "content": "This is a test article for debugging the result structure. It contains some content to analyze."
        }
    ]
    
    try:
        # 创建分析器并分析
        analyzer = create_analyzer(config)
        results = analyzer.analyze_batch(test_articles)
        
        print("✅ Analysis completed successfully")
        print(f"📊 Results type: {type(results)}")
        print(f"📊 Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        if 'results' in results:
            print(f"📊 Number of results: {len(results['results'])}")
            
            if results['results']:
                first_result = results['results'][0]
                print(f"📊 First result type: {type(first_result)}")
                
                if isinstance(first_result, dict):
                    print(f"📊 First result keys: {list(first_result.keys())}")
                    print(f"📊 Sample values:")
                    for key, value in first_result.items():
                        if isinstance(value, (int, float, str)):
                            print(f"   {key}: {value}")
                        else:
                            print(f"   {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
                else:
                    print(f"📊 First result attributes: {dir(first_result)}")
                    print(f"📊 Sample attribute values:")
                    for attr in ['framing_intensity', 'pseudo_label', 'components', 'evidence']:
                        if hasattr(first_result, attr):
                            value = getattr(first_result, attr)
                            print(f"   {attr}: {value}")
        
        # 保存完整结构到文件
        debug_file = Path("results/debug/result_structure.json")
        debug_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"💾 Full results saved to: {debug_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    debug_result_structure()