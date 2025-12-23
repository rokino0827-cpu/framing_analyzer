#!/usr/bin/env python3
"""
快速Quote检测测试
快速检查quote分量是否正常工作

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/quick_quote_test.py
"""

import sys
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

def quick_quote_test():
    """快速测试quote功能"""
    
    print("🔍 Quick Quote Test")
    print("="*40)
    
    # 创建配置
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1
    config.teacher.model_local_path = "bias_detector_data"
    config.output.include_components = True
    config.output.include_raw_scores = True  # 包含原始分数用于调试
    
    # 测试文章 - 确保包含明显的引号
    test_article = {
        "id": "quote_test",
        "title": "Government Policy Announcement",
        "content": '''The government announced a major policy change today. The Prime Minister stated "this is the most important reform in decades." Opposition leaders immediately responded. "We completely disagree with this decision," said the opposition leader. "This policy will harm working families," added another critic. The implementation begins next month. Economists predict "significant market impacts" from the new policy.'''
    }
    
    print(f"📄 Test Article: {test_article['title']}")
    print(f"📝 Content preview: {test_article['content'][:100]}...")
    
    # 手动检查引号
    content = test_article['content']
    quote_count = content.count('"') + content.count('"') + content.count('"')
    print(f"🔍 Manual quote count: {quote_count} quote marks found")
    
    try:
        # 创建分析器并分析
        analyzer = create_analyzer(config)
        results = analyzer.analyze_batch([test_article])
        
        if 'results' in results and results['results']:
            result = results['results'][0]
            
            print(f"\n📊 Analysis Results:")
            print(f"   Framing Intensity: {result.get('framing_intensity', 0):.3f}")
            
            if 'components' in result:
                components = result['components']
                print(f"\n🧩 Component Scores:")
                print(f"   Headline: {components.get('headline', 0):.3f}")
                print(f"   Lede: {components.get('lede', 0):.3f}")
                print(f"   Quotes: {components.get('quotes', 0):.3f}")
                print(f"   Narration: {components.get('narration', 0):.3f}")
                
                quotes_score = components.get('quotes', 0)
                
                if quotes_score > 0:
                    print(f"\n✅ SUCCESS: Quotes component = {quotes_score:.3f}")
                    print("Quote detection is working correctly!")
                else:
                    print(f"\n❌ PROBLEM: Quotes component = {quotes_score}")
                    print("Quote detection may have issues!")
                    
                    # 检查原始数据
                    if 'raw_scores' in result:
                        raw_scores = result['raw_scores']
                        if 'zone_fragments' in raw_scores:
                            zone_fragments = raw_scores['zone_fragments']
                            quotes_fragments = zone_fragments.get('quotes', [])
                            print(f"   Raw quotes fragments: {len(quotes_fragments)}")
                            
                            if not quotes_fragments:
                                print("   ⚠️  No quote fragments generated!")
                                print("   This suggests quote detection failed at the text processing stage")
                            else:
                                print(f"   ✅ {len(quotes_fragments)} quote fragments found")
                                for i, frag in enumerate(quotes_fragments[:3]):
                                    print(f"      {i+1}. {frag.get('text', '')[:50]}...")
            else:
                print("❌ No components in result")
        else:
            print("❌ No results from analysis")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_quote_test()