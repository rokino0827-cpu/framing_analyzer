#!/usr/bin/env python3
"""
Quote分量异常为0问题诊断脚本
检查quote检测逻辑是否正常工作

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/debug_quotes_issue.py
"""

import sys
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer
from framing_analyzer.text_processor import TextProcessor, StructureZoneExtractor

def test_quote_patterns():
    """测试引号模式匹配"""
    
    print("🔍 Testing quote patterns...")
    
    # 测试句子
    test_sentences = [
        'He said "This is a test quote."',  # 英文双引号
        "She replied 'I agree with that.'",  # 英文单引号
        '官员表示"这是一个重要决定"。',  # 中文双引号
        "专家认为'这种方法很有效'。",  # 中文单引号
        'This is a normal sentence without quotes.',  # 无引号
        'The "quoted text" is in the middle.',  # 中间有引号
        '"Multiple quotes" and "more quotes" here.',  # 多个引号
        "Mixed 'quotes' and \"other quotes\" together.",  # 混合引号
    ]
    
    # 默认引号模式
    quote_patterns = [
        r'"([^"]+)"',      # 英文双引号
        r"'([^']+)'",      # 英文单引号
        r'"([^"]+)"',      # 中文双引号
        r"'([^']+)'",      # 中文单引号
    ]
    
    compiled_patterns = [re.compile(pattern) for pattern in quote_patterns]
    
    print(f"Testing {len(test_sentences)} sentences with {len(quote_patterns)} patterns:")
    
    quote_found = False
    for i, sentence in enumerate(test_sentences):
        has_quote = False
        matched_patterns = []
        
        for j, pattern in enumerate(compiled_patterns):
            if pattern.search(sentence):
                has_quote = True
                matched_patterns.append(j)
        
        if has_quote:
            quote_found = True
        
        status = "✅ QUOTE" if has_quote else "❌ NO QUOTE"
        print(f"  {i+1}. {status}: {sentence}")
        if matched_patterns:
            print(f"     Matched patterns: {matched_patterns}")
    
    if quote_found:
        print("✅ Quote patterns are working correctly")
    else:
        print("❌ No quotes detected - patterns may be broken")
    
    return quote_found

def test_zone_extraction():
    """测试结构区提取"""
    
    print("\n🔍 Testing zone extraction...")
    
    # 创建配置
    config = AnalyzerConfig()
    
    # 测试文章
    test_articles = [
        {
            "title": "Government Announces New Policy",
            "content": '''The government announced a new policy yesterday. Officials said this would improve the economy. "This is a significant step forward," stated the minister. The policy will be implemented next year. Critics argue "the timing is not right" for such changes. The implementation will require careful planning.'''
        },
        {
            "title": "Technology Breakthrough Reported", 
            "content": '''Scientists have made a breakthrough in AI research. The new algorithm shows promising results. Dr. Smith explained "this could revolutionize the field." The research was published in Nature journal. "We are excited about the possibilities," said the lead researcher. Further testing is needed before commercial applications.'''
        },
        {
            "title": "Economic Report Released",
            "content": '''The quarterly economic report was released today. GDP growth exceeded expectations this quarter. The finance minister noted that "economic indicators are positive." Unemployment rates have decreased significantly. Analysts believe "the trend will continue" through next year. Market confidence remains strong.'''
        }
    ]
    
    # 初始化处理器
    text_processor = TextProcessor(config)
    zone_extractor = StructureZoneExtractor(config)
    
    total_quotes = 0
    articles_with_quotes = 0
    
    for i, article_data in enumerate(test_articles):
        print(f"\n📄 Article {i+1}: {article_data['title']}")
        
        # 处理文章
        processed_article = text_processor.process_article(
            article_data['content'], 
            article_data['title']
        )
        processed_article = zone_extractor.divide_into_zones(processed_article)
        
        # 检查各个区域
        zones = processed_article.zones
        
        print(f"   Headline: {len(zones['headline'])} sentences")
        print(f"   Lede: {len(zones['lede'])} sentences") 
        print(f"   Quotes: {len(zones['quotes'])} sentences")
        print(f"   Narration: {len(zones['narration'])} sentences")
        
        if zones['quotes']:
            articles_with_quotes += 1
            total_quotes += len(zones['quotes'])
            print(f"   📝 Quote sentences:")
            for j, quote in enumerate(zones['quotes']):
                print(f"      {j+1}. {quote}")
        else:
            print(f"   ❌ No quotes detected")
            
            # 手动检查是否有引号
            content = article_data['content']
            manual_quotes = []
            if '"' in content:
                manual_quotes.append('English double quotes')
            if "'" in content:
                manual_quotes.append('English single quotes')
            if '"' in content or '"' in content:
                manual_quotes.append('Chinese quotes')
            
            if manual_quotes:
                print(f"   ⚠️  Manual check found: {', '.join(manual_quotes)}")
    
    print(f"\n📊 Summary:")
    print(f"   Total articles: {len(test_articles)}")
    print(f"   Articles with quotes: {articles_with_quotes}")
    print(f"   Total quotes detected: {total_quotes}")
    print(f"   Quote detection rate: {articles_with_quotes/len(test_articles)*100:.1f}%")
    
    return articles_with_quotes > 0

def test_real_data_quotes():
    """测试真实数据中的quote检测"""
    
    print("\n🔍 Testing quote detection on real data...")
    
    data_path = Path("data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv")
    
    if not data_path.exists():
        print("⚠️  Real data file not found, skipping real data test")
        return True
    
    # 读取少量真实数据
    df = pd.read_csv(data_path, encoding="utf-8")
    df = df[df["content"].notna() & df["title"].notna()].head(10)
    
    config = AnalyzerConfig()
    text_processor = TextProcessor(config)
    zone_extractor = StructureZoneExtractor(config)
    
    articles_with_quotes = 0
    total_quotes = 0
    
    for idx, row in df.iterrows():
        title = str(row["title"])
        content = str(row["content"])
        
        # 处理文章
        processed_article = text_processor.process_article(content, title)
        processed_article = zone_extractor.divide_into_zones(processed_article)
        
        quotes_count = len(processed_article.zones['quotes'])
        if quotes_count > 0:
            articles_with_quotes += 1
            total_quotes += quotes_count
        
        print(f"   Article {idx}: {quotes_count} quotes - {title[:50]}...")
    
    print(f"\n📊 Real data summary:")
    print(f"   Articles tested: {len(df)}")
    print(f"   Articles with quotes: {articles_with_quotes}")
    print(f"   Total quotes: {total_quotes}")
    print(f"   Average quotes per article: {total_quotes/len(df):.2f}")
    
    return articles_with_quotes > 0

def test_full_analysis_pipeline():
    """测试完整分析流程中的quote分量"""
    
    print("\n🔍 Testing full analysis pipeline...")
    
    # 创建配置
    config = AnalyzerConfig()
    config.teacher.bias_class_index = 1
    config.teacher.model_local_path = "bias_detector_data"
    config.output.include_components = True
    
    # 测试文章（确保包含引号）
    test_articles = [
        {
            "id": "quote_test_1",
            "title": "Policy Announcement with Quotes",
            "content": '''The government made an important announcement today. The Prime Minister said "this policy will benefit all citizens." Opposition leaders criticized the decision. "We strongly disagree with this approach," stated the opposition leader. The policy will take effect next month. Experts believe "the implementation will face challenges" in the coming weeks.'''
        }
    ]
    
    try:
        # 创建分析器
        analyzer = create_analyzer(config)
        
        # 分析文章
        results = analyzer.analyze_batch(test_articles)
        
        if 'results' in results and results['results']:
            result = results['results'][0]
            
            if 'components' in result:
                components = result['components']
                
                print(f"📊 Component scores:")
                print(f"   Headline: {components.get('headline', 0):.3f}")
                print(f"   Lede: {components.get('lede', 0):.3f}")
                print(f"   Quotes: {components.get('quotes', 0):.3f}")
                print(f"   Narration: {components.get('narration', 0):.3f}")
                
                quotes_score = components.get('quotes', 0)
                if quotes_score > 0:
                    print("✅ Quotes component has non-zero score")
                    return True
                else:
                    print("❌ Quotes component is zero!")
                    return False
            else:
                print("❌ No components in result")
                return False
        else:
            print("❌ No results from analysis")
            return False
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def diagnose_quote_issue():
    """综合诊断quote问题"""
    
    print("="*60)
    print("🔍 QUOTE COMPONENT DIAGNOSIS")
    print("="*60)
    
    # 测试1: 引号模式匹配
    patterns_ok = test_quote_patterns()
    
    # 测试2: 结构区提取
    extraction_ok = test_zone_extraction()
    
    # 测试3: 真实数据测试
    real_data_ok = test_real_data_quotes()
    
    # 测试4: 完整分析流程
    pipeline_ok = test_full_analysis_pipeline()
    
    print("\n" + "="*60)
    print("📊 DIAGNOSIS SUMMARY")
    print("="*60)
    
    print(f"Quote patterns working: {'✅' if patterns_ok else '❌'}")
    print(f"Zone extraction working: {'✅' if extraction_ok else '❌'}")
    print(f"Real data detection: {'✅' if real_data_ok else '❌'}")
    print(f"Full pipeline working: {'✅' if pipeline_ok else '❌'}")
    
    if all([patterns_ok, extraction_ok, pipeline_ok]):
        print("\n🎉 Quote detection appears to be working correctly!")
        print("If you're still seeing zero quote scores, it might be due to:")
        print("  - Articles in your dataset don't contain quotes")
        print("  - Quote patterns don't match the quote styles in your data")
        print("  - Quotes are being classified into other zones (lede/narration)")
    else:
        print("\n❌ Quote detection has issues:")
        if not patterns_ok:
            print("  - Quote regex patterns are not working")
        if not extraction_ok:
            print("  - Zone extraction logic has problems")
        if not pipeline_ok:
            print("  - Full analysis pipeline has issues")
    
    print("="*60)

def main():
    """主函数"""
    diagnose_quote_issue()

if __name__ == "__main__":
    main()