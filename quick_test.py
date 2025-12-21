#!/usr/bin/env python3
"""
快速测试脚本 - 用于日常开发和调试

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/quick_test.py
"""

import sys
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from framing_analyzer import (
    AnalyzerConfig, 
    create_analyzer,
    verify_bias_class_index
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """快速测试主要功能"""
    
    print("🚀 Starting quick test...")
    
    # 1. 验证bias_class_index
    print("\n1️⃣  Verifying bias class index...")
    try:
        result = verify_bias_class_index()
        if result and 'config_suggestion' in result:
            bias_index = result['config_suggestion']['bias_class_index']
            print(f"✅ Recommended bias_class_index: {bias_index}")
        else:
            bias_index = 1
            print(f"⚠️  Using default bias_class_index: {bias_index}")
    except Exception as e:
        print(f"⚠️  Verification failed, using default: {e}")
        bias_index = 1
    
    # 2. 创建配置
    print("\n2️⃣  Creating configuration...")
    config = AnalyzerConfig()
    config.teacher.bias_class_index = bias_index
    config.teacher.model_local_path = str(PROJECT_ROOT / "bias_detector_data")
    config.output.output_dir = str(PROJECT_ROOT / "results/quick_test")
    print("✅ Configuration created")
    
    # 3. 测试数据
    print("\n3️⃣  Loading test data...")
    data_path = PROJECT_ROOT / "data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path, encoding="utf-8")
        df = df[df["content"].notna() & df["title"].notna()].head(5)
        
        articles = []
        for idx, row in df.iterrows():
            articles.append({
                "id": f"test_{idx}",
                "title": str(row["title"]),
                "content": str(row["content"])
            })
        print(f"✅ Loaded {len(articles)} test articles")
    else:
        # 使用内置测试数据
        articles = [
            {
                "id": "test_1",
                "title": "Economic Policy Update",
                "content": "The government announced new economic policies yesterday. These measures are expected to impact various sectors of the economy. Officials stated that the implementation will begin next quarter."
            },
            {
                "id": "test_2", 
                "title": "Technology Innovation Report",
                "content": "A new breakthrough in artificial intelligence has been reported by researchers. The technology promises to revolutionize how we process information. Industry experts are optimistic about its potential applications."
            },
            {
                "id": "test_3",
                "title": "Climate Change Discussion",
                "content": "Scientists continue to study the effects of climate change on global weather patterns. Recent data shows significant changes in temperature and precipitation. Policymakers are considering various response strategies."
            }
        ]
        print(f"✅ Using {len(articles)} built-in test articles")
    
    # 4. 基础分析测试
    print("\n4️⃣  Testing basic analysis...")
    start_time = time.time()
    
    try:
        analyzer = create_analyzer(config)
        results = analyzer.analyze_batch(articles)
        
        analysis_time = time.time() - start_time
        
        print(f"✅ Analysis completed in {analysis_time:.2f}s")
        print(f"📊 Processed {len(results['results'])} articles")
        
        # 显示结果摘要
        framing_scores = [r.framing_score for r in results['results']]
        print(f"📈 Framing scores: {[f'{s:.3f}' for s in framing_scores]}")
        print(f"📊 Average score: {np.mean(framing_scores):.3f}")
        
        # 显示第一篇文章的详细结果
        if results['results']:
            first_result = results['results'][0]
            print(f"\n📄 Sample result (Article: {first_result.article_id}):")
            print(f"   Framing Score: {first_result.framing_score:.3f}")
            print(f"   Bias Intensity: {first_result.bias_intensity}")
            if hasattr(first_result, 'evidence') and first_result.evidence:
                print(f"   Evidence Count: {len(first_result.evidence)}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False
    
    # 5. 省略检测测试（可选）
    print("\n5️⃣  Testing omission detection (optional)...")
    try:
        omission_config = config
        omission_config.omission.enabled = True
        
        omission_analyzer = create_analyzer(omission_config)
        omission_results = omission_analyzer.analyze_batch(articles[:2])  # 只测试2篇
        
        omission_count = sum(1 for r in omission_results['results'] 
                           if hasattr(r, 'omission_result') and r.omission_result)
        
        print(f"✅ Omission detection test completed")
        print(f"📊 Articles with omissions: {omission_count}/{len(omission_results['results'])}")
        
    except Exception as e:
        print(f"⚠️  Omission detection test failed: {e}")
    
    print("\n🎉 Quick test completed successfully!")
    print("\n💡 To run comprehensive test:")
    print("   python framing_analyzer/comprehensive_test.py --sample 20")
    
    return True

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
