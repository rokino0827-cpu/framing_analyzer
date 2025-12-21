#!/usr/bin/env python3
"""
性能基准测试脚本
测试不同配置下的性能表现

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/benchmark_test.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np

# 设置路径
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

# 配置日志
logging.basicConfig(level=logging.WARNING)  # 减少日志噪音
logger = logging.getLogger(__name__)

class BenchmarkTest:
    """性能基准测试"""
    
    def __init__(self):
        self.data_path = PROJECT_ROOT / "data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv"
        self.results = {}
    
    def load_test_data(self, sample_sizes: List[int]) -> Dict[int, List[Dict]]:
        """加载不同大小的测试数据集"""
        print("📁 Loading benchmark data...")
        
        if not self.data_path.exists():
            # 使用内置数据
            base_article = {
                "id": "benchmark_article",
                "title": "Benchmark Test Article",
                "content": "This is a benchmark test article for performance testing. " * 20
            }
            
            datasets = {}
            for size in sample_sizes:
                datasets[size] = [
                    {**base_article, "id": f"benchmark_{i}"}
                    for i in range(size)
                ]
            return datasets
        
        # 从CSV加载
        df = pd.read_csv(self.data_path, encoding="utf-8")
        df = df[df["content"].notna() & df["title"].notna()]
        
        datasets = {}
        for size in sample_sizes:
            if size <= len(df):
                sample_df = df.head(size)
                articles = []
                for idx, row in sample_df.iterrows():
                    articles.append({
                        "id": f"benchmark_{idx}",
                        "title": str(row["title"]),
                        "content": str(row["content"])
                    })
                datasets[size] = articles
            else:
                print(f"⚠️  Requested size {size} > available data {len(df)}")
        
        return datasets
    
    def create_benchmark_configs(self) -> Dict[str, AnalyzerConfig]:
        """创建不同的基准配置"""
        configs = {}
        
        # 基础配置
        base_config = AnalyzerConfig()
        base_config.teacher.bias_class_index = 1
        base_config.teacher.model_local_path = str(PROJECT_ROOT / "bias_detector_data")
        base_config.output.generate_plots = False  # 关闭图表生成以提高速度
        
        # 1. 快速配置（小batch）
        fast_config = base_config
        fast_config.teacher.batch_size = 8
        fast_config.scoring.evidence_count = 3
        configs["fast"] = fast_config
        
        # 2. 标准配置
        standard_config = base_config
        standard_config.teacher.batch_size = 16
        standard_config.scoring.evidence_count = 5
        configs["standard"] = standard_config
        
        # 3. 高精度配置（大batch）
        precision_config = base_config
        precision_config.teacher.batch_size = 32
        precision_config.scoring.evidence_count = 10
        configs["precision"] = precision_config
        
        # 4. 省略检测配置
        omission_config = base_config
        omission_config.omission.enabled = True
        omission_config.teacher.batch_size = 16
        configs["omission"] = omission_config
        
        return configs
    
    def benchmark_config(self, config_name: str, config: AnalyzerConfig, 
                        datasets: Dict[int, List[Dict]]) -> Dict:
        """对单个配置进行基准测试"""
        print(f"\n🧪 Benchmarking config: {config_name}")
        
        config_results = {
            'config_name': config_name,
            'batch_size': config.teacher.batch_size,
            'omission_enabled': config.omission.enabled,
            'results': {}
        }
        
        for size, articles in datasets.items():
            print(f"  📊 Testing {size} articles...")
            
            try:
                # 创建分析器
                analyzer = create_analyzer(config)
                
                # 计时分析
                start_time = time.time()
                results = analyzer.analyze_batch(articles)
                end_time = time.time()
                
                analysis_time = end_time - start_time
                
                # 统计结果
                framing_scores = [r.framing_score for r in results['results']]
                
                config_results['results'][size] = {
                    'total_time': analysis_time,
                    'time_per_article': analysis_time / size,
                    'articles_per_second': size / analysis_time,
                    'success_count': len(results['results']),
                    'avg_framing_score': np.mean(framing_scores),
                    'score_std': np.std(framing_scores)
                }
                
                print(f"    ✅ {size} articles in {analysis_time:.2f}s ({size/analysis_time:.1f} articles/s)")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                config_results['results'][size] = {
                    'error': str(e)
                }
        
        return config_results
    
    def run_benchmark(self):
        """运行完整基准测试"""
        print("🚀 Starting performance benchmark...")
        
        # 测试数据大小
        sample_sizes = [1, 5, 10, 20, 50]
        
        # 加载数据
        datasets = self.load_test_data(sample_sizes)
        available_sizes = list(datasets.keys())
        print(f"📊 Available test sizes: {available_sizes}")
        
        # 创建配置
        configs = self.create_benchmark_configs()
        
        # 运行基准测试
        benchmark_results = {
            'timestamp': time.time(),
            'test_sizes': available_sizes,
            'configs': {}
        }
        
        for config_name, config in configs.items():
            try:
                result = self.benchmark_config(config_name, config, datasets)
                benchmark_results['configs'][config_name] = result
            except Exception as e:
                print(f"❌ Config {config_name} failed: {e}")
                benchmark_results['configs'][config_name] = {'error': str(e)}
        
        # 保存结果
        output_dir = PROJECT_ROOT / "results/benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / "benchmark_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # 打印摘要
        self.print_benchmark_summary(benchmark_results)
        
        return benchmark_results
    
    def print_benchmark_summary(self, results: Dict):
        """打印基准测试摘要"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("="*60)
        
        # 找到最大测试大小
        max_size = 0
        for config_data in results['configs'].values():
            if 'results' in config_data:
                sizes = [int(s) for s in config_data['results'].keys() if s != 'error']
                if sizes:
                    max_size = max(max_size, max(sizes))
        
        if max_size == 0:
            print("❌ No successful benchmark results")
            return
        
        print(f"📈 Performance at {max_size} articles:")
        print("-" * 40)
        
        performance_data = []
        for config_name, config_data in results['configs'].items():
            if 'results' in config_data and str(max_size) in config_data['results']:
                result = config_data['results'][str(max_size)]
                if 'error' not in result:
                    performance_data.append({
                        'config': config_name,
                        'time': result['total_time'],
                        'speed': result['articles_per_second'],
                        'batch_size': config_data.get('batch_size', 'N/A')
                    })
        
        # 按速度排序
        performance_data.sort(key=lambda x: x['speed'], reverse=True)
        
        for i, data in enumerate(performance_data):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
            print(f"{rank} {data['config']:12} | {data['speed']:6.1f} art/s | {data['time']:6.2f}s | batch={data['batch_size']}")
        
        print("\n💡 Recommendations:")
        if performance_data:
            fastest = performance_data[0]
            print(f"   🚀 Fastest: {fastest['config']} config ({fastest['speed']:.1f} articles/s)")
            
            # 寻找平衡点
            balanced = None
            for data in performance_data:
                if data['config'] == 'standard':
                    balanced = data
                    break
            
            if balanced:
                print(f"   ⚖️  Balanced: {balanced['config']} config ({balanced['speed']:.1f} articles/s)")
        
        print("="*60)

def main():
    benchmark = BenchmarkTest()
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
