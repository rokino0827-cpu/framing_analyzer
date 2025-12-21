#!/usr/bin/env python3
"""
省略检测融合权重优化脚本
通过网格搜索找到最佳的fusion_weight参数
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

# 配置日志
logging.basicConfig(level=logging.WARNING)  # 减少噪音
logger = logging.getLogger(__name__)

class FusionWeightOptimizer:
    """融合权重优化器"""
    
    def __init__(self, data_path: str = "data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv"):
        self.data_path = Path(data_path)
        self.results = {}
    
    def load_evaluation_data(self, max_articles: int = 100) -> Tuple[List[Dict], List[float]]:
        """加载带有ground truth的评估数据"""
        
        print(f"📁 Loading evaluation data from: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 读取数据
        df = pd.read_csv(self.data_path, encoding="utf-8")
        
        # 过滤有ground truth的数据
        df = df[df["bias_label"].notna() & df["bias_probability"].notna()]
        df = df[df["content"].notna() & df["title"].notna()]
        df = df[df["content"].str.len() > 100]
        
        print(f"📊 Found {len(df)} articles with ground truth")
        
        # 采样
        if len(df) > max_articles:
            df = df.sample(n=max_articles, random_state=42)
            print(f"📊 Sampled {len(df)} articles for evaluation")
        
        # 转换为标准格式
        articles = []
        ground_truth = []
        
        for idx, row in df.iterrows():
            articles.append({
                "id": f"eval_{idx}",
                "title": str(row["title"]),
                "content": str(row["content"])
            })
            ground_truth.append(float(row["bias_probability"]))
        
        return articles, ground_truth
    
    def evaluate_fusion_weight(self, articles: List[Dict], ground_truth: List[float], 
                              fusion_weight: float) -> Dict:
        """评估特定融合权重的性能"""
        
        print(f"🧪 Testing fusion_weight = {fusion_weight:.2f}")
        
        try:
            # 创建配置
            config = AnalyzerConfig()
            config.teacher.bias_class_index = 1
            config.teacher.model_local_path = "bias_detector_data"
            config.teacher.batch_size = 16
            config.output.generate_plots = False  # 关闭图表生成
            
            # 省略检测配置
            config.omission.enabled = True
            config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
            config.omission.fusion_weight = fusion_weight
            config.omission.key_topics_count = 10
            
            # 分析
            start_time = time.time()
            analyzer = create_analyzer(config)
            results = analyzer.analyze_batch(articles)
            analysis_time = time.time() - start_time
            
            # 提取预测分数
            predicted_scores = []
            omission_scores = []
            
            for result in results['results']:
                framing_intensity = result.get('framing_intensity', 0.0)
                omission_score = result.get('omission_score')
                
                predicted_scores.append(framing_intensity)
                omission_scores.append(omission_score if omission_score is not None else 0.0)
            
            # 计算评估指标
            correlation, p_value = pearsonr(ground_truth, predicted_scores)
            
            # 转换为二分类进行AUC计算
            binary_gt = [1 if gt > 0.5 else 0 for gt in ground_truth]
            binary_pred = [1 if pred > 0.5 else 0 for pred in predicted_scores]
            
            try:
                auc_score = roc_auc_score(binary_gt, predicted_scores)
            except:
                auc_score = 0.0
            
            accuracy = accuracy_score(binary_gt, binary_pred)
            
            # 省略检测统计
            omission_rate = sum(1 for os in omission_scores if os > 0.1) / len(omission_scores)
            avg_omission_score = np.mean([os for os in omission_scores if os > 0])
            
            result = {
                'fusion_weight': fusion_weight,
                'correlation': correlation,
                'p_value': p_value,
                'auc_score': auc_score,
                'accuracy': accuracy,
                'analysis_time': analysis_time,
                'omission_rate': omission_rate,
                'avg_omission_score': avg_omission_score,
                'avg_predicted_score': np.mean(predicted_scores),
                'std_predicted_score': np.std(predicted_scores)
            }
            
            print(f"   📊 Correlation: {correlation:.3f} (p={p_value:.3f})")
            print(f"   📊 AUC: {auc_score:.3f}")
            print(f"   📊 Accuracy: {accuracy:.3f}")
            print(f"   📊 Omission rate: {omission_rate:.3f}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return {
                'fusion_weight': fusion_weight,
                'error': str(e)
            }
    
    def grid_search(self, articles: List[Dict], ground_truth: List[float], 
                   weight_range: Tuple[float, float] = (0.0, 0.5),
                   num_points: int = 11) -> Dict:
        """网格搜索最佳融合权重"""
        
        print(f"🔍 Grid search for optimal fusion weight...")
        print(f"   Range: {weight_range[0]:.1f} - {weight_range[1]:.1f}")
        print(f"   Points: {num_points}")
        
        # 生成权重网格
        weights = np.linspace(weight_range[0], weight_range[1], num_points)
        
        results = []
        best_result = None
        best_correlation = -1
        
        for weight in weights:
            result = self.evaluate_fusion_weight(articles, ground_truth, weight)
            results.append(result)
            
            # 跟踪最佳结果
            if 'correlation' in result and result['correlation'] > best_correlation:
                best_correlation = result['correlation']
                best_result = result
        
        # 保存结果
        optimization_results = {
            'best_result': best_result,
            'all_results': results,
            'search_config': {
                'weight_range': weight_range,
                'num_points': num_points,
                'num_articles': len(articles)
            }
        }
        
        return optimization_results
    
    def save_results(self, results: Dict, output_path: str = "results/fusion_optimization"):
        """保存优化结果"""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_file = output_dir / "fusion_weight_optimization.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"💾 Results saved to: {results_file}")
        
        # 生成摘要报告
        self.generate_report(results, output_dir)
    
    def generate_report(self, results: Dict, output_dir: Path):
        """生成优化报告"""
        
        report_file = output_dir / "optimization_report.md"
        
        best = results['best_result']
        all_results = [r for r in results['all_results'] if 'correlation' in r]
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Fusion Weight Optimization Report\n\n")
            
            f.write("## Best Configuration\n\n")
            f.write(f"- **Optimal fusion_weight**: {best['fusion_weight']:.3f}\n")
            f.write(f"- **Correlation**: {best['correlation']:.3f} (p={best['p_value']:.3f})\n")
            f.write(f"- **AUC Score**: {best['auc_score']:.3f}\n")
            f.write(f"- **Accuracy**: {best['accuracy']:.3f}\n")
            f.write(f"- **Omission Rate**: {best['omission_rate']:.3f}\n\n")
            
            f.write("## All Results\n\n")
            f.write("| Fusion Weight | Correlation | AUC | Accuracy | Omission Rate |\n")
            f.write("|---------------|-------------|-----|----------|---------------|\n")
            
            for result in all_results:
                f.write(f"| {result['fusion_weight']:.3f} | {result['correlation']:.3f} | "
                       f"{result['auc_score']:.3f} | {result['accuracy']:.3f} | "
                       f"{result['omission_rate']:.3f} |\n")
            
            f.write("\n## Recommendations\n\n")
            
            if best['correlation'] > 0.1:
                f.write(f"✅ **Recommended**: Use fusion_weight = {best['fusion_weight']:.3f}\n\n")
                f.write("This configuration shows the best correlation with ground truth labels.\n\n")
            else:
                f.write("⚠️ **Warning**: Low correlation with ground truth across all weights.\n\n")
                f.write("Consider:\n")
                f.write("- Checking data quality\n")
                f.write("- Adjusting other hyperparameters\n")
                f.write("- Using more training data\n\n")
            
            f.write("## Configuration\n\n")
            f.write("```python\n")
            f.write("config.omission.enabled = True\n")
            f.write(f"config.omission.fusion_weight = {best['fusion_weight']:.3f}\n")
            f.write("config.omission.embedding_model_name_or_path = 'all-MiniLM-L6-v2'\n")
            f.write("```\n")
        
        print(f"📄 Report saved to: {report_file}")

def main():
    """主函数"""
    
    print("="*60)
    print("🔍 FUSION WEIGHT OPTIMIZATION")
    print("="*60)
    
    optimizer = FusionWeightOptimizer()
    
    try:
        # 加载评估数据
        articles, ground_truth = optimizer.load_evaluation_data(max_articles=50)  # 限制数量以加快测试
        
        # 网格搜索
        results = optimizer.grid_search(articles, ground_truth, 
                                      weight_range=(0.0, 0.4), 
                                      num_points=9)
        
        # 保存结果
        optimizer.save_results(results)
        
        # 打印最佳结果
        best = results['best_result']
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"🏆 Best fusion_weight: {best['fusion_weight']:.3f}")
        print(f"📊 Best correlation: {best['correlation']:.3f}")
        print(f"📊 AUC Score: {best['auc_score']:.3f}")
        print(f"📊 Accuracy: {best['accuracy']:.3f}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()