#!/usr/bin/env python3
"""
全面的大规模测试脚本
测试所有已开发的framing_analyzer模块功能

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py [options]

选项：
    --sample N          只测试前N篇文章（默认：50）
    --full              测试全部数据（可能很慢）
    --enable-omission   启用省略检测功能
    --enable-relative   启用相对框架分析
    --output-dir DIR    输出目录（默认：results/comprehensive_test）
    --config-bias-index N  设置bias_class_index（默认：1）
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import (
    AnalyzerConfig, 
    create_analyzer,
    verify_bias_class_index,
    FramingAnalyzer,
    RelativeFramingAnalyzer
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveTest:
    """全面测试类"""
    
    def __init__(self, args):
        self.args = args
        self.data_path = Path("data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv")
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 测试结果
        self.test_results = {
            'start_time': time.time(),
            'config': None,
            'data_stats': {},
            'basic_analysis': {},
            'omission_analysis': {},
            'relative_analysis': {},
            'performance_stats': {},
            'article_results': [],
            'errors': []
        }
    
    def load_data(self) -> List[Dict]:
        """加载测试数据"""
        logger.info(f"📁 Loading data from: {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        # 读取数据
        df = pd.read_csv(self.data_path, encoding="utf-8")
        logger.info(f"📊 Original dataset: {len(df)} articles")
        
        # 数据清洗
        df = df[df["content"].notna() & df["title"].notna()]
        df = df[df["content"].str.len() > 100]  # 过滤太短的文章
        logger.info(f"📊 After cleaning: {len(df)} articles")
        
        # 采样
        if self.args.full:
            sample_size = len(df)
        else:
            sample_size = min(self.args.sample, len(df))
        
        df_sample = df.head(sample_size)
        logger.info(f"📊 Test sample: {len(df_sample)} articles")
        
        # 转换为标准格式
        articles = []
        for idx, row in df_sample.iterrows():
            article = {
                "id": row.get("url") or f"article_{idx}",
                "title": str(row["title"]),
                "content": str(row["content"]),
                "publication": row.get("publication", "unknown"),
                "date": row.get("date", "unknown"),
                "author": row.get("author", "unknown"),
                "section": row.get("section", "unknown"),
            }
            
            # 添加已有的bias标签（如果存在）
            if "bias_label" in df.columns and pd.notna(row["bias_label"]):
                article["bias_label"] = row["bias_label"]
            if "bias_probability" in df.columns and pd.notna(row["bias_probability"]):
                article["bias_probability"] = float(row["bias_probability"])
            
            articles.append(article)
        
        # 记录数据统计
        self.test_results['data_stats'] = {
            'total_articles': len(articles),
            'avg_content_length': np.mean([len(a['content']) for a in articles]),
            'publications': list(set(a['publication'] for a in articles)),
            'has_bias_labels': sum(1 for a in articles if 'bias_label' in a)
        }
        
        return articles
    
    def create_test_config(self) -> AnalyzerConfig:
        """创建测试配置"""
        logger.info("🔧 Creating test configuration...")
        
        config = AnalyzerConfig()
        
        # 配置bias_class_index
        config.teacher.bias_class_index = self.args.config_bias_index
        config.teacher.model_local_path = "bias_detector_data"
        config.teacher.batch_size = 16
        
        # 输出配置
        config.output.output_dir = str(self.output_dir)
        config.output.generate_plots = True
        config.output.include_evidence = True
        config.output.include_statistics = True
        
        # 省略检测配置
        if self.args.enable_omission:
            config.omission.enabled = True
            config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"  # 确保使用本地模型
            config.omission.key_topics_count = 10
            config.omission.similarity_threshold = 0.5
            config.omission.fusion_weight = 0.2  # 设置融合权重
            config.omission.guidance_threshold = 0.45
            config.omission.omission_effect_threshold = 0.4
            config.omission.min_topic_frequency = 2
            logger.info("✅ Omission detection enabled with tighter thresholds (guidance=0.45, effect>=0.4)")
        
        # 相对框架分析配置
        if self.args.enable_relative:
            config.relative_framing.enabled = True
            config.relative_framing.similarity_threshold = 0.3
            config.relative_framing.min_cluster_size = 2
            logger.info("✅ Relative framing analysis enabled")
        
        self.test_results['config'] = {
            'bias_class_index': config.teacher.bias_class_index,
            'omission_enabled': config.omission.enabled,
            'omission_fusion_weight': config.omission.fusion_weight if config.omission.enabled else None,
            'omission_guidance_threshold': config.omission.guidance_threshold if config.omission.enabled else None,
            'omission_effect_threshold': config.omission.omission_effect_threshold if config.omission.enabled else None,
            'relative_enabled': config.relative_framing.enabled,
            'batch_size': config.teacher.batch_size
        }
        
        return config
    
    def verify_model_setup(self):
        """验证模型设置"""
        logger.info("🔍 Verifying model setup...")
        
        try:
            result = verify_bias_class_index()
            if result and 'config_suggestion' in result:
                suggested_index = result['config_suggestion']['bias_class_index']
                if suggested_index != self.args.config_bias_index:
                    logger.warning(f"⚠️  Suggested bias_class_index: {suggested_index}, using: {self.args.config_bias_index}")
            logger.info("✅ Model verification completed")
        except Exception as e:
            logger.warning(f"⚠️  Model verification failed: {e}")
            self.test_results['errors'].append(f"Model verification: {str(e)}")
    
    def test_basic_analysis(self, articles: List[Dict], config: AnalyzerConfig):
        """测试基础分析功能"""
        logger.info("🧪 Testing basic framing analysis...")
        
        start_time = time.time()
        
        try:
            analyzer = create_analyzer(config)
            results = analyzer.analyze_batch(articles)
            
            analysis_time = time.time() - start_time
            
            # 统计结果 - 修复字段访问
            if 'results' in results and results['results']:
                # 结果是字典格式，不是对象
                framing_intensities = []
                pseudo_labels = []
                
                for result_dict in results['results']:
                    if isinstance(result_dict, dict):
                        framing_intensities.append(result_dict.get('framing_intensity', 0.0))
                        pseudo_labels.append(result_dict.get('pseudo_label', 'uncertain'))
                    else:
                        # 如果是对象格式
                        framing_intensities.append(getattr(result_dict, 'framing_intensity', 0.0))
                        pseudo_labels.append(getattr(result_dict, 'pseudo_label', 'uncertain'))
                
                self.test_results['basic_analysis'] = {
                    'success': True,
                    'total_articles': len(results['results']),
                    'analysis_time': analysis_time,
                    'avg_time_per_article': analysis_time / len(articles),
                    'framing_intensity_stats': {
                        'mean': np.mean(framing_intensities),
                        'std': np.std(framing_intensities),
                        'min': np.min(framing_intensities),
                        'max': np.max(framing_intensities)
                    },
                    'pseudo_label_distribution': {
                        label: sum(1 for pl in pseudo_labels if pl == label)
                        for label in set(pseudo_labels)
                    }
                }
                
                logger.info(f"✅ Basic analysis completed in {analysis_time:.2f}s")
                logger.info(f"📊 Average framing intensity: {np.mean(framing_intensities):.3f}")
            else:
                raise ValueError("No results returned from analyzer")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Basic analysis failed: {e}")
            self.test_results['basic_analysis'] = {
                'success': False,
                'error': str(e)
            }
            self.test_results['errors'].append(f"Basic analysis: {str(e)}")
            return None
    
    def test_omission_analysis(self, articles: List[Dict], config: AnalyzerConfig):
        """测试省略检测功能"""
        if not self.args.enable_omission:
            logger.info("⏭️  Skipping omission analysis (not enabled)")
            return None
        
        logger.info("🧪 Testing omission detection...")
        
        start_time = time.time()
        
        try:
            # 启用省略检测的配置
            omission_config = config
            omission_config.omission.enabled = True
            
            analyzer = create_analyzer(omission_config)
            results = analyzer.analyze_batch(articles)
            
            analysis_time = time.time() - start_time
            
            # 统计省略检测结果 - 修复字段访问
            omission_count = 0
            omission_scores = []
            missing_topic_counts = []
            applied_flags = []
            effect_threshold = getattr(config.omission, 'omission_effect_threshold', 0.5)
            if 'results' in results and results['results']:
                for result_dict in results['results']:
                    # 统一用dict接口
                    if not isinstance(result_dict, dict):
                        continue
                    
                    score = result_dict.get('omission_score')
                    stats = result_dict.get('statistics', {}) if isinstance(result_dict, dict) else {}
                    
                    if score is not None:
                        omission_scores.append(score)
                        if score >= effect_threshold:
                            omission_count += 1
                    
                    if stats:
                        missing_topic_counts.append(stats.get('key_topics_missing_count', 0))
                        if 'omission_applied' in stats:
                            applied_flags.append(bool(stats['omission_applied']))
            
            omission_score_stats = {}
            omission_missing_stats = {}
            if omission_scores:
                omission_score_stats = {
                    'mean': float(np.mean(omission_scores)),
                    'std': float(np.std(omission_scores)),
                    'min': float(np.min(omission_scores)),
                    'median': float(np.median(omission_scores)),
                    'p95': float(np.percentile(omission_scores, 95)),
                    'max': float(np.max(omission_scores))
                }
            if missing_topic_counts:
                omission_missing_stats = {
                    'min': int(np.min(missing_topic_counts)),
                    'median': float(np.median(missing_topic_counts)),
                    'p95': float(np.percentile(missing_topic_counts, 95)),
                    'max': int(np.max(missing_topic_counts))
                }
            applied_rate = (len([f for f in applied_flags if f]) / len(applied_flags)) if applied_flags else 0.0
            
            self.test_results['omission_analysis'] = {
                'success': True,
                'total_articles': len(results['results']) if results and 'results' in results else 0,
                'articles_with_omissions': omission_count,
                'analysis_time': analysis_time,
                'omission_rate': omission_count / len(articles) if articles else 0,
                'omission_score_stats': omission_score_stats,
                'omission_missing_stats': omission_missing_stats,
                'omission_applied_rate': applied_rate,
                'omission_effect_threshold': effect_threshold
            }
            
            # 保存每篇文章的精简结果，方便后续手动排查
            if results and 'results' in results:
                self.test_results['article_results'] = self._extract_article_summaries(
                    results['results'],
                    effect_threshold
                )
            
            logger.info(f"✅ Omission analysis completed in {analysis_time:.2f}s")
            logger.info(f"📊 Omission detection rate: {omission_count}/{len(articles)} (threshold >= {effect_threshold})")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Omission analysis failed: {e}")
            self.test_results['omission_analysis'] = {
                'success': False,
                'error': str(e)
            }
            self.test_results['errors'].append(f"Omission analysis: {str(e)}")
            return None
    
    def test_relative_analysis(self, articles: List[Dict], config: AnalyzerConfig):
        """测试相对框架分析"""
        if not self.args.enable_relative:
            logger.info("⏭️  Skipping relative analysis (not enabled)")
            return None
        
        logger.info("🧪 Testing relative framing analysis...")
        
        start_time = time.time()
        
        try:
            # 相对框架分析需要足够的文章数量
            if len(articles) < 5:
                logger.warning("⚠️  Too few articles for relative analysis, skipping")
                return None
            
            relative_analyzer = RelativeFramingAnalyzer(config)
            results = relative_analyzer.analyze_batch(articles)
            
            analysis_time = time.time() - start_time
            
            self.test_results['relative_analysis'] = {
                'success': True,
                'total_articles': len(articles),
                'analysis_time': analysis_time,
                'clusters_found': len(results.get('clusters', [])) if results else 0
            }
            
            logger.info(f"✅ Relative analysis completed in {analysis_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Relative analysis failed: {e}")
            self.test_results['relative_analysis'] = {
                'success': False,
                'error': str(e)
            }
            self.test_results['errors'].append(f"Relative analysis: {str(e)}")
            return None
    
    def evaluate_against_bias_labels(self, results, articles: List[Dict]):
        """与bias标签对比评估"""
        if not results or 'results' not in results:
            return
        
        logger.info("📊 Evaluating against bias labels...")
        
        labeled_articles = [a for a in articles if 'bias_label' in a or 'bias_probability' in a]
        
        if not labeled_articles:
            logger.info("⏭️  No bias labels available")
            return
        
        # 创建ID到结果的映射 - 修复字段访问
        results_by_id = {}
        for result_dict in results['results']:
            if isinstance(result_dict, dict):
                article_id = result_dict.get('id')
                if article_id:
                    results_by_id[article_id] = result_dict
            else:
                # 如果是对象格式
                article_id = getattr(result_dict, 'id', None) or getattr(result_dict, 'article_id', None)
                if article_id:
                    results_by_id[article_id] = result_dict
        
        comparisons = []
        for article in labeled_articles:
            if article['id'] in results_by_id:
                result = results_by_id[article['id']]
                
                # 获取预测值 - 修复字段访问
                if isinstance(result, dict):
                    predicted_intensity = result.get('framing_intensity', 0.0)
                    predicted_label = result.get('pseudo_label', 'uncertain')
                else:
                    predicted_intensity = getattr(result, 'framing_intensity', 0.0)
                    predicted_label = getattr(result, 'pseudo_label', 'uncertain')
                
                # 使用bias_probability作为ground truth，如果没有则使用bias_label
                ground_truth_value = None
                if 'bias_probability' in article:
                    ground_truth_value = article['bias_probability']
                elif 'bias_label' in article:
                    ground_truth_value = article['bias_label']
                
                if ground_truth_value is not None:
                    comparisons.append({
                        'ground_truth': ground_truth_value,
                        'predicted_intensity': predicted_intensity,
                        'predicted_label': predicted_label
                    })
        
        if comparisons:
            # 简单的相关性分析
            gt_scores = [c['ground_truth'] for c in comparisons]
            pred_scores = [c['predicted_intensity'] for c in comparisons]
            
            correlation = np.corrcoef(gt_scores, pred_scores)[0, 1] if len(gt_scores) > 1 else 0
            
            self.test_results['evaluation'] = {
                'total_comparisons': len(comparisons),
                'correlation': correlation,
                'avg_predicted_intensity': np.mean(pred_scores),
                'avg_ground_truth': np.mean(gt_scores)
            }
            
            logger.info(f"📊 Correlation with bias labels: {correlation:.3f}")

    def _extract_article_summaries(self, results: List[Dict], omission_threshold: float) -> List[Dict]:
        """提取每篇文章的精简结果，便于调试和分布统计"""
        summaries = []
        threshold = omission_threshold if omission_threshold is not None else 0.0
        for res in results:
            stats = res.get('statistics', {}) if isinstance(res, dict) else {}
            omission_score = res.get('omission_score') if isinstance(res, dict) else None
            summaries.append({
                'id': res.get('id') if isinstance(res, dict) else None,
                'title': res.get('title') if isinstance(res, dict) else None,
                'framing_intensity': res.get('framing_intensity') if isinstance(res, dict) else None,
                'pseudo_label': res.get('pseudo_label') if isinstance(res, dict) else None,
                'omission_score': omission_score,
                'omission_score_effective': stats.get('omission_score_effective'),
                'omission_applied': stats.get('omission_applied'),
                'key_topics_missing_count': stats.get('key_topics_missing_count'),
                'omission_locations_count': stats.get('omission_locations_count'),
                'considered_omission': omission_score is not None and omission_score >= threshold
            })
        return summaries
    
    def save_results(self):
        """保存测试结果"""
        self.test_results['end_time'] = time.time()
        self.test_results['total_duration'] = self.test_results['end_time'] - self.test_results['start_time']
        
        # 保存详细结果
        results_file = self.output_dir / "comprehensive_test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Results saved to: {results_file}")
        
        # 打印摘要
        self.print_summary()
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        
        print(f"📊 Data: {self.test_results['data_stats']['total_articles']} articles")
        print(f"⏱️  Total time: {self.test_results['total_duration']:.2f}s")
        
        if self.test_results['basic_analysis'].get('success'):
            ba = self.test_results['basic_analysis']
            print(f"✅ Basic Analysis: {ba['total_articles']} articles in {ba['analysis_time']:.2f}s")
            print(f"   Average framing intensity: {ba['framing_intensity_stats']['mean']:.3f}")
        
        if self.test_results['omission_analysis'].get('success'):
            oa = self.test_results['omission_analysis']
            print(f"✅ Omission Analysis: {oa['articles_with_omissions']}/{oa['total_articles']} articles with omissions (threshold >= {oa.get('omission_effect_threshold', 0)})")
            if oa.get('omission_score_stats'):
                stats = oa['omission_score_stats']
                print(f"   Omission score median/p95: {stats.get('median', 0):.3f}/{stats.get('p95', 0):.3f}")
            if oa.get('omission_missing_stats'):
                miss_stats = oa['omission_missing_stats']
                print(f"   Missing topics min/med/p95: {miss_stats.get('min', 0)}/{miss_stats.get('median', 0):.1f}/{miss_stats.get('p95', 0):.1f}")
        
        if self.test_results['relative_analysis'].get('success'):
            ra = self.test_results['relative_analysis']
            print(f"✅ Relative Analysis: {ra['clusters_found']} clusters found")
        
        if 'evaluation' in self.test_results:
            ev = self.test_results['evaluation']
            print(f"📊 Bias Label Correlation: {ev['correlation']:.3f}")
        
        if self.test_results['errors']:
            print(f"❌ Errors: {len(self.test_results['errors'])}")
            for error in self.test_results['errors']:
                print(f"   - {error}")
        
        print("="*60)
    
    def run(self):
        """运行全面测试"""
        logger.info("🚀 Starting comprehensive test...")
        
        try:
            # 1. 验证模型设置
            self.verify_model_setup()
            
            # 2. 加载数据
            articles = self.load_data()
            
            # 3. 创建配置
            config = self.create_test_config()
            
            # 4. 基础分析测试
            basic_results = self.test_basic_analysis(articles, config)
            
            # 5. 省略检测测试
            omission_results = self.test_omission_analysis(articles, config)
            
            # 6. 相对框架分析测试
            relative_results = self.test_relative_analysis(articles, config)
            
            # 7. 评估
            if basic_results:
                self.evaluate_against_bias_labels(basic_results, articles)
            
            # 8. 保存结果
            self.save_results()
            
            logger.info("🎉 Comprehensive test completed successfully!")
            
        except Exception as e:
            logger.error(f"💥 Test failed: {e}")
            self.test_results['errors'].append(f"Main execution: {str(e)}")
            self.save_results()
            raise

def main():
    parser = argparse.ArgumentParser(description="Comprehensive framing analyzer test")
    parser.add_argument("--sample", type=int, default=50, help="Number of articles to test (default: 50)")
    parser.add_argument("--full", action="store_true", help="Test all articles in dataset")
    parser.add_argument("--enable-omission", action="store_true", dest="enable_omission", help="Enable omission detection")
    parser.add_argument("--disable-omission", action="store_false", dest="enable_omission", help="Disable omission detection")
    parser.add_argument("--enable-relative", action="store_true", dest="enable_relative", help="Enable relative framing analysis")
    parser.add_argument("--disable-relative", action="store_false", dest="enable_relative", help="Disable relative framing analysis")
    parser.add_argument("--output-dir", default="results/comprehensive_test", help="Output directory")
    parser.add_argument("--config-bias-index", type=int, default=1, help="Bias class index (default: 1)")
    
    parser.set_defaults(enable_omission=True, enable_relative=True)
    args = parser.parse_args()
    
    # 运行测试
    test = ComprehensiveTest(args)
    test.run()

if __name__ == "__main__":
    main()
