#!/usr/bin/env python3
"""
省略检测融合测试脚本
专门测试省略检测的启用和融合功能

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/test_omission_fusion.py
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from framing_analyzer import AnalyzerConfig, create_analyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OmissionFusionTest:
    """省略检测融合测试类"""
    
    def __init__(self):
        self.data_path = Path("data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv")
        self.output_dir = Path("results/omission_fusion_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_results = {
            'step1_omission_enablement': {},
            'step2_fusion_verification': {},
            'fusion_weight_optimization': {},
            'performance_comparison': {}
        }
    
    def load_test_data(self, sample_size: int = 20) -> List[Dict]:
        """加载测试数据"""
        logger.info(f"📁 Loading test data (sample_size={sample_size})")
        
        if not self.data_path.exists():
            # 如果没有大数据集，使用内置测试数据
            logger.warning("Large dataset not found, using built-in test data")
            return self._get_builtin_test_data()
        
        df = pd.read_csv(self.data_path, encoding="utf-8")
        df = df[df["content"].notna() & df["title"].notna()]
        df = df[df["content"].str.len() > 200]  # 确保文章足够长
        
        df_sample = df.head(sample_size)
        
        articles = []
        for idx, row in df_sample.iterrows():
            article = {
                "id": row.get("url") or f"article_{idx}",
                "title": str(row["title"]),
                "content": str(row["content"]),
                "publication": row.get("publication", "unknown"),
                "date": row.get("date", "unknown"),
            }
            
            # 添加ground truth（如果存在）
            if "bias_label" in row and pd.notna(row["bias_label"]):
                article["ground_truth_bias"] = row["bias_label"]
            
            articles.append(article)
        
        logger.info(f"✅ Loaded {len(articles)} articles")
        return articles
    
    def _get_builtin_test_data(self) -> List[Dict]:
        """内置测试数据（用于没有大数据集的情况）"""
        return [
            {
                "id": "test_1",
                "title": "Economic Policy Changes Announced by Government",
                "content": "The government announced significant changes to economic policy yesterday during a press conference. The new measures include comprehensive tax reforms and strategic spending cuts across multiple departments. Officials stated that these changes will help reduce the national deficit and improve fiscal responsibility. The policy affects multiple sectors including healthcare, education, and infrastructure development. Critics argue that the cuts may harm essential services, while supporters believe the reforms are necessary for long-term economic stability."
            },
            {
                "id": "test_2", 
                "title": "Economic Reforms Impact Multiple Sectors Nationwide",
                "content": "Recent economic reforms have begun affecting various sectors of the economy in unprecedented ways. Healthcare providers are expressing serious concerns about potential funding cuts that could impact patient care. Education officials are worried about reduced budgets affecting classroom resources and teacher salaries. The manufacturing sector, however, sees potential benefits from the proposed tax changes. Labor unions have called for protests, while business groups have praised the government's fiscal discipline."
            },
            {
                "id": "test_3",
                "title": "Policy Implementation Timeline Released to Public",
                "content": "The government released a detailed timeline for implementing the new economic policies over the next two years. The first phase begins next month with comprehensive tax code changes affecting both individuals and corporations. Healthcare funding adjustments will follow in the second quarter, with education budget modifications scheduled for the summer. Infrastructure projects will be reviewed and potentially scaled back in the fall. Opposition parties have criticized the rushed timeline, calling for more public consultation."
            },
            {
                "id": "test_4",
                "title": "Citizens React to Economic Policy Announcements",
                "content": "Public reaction to the new economic policies has been mixed, with various groups expressing different concerns. Small business owners worry about increased tax burdens, while large corporations welcome the simplified tax structure. Healthcare workers have organized rallies to protest potential funding cuts. Teachers' unions are planning strikes if education budgets are reduced. Consumer advocacy groups warn that the policies may lead to higher prices for essential goods and services."
            },
            {
                "id": "test_5",
                "title": "International Response to Economic Policy Changes",
                "content": "International observers and trading partners have begun responding to the announced economic policy changes. The International Monetary Fund expressed cautious optimism about the fiscal reforms. European Union officials noted concerns about potential trade implications. Asian markets showed mixed reactions, with some sectors benefiting from anticipated policy changes. Credit rating agencies are reviewing the country's fiscal outlook in light of the new policies."
            }
        ]
    
    def step1_test_omission_enablement(self, articles: List[Dict]) -> Dict:
        """步骤1：测试省略检测是否正确启用"""
        logger.info("🔍 Step 1: Testing omission detection enablement")
        
        # 创建启用省略检测的配置
        config = AnalyzerConfig()
        config.teacher.bias_class_index = 1
        config.teacher.model_local_path = "bias_detector_data"
        config.output.output_dir = str(self.output_dir / "step1")
        
        # 启用省略检测
        config.omission.enabled = True
        config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
        config.omission.fusion_weight = 0.2
        config.omission.key_topics_count = 10
        config.omission.similarity_threshold = 0.5
        
        logger.info(f"✅ Configuration created with omission.enabled = {config.omission.enabled}")
        
        try:
            # 创建分析器
            analyzer = create_analyzer(config)
            
            # 验证省略检测器是否初始化
            if not hasattr(analyzer, 'omission_detector') or not analyzer.omission_detector:
                raise ValueError("Omission detector not initialized")
            
            logger.info("✅ Omission detector initialized successfully")
            
            # 分析文章
            start_time = time.time()
            results = analyzer.analyze_batch(articles)
            analysis_time = time.time() - start_time
            
            # 检查结果
            omission_fields_count = 0
            fusion_evidence_count = 0
            
            if 'results' in results and results['results']:
                for result in results['results']:
                    # 检查省略字段
                    if result.get('omission_score') is not None:
                        omission_fields_count += 1
                    
                    # 检查融合证据（framing_intensity应该受到omission_score影响）
                    if (result.get('omission_score') is not None and 
                        result.get('framing_intensity') is not None):
                        fusion_evidence_count += 1
            
            # 记录结果
            step1_results = {
                'success': True,
                'total_articles': len(articles),
                'articles_with_omission_fields': omission_fields_count,
                'articles_with_fusion_evidence': fusion_evidence_count,
                'analysis_time': analysis_time,
                'omission_detection_rate': omission_fields_count / len(articles) * 100,
                'fusion_rate': fusion_evidence_count / len(articles) * 100,
                'config_used': {
                    'omission_enabled': config.omission.enabled,
                    'fusion_weight': config.omission.fusion_weight,
                    'embedding_model': config.omission.embedding_model_name_or_path
                }
            }
            
            # 保存详细结果
            output_file = self.output_dir / "step1" / "omission_enablement_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'step1_results': step1_results,
                    'detailed_results': results
                }, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"✅ Step 1 completed: {omission_fields_count}/{len(articles)} articles with omission fields")
            logger.info(f"📊 Omission detection rate: {step1_results['omission_detection_rate']:.1f}%")
            
            return step1_results
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def step2_test_fusion_verification(self, articles: List[Dict]) -> Dict:
        """步骤2：测试线性融合是否正确工作"""
        logger.info("🧪 Step 2: Testing linear fusion verification")
        
        fusion_weights = [0.0, 0.1, 0.2, 0.3]
        fusion_results = {}
        
        for weight in fusion_weights:
            logger.info(f"   Testing fusion_weight = {weight}")
            
            try:
                # 创建配置
                config = AnalyzerConfig()
                config.teacher.bias_class_index = 1
                config.teacher.model_local_path = "bias_detector_data"
                config.output.output_dir = str(self.output_dir / f"step2_weight_{weight}")
                
                # 省略检测配置
                config.omission.enabled = True
                config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
                config.omission.fusion_weight = weight
                config.omission.key_topics_count = 10
                
                # 分析
                analyzer = create_analyzer(config)
                results = analyzer.analyze_batch(articles)
                
                # 提取分数
                if 'results' in results and results['results']:
                    framing_intensities = []
                    omission_scores = []
                    
                    for result in results['results']:
                        if not result.get('error'):
                            framing_intensities.append(result.get('framing_intensity', 0.0))
                            omission_scores.append(result.get('omission_score'))
                    
                    fusion_results[weight] = {
                        'avg_framing_intensity': np.mean(framing_intensities) if framing_intensities else 0.0,
                        'std_framing_intensity': np.std(framing_intensities) if framing_intensities else 0.0,
                        'avg_omission_score': np.mean([s for s in omission_scores if s is not None]),
                        'articles_with_omission': sum(1 for s in omission_scores if s is not None),
                        'sample_results': [
                            {
                                'id': result.get('id'),
                                'framing_intensity': result.get('framing_intensity'),
                                'omission_score': result.get('omission_score')
                            }
                            for result in results['results'][:3]  # 前3个样本
                            if not result.get('error')
                        ]
                    }
                
                logger.info(f"     ✅ Weight {weight}: avg_framing_intensity = {fusion_results[weight]['avg_framing_intensity']:.3f}")
                
            except Exception as e:
                logger.error(f"     ❌ Weight {weight} failed: {e}")
                fusion_results[weight] = {'error': str(e)}
        
        # 分析融合效果
        step2_results = {
            'success': True,
            'fusion_weights_tested': fusion_weights,
            'fusion_results': fusion_results,
            'fusion_analysis': self._analyze_fusion_effects(fusion_results)
        }
        
        # 保存结果
        output_file = self.output_dir / "step2_fusion_verification.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(step2_results, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info("✅ Step 2 completed: Linear fusion verification")
        
        return step2_results
    
    def _analyze_fusion_effects(self, fusion_results: Dict) -> Dict:
        """分析融合效果"""
        analysis = {
            'weight_impact': {},
            'recommendations': []
        }
        
        # 分析不同权重的影响
        valid_results = {w: r for w, r in fusion_results.items() if 'error' not in r}
        
        if len(valid_results) >= 2:
            weights = sorted(valid_results.keys())
            intensities = [valid_results[w]['avg_framing_intensity'] for w in weights]
            
            # 计算变化趋势
            if len(intensities) > 1:
                intensity_change = intensities[-1] - intensities[0]
                analysis['weight_impact'] = {
                    'min_weight': weights[0],
                    'max_weight': weights[-1],
                    'intensity_change': intensity_change,
                    'change_direction': 'increasing' if intensity_change > 0 else 'decreasing'
                }
                
                # 推荐
                if abs(intensity_change) > 0.05:
                    analysis['recommendations'].append(
                        f"Fusion weight has significant impact ({intensity_change:.3f} change)"
                    )
                else:
                    analysis['recommendations'].append(
                        "Fusion weight has minimal impact, consider using default 0.2"
                    )
        
        return analysis
    
    def test_performance_comparison(self, articles: List[Dict]) -> Dict:
        """测试性能对比（有无省略检测）"""
        logger.info("⚡ Testing performance comparison (with/without omission)")
        
        comparison_results = {}
        
        # 测试无省略检测
        try:
            config_no_omission = AnalyzerConfig()
            config_no_omission.teacher.bias_class_index = 1
            config_no_omission.teacher.model_local_path = "bias_detector_data"
            config_no_omission.omission.enabled = False
            
            analyzer_no_omission = create_analyzer(config_no_omission)
            
            start_time = time.time()
            results_no_omission = analyzer_no_omission.analyze_batch(articles)
            time_no_omission = time.time() - start_time
            
            comparison_results['without_omission'] = {
                'analysis_time': time_no_omission,
                'avg_time_per_article': time_no_omission / len(articles),
                'success': True
            }
            
            logger.info(f"   Without omission: {time_no_omission:.2f}s")
            
        except Exception as e:
            comparison_results['without_omission'] = {'error': str(e), 'success': False}
        
        # 测试有省略检测
        try:
            config_with_omission = AnalyzerConfig()
            config_with_omission.teacher.bias_class_index = 1
            config_with_omission.teacher.model_local_path = "bias_detector_data"
            config_with_omission.omission.enabled = True
            config_with_omission.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
            config_with_omission.omission.fusion_weight = 0.2
            
            analyzer_with_omission = create_analyzer(config_with_omission)
            
            start_time = time.time()
            results_with_omission = analyzer_with_omission.analyze_batch(articles)
            time_with_omission = time.time() - start_time
            
            comparison_results['with_omission'] = {
                'analysis_time': time_with_omission,
                'avg_time_per_article': time_with_omission / len(articles),
                'success': True
            }
            
            logger.info(f"   With omission: {time_with_omission:.2f}s")
            
        except Exception as e:
            comparison_results['with_omission'] = {'error': str(e), 'success': False}
        
        # 计算性能影响
        if (comparison_results['without_omission'].get('success') and 
            comparison_results['with_omission'].get('success')):
            
            time_overhead = (comparison_results['with_omission']['analysis_time'] - 
                           comparison_results['without_omission']['analysis_time'])
            overhead_percentage = (time_overhead / 
                                 comparison_results['without_omission']['analysis_time']) * 100
            
            comparison_results['performance_impact'] = {
                'time_overhead_seconds': time_overhead,
                'overhead_percentage': overhead_percentage,
                'acceptable': overhead_percentage < 100  # 小于100%增长认为可接受
            }
            
            logger.info(f"   Performance overhead: {time_overhead:.2f}s ({overhead_percentage:.1f}%)")
        
        return comparison_results
    
    def run_comprehensive_test(self, sample_size: int = 15):
        """运行全面测试"""
        logger.info("🚀 Starting comprehensive omission fusion test")
        logger.info("="*60)
        
        try:
            # 加载数据
            articles = self.load_test_data(sample_size)
            
            # 步骤1：省略检测启用测试
            logger.info("\n" + "="*40)
            self.test_results['step1_omission_enablement'] = self.step1_test_omission_enablement(articles)
            
            # 步骤2：融合验证测试
            logger.info("\n" + "="*40)
            self.test_results['step2_fusion_verification'] = self.step2_test_fusion_verification(articles)
            
            # 性能对比测试
            logger.info("\n" + "="*40)
            self.test_results['performance_comparison'] = self.test_performance_comparison(articles)
            
            # 保存综合结果
            self._save_comprehensive_results()
            
            # 打印总结
            self._print_comprehensive_summary()
            
            logger.info("🎉 Comprehensive omission fusion test completed!")
            
        except Exception as e:
            logger.error(f"💥 Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_comprehensive_results(self):
        """保存综合结果"""
        output_file = self.output_dir / "comprehensive_omission_fusion_results.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_timestamp': time.time(),
                'test_results': self.test_results
            }, f, indent=2, default=str, ensure_ascii=False)
        
        logger.info(f"💾 Comprehensive results saved to: {output_file}")
    
    def _print_comprehensive_summary(self):
        """打印综合总结"""
        print("\n" + "="*60)
        print("🎯 OMISSION FUSION TEST SUMMARY")
        print("="*60)
        
        # 步骤1结果
        step1 = self.test_results['step1_omission_enablement']
        if step1.get('success'):
            print(f"✅ Step 1 - Omission Enablement:")
            print(f"   Omission detection rate: {step1['omission_detection_rate']:.1f}%")
            print(f"   Fusion rate: {step1['fusion_rate']:.1f}%")
        else:
            print(f"❌ Step 1 - Omission Enablement: FAILED")
            print(f"   Error: {step1.get('error', 'Unknown error')}")
        
        # 步骤2结果
        step2 = self.test_results['step2_fusion_verification']
        if step2.get('success'):
            print(f"✅ Step 2 - Fusion Verification:")
            fusion_results = step2['fusion_results']
            for weight, result in fusion_results.items():
                if 'error' not in result:
                    print(f"   Weight {weight}: avg_intensity = {result['avg_framing_intensity']:.3f}")
        else:
            print(f"❌ Step 2 - Fusion Verification: FAILED")
        
        # 性能对比结果
        perf = self.test_results['performance_comparison']
        if perf.get('with_omission', {}).get('success') and perf.get('without_omission', {}).get('success'):
            print(f"⚡ Performance Comparison:")
            print(f"   Without omission: {perf['without_omission']['analysis_time']:.2f}s")
            print(f"   With omission: {perf['with_omission']['analysis_time']:.2f}s")
            if 'performance_impact' in perf:
                print(f"   Overhead: {perf['performance_impact']['overhead_percentage']:.1f}%")
        
        print("="*60)

def main():
    """主函数"""
    test = OmissionFusionTest()
    test.run_comprehensive_test(sample_size=15)

if __name__ == "__main__":
    main()