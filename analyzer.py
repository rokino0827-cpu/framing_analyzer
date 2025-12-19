"""
主分析器 - 整合所有组件的核心分析器
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from tqdm import tqdm
import json
import os
from datetime import datetime

from .text_processor import TextProcessor, StructureZoneExtractor, FragmentGenerator
from .bias_teacher import TeacherInference
from .framing_scorer import FramingAnalysisEngine, FramingResult
from .relative_framing import RelativeFramingAnalyzer
from .omission_detector import OmissionDetector
from .utils import setup_logging, save_results, validate_input_data

logger = logging.getLogger(__name__)

class FramingAnalyzer:
    """框架偏见分析器主类"""
    
    def __init__(self, config):
        self.config = config
        
        # 设置日志
        setup_logging(config.log_level)
        
        # 初始化组件
        self.text_processor = TextProcessor(config)
        self.zone_extractor = StructureZoneExtractor(config)
        self.fragment_generator = FragmentGenerator(config)
        self.teacher_inference = TeacherInference(config)
        self.framing_engine = FramingAnalysisEngine(config)
        
        # 可选的相对框架分析器
        self.relative_analyzer = None
        if config.relative_framing.enabled:
            self.relative_analyzer = RelativeFramingAnalyzer(config)
        
        # 可选的省略感知检测器
        self.omission_detector = None
        if config.omission.enabled:
            self.omission_detector = OmissionDetector(config)
        
        logger.info("FramingAnalyzer initialized successfully")
    
    def analyze_article(self, content: str, title: str = "", 
                       article_id: Optional[str] = None,
                       event_cluster: Optional[List[Dict]] = None) -> FramingResult:
        """分析单篇文章
        
        Args:
            content: 文章内容
            title: 文章标题
            event_cluster: 同事件文章集群（用于省略检测）
        """
        
        # Step 1 & 2: 文本预处理和结构区划分
        processed_article = self.text_processor.process_article(content, title)
        processed_article = self.zone_extractor.divide_into_zones(processed_article)
        
        # Step 3: 生成片段
        fragments = self.fragment_generator.create_fragments(processed_article)
        
        # Step 4: Teacher推理
        zone_fragments = self.teacher_inference.process_article_fragments(fragments)
        
        # Step 9: 省略检测（如果启用且有事件集群）
        omission_result = None
        if self.omission_detector and event_cluster:
            omission_result = self.omission_detector.detect_omissions(
                article_id=article_id or "unknown",
                processed_article=processed_article,
                cluster=event_cluster
            )
        
        # Step 5-8: 框架分析（包含省略结果）
        result = self.framing_engine.analyze_article(zone_fragments, omission_result=omission_result)
        
        return result
    
    def analyze_batch(self, articles: List[Dict], output_path: Optional[str] = None) -> Dict:
        """批量分析文章
        
        Args:
            articles: 文章列表，每个元素包含 {'content': str, 'title': str, 'id': str}
            output_path: 输出文件路径（可选）
        
        Returns:
            包含所有结果的字典
        """
        
        logger.info(f"Starting batch analysis of {len(articles)} articles")
        
        # 验证输入数据
        articles = validate_input_data(articles)
        
        # 事件聚类（如果启用省略检测）
        event_clusters = {}
        article_to_cluster = {}  # 文章ID到集群的映射
        if self.omission_detector:
            logger.info("Performing event clustering for omission detection")
            event_clusters = self.omission_detector.cluster_articles_by_event(articles)
            logger.info(f"Created {len(event_clusters)} event clusters")
            
            # 构建文章ID到集群的映射，避免O(N²)查找
            for cluster_id, cluster_articles in event_clusters.items():
                for article in cluster_articles:
                    article_to_cluster[article.get('id')] = cluster_articles
        
        # 单轮推理：完整分析所有文章
        logger.info("Single pass: Complete analysis for all articles")
        results = []
        framing_scores = []
        
        for i, article in enumerate(tqdm(articles, desc="Analyzing articles")):
            try:
                # 获取该文章的事件集群
                event_cluster = None
                if self.omission_detector:
                    event_cluster = article_to_cluster.get(article.get('id'))
                
                result = self.analyze_article(
                    article['content'], 
                    article.get('title', ''),
                    article.get('id'),
                    event_cluster
                )
                
                # 收集framing分数用于阈值拟合
                framing_scores.append(result.framing_intensity)
                
                # 添加文章元信息（伪标签先占位）
                result_dict = self._format_result(result, article, i)
                results.append(result_dict)
                
            except Exception as e:
                logger.error(f"Error processing article {article.get('id', 'unknown')}: {e}")
                # 添加错误结果，不计入framing_scores
                error_result = self._create_error_result(article, i, str(e))
                results.append(error_result)
        
        # 拟合伪标签阈值（只用成功的样本）
        if framing_scores:
            self.framing_engine.fit_pseudo_label_thresholds(framing_scores)
            threshold_info = self.framing_engine.get_threshold_info()
            logger.info(f"Fitted thresholds: {threshold_info}")
            
            # 重新生成所有伪标签（仅内存操作，无推理）
            logger.info("Updating pseudo labels with fitted thresholds")
            for result in results:
                if not result.get('error'):
                    result['pseudo_label'] = self.framing_engine.label_generator.generate_pseudo_label(
                        result['framing_intensity']
                    )
        else:
            logger.warning("No valid framing scores for threshold fitting")
            threshold_info = {'thresholds': {'positive': 0.8, 'negative': 0.2}}
        
        # 相对框架分析（可选）
        if self.relative_analyzer:
            logger.info("Computing relative framing scores")
            results = self.relative_analyzer.compute_relative_scores(results)
        
        # 生成批量统计
        batch_stats = self._compute_batch_statistics(results, threshold_info)
        
        # 组装最终结果
        final_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_articles': len(articles),
                'successful_analyses': len([r for r in results if not r.get('error')]),
                'failed_analyses': len([r for r in results if r.get('error')]),
                'config': self._serialize_config()
            },
            'threshold_info': threshold_info,
            'batch_statistics': batch_stats,
            'results': results
        }
        
        # 保存结果
        if output_path:
            save_results(final_results, output_path, self.config.output)
            logger.info(f"Results saved to {output_path}")
        
        logger.info("Batch analysis completed successfully")
        return final_results
    
    def analyze_from_csv(self, csv_path: str, 
                        content_column: str = 'content',
                        title_column: str = 'title',
                        id_column: str = 'id',
                        output_path: Optional[str] = None) -> Dict:
        """从CSV文件分析文章"""
        
        logger.info(f"Loading articles from CSV: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} articles from CSV")
        
        # 转换为标准格式
        articles = []
        for i, row in df.iterrows():
            article = {
                'content': str(row.get(content_column, '')),
                'title': str(row.get(title_column, '')),
                'id': str(row.get(id_column, f'article_{i}'))
            }
            
            # 添加其他列作为元数据
            for col in df.columns:
                if col not in [content_column, title_column, id_column]:
                    article[col] = row[col]
            
            articles.append(article)
        
        # 批量分析
        return self.analyze_batch(articles, output_path)
    
    def _format_result(self, result: FramingResult, article: Dict, index: int) -> Dict:
        """格式化单个结果"""
        
        formatted = {
            'id': article.get('id', f'article_{index}'),
            'title': article.get('title', ''),
            'framing_intensity': result.framing_intensity,
            'pseudo_label': result.pseudo_label
        }
        
        # 添加组件分数
        if self.config.output.include_components:
            formatted['components'] = result.components
        
        # 添加证据片段
        if self.config.output.include_evidence:
            formatted['evidence'] = result.evidence
        
        # 添加统计信息
        if self.config.output.include_statistics:
            formatted['statistics'] = result.statistics
        
        # 添加原始分数
        if self.config.output.include_raw_scores and result.raw_scores:
            formatted['raw_scores'] = result.raw_scores
        
        # 添加省略检测结果
        if hasattr(result, 'omission_score') and result.omission_score is not None:
            formatted['omission_score'] = result.omission_score
            if self.config.output.include_evidence and hasattr(result, 'omission_evidence'):
                formatted['omission_evidence'] = result.omission_evidence
        
        # 添加文章元数据
        for key, value in article.items():
            if key not in ['content', 'title', 'id']:
                formatted[f'meta_{key}'] = value
        
        return formatted
    
    def _create_error_result(self, article: Dict, index: int, error_msg: str) -> Dict:
        """创建错误结果"""
        return {
            'id': article.get('id', f'article_{index}'),
            'title': article.get('title', ''),
            'error': True,
            'error_message': error_msg,
            'framing_intensity': 0.0,
            'pseudo_label': 'error'
        }
    
    def _compute_batch_statistics(self, results: List[Dict], threshold_info: Dict) -> Dict:
        """计算批量统计"""
        
        # 过滤掉错误结果
        valid_results = [r for r in results if not r.get('error')]
        
        if not valid_results:
            return {'error': 'No valid results to compute statistics'}
        
        # 提取分数
        framing_scores = [r['framing_intensity'] for r in valid_results]
        pseudo_labels = [r['pseudo_label'] for r in valid_results]
        
        # 基础统计
        stats = {
            'framing_intensity': {
                'mean': np.mean(framing_scores),
                'std': np.std(framing_scores),
                'min': np.min(framing_scores),
                'max': np.max(framing_scores),
                'median': np.median(framing_scores),
                'q25': np.percentile(framing_scores, 25),
                'q75': np.percentile(framing_scores, 75)
            },
            'pseudo_label_distribution': {
                label: pseudo_labels.count(label) for label in set(pseudo_labels)
            },
            'pseudo_label_percentages': {
                label: (pseudo_labels.count(label) / len(pseudo_labels)) * 100 
                for label in set(pseudo_labels)
            }
        }
        
        # 组件统计（如果包含）
        if self.config.output.include_components and 'components' in valid_results[0]:
            component_stats = {}
            for component in ['headline', 'lede', 'narration', 'quotes']:
                component_scores = [r['components'][component] for r in valid_results 
                                  if component in r['components']]
                if component_scores:
                    component_stats[component] = {
                        'mean': np.mean(component_scores),
                        'std': np.std(component_scores),
                        'min': np.min(component_scores),
                        'max': np.max(component_scores)
                    }
            stats['components'] = component_stats
        
        # 省略检测统计（如果包含）
        if 'omission_score' in valid_results[0]:
            omission_scores = [r['omission_score'] for r in valid_results 
                             if 'omission_score' in r and r['omission_score'] is not None]
            if omission_scores:
                stats['omission_detection'] = {
                    'mean': np.mean(omission_scores),
                    'std': np.std(omission_scores),
                    'min': np.min(omission_scores),
                    'max': np.max(omission_scores),
                    'median': np.median(omission_scores),
                    'articles_with_omissions': len([s for s in omission_scores if s > 0.5]),
                    'omission_rate': len([s for s in omission_scores if s > 0.5]) / len(omission_scores) * 100
                }
        
        return stats
    
    def _serialize_config(self) -> Dict:
        """序列化配置"""
        from dataclasses import asdict
        config_dict = {
            'processing': asdict(self.config.processing),
            'teacher': asdict(self.config.teacher),
            'scoring': asdict(self.config.scoring),
            'output': asdict(self.config.output),
            'relative_framing': asdict(self.config.relative_framing)
        }
        
        # 添加省略检测配置（如果存在）
        if hasattr(self.config, 'omission'):
            config_dict['omission'] = asdict(self.config.omission)
        
        return config_dict
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'teacher_model': self.config.teacher.model_name,
            'max_length': self.config.teacher.max_length,
            'fragment_mode': self.config.teacher.fragment_mode,
            'device': self.teacher_inference.teacher.device,
            'version': '1.0.0'
        }
    
    def validate_setup(self) -> Dict:
        """验证设置"""
        validation_results = {
            'model_loaded': False,
            'tokenizer_loaded': False,
            'device_available': False,
            'config_valid': False,
            'errors': []
        }
        
        try:
            # 检查模型
            if self.teacher_inference.teacher.model is not None:
                validation_results['model_loaded'] = True
            
            # 检查tokenizer
            if self.teacher_inference.teacher.tokenizer is not None:
                validation_results['tokenizer_loaded'] = True
            
            # 检查设备
            import torch
            device = self.teacher_inference.teacher.device
            if device == 'cuda' and torch.cuda.is_available():
                validation_results['device_available'] = True
            elif device == 'cpu':
                validation_results['device_available'] = True
            
            # 检查配置
            validation_results['config_valid'] = True
            
        except Exception as e:
            validation_results['errors'].append(str(e))
        
        validation_results['overall_status'] = all([
            validation_results['model_loaded'],
            validation_results['tokenizer_loaded'],
            validation_results['device_available'],
            validation_results['config_valid']
        ])
        
        return validation_results