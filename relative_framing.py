"""
相对框架分析模块 - Step 9 (可选)
实现同事件相对framing分析
"""

import copy
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class RelativeFramingAnalyzer:
    """相对框架分析器"""
    
    def __init__(self, config):
        # 保留完整配置以便需要时触发基础分析
        self.analyzer_config = config
        self.config = config.relative_framing
        
        # TF-IDF向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            stop_words='english',
            lowercase=True
        )

    def analyze_batch(self, data: Union[List[Dict], Dict], output_path: Optional[str] = None) -> Dict:
        """批量执行相对框架分析
        
        兼容两类输入：
        1) 直接传入已包含framing_intensity的结果列表或带有'results'键的结果字典
        2) 传入原始文章列表，内部会先跑一次基础FramingAnalyzer获取分数
        """

        if not self.config.enabled:
            logger.info("Relative framing disabled; returning original data")
            return data if isinstance(data, dict) else {'results': data}

        # 标准化输入
        results_wrapper: Dict = {}
        base_results: List[Dict] = []

        if isinstance(data, dict) and 'results' in data:
            results_wrapper = data
            base_results = data.get('results', [])
        elif isinstance(data, list):
            base_results = data
            results_wrapper = {'results': base_results}
        else:
            raise ValueError("analyze_batch expects a list of articles or a dict with 'results'")

        # 如果没有framing_intensity，则先执行基础分析以获得分数
        needs_framing = not all(isinstance(item, dict) and 'framing_intensity' in item for item in base_results)
        if needs_framing:
            from .analyzer import FramingAnalyzer  # 延迟导入避免循环

            temp_config = copy.deepcopy(self.analyzer_config)
            # 关闭相对分析以避免重复计算
            temp_config.relative_framing.enabled = False

            core_analyzer = FramingAnalyzer(temp_config)
            analysis_output = core_analyzer.analyze_batch(base_results, output_path=output_path)
            results_wrapper = analysis_output
            base_results = analysis_output.get('results', [])

        # 计算相对分数
        scored_results = self.compute_relative_scores(base_results)
        results_wrapper['results'] = scored_results

        # 聚合簇信息与统计
        cluster_info = EventClusterAnalyzer.analyze_clusters(scored_results)
        extreme_cases = EventClusterAnalyzer.find_extreme_relative_cases(scored_results)

        # 将簇转换为列表，便于外部统计
        cluster_sizes: Dict[str, int] = {}
        for result in scored_results:
            relative_meta = result.get('relative_framing') or {}
            cluster_id = relative_meta.get('cluster_id')
            if cluster_id:
                cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1

        results_wrapper.update({
            'clusters': [{'cluster_id': cid, 'size': size} for cid, size in cluster_sizes.items()],
            'cluster_analysis': cluster_info,
            'extreme_cases': extreme_cases
        })

        return results_wrapper
    
    def compute_relative_scores(self, results: List[Dict]) -> List[Dict]:
        """计算相对框架分数"""
        
        if not self.config.enabled:
            return results
        
        logger.info("Computing relative framing scores")
        
        # 提取标题用于相似度计算
        titles = [result.get('title', '') for result in results]
        
        # 过滤空标题
        valid_indices = [i for i, title in enumerate(titles) if title.strip()]
        if len(valid_indices) < 2:
            logger.warning("Not enough valid titles for relative framing analysis")
            return results
        
        # 构建事件簇
        clusters = self._build_event_clusters(results, valid_indices)
        
        # 计算相对分数
        for cluster in clusters:
            if len(cluster) >= self.config.min_cluster_size:
                self._compute_cluster_relative_scores(results, cluster)
        
        return results
    
    def _build_event_clusters(self, results: List[Dict], valid_indices: List[int]) -> List[List[int]]:
        """构建事件簇"""
        
        # 提取有效标题
        valid_titles = [results[i]['title'] for i in valid_indices]
        
        # TF-IDF向量化
        try:
            tfidf_matrix = self.vectorizer.fit_transform(valid_titles)
        except Exception as e:
            logger.error(f"TF-IDF vectorization failed: {e}")
            return []
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # 基于相似度构建簇
        clusters = []
        used_indices = set()
        
        for i, idx in enumerate(valid_indices):
            if idx in used_indices:
                continue
            
            # 找到相似的文章
            similar_indices = []
            for j, other_idx in enumerate(valid_indices):
                if i != j and similarity_matrix[i, j] >= self.config.similarity_threshold:
                    # 检查时间窗口（如果有时间信息）
                    if self._within_time_window(results[idx], results[other_idx]):
                        similar_indices.append(other_idx)
            
            # 如果找到相似文章，创建簇
            if similar_indices:
                cluster = [idx] + similar_indices
                clusters.append(cluster)
                used_indices.update(cluster)
        
        logger.info(f"Built {len(clusters)} event clusters")
        return clusters
    
    def _within_time_window(self, article1: Dict, article2: Dict) -> bool:
        """检查两篇文章是否在时间窗口内"""
        
        # 尝试从元数据中提取时间信息
        time_fields = ['date', 'publish_date', 'timestamp', 'created_at']
        
        time1 = None
        time2 = None
        
        for field in time_fields:
            if f'meta_{field}' in article1:
                time1 = self._parse_time(article1[f'meta_{field}'])
                break
        
        for field in time_fields:
            if f'meta_{field}' in article2:
                time2 = self._parse_time(article2[f'meta_{field}'])
                break
        
        # 如果没有时间信息，假设在时间窗口内
        if time1 is None or time2 is None:
            return True
        
        # 检查时间差
        time_diff = abs((time1 - time2).days)
        return time_diff <= self.config.time_window_days
    
    def _parse_time(self, time_str) -> Optional[datetime]:
        """解析时间字符串"""
        
        if pd.isna(time_str):
            return None
        
        time_str = str(time_str)
        
        # 常见时间格式
        formats = [
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue
        
        # 如果都失败了，尝试pandas的解析
        try:
            return pd.to_datetime(time_str)
        except:
            return None
    
    def _compute_cluster_relative_scores(self, results: List[Dict], cluster: List[int]):
        """计算簇内相对分数"""
        
        # 提取簇内的framing分数
        cluster_scores = [results[i]['framing_intensity'] for i in cluster]
        
        # 计算中位数
        median_score = np.median(cluster_scores)
        
        # 计算相对分数
        for idx in cluster:
            absolute_score = results[idx]['framing_intensity']
            relative_score = absolute_score - median_score
            
            # 添加相对分数到结果中
            if 'relative_framing' not in results[idx]:
                results[idx]['relative_framing'] = {}
            
            results[idx]['relative_framing'].update({
                'relative_score': relative_score,
                'cluster_median': median_score,
                'cluster_size': len(cluster),
                'cluster_id': f"cluster_{min(cluster)}"  # 使用最小索引作为簇ID
            })
        
        logger.debug(f"Computed relative scores for cluster of size {len(cluster)}")

class EventClusterAnalyzer:
    """事件簇分析器"""
    
    @staticmethod
    def analyze_clusters(results: List[Dict]) -> Dict:
        """分析事件簇的统计信息"""
        
        # 收集有相对分数的结果
        relative_results = [r for r in results if 'relative_framing' in r]
        
        if not relative_results:
            return {'error': 'No relative framing results found'}
        
        # 按簇分组
        clusters = {}
        for result in relative_results:
            cluster_id = result['relative_framing']['cluster_id']
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(result)
        
        # 计算簇统计
        cluster_stats = {}
        for cluster_id, cluster_results in clusters.items():
            relative_scores = [r['relative_framing']['relative_score'] for r in cluster_results]
            absolute_scores = [r['framing_intensity'] for r in cluster_results]
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_results),
                'median_absolute': np.median(absolute_scores),
                'relative_score_range': (min(relative_scores), max(relative_scores)),
                'relative_score_std': np.std(relative_scores),
                'titles': [r['title'] for r in cluster_results]
            }
        
        # 整体统计
        all_relative_scores = [r['relative_framing']['relative_score'] for r in relative_results]
        
        overall_stats = {
            'total_clusters': len(clusters),
            'total_articles_in_clusters': len(relative_results),
            'relative_score_distribution': {
                'mean': np.mean(all_relative_scores),
                'std': np.std(all_relative_scores),
                'min': np.min(all_relative_scores),
                'max': np.max(all_relative_scores),
                'median': np.median(all_relative_scores)
            }
        }
        
        return {
            'overall_stats': overall_stats,
            'cluster_stats': cluster_stats
        }
    
    @staticmethod
    def find_extreme_relative_cases(results: List[Dict], top_k: int = 5) -> Dict:
        """找到相对分数极端的案例"""
        
        relative_results = [r for r in results if 'relative_framing' in r]
        
        if not relative_results:
            return {'error': 'No relative framing results found'}
        
        # 按相对分数排序
        sorted_by_relative = sorted(relative_results, 
                                  key=lambda x: x['relative_framing']['relative_score'])
        
        return {
            'most_negative_relative': sorted_by_relative[:top_k],
            'most_positive_relative': sorted_by_relative[-top_k:],
            'most_neutral_relative': sorted(relative_results, 
                                          key=lambda x: abs(x['relative_framing']['relative_score']))[:top_k]
        }
