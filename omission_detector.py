"""
省略检测器模块
实现基于省略感知图的省略模式检测
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

from .omission_graph import OmissionGraph, OmissionResult, GraphNode, OmissionAwareGraphBuilder, EntityExtractor

logger = logging.getLogger(__name__)

class OmissionDetector:
    """省略检测器"""
    
    def __init__(self, config):
        self.config = config
        self.graph_builder = OmissionAwareGraphBuilder(config)
        self.entity_extractor = EntityExtractor(self.graph_builder.nlp)
        
        # TF-IDF向量化器用于主题提取
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=2
        )
    
    def detect_omissions(self, article: Dict, cluster: List[Dict]) -> OmissionResult:
        """检测文章中的省略模式"""
        
        article_id = article['id']
        
        try:
            # Step 1: 识别簇内关键主题
            key_topics = self.identify_key_topics(cluster)
            
            # Step 2: 分析当前文章的主题覆盖
            article_topics = self._extract_article_topics(article)
            
            # Step 3: 计算省略分数
            omission_score = self.compute_omission_score(article, key_topics)
            
            # Step 4: 识别缺失和覆盖的主题
            key_topics_missing = [topic for topic in key_topics if topic not in article_topics]
            key_topics_covered = [topic for topic in key_topics if topic in article_topics]
            
            # Step 5: 分析省略位置
            omission_locations = self._analyze_omission_locations(article, key_topics_missing)
            
            # Step 6: 生成省略证据
            evidence = self.extract_omission_evidence(article, key_topics_missing, cluster)
            
            # Step 7: 计算簇内主题覆盖率
            cluster_coverage = self._compute_cluster_coverage(cluster, key_topics)
            
            return OmissionResult(
                article_id=article_id,
                omission_score=omission_score,
                key_topics_missing=key_topics_missing,
                key_topics_covered=key_topics_covered,
                omission_locations=omission_locations,
                evidence=evidence,
                cluster_coverage=cluster_coverage
            )
            
        except Exception as e:
            logger.error(f"Omission detection failed for article {article_id}: {e}")
            return self._create_empty_result(article_id)
    
    def identify_key_topics(self, cluster: List[Dict]) -> List[str]:
        """识别簇内关键主题"""
        if len(cluster) < 2:
            return []
        
        try:
            # 收集所有文章的文本
            all_texts = []
            for article in cluster:
                # 优先使用headline和lede
                text_parts = []
                if article.get('title'):
                    text_parts.append(article['title'])
                
                content = article.get('content', '')
                if content:
                    # 取前500字符作为lede近似
                    text_parts.append(content[:500])
                
                if text_parts:
                    all_texts.append(' '.join(text_parts))
            
            if len(all_texts) < 2:
                return []
            
            # 使用TF-IDF提取关键主题
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # 计算每个主题在簇中的重要性
            topic_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 选择top主题
            top_indices = np.argsort(topic_scores)[-20:]  # 取前20个主题
            key_topics = [feature_names[i] for i in top_indices if topic_scores[i] > 0.1]
            
            # 补充实体提取
            entity_topics = self._extract_cluster_entities(cluster)
            
            # 合并并去重
            all_topics = list(set(key_topics + entity_topics))
            
            logger.debug(f"Identified {len(all_topics)} key topics for cluster")
            return all_topics[:15]  # 限制主题数量
            
        except Exception as e:
            logger.error(f"Failed to identify key topics: {e}")
            return []
    
    def compute_omission_score(self, article: Dict, key_topics: List[str]) -> float:
        """计算省略分数"""
        if not key_topics:
            return 0.0
        
        try:
            # 提取文章的不同区域文本
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # 估算lede（前4句或前300字符）
            sentences = content.split('.')[:4]
            lede = ' '.join(sentences).lower()
            
            # 计算各区域的主题覆盖率
            headline_coverage = self._compute_topic_coverage(title, key_topics)
            lede_coverage = self._compute_topic_coverage(lede, key_topics)
            full_coverage = self._compute_topic_coverage(content, key_topics)
            
            # 加权计算省略分数
            # 重点关注headline和lede的省略
            headline_omission = 1.0 - headline_coverage
            lede_omission = 1.0 - lede_coverage
            full_omission = 1.0 - full_coverage
            
            # 加权融合（headline和lede权重更高）
            omission_score = (
                0.4 * headline_omission +
                0.4 * lede_omission +
                0.2 * full_omission
            )
            
            return max(0.0, min(1.0, omission_score))
            
        except Exception as e:
            logger.error(f"Failed to compute omission score: {e}")
            return 0.0
    
    def extract_omission_evidence(self, article: Dict, omissions: List[str], cluster: List[Dict]) -> List[Dict]:
        """提取省略证据"""
        evidence = []
        
        if not omissions or len(cluster) < 2:
            return evidence
        
        try:
            # 找到其他文章中覆盖这些主题的片段
            for omitted_topic in omissions[:5]:  # 限制数量
                supporting_fragments = self._find_supporting_fragments(omitted_topic, cluster, article['id'])
                
                if supporting_fragments:
                    evidence_item = {
                        'omitted_topic': omitted_topic,
                        'evidence_type': 'omission',
                        'supporting_articles': len(supporting_fragments),
                        'examples': supporting_fragments[:3],  # 最多3个例子
                        'coverage_rate': len(supporting_fragments) / max(1, len(cluster) - 1)
                    }
                    evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to extract omission evidence: {e}")
            return []
    
    def _extract_article_topics(self, article: Dict) -> List[str]:
        """提取文章主题"""
        try:
            # 合并标题和内容
            text_parts = []
            if article.get('title'):
                text_parts.append(article['title'])
            if article.get('content'):
                text_parts.append(article['content'])
            
            full_text = ' '.join(text_parts)
            
            # 使用实体提取器
            entities = self.entity_extractor.extract_entities(full_text)
            
            # 简单的关键词提取
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())
            word_counts = Counter(words)
            
            # 过滤停用词
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            keywords = [word for word, count in word_counts.most_common(20) if word not in stop_words]
            
            return list(set(entities + keywords))
            
        except Exception as e:
            logger.error(f"Failed to extract article topics: {e}")
            return []
    
    def _analyze_omission_locations(self, article: Dict, missing_topics: List[str]) -> List[str]:
        """分析省略发生的位置"""
        locations = []
        
        if not missing_topics:
            return locations
        
        try:
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # 估算不同区域
            sentences = content.split('.')
            lede = ' '.join(sentences[:4]).lower() if sentences else ''
            narration = ' '.join(sentences[4:]).lower() if len(sentences) > 4 else ''
            
            # 检查每个区域是否缺失关键主题
            for topic in missing_topics[:5]:  # 限制检查数量
                topic_lower = topic.lower()
                
                if topic_lower not in title:
                    locations.append('headline')
                
                if topic_lower not in lede:
                    locations.append('lede')
                
                if topic_lower not in narration:
                    locations.append('narration')
            
            return list(set(locations))
            
        except Exception as e:
            logger.error(f"Failed to analyze omission locations: {e}")
            return []
    
    def _compute_topic_coverage(self, text: str, topics: List[str]) -> float:
        """计算文本对主题的覆盖率"""
        if not topics or not text:
            return 0.0
        
        covered_count = 0
        for topic in topics:
            if topic.lower() in text:
                covered_count += 1
        
        return covered_count / len(topics)
    
    def _extract_cluster_entities(self, cluster: List[Dict]) -> List[str]:
        """提取簇内实体"""
        all_entities = []
        
        for article in cluster:
            text_parts = []
            if article.get('title'):
                text_parts.append(article['title'])
            if article.get('content'):
                text_parts.append(article['content'][:500])  # 限制长度
            
            full_text = ' '.join(text_parts)
            entities = self.entity_extractor.extract_entities(full_text)
            all_entities.extend(entities)
        
        # 统计实体频率，选择出现在多篇文章中的实体
        entity_counts = Counter(all_entities)
        min_frequency = max(2, len(cluster) // 2)  # 至少出现在一半文章中
        
        frequent_entities = [entity for entity, count in entity_counts.items() 
                           if count >= min_frequency]
        
        return frequent_entities[:10]  # 限制数量
    
    def _find_supporting_fragments(self, topic: str, cluster: List[Dict], exclude_article_id: str) -> List[Dict]:
        """找到支持某个主题的片段"""
        supporting_fragments = []
        
        for article in cluster:
            if article['id'] == exclude_article_id:
                continue
            
            # 检查标题
            title = article.get('title', '')
            if topic.lower() in title.lower():
                supporting_fragments.append({
                    'article_id': article['id'],
                    'text': title,
                    'location': 'headline',
                    'relevance': 1.0
                })
            
            # 检查内容前部分
            content = article.get('content', '')
            if content:
                sentences = content.split('.')[:6]  # 前6句
                for i, sentence in enumerate(sentences):
                    if topic.lower() in sentence.lower():
                        supporting_fragments.append({
                            'article_id': article['id'],
                            'text': sentence.strip(),
                            'location': 'lede' if i < 4 else 'narration',
                            'relevance': 0.8 if i < 4 else 0.6
                        })
                        break  # 每篇文章最多一个支持片段
        
        return supporting_fragments
    
    def _compute_cluster_coverage(self, cluster: List[Dict], key_topics: List[str]) -> Dict[str, float]:
        """计算簇内各主题的覆盖率"""
        coverage = {}
        
        if not key_topics:
            return coverage
        
        for topic in key_topics:
            covered_articles = 0
            
            for article in cluster:
                text_parts = []
                if article.get('title'):
                    text_parts.append(article['title'])
                if article.get('content'):
                    text_parts.append(article['content'])
                
                full_text = ' '.join(text_parts).lower()
                
                if topic.lower() in full_text:
                    covered_articles += 1
            
            coverage[topic] = covered_articles / len(cluster) if cluster else 0.0
        
        return coverage
    
    def _create_empty_result(self, article_id: str) -> OmissionResult:
        """创建空的省略检测结果"""
        return OmissionResult(
            article_id=article_id,
            omission_score=0.0,
            key_topics_missing=[],
            key_topics_covered=[],
            omission_locations=[],
            evidence=[],
            cluster_coverage={}
        )