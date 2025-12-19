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
        
        # 复用TextProcessor实例，避免重复创建
        self.text_processor = None
        
        # TF-IDF向量化器用于主题提取
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            min_df=1  # 改为1，避免小簇时空词表
        )
    
    def cluster_articles_by_event(self, articles: List[Dict]) -> Dict[str, List[Dict]]:
        """
        对文章进行事件聚类
        复用 TF-IDF 标题相似度聚类逻辑
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np
        
        if len(articles) <= 1:
            return {"cluster_0": articles}
        
        # 提取标题
        titles = [article.get('title', '') for article in articles]
        
        # 过滤空标题
        valid_indices = [i for i, title in enumerate(titles) if title.strip()]
        if len(valid_indices) <= 1:
            return {"cluster_0": articles}
        
        valid_titles = [titles[i] for i in valid_indices]
        valid_articles = [articles[i] for i in valid_indices]
        
        # TF-IDF 向量化
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(valid_titles)
            
            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 转换为距离矩阵
            distance_matrix = 1 - similarity_matrix
            
            # 层次聚类 - 使用距离阈值而非固定簇数
            distance_threshold = 1 - self.config.omission.similarity_threshold
            
            try:
                # 尝试使用distance_threshold + metric（sklearn >= 0.24）
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold,
                    metric='precomputed',
                    linkage='average',
                    compute_full_tree=True
                )
            except TypeError:
                try:
                    # 尝试使用affinity（sklearn旧版本）
                    clustering = AgglomerativeClustering(
                        n_clusters=None,
                        distance_threshold=distance_threshold,
                        affinity='precomputed',
                        linkage='average',
                        compute_full_tree=True
                    )
                except TypeError:
                    # 最终回退到固定簇数
                    n_clusters = min(max(2, len(valid_articles) // 3), len(valid_articles))
                    try:
                        clustering = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            metric='precomputed',
                            linkage='average'
                        )
                    except TypeError:
                        clustering = AgglomerativeClustering(
                            n_clusters=n_clusters,
                            affinity='precomputed',
                            linkage='average'
                        )
                    logger.warning(f"Using fixed n_clusters={n_clusters} due to sklearn version compatibility")
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # 组织聚类结果
            clusters = {}
            for i, label in enumerate(cluster_labels):
                cluster_key = f"cluster_{label}"
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append(valid_articles[i])
            
            # 添加无效标题的文章到最大的聚类中
            if len(valid_indices) < len(articles):
                largest_cluster = max(clusters.keys(), key=lambda k: len(clusters[k]))
                for i, article in enumerate(articles):
                    if i not in valid_indices:
                        clusters[largest_cluster].append(article)
            
            logger.info(f"Created {len(clusters)} event clusters from {len(articles)} articles")
            return clusters
            
        except Exception as e:
            logger.warning(f"Clustering failed: {e}, returning single cluster")
            return {"cluster_0": articles}
    
    def detect_omissions(self, article_id: str, processed_article, cluster: List[Dict]) -> OmissionResult:
        """检测文章中的省略模式"""
        
        try:
            # Step 1: 识别簇内关键主题
            key_topics = self.identify_key_topics(cluster)
            
            # Step 2: 分析当前文章的主题覆盖
            article_topics = self._extract_article_topics_from_processed(processed_article)
            
            # Step 3: 计算省略分数
            omission_score = self.compute_omission_score_from_processed(processed_article, key_topics)
            
            # Step 4: 识别缺失和覆盖的主题（基于文本覆盖而非列表成员关系）
            full_text = (processed_article.title or "") + " " + (processed_article.content or "")
            key_topics_missing = [topic for topic in key_topics if not self._topic_in_text(full_text, topic)]
            key_topics_covered = [topic for topic in key_topics if self._topic_in_text(full_text, topic)]
            
            # Step 5: 分析省略位置
            omission_locations = self._analyze_omission_locations_from_processed(processed_article, key_topics_missing)
            
            # Step 6: 生成省略证据
            evidence = self.extract_omission_evidence_from_processed(processed_article, key_topics_missing, cluster, article_id)
            
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
            
            # 计算每个主题在簇中的重要性（稀疏矩阵友好）
            topic_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
            
            # 选择top主题，保持稳定排序
            max_topics = self.config.omission.key_topics_count * 2  # 取更多候选主题
            top_indices = np.argsort(topic_scores)[-max_topics:]
            key_topics_with_scores = [(feature_names[i], topic_scores[i]) for i in top_indices if topic_scores[i] > 0.1]
            
            # 按分数降序排序，分数相同时按字母序排序（保证稳定性）
            key_topics_with_scores.sort(key=lambda x: (-x[1], x[0]))
            key_topics = [topic for topic, score in key_topics_with_scores]
            
            # 补充实体提取
            entity_topics = self._extract_cluster_entities(cluster)
            
            # 合并，保持TF-IDF主题的优先级
            all_topics = key_topics + [entity for entity in entity_topics if entity not in key_topics]
            
            logger.debug(f"Identified {len(all_topics)} key topics for cluster")
            return all_topics[:self.config.omission.key_topics_count]
            
        except Exception as e:
            logger.error(f"Failed to identify key topics: {e}")
            return []
    
    def compute_omission_score_from_processed(self, processed_article, key_topics: List[str]) -> float:
        """从ProcessedArticle计算省略分数"""
        if not key_topics:
            return 0.0
        
        try:
            # 提取文章的不同区域文本
            title = processed_article.title.lower() if processed_article.title else ""
            content = processed_article.content.lower() if processed_article.content else ""
            
            # 使用已经切分好的句子来估算lede
            lede_sentences = processed_article.sentences[:4] if processed_article.sentences else []
            lede = ' '.join(lede_sentences).lower()
            
            # 计算各区域的主题覆盖率
            headline_coverage = self._compute_topic_coverage(title, key_topics)
            lede_coverage = self._compute_topic_coverage(lede, key_topics)
            full_coverage = self._compute_topic_coverage(content, key_topics)
            
            # 加权计算省略分数
            # 重点关注headline和lede的省略
            headline_omission = 1.0 - headline_coverage
            lede_omission = 1.0 - lede_coverage
            full_omission = 1.0 - full_coverage
            
            # 加权融合（使用配置中的权重）
            omission_score = (
                self.config.omission.omission_weight_headline * headline_omission +
                self.config.omission.omission_weight_lede * lede_omission +
                self.config.omission.omission_weight_full * full_omission
            )
            
            return max(0.0, min(1.0, omission_score))
            
        except Exception as e:
            logger.error(f"Failed to compute omission score: {e}")
            return 0.0

    def compute_omission_score(self, article: Dict, key_topics: List[str]) -> float:
        """计算省略分数"""
        if not key_topics:
            return 0.0
        
        try:
            # 提取文章的不同区域文本
            title = article.get('title', '').lower()
            content = article.get('content', '').lower()
            
            # 使用TextProcessor进行句子切分
            if self.text_processor is None:
                from .text_processor import TextProcessor
                self.text_processor = TextProcessor(self.config)
            sentences, _ = self.text_processor.split_sentences(content)
            
            # 估算lede（前4句）
            lede_sentences = sentences[:4] if sentences else []
            lede = ' '.join(lede_sentences).lower()
            
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
    
    def extract_omission_evidence_from_processed(self, processed_article, omissions: List[str], cluster: List[Dict], article_id: str) -> List[Dict]:
        """从ProcessedArticle提取省略证据"""
        evidence = []
        
        if not omissions or len(cluster) < 2:
            return evidence
        
        try:
            # 找到其他文章中覆盖这些主题的片段
            for omitted_topic in omissions[:self.config.omission.max_evidence_count]:  # 使用配置限制数量
                supporting_fragments = self._find_supporting_fragments(omitted_topic, cluster, article_id)
                
                if supporting_fragments:
                    # 统计唯一文章数而非片段数
                    unique_article_ids = {f['article_id'] for f in supporting_fragments}
                    evidence_item = {
                        'omitted_topic': omitted_topic,
                        'evidence_type': 'omission',
                        'supporting_articles': len(unique_article_ids),
                        'examples': supporting_fragments[:self.config.omission.max_examples_per_evidence],  # 使用配置限制例子数量
                        'coverage_rate': len(unique_article_ids) / max(1, len(cluster) - 1)
                    }
                    evidence.append(evidence_item)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Failed to extract omission evidence: {e}")
            return []

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
    
    def _extract_article_topics_from_processed(self, processed_article) -> List[str]:
        """从ProcessedArticle提取文章主题"""
        try:
            # 合并标题和内容
            text_parts = []
            if processed_article.title:
                text_parts.append(processed_article.title)
            if processed_article.content:
                text_parts.append(processed_article.content)
            
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
    
    def _analyze_omission_locations_from_processed(self, processed_article, missing_topics: List[str]) -> List[str]:
        """从ProcessedArticle分析省略发生的位置"""
        locations = []
        
        if not missing_topics:
            return locations
        
        try:
            title = processed_article.title.lower() if processed_article.title else ""
            
            # 使用已经切分好的句子
            sentences = processed_article.sentences
            lede = ' '.join(sentences[:4]).lower() if sentences else ''
            narration = ' '.join(sentences[4:]).lower() if len(sentences) > 4 else ''
            
            # 检查每个区域是否缺失关键主题
            for topic in missing_topics[:self.config.omission.max_evidence_count]:  # 使用配置限制检查数量
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

    def _analyze_omission_locations(self, article: Dict, missing_topics: List[str]) -> List[str]:
        """分析省略发生的位置"""
        locations = []
        
        if not missing_topics:
            return locations
        
        try:
            title = article.get('title', '').lower()
            content = article.get('content', '')
            
            # 使用TextProcessor进行句子切分
            if self.text_processor is None:
                from .text_processor import TextProcessor
                self.text_processor = TextProcessor(self.config)
            sentences, _ = self.text_processor.split_sentences(content)
            
            # 估算不同区域
            lede = ' '.join(sentences[:4]).lower() if sentences else ''
            narration = ' '.join(sentences[4:]).lower() if len(sentences) > 4 else ''
            
            # 检查每个区域是否缺失关键主题
            for topic in missing_topics[:self.config.omission.max_evidence_count]:  # 使用配置限制检查数量
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
    
    def _topic_in_text(self, text: str, topic: str) -> bool:
        """判断主题是否在文本中（支持短语和单词边界匹配）"""
        import re
        if not text or not topic:
            return False
        
        text_lower = text.lower()
        topic_lower = topic.lower()
        
        if ' ' in topic_lower:
            # 多词主题，直接搜索
            return topic_lower in text_lower
        else:
            # 单词主题，使用词边界
            pattern = r'\b' + re.escape(topic_lower) + r'\b'
            return re.search(pattern, text_lower) is not None

    def _compute_topic_coverage(self, text: str, topics: List[str]) -> float:
        """计算文本对主题的覆盖率"""
        if not topics or not text:
            return 0.0
        
        import re
        covered_count = 0
        text_lower = text.lower()
        
        for topic in topics:
            topic_lower = topic.lower()
            
            # 使用词边界匹配，避免子字符串误匹配
            if ' ' in topic_lower:
                # 多词主题，直接搜索
                if topic_lower in text_lower:
                    covered_count += 1
            else:
                # 单词主题，使用词边界
                pattern = r'\b' + re.escape(topic_lower) + r'\b'
                if re.search(pattern, text_lower):
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
    
    def _find_supporting_fragments_exclude_title(self, topic: str, cluster: List[Dict], exclude_title: str) -> List[Dict]:
        """找到支持某个主题的片段（通过标题排除当前文章）"""
        supporting_fragments = []
        
        for article in cluster:
            # 通过标题排除当前文章（简单但有效的方法）
            if article.get('title', '') == exclude_title:
                continue
            
            # 检查标题
            title = article.get('title', '')
            if topic.lower() in title.lower():
                supporting_fragments.append({
                    'article_id': article.get('id', 'unknown'),
                    'text': title,
                    'location': 'headline',
                    'relevance': 1.0
                })
            
            # 检查内容前部分
            content = article.get('content', '')
            if content:
                # 使用TextProcessor进行句子切分
                if self.text_processor is None:
                    from .text_processor import TextProcessor
                    self.text_processor = TextProcessor(self.config)
                sentences, _ = self.text_processor.split_sentences(content)
                sentences = sentences[:6]  # 前6句
                
                for i, sentence in enumerate(sentences):
                    if topic.lower() in sentence.lower():
                        supporting_fragments.append({
                            'article_id': article.get('id', 'unknown'),
                            'text': sentence.strip(),
                            'location': 'lede' if i < 4 else 'narration',
                            'relevance': 0.8 if i < 4 else 0.6
                        })
                        break  # 每篇文章最多一个支持片段
        
        return supporting_fragments

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
                # 使用TextProcessor进行句子切分
                if self.text_processor is None:
                    from .text_processor import TextProcessor
                    self.text_processor = TextProcessor(self.config)
                sentences, _ = self.text_processor.split_sentences(content)
                sentences = sentences[:6]  # 前6句
                
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