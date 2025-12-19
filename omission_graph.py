"""
省略感知图模块 - OmiGraph功能实现
基于OmiGraph论文的省略感知图推理方法
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import spacy
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """图节点数据结构"""
    node_id: str
    article_id: str
    fragment_text: str
    zone: str  # headline/lede/narration/quotes
    bias_score: float
    embedding: np.ndarray
    position: Dict  # 片段位置信息
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.node_id == other.node_id

@dataclass
class GraphEdge:
    """图边数据结构"""
    edge_id: str
    source: GraphNode
    target: GraphNode
    edge_type: str  # "intra_sequential", "intra_guidance", "inter_similar"
    weight: float

@dataclass
class OmissionResult:
    """省略检测结果"""
    article_id: str
    omission_score: float  # 0~1，越高表示省略越严重
    key_topics_missing: List[str]  # 缺失的关键主题
    key_topics_covered: List[str]  # 覆盖的关键主题
    omission_locations: List[str]  # 省略发生的位置（headline/lede/narration）
    evidence: List[Dict]  # 省略证据
    cluster_coverage: Dict[str, float]  # 簇内各主题的覆盖率

class OmissionGraph:
    """省略感知图数据结构"""
    
    def __init__(self):
        self.nodes: List[GraphNode] = []
        self.edges: List[GraphEdge] = []
        self.node_features: Dict[str, np.ndarray] = {}
        self.graph = nx.DiGraph()  # 使用NetworkX作为底层图结构
        self._node_map: Dict[str, GraphNode] = {}
    
    def add_node(self, fragment: Dict, article_id: str, zone: str) -> GraphNode:
        """添加图节点"""
        node_id = f"{article_id}_{zone}_{len(self.nodes)}"
        
        node = GraphNode(
            node_id=node_id,
            article_id=article_id,
            fragment_text=fragment['text'],
            zone=zone,
            bias_score=fragment.get('bias_score', 0.0),
            embedding=np.zeros(384),  # 将在后续设置
            position=self._extract_position_info(fragment)
        )
        
        self.nodes.append(node)
        self._node_map[node_id] = node
        self.graph.add_node(node_id, **{
            'article_id': article_id,
            'zone': zone,
            'bias_score': fragment.get('bias_score', 0.0),
            'text': fragment['text']
        })
        
        return node
    
    def add_edge(self, source: GraphNode, target: GraphNode, edge_type: str, weight: float) -> GraphEdge:
        """添加图边"""
        edge_id = f"{source.node_id}_{target.node_id}_{edge_type}"
        
        edge = GraphEdge(
            edge_id=edge_id,
            source=source,
            target=target,
            edge_type=edge_type,
            weight=weight
        )
        
        self.edges.append(edge)
        self.graph.add_edge(source.node_id, target.node_id, 
                           edge_type=edge_type, weight=weight)
        
        return edge
    
    def get_neighbors(self, node: GraphNode, edge_type: Optional[str] = None) -> List[GraphNode]:
        """获取节点的邻居"""
        neighbors = []
        
        for edge in self.edges:
            if edge.source == node:
                if edge_type is None or edge.edge_type == edge_type:
                    neighbors.append(edge.target)
        
        return neighbors
    
    def compute_node_centrality(self) -> Dict[GraphNode, float]:
        """计算节点中心性"""
        centrality_scores = nx.betweenness_centrality(self.graph)
        
        result = {}
        for node_id, score in centrality_scores.items():
            if node_id in self._node_map:
                result[self._node_map[node_id]] = score
        
        return result
    
    def _extract_position_info(self, fragment: Dict) -> Dict:
        """提取片段位置信息"""
        position = {
            'fragment_type': fragment.get('fragment_type', 'unknown')
        }
        
        if 'sentence_idx' in fragment:
            position['sentence_idx'] = fragment['sentence_idx']
        
        if 'chunk_idx' in fragment:
            position['chunk_idx'] = fragment['chunk_idx']
        
        if 'char_range' in fragment:
            position['char_range'] = fragment['char_range']
        
        if 'sentence_range' in fragment:
            position['sentence_range'] = fragment['sentence_range']
        
        return position

class OmissionAwareGraphBuilder:
    """省略感知图构建器"""
    
    def __init__(self, config):
        self.config = config
        
        # 初始化嵌入模型
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.embedding_model = None
        
        # 初始化NLP模型用于实体识别
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            self.nlp = None
    
    def build_graph(self, cluster_fragments: List[Dict]) -> OmissionGraph:
        """构建省略感知图"""
        graph = OmissionGraph()
        
        # Step 1: 添加所有节点
        article_nodes = defaultdict(list)
        
        for fragment_data in cluster_fragments:
            article_id = fragment_data['article_id']
            fragments = fragment_data['fragments']
            
            for zone, zone_fragments in fragments.items():
                for fragment in zone_fragments:
                    node = graph.add_node(fragment, article_id, zone)
                    article_nodes[article_id].append(node)
        
        # Step 2: 计算节点嵌入
        self._compute_node_embeddings(graph)
        
        # Step 3: 添加文内边
        for article_id, nodes in article_nodes.items():
            self._add_intra_article_edges(graph, nodes)
        
        # Step 4: 添加跨文边
        self._add_inter_article_edges(graph, list(article_nodes.values()))
        
        logger.info(f"Built omission-aware graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        return graph
    
    def _compute_node_embeddings(self, graph: OmissionGraph) -> None:
        """计算节点嵌入"""
        if self.embedding_model is None:
            logger.warning("No embedding model available, using zero embeddings")
            for node in graph.nodes:
                node.embedding = np.zeros(384)
            return
        
        # 批量计算嵌入
        texts = [node.fragment_text for node in graph.nodes]
        
        try:
            embeddings = self.embedding_model.encode(texts, batch_size=32, show_progress_bar=False)
            
            for i, node in enumerate(graph.nodes):
                node.embedding = embeddings[i]
                graph.node_features[node.node_id] = embeddings[i]
                
        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}")
            for node in graph.nodes:
                node.embedding = np.zeros(384)
    
    def _add_intra_article_edges(self, graph: OmissionGraph, article_nodes: List[GraphNode]) -> None:
        """添加文内边"""
        # 按zone分组
        zone_nodes = defaultdict(list)
        for node in article_nodes:
            zone_nodes[node.zone].append(node)
        
        # 1. 同zone顺序边
        for zone, nodes in zone_nodes.items():
            # 按位置排序
            sorted_nodes = sorted(nodes, key=lambda n: n.position.get('sentence_idx', 0))
            
            for i in range(len(sorted_nodes) - 1):
                graph.add_edge(sorted_nodes[i], sorted_nodes[i + 1], 
                             "intra_sequential", 1.0)
        
        # 2. lede → narration 引导边
        lede_nodes = zone_nodes.get('lede', [])
        narration_nodes = zone_nodes.get('narration', [])
        
        for lede_node in lede_nodes:
            for narr_node in narration_nodes[:3]:  # 只连接前3个narration节点
                # 计算语义相似度作为权重
                similarity = self._compute_similarity(lede_node, narr_node)
                if similarity > 0.3:  # 阈值过滤
                    graph.add_edge(lede_node, narr_node, "intra_guidance", similarity)
    
    def _add_inter_article_edges(self, graph: OmissionGraph, all_article_nodes: List[List[GraphNode]]) -> None:
        """添加跨文边"""
        # 收集所有节点按zone分组
        all_nodes_by_zone = defaultdict(list)
        
        for article_nodes in all_article_nodes:
            for node in article_nodes:
                all_nodes_by_zone[node.zone].append(node)
        
        # 在同zone内建立相似片段连边
        for zone, nodes in all_nodes_by_zone.items():
            if len(nodes) < 2:
                continue
            
            # 计算相似度矩阵
            similarities = self._compute_similarity_matrix(nodes)
            
            # 添加高相似度边
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    # 确保不是同一篇文章
                    if nodes[i].article_id != nodes[j].article_id:
                        similarity = similarities[i, j]
                        if similarity > 0.5:  # 相似度阈值
                            graph.add_edge(nodes[i], nodes[j], "inter_similar", similarity)
                            graph.add_edge(nodes[j], nodes[i], "inter_similar", similarity)
    
    def _compute_similarity(self, node1: GraphNode, node2: GraphNode) -> float:
        """计算两个节点的相似度"""
        if node1.embedding is None or node2.embedding is None:
            return 0.0
        
        try:
            similarity = cosine_similarity(
                node1.embedding.reshape(1, -1),
                node2.embedding.reshape(1, -1)
            )[0, 0]
            return max(0.0, float(similarity))
        except:
            return 0.0
    
    def _compute_similarity_matrix(self, nodes: List[GraphNode]) -> np.ndarray:
        """计算节点相似度矩阵"""
        if not nodes:
            return np.array([])
        
        embeddings = np.array([node.embedding for node in nodes])
        
        try:
            similarities = cosine_similarity(embeddings)
            return np.maximum(0.0, similarities)
        except:
            return np.zeros((len(nodes), len(nodes)))

class EntityExtractor:
    """实体提取器"""
    
    def __init__(self, nlp_model=None):
        self.nlp = nlp_model
    
    def extract_entities(self, text: str) -> List[str]:
        """提取文本中的实体"""
        if self.nlp is None:
            # 简单的关键词提取作为后备
            return self._simple_keyword_extraction(text)
        
        try:
            doc = self.nlp(text)
            entities = []
            
            # 提取命名实体
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT']:
                    entities.append(ent.text.lower())
            
            # 提取名词短语
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # 限制长度
                    entities.append(chunk.text.lower())
            
            return list(set(entities))
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return self._simple_keyword_extraction(text)
    
    def _simple_keyword_extraction(self, text: str) -> List[str]:
        """简单的关键词提取"""
        # 移除标点符号并转小写
        import re
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # 过滤停用词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [word for word in words if word not in stop_words]
        
        # 返回最频繁的关键词
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(20)]