"""
框架评分模块 - Step 5, 6, 7, 8
从片段分数构造framing代理指标，合成文章级分数，生成弱标签和证据
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FramingResult:
    """框架分析结果"""
    # 主要输出
    framing_intensity: float  # F: 0~1
    pseudo_label: str  # "positive", "negative", "uncertain"
    
    # 组件分数
    components: Dict[str, float]  # headline, lede, narration, quotes得分
    
    # 证据片段
    evidence: List[Dict]  # Top-K证据片段
    
    # 统计信息
    statistics: Dict[str, float]  # 各种统计量
    
    # 省略感知结果（新增）
    omission_score: Optional[float] = None  # 省略分数
    omission_evidence: Optional[List[Dict]] = None  # 省略证据
    
    # 原始数据（可选）
    raw_scores: Optional[Dict] = None

class FramingScorer:
    """框架评分器 - Step 5 & 6"""
    
    def __init__(self, config):
        self.config = config.scoring
    
    def compute_zone_scores(self, zone_fragments: Dict[str, List[Dict]]) -> Dict[str, float]:
        """计算各结构区得分 - Step 5.1"""
        zone_scores = {}
        
        # Z1: Headline
        if "headline" in zone_fragments and zone_fragments["headline"]:
            headline_scores = [f['bias_score'] for f in zone_fragments["headline"]]
            zone_scores['headline'] = max(headline_scores)
        else:
            zone_scores['headline'] = 0.0
        
        # Z2: Lede
        if "lede" in zone_fragments and zone_fragments["lede"]:
            lede_scores = [f['bias_score'] for f in zone_fragments["lede"]]
            zone_scores['lede'] = self._topk_mean(lede_scores, self.config.lede_topk)
        else:
            zone_scores['lede'] = 0.0
        
        # Z3: Quotes
        if "quotes" in zone_fragments and zone_fragments["quotes"]:
            quote_scores = [f['bias_score'] for f in zone_fragments["quotes"]]
            zone_scores['quotes'] = self._topk_mean(quote_scores, self.config.quotes_topk)
        else:
            zone_scores['quotes'] = 0.0
        
        # Z4: Narration
        if "narration" in zone_fragments and zone_fragments["narration"]:
            narr_scores = [f['bias_score'] for f in zone_fragments["narration"]]
            zone_scores['narration'] = self._topk_mean(narr_scores, self.config.narration_topk)
        else:
            zone_scores['narration'] = 0.0
        
        return zone_scores
    
    def compute_structural_indicators(self, zone_fragments: Dict[str, List[Dict]]) -> Dict[str, float]:
        """计算结构指标 - Step 5.2 & 5.3"""
        indicators = {}
        
        # 收集所有片段分数
        all_scores = []
        for zone, fragments in zone_fragments.items():
            all_scores.extend([f['bias_score'] for f in fragments])
        
        if not all_scores:
            return {
                'quote_gap': 0.0,
                'peak': 0.0,
                'top3_mean': 0.0,
                'frac_high': 0.0
            }
        
        # 5.2 结构失衡 (quote_gap)
        narr_scores = [f['bias_score'] for f in zone_fragments.get("narration", [])]
        quote_scores = [f['bias_score'] for f in zone_fragments.get("quotes", [])]
        
        narr_mean = np.mean(narr_scores) if narr_scores else 0.0
        quote_mean = np.mean(quote_scores) if quote_scores else 0.0
        indicators['quote_gap'] = abs(narr_mean - quote_mean)
        
        # 5.3 稀疏强信号
        indicators['peak'] = max(all_scores)
        indicators['top3_mean'] = self._topk_mean(all_scores, self.config.global_topk)
        indicators['frac_high'] = sum(1 for score in all_scores 
                                     if score > self.config.high_bias_threshold) / len(all_scores)
        
        return indicators
    
    def compute_framing_intensity(self, zone_scores: Dict[str, float], 
                                 indicators: Dict[str, float],
                                 omission_score: Optional[float] = None,
                                 omission_config: Optional[object] = None) -> float:
        """合成文章级Framing-Intensity分数 - Step 6"""
        
        # 基础框架强度计算
        base_weights_sum = (self.config.lede_weight + self.config.headline_weight + 
                           self.config.narration_weight + self.config.quote_gap_weight + 
                           self.config.sparse_signal_weight)
        
        F = (
            self.config.lede_weight * zone_scores.get('lede', 0.0) +
            self.config.headline_weight * zone_scores.get('headline', 0.0) +
            self.config.narration_weight * zone_scores.get('narration', 0.0) +
            self.config.quote_gap_weight * indicators.get('quote_gap', 0.0) +
            self.config.sparse_signal_weight * indicators.get('frac_high', 0.0)
        )
        
        # 如果启用省略感知，使用线性融合
        if omission_score is not None:
            # 获取融合权重，默认0.2
            omission_weight = getattr(omission_config, 'fusion_weight', 0.2) if omission_config else 0.2
            # 线性融合: final_intensity = (1 - α) * framing_intensity + α * omission_intensity
            F = (1 - omission_weight) * F + omission_weight * omission_score
        
        # 确保在[0, 1]范围内
        return max(0.0, min(1.0, F))
    
    def _topk_mean(self, scores: List[float], k: int) -> float:
        """计算Top-K平均值"""
        if not scores:
            return 0.0
        
        k = min(k, len(scores))
        top_scores = sorted(scores, reverse=True)[:k]
        return np.mean(top_scores)

class PseudoLabelGenerator:
    """伪标签生成器 - Step 7"""
    
    def __init__(self, config):
        self.config = config.scoring
        self.thresholds = None
    
    def fit_thresholds(self, framing_scores: List[float]):
        """根据分数分布拟合阈值"""
        if not framing_scores:
            logger.warning("No framing scores provided for threshold fitting")
            self.thresholds = {'positive': 0.8, 'negative': 0.2}
            return
        
        scores_array = np.array(framing_scores)
        
        positive_threshold = np.percentile(scores_array, self.config.positive_threshold_percentile)
        negative_threshold = np.percentile(scores_array, self.config.negative_threshold_percentile)
        
        self.thresholds = {
            'positive': positive_threshold,
            'negative': negative_threshold
        }
        
        logger.info(f"Fitted thresholds - Positive: {positive_threshold:.3f}, "
                   f"Negative: {negative_threshold:.3f}")
    
    def generate_pseudo_label(self, framing_intensity: float) -> str:
        """生成伪标签"""
        if self.thresholds is None:
            logger.warning("Thresholds not fitted, using default values")
            self.thresholds = {'positive': 0.8, 'negative': 0.2}
        
        if framing_intensity >= self.thresholds['positive']:
            return "positive"
        elif framing_intensity <= self.thresholds['negative']:
            return "negative"
        else:
            return "uncertain"

class EvidenceExtractor:
    """证据片段提取器 - Step 8"""
    
    def __init__(self, config):
        self.config = config.scoring
    
    def extract_evidence(self, zone_fragments: Dict[str, List[Dict]]) -> List[Dict]:
        """提取Top-K证据片段，优先保留headline/lede的证据"""
        # 收集所有片段
        all_fragments = []
        priority_zones = ['headline', 'lede']
        priority_fragments = []
        other_fragments = []
        
        for zone, fragments in zone_fragments.items():
            for fragment in fragments:
                fragment_copy = fragment.copy()
                fragment_copy['zone'] = zone
                
                if zone in priority_zones:
                    priority_fragments.append(fragment_copy)
                else:
                    other_fragments.append(fragment_copy)
                all_fragments.append(fragment_copy)
        
        # 按bias_score排序
        priority_fragments.sort(key=lambda x: x['bias_score'], reverse=True)
        other_fragments.sort(key=lambda x: x['bias_score'], reverse=True)
        all_fragments.sort(key=lambda x: x['bias_score'], reverse=True)
        
        # 构建证据列表：优先保留至少1条headline/lede证据
        evidence_fragments = []
        seen_keys = set()
        
        def _frag_key(f: Dict) -> tuple:
            """生成片段的唯一标识"""
            return (
                f.get('zone'),
                f.get('fragment_type'),
                f.get('sentence_idx'),
                f.get('chunk_idx'),
                f.get('window_idx'),
                tuple(f.get('char_range', (None, None))),
                tuple(f.get('sentence_range', (None, None))),
                f.get('text', '')[:50],  # 兜底用文本前50字符
            )
        
        # 先添加最高分的priority证据（如果存在）
        if priority_fragments:
            key = _frag_key(priority_fragments[0])
            evidence_fragments.append(priority_fragments[0])
            seen_keys.add(key)
        
        # 剩余位置从全局最高分中补充
        for fragment in all_fragments:
            if len(evidence_fragments) >= self.config.evidence_count:
                break
            
            key = _frag_key(fragment)
            if key not in seen_keys:
                evidence_fragments.append(fragment)
                seen_keys.add(key)
        
        # 格式化证据
        evidence = []
        for i, fragment in enumerate(evidence_fragments):
            evidence_item = {
                'rank': i + 1,
                'text': fragment['text'],
                'bias_score': fragment['bias_score'],
                'zone': fragment['zone'],
                'confidence': fragment.get('confidence', 0.0),
                'position': self._get_fragment_position(fragment)
            }
            evidence.append(evidence_item)
        
        return evidence
    
    def _get_fragment_position(self, fragment: Dict) -> Dict:
        """获取片段位置信息"""
        position = {
            'fragment_type': fragment['fragment_type']
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

class FramingAnalysisEngine:
    """框架分析引擎 - 整合所有评分组件"""
    
    def __init__(self, config):
        self.config = config
        self.scorer = FramingScorer(config)
        self.label_generator = PseudoLabelGenerator(config)
        self.evidence_extractor = EvidenceExtractor(config)
    
    def analyze_article(self, zone_fragments: Dict[str, List[Dict]], 
                       omission_result=None) -> FramingResult:
        """分析单篇文章"""
        
        # Step 5: 计算结构区得分和指标
        zone_scores = self.scorer.compute_zone_scores(zone_fragments)
        structural_indicators = self.scorer.compute_structural_indicators(zone_fragments)
        
        # Step 6: 计算framing强度（可能包含省略分数）
        omission_score = omission_result.omission_score if omission_result else None
        
        # 获取省略配置用于融合权重
        omission_config = getattr(self.config, 'omission', None)
        
        framing_intensity = self.scorer.compute_framing_intensity(
            zone_scores, structural_indicators, omission_score, omission_config
        )
        
        # Step 7: 生成伪标签（需要先拟合阈值）
        pseudo_label = self.label_generator.generate_pseudo_label(framing_intensity)
        
        # Step 8: 提取证据片段
        evidence = self.evidence_extractor.extract_evidence(zone_fragments)
        
        # 组装结果
        components = {
            'headline': zone_scores.get('headline', 0.0),
            'lede': zone_scores.get('lede', 0.0),
            'narration': zone_scores.get('narration', 0.0),
            'quotes': zone_scores.get('quotes', 0.0)
        }
        
        statistics = {
            **structural_indicators,
            'total_fragments': sum(len(fragments) for fragments in zone_fragments.values()),
            'zones_with_content': len([z for z, f in zone_fragments.items() if f])
        }
        
        # 添加省略相关统计
        if omission_result:
            statistics.update({
                'omission_score': omission_result.omission_score,
                'key_topics_missing_count': len(omission_result.key_topics_missing),
                'key_topics_covered_count': len(omission_result.key_topics_covered),
                'omission_locations_count': len(omission_result.omission_locations)
            })
        
        raw_scores = {
            'zone_fragments': zone_fragments,
            'zone_scores': zone_scores,
            'structural_indicators': structural_indicators,
            'omission_result': omission_result
        } if self.config.output.include_raw_scores else None
        
        return FramingResult(
            framing_intensity=framing_intensity,
            pseudo_label=pseudo_label,
            components=components,
            evidence=evidence,
            statistics=statistics,
            omission_score=omission_score,
            omission_evidence=omission_result.evidence if omission_result else None,
            raw_scores=raw_scores
        )
    
    def fit_pseudo_label_thresholds(self, framing_scores: List[float]):
        """拟合伪标签阈值"""
        self.label_generator.fit_thresholds(framing_scores)
    
    def get_threshold_info(self) -> Dict:
        """获取阈值信息"""
        return {
            'thresholds': self.label_generator.thresholds,
            'positive_percentile': self.config.scoring.positive_threshold_percentile,
            'negative_percentile': self.config.scoring.negative_threshold_percentile
        }