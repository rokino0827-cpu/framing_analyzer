"""
Bias Teacher模块 - Step 4
使用原版bias_detector进行批量推理
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BiasTeacher:
    """Bias Teacher - 使用预训练的bias_detector模型"""
    
    def __init__(self, config):
        self.config = config.teacher
        self.device = self._setup_device()
        
        # 加载模型和tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """加载bias_detector模型"""
        logger.info(f"Loading bias_detector model: {self.config.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_fragments(self, fragments: List[Dict]) -> List[Dict]:
        """批量预测片段的偏见分数"""
        if not fragments:
            return []
        
        logger.info(f"Predicting bias scores for {len(fragments)} fragments")
        
        # 按token长度排序以优化batching
        sorted_fragments = self._sort_fragments_by_length(fragments)
        
        # 批量推理
        results = []
        for i in tqdm(range(0, len(sorted_fragments), self.config.batch_size), 
                     desc="Bias prediction"):
            batch = sorted_fragments[i:i + self.config.batch_size]
            batch_results = self._predict_batch(batch)
            results.extend(batch_results)
        
        # 恢复原始顺序
        results = self._restore_original_order(results, fragments)
        
        return results
    
    def _sort_fragments_by_length(self, fragments: List[Dict]) -> List[Dict]:
        """按文本长度排序片段以优化batching"""
        # 添加原始索引
        for i, fragment in enumerate(fragments):
            fragment['original_idx'] = i
        
        # 按估算token数排序
        return sorted(fragments, key=lambda x: x.get('estimated_tokens', 0))
    
    def _restore_original_order(self, results: List[Dict], original_fragments: List[Dict]) -> List[Dict]:
        """恢复原始顺序"""
        # 创建索引映射
        idx_to_result = {result['original_idx']: result for result in results}
        
        # 按原始顺序返回
        return [idx_to_result[i] for i in range(len(original_fragments))]
    
    def _predict_batch(self, batch: List[Dict]) -> List[Dict]:
        """预测一个batch的片段"""
        texts = [fragment['text'] for fragment in batch]
        
        # Tokenization
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 计算概率
            probabilities = torch.softmax(logits, dim=-1)
            bias_scores = probabilities[:, 1].cpu().numpy()  # 假设索引1是bias类
        
        # 组装结果
        results = []
        for i, fragment in enumerate(batch):
            result = fragment.copy()
            result['bias_score'] = float(bias_scores[i])
            result['confidence'] = float(np.max(probabilities[i].cpu().numpy()))
            results.append(result)
        
        return results

class TeacherInference:
    """Teacher推理管理器"""
    
    def __init__(self, config):
        self.config = config
        self.teacher = BiasTeacher(config)
    
    def process_article_fragments(self, fragments: List[Dict]) -> Dict[str, List[Dict]]:
        """处理文章片段并按zone分组"""
        # 批量预测
        scored_fragments = self.teacher.predict_fragments(fragments)
        
        # 按zone分组
        zone_fragments = {}
        for fragment in scored_fragments:
            zone = fragment['zone']
            if zone not in zone_fragments:
                zone_fragments[zone] = []
            zone_fragments[zone].append(fragment)
        
        # 处理滑窗片段的聚合
        zone_fragments = self._aggregate_sliding_windows(zone_fragments)
        
        return zone_fragments
    
    def _aggregate_sliding_windows(self, zone_fragments: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """聚合滑窗片段的分数"""
        for zone, fragments in zone_fragments.items():
            # 找出需要聚合的滑窗片段
            sliding_groups = {}
            non_sliding = []
            
            for fragment in fragments:
                if fragment['fragment_type'] == 'sliding_window':
                    sentence_idx = fragment['sentence_idx']
                    if sentence_idx not in sliding_groups:
                        sliding_groups[sentence_idx] = []
                    sliding_groups[sentence_idx].append(fragment)
                else:
                    non_sliding.append(fragment)
            
            # 聚合滑窗片段
            aggregated_fragments = non_sliding.copy()
            
            for sentence_idx, sliding_fragments in sliding_groups.items():
                if len(sliding_fragments) == 1:
                    # 只有一个片段，直接使用
                    aggregated_fragments.append(sliding_fragments[0])
                else:
                    # 多个片段，需要聚合
                    aggregated_fragment = self._aggregate_sliding_fragment_scores(
                        sliding_fragments
                    )
                    aggregated_fragments.append(aggregated_fragment)
            
            zone_fragments[zone] = aggregated_fragments
        
        return zone_fragments
    
    def _aggregate_sliding_fragment_scores(self, sliding_fragments: List[Dict]) -> Dict:
        """聚合滑窗片段的分数"""
        scores = [f['bias_score'] for f in sliding_fragments]
        
        if self.config.teacher.long_sentence_aggregation == "max":
            aggregated_score = max(scores)
        elif self.config.teacher.long_sentence_aggregation == "mean":
            aggregated_score = np.mean(scores)
        elif self.config.teacher.long_sentence_aggregation == "top2_mean":
            top2_scores = sorted(scores, reverse=True)[:2]
            aggregated_score = np.mean(top2_scores)
        else:
            aggregated_score = max(scores)  # 默认使用max
        
        # 创建聚合后的片段
        base_fragment = sliding_fragments[0].copy()
        base_fragment['bias_score'] = aggregated_score
        base_fragment['fragment_type'] = 'aggregated_sliding_window'
        base_fragment['num_windows'] = len(sliding_fragments)
        base_fragment['window_scores'] = scores
        
        return base_fragment

class BatchInferenceOptimizer:
    """批量推理优化器"""
    
    @staticmethod
    def optimize_batching(fragments: List[Dict], batch_size: int) -> List[List[Dict]]:
        """优化批量处理"""
        # 按长度分组
        length_groups = {}
        for fragment in fragments:
            length = fragment.get('estimated_tokens', 0)
            # 将长度归类到最近的50的倍数
            length_bucket = (length // 50) * 50
            if length_bucket not in length_groups:
                length_groups[length_bucket] = []
            length_groups[length_bucket].append(fragment)
        
        # 创建优化的批次
        batches = []
        for length_bucket in sorted(length_groups.keys()):
            group_fragments = length_groups[length_bucket]
            
            # 将同长度组的片段分批
            for i in range(0, len(group_fragments), batch_size):
                batch = group_fragments[i:i + batch_size]
                batches.append(batch)
        
        return batches
    
    @staticmethod
    def estimate_memory_usage(fragments: List[Dict], batch_size: int) -> Dict:
        """估算内存使用"""
        max_length = max(f.get('estimated_tokens', 0) for f in fragments)
        avg_length = np.mean([f.get('estimated_tokens', 0) for f in fragments])
        
        # 简单的内存估算（基于经验值）
        estimated_memory_mb = (batch_size * max_length * 4) / (1024 * 1024)  # 4 bytes per token
        
        return {
            'max_length': max_length,
            'avg_length': avg_length,
            'estimated_memory_mb': estimated_memory_mb,
            'recommended_batch_size': min(batch_size, max(1, int(2048 / max_length)))
        }