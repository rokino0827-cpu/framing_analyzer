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
    
    def verify_bias_class_with_examples(self, test_texts: Optional[List[str]] = None) -> Dict:
        """
        使用对照句验证bias类别索引
        
        Args:
            test_texts: 测试文本列表，如果为None则使用默认测试句
            
        Returns:
            包含验证结果的字典
        """
        if test_texts is None:
            test_texts = [
                "This is a factual report about the event.",  # 更中性
                "Those people are disgusting and should be punished.",  # 更带偏见
            ]
        
        logger.info("Verifying bias class index with test examples...")
        
        # Tokenization
        inputs = self.tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            num_labels = getattr(self.model.config, "num_labels", None)
            id2label = getattr(self.model.config, "id2label", {})
            
            if num_labels == 1:
                # 单输出模型
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                result = {
                    'model_type': 'single_output',
                    'num_labels': num_labels,
                    'test_texts': test_texts,
                    'scores': probs.tolist(),
                    'recommendation': 'For single output models, higher scores indicate more bias'
                }
            else:
                # 多分类模型
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                result = {
                    'model_type': 'multi_class',
                    'num_labels': num_labels,
                    'id2label': id2label,
                    'test_texts': test_texts,
                    'probabilities': probs.tolist(),
                    'recommendation': {}
                }
                
                # 分析哪个类别在偏见句子中概率更高
                if len(probs) >= 2:
                    neutral_probs = probs[0]
                    biased_probs = probs[1]
                    
                    for idx in range(num_labels):
                        label = id2label.get(idx, f'LABEL_{idx}')
                        neutral_prob = neutral_probs[idx]
                        biased_prob = biased_probs[idx]
                        
                        if biased_prob > neutral_prob:
                            result['recommendation'][idx] = {
                                'label': label,
                                'likely_bias_class': True,
                                'neutral_prob': float(neutral_prob),
                                'biased_prob': float(biased_prob),
                                'difference': float(biased_prob - neutral_prob)
                            }
                        else:
                            result['recommendation'][idx] = {
                                'label': label,
                                'likely_bias_class': False,
                                'neutral_prob': float(neutral_prob),
                                'biased_prob': float(biased_prob),
                                'difference': float(biased_prob - neutral_prob)
                            }
        
        # 打印结果
        logger.info("=== Bias Class Verification Results ===")
        logger.info(f"Model type: {result['model_type']}")
        logger.info(f"Number of labels: {result['num_labels']}")
        
        if result['model_type'] == 'single_output':
            logger.info(f"Scores: {result['scores']}")
            logger.info("Higher scores indicate more bias")
        else:
            logger.info(f"Label mapping: {result['id2label']}")
            logger.info("Probabilities for each test text:")
            for i, text in enumerate(test_texts):
                logger.info(f"  Text {i+1}: '{text[:50]}...'")
                logger.info(f"    Probabilities: {result['probabilities'][i]}")
            
            logger.info("Recommendation for bias class:")
            for idx, rec in result['recommendation'].items():
                if rec['likely_bias_class']:
                    logger.info(f"  ✓ Index {idx} ({rec['label']}) - likely BIAS class")
                    logger.info(f"    Difference: {rec['difference']:.3f}")
                else:
                    logger.info(f"    Index {idx} ({rec['label']}) - likely neutral class")
        
        return result
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA未就绪，自动回退CPU")
                device = "cpu"
        
        logger.info(f"Using device: {device}")
        return device
    
    def _load_model(self):
        """加载bias_detector模型"""
        # 优先使用本地路径，避免离线环境拉取失败
        model_path = self.config.model_local_path or self.config.model_name
        logger.info(f"Loading bias_detector model from: {model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            try:
                self.model.to(self.device)
            except RuntimeError as cuda_err:
                if self.device.startswith("cuda"):
                    logger.warning(f"CUDA不可用，自动回退CPU: {cuda_err}")
                    self.device = "cpu"
                    self.model.to(self.device)
                else:
                    raise
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("如果需要联网下载模型，请清除local_files_only或更新config.teacher.model_local_path")
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
        # 创建副本并添加原始索引，避免污染输入
        fragments_with_idx = []
        for i, fragment in enumerate(fragments):
            fragment_copy = fragment.copy()
            fragment_copy['original_idx'] = i
            fragments_with_idx.append(fragment_copy)
        
        # 按估算token数排序
        return sorted(fragments_with_idx, key=lambda x: x.get('estimated_tokens', 0))
    
    def _restore_original_order(self, results: List[Dict], original_fragments: List[Dict]) -> List[Dict]:
        """恢复原始顺序"""
        # 创建索引映射
        idx_to_result = {result['original_idx']: result for result in results}
        
        # 按原始顺序返回
        return [idx_to_result[i] for i in range(len(original_fragments))]
    
    def _get_bias_class_index(self) -> int:
        """动态推断bias类的索引，优先使用配置"""
        if not hasattr(self, '_bias_class_idx'):
            # 1) 用户显式指定（最靠谱）
            if getattr(self.config, "bias_class_index", None) is not None:
                self._bias_class_idx = int(self.config.bias_class_index)
                logger.info(f"Using configured bias class index: {self._bias_class_idx}")
                return self._bias_class_idx
            
            if getattr(self.config, "bias_class_name", None):
                name = self.config.bias_class_name.lower()
                id2label = getattr(self.model.config, "id2label", {}) or {}
                for idx, label in id2label.items():
                    if label.lower() == name:
                        self._bias_class_idx = int(idx)
                        logger.info(f"Found configured bias class '{name}' at index {idx}")
                        return self._bias_class_idx
                logger.warning(f"Configured bias class name '{name}' not found in model labels")
            
            # 2) 再走原来的"关键词猜测"
            id2label = getattr(self.model.config, "id2label", {}) or {}
            
            # 先打印模型信息用于调试
            num_labels = getattr(self.model.config, "num_labels", None)
            logger.info(f"Model info: num_labels={num_labels}")
            logger.info(f"id2label={id2label}")
            logger.info(f"label2id={getattr(self.model.config, 'label2id', None)}")
            
            for idx, label in id2label.items():
                if any(k in label.lower() for k in ["bias", "biased"]):
                    self._bias_class_idx = int(idx)
                    logger.info(f"Found bias class at index {idx}: {label}")
                    return self._bias_class_idx
            
            for idx, label in id2label.items():
                if "positive" in label.lower():
                    self._bias_class_idx = int(idx)
                    logger.warning(f"Using positive class as bias class at index {idx}: {label}")
                    return self._bias_class_idx
            
            # 3) 最后兜底：二分类默认 1，否则报错更好（避免 silent bug）
            if num_labels == 2:
                self._bias_class_idx = 1
                logger.warning("Could not determine bias class index, using default index 1")
                return self._bias_class_idx
            
            raise ValueError(
                f"Cannot determine bias class index automatically (num_labels={num_labels}). "
                "Please set teacher.bias_class_index in config."
            )
        
        return self._bias_class_idx
    
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
        
        # 推理 - 使用inference_mode获得更好性能
        with torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # 处理不同的模型输出格式
            num_labels = getattr(self.model.config, "num_labels", None)
            
            if num_labels == 1:
                # 单输出模型（regression/单logit），使用sigmoid
                bias_scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                # 对于单输出模型，confidence就是预测的确定性
                all_confidences = torch.abs(logits.squeeze(-1) - 0.5).cpu().numpy() * 2  # 转换到[0,1]
            else:
                # 多分类模型，使用softmax
                probabilities = torch.softmax(logits, dim=-1)
                
                # 动态推断bias类的索引
                bias_class_idx = self._get_bias_class_index()
                bias_scores = probabilities[:, bias_class_idx].cpu().numpy()
                
                # 批量计算confidence并转移到CPU (P1优化: 避免逐个转移)
                all_confidences = torch.max(probabilities, dim=-1)[0].cpu().numpy()
        
        # 组装结果
        results = []
        for i, fragment in enumerate(batch):
            result = fragment.copy()
            result['bias_score'] = float(bias_scores[i])
            result['confidence'] = float(all_confidences[i])
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
