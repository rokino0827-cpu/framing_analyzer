"""
SV2000 Framing Head - 基于Semetko & Valkenburg (2000)的框架预测组件
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SVFramingHead:
    """SV2000框架预测头 - 主要的框架分析组件"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        
        # 加载编码器和分类器
        self.encoder = None
        self.frame_classifier = None
        self._load_components()
        
        logger.info("SVFramingHead initialized successfully")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA不可用，自动回退CPU")
                device = "cpu"
        
        logger.info(f"SVFramingHead using device: {device}")
        return device
    
    def _load_components(self):
        """加载编码器和分类器组件"""
        try:
            # 加载编码器（优先使用本地路径）
            encoder_path = self.config.encoder_local_path or self.config.encoder_name
            logger.info(f"Loading encoder from: {encoder_path}")
            
            if "sentence-transformers" in self.config.encoder_name or "all-MiniLM" in self.config.encoder_name:
                # 使用sentence-transformers
                self.encoder = SentenceTransformer(encoder_path)
                self.encoder.to(self.device)
                hidden_size = self.encoder.get_sentence_embedding_dimension()
            else:
                # 使用标准transformers
                self.tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)
                self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
                self.encoder.to(self.device)
                hidden_size = self.encoder.config.hidden_size
            
            # 创建框架分类器
            self.frame_classifier = nn.Sequential(
                nn.Dropout(self.config.dropout_rate),
                nn.Linear(hidden_size, 5)  # 5个SV2000框架
            ).to(self.device)
            
            # 初始化分类器权重
            nn.init.xavier_uniform_(self.frame_classifier[1].weight)
            nn.init.zeros_(self.frame_classifier[1].bias)
            
            logger.info(f"Frame classifier initialized with hidden_size={hidden_size}")
            
        except Exception as e:
            logger.error(f"Failed to load SV2000 components: {e}")
            raise
    
    def predict_frames(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """预测SV2000框架分数
        
        Args:
            texts: 输入文本列表
            
        Returns:
            包含5个框架分数和平均分数的字典
        """
        if not texts:
            return self._empty_predictions(0)
        
        try:
            # 获取文本嵌入
            embeddings = self._encode_texts(texts)
            
            # 框架分类预测
            with torch.inference_mode():
                frame_logits = self.frame_classifier(embeddings)
                frame_probs = torch.sigmoid(frame_logits)  # 多标签分类
                
                # 转换为numpy数组
                frame_probs_np = frame_probs.cpu().numpy()
                
                # 计算平均框架分数
                frame_avg = np.mean(frame_probs_np, axis=1)
                
                return {
                    'sv_conflict_pred': frame_probs_np[:, 0],
                    'sv_human_pred': frame_probs_np[:, 1],
                    'sv_econ_pred': frame_probs_np[:, 2],
                    'sv_moral_pred': frame_probs_np[:, 3],
                    'sv_resp_pred': frame_probs_np[:, 4],
                    'sv_frame_avg_pred': frame_avg
                }
                
        except Exception as e:
            logger.error(f"Error in frame prediction: {e}")
            return self._empty_predictions(len(texts))
    
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """编码文本为嵌入向量"""
        if hasattr(self.encoder, 'encode'):
            # sentence-transformers接口
            embeddings = self.encoder.encode(
                texts, 
                convert_to_tensor=True,
                device=self.device,
                batch_size=self.config.batch_size
            )
        else:
            # 标准transformers接口
            embeddings_list = []
            
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.inference_mode():
                    outputs = self.encoder(**inputs)
                    # 使用[CLS] token或平均池化
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                    embeddings_list.append(batch_embeddings)
            
            embeddings = torch.cat(embeddings_list, dim=0)
        
        return embeddings
    
    def _empty_predictions(self, batch_size: int) -> Dict[str, np.ndarray]:
        """返回空预测结果"""
        return {
            'sv_conflict_pred': np.zeros(batch_size),
            'sv_human_pred': np.zeros(batch_size),
            'sv_econ_pred': np.zeros(batch_size),
            'sv_moral_pred': np.zeros(batch_size),
            'sv_resp_pred': np.zeros(batch_size),
            'sv_frame_avg_pred': np.zeros(batch_size)
        }
    
    def train_step(self, texts: List[str], targets: np.ndarray, optimizer, criterion) -> float:
        """单步训练
        
        Args:
            texts: 输入文本
            targets: 目标标签 (batch_size, 5)
            optimizer: 优化器
            criterion: 损失函数
            
        Returns:
            训练损失
        """
        self.frame_classifier.train()
        
        # 获取嵌入
        embeddings = self._encode_texts(texts)
        
        # 前向传播
        logits = self.frame_classifier(embeddings)
        
        # 计算损失
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        loss = criterion(logits, targets_tensor)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'frame_classifier_state_dict': self.frame_classifier.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"SV2000 model saved to {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.frame_classifier.load_state_dict(checkpoint['frame_classifier_state_dict'])
        logger.info(f"SV2000 model loaded from {path}")
    
    def eval_mode(self):
        """设置为评估模式"""
        if self.frame_classifier:
            self.frame_classifier.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        if self.frame_classifier:
            self.frame_classifier.train()