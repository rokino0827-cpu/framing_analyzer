"""
SV2000 Trainer - SV2000框架预测模型训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import os
from pathlib import Path

from .sv_framing_head import SVFramingHead
from .sv2000_data_loader import SV2000DataLoader
from .fusion_scorer import FusionWeightOptimizer

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """多标签Focal Loss，缓解易样本主导"""

    def __init__(self, gamma: float = 2.0, reduction: str = "none"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        modulating = (1 - p_t) ** self.gamma
        loss = modulating * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

class SVFramingTrainer:
    """SV2000框架预测训练器"""
    
    def __init__(self, config):
        # 仅保留SV框架相关配置，避免直接依赖AnalyzerConfig顶层属性
        self.config = config.sv_framing if hasattr(config, "sv_framing") else config
        self.analyzer_config = config
        self.device = self._setup_device()
        
        # 初始化模型
        self.model = SVFramingHead(self.config)
        self.enable_encoder_grad = getattr(self.model, "enable_encoder_grad", False)
        
        # 初始化优化器和损失函数
        self.optimizer = self._setup_optimizer()
        self.criterion = None  # 根据训练数据动态设置
        self.base_pos_weight: Optional[torch.Tensor] = None
        self.scheduler = self._setup_scheduler()
        self.frame_loss_weights = self._build_frame_loss_weights()
        self.item_loss_weights = self._build_item_loss_weights(self.frame_loss_weights)
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_correlation = -1.0
        self.best_val_metrics = None
        self.best_epoch = None
        self.patience_counter = 0
        
        logger.info("SVFramingTrainer initialized")
    
    def _setup_device(self) -> str:
        """设置计算设备"""
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("CUDA不可用，自动回退CPU")
                device = "cpu"
        
        logger.info(f"Training device: {device}")
        return device
    
    def _setup_optimizer(self):
        """设置优化器"""
        param_groups = []
        head_params = list(self.model.frame_classifier.parameters())
        head_lr = getattr(self.config, "head_learning_rate", self.config.learning_rate)
        # 防止头部学习率过低导致欠拟合
        min_head_lr = getattr(self.config, "min_head_learning_rate", 1e-3)
        head_lr = max(head_lr, min_head_lr)
        logger.info("Using head learning rate %.2e (min floor %.2e)", head_lr, min_head_lr)
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'weight_decay': 0.01
            })

        enable_encoder_grad = getattr(self, "enable_encoder_grad", None)
        if enable_encoder_grad is None:
            enable_encoder_grad = getattr(self.config, "fine_tune_encoder", False) or getattr(self.config, "trainable_encoder_layers", 0) > 0

        if enable_encoder_grad and hasattr(self.model, "encoder"):
            encoder_params = [p for p in self.model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({
                    'params': encoder_params,
                    'lr': getattr(self.config, "encoder_learning_rate", self.config.learning_rate),
                    'weight_decay': 0.01
                })
            else:
                logger.warning("fine_tune_encoder 已启用但未找到可训练的编码器参数")

        base_lr = head_lr
        return optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01)
    
    def _setup_loss_function(self, pos_weight: Optional[torch.Tensor] = None):
        """设置损失函数"""
        loss_type = getattr(self.config, "loss_type", "bce")
        self.base_pos_weight = pos_weight.to(self.device) if pos_weight is not None else None

        if loss_type == "focal":
            gamma = getattr(self.config, "focal_gamma", 2.0)
            logger.info("Using FocalLoss gamma=%.2f", gamma)

            def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                loss = FocalLoss(gamma=gamma, reduction="none")(logits, targets)
                aligned_pos_weight = self._align_pos_weight(self.base_pos_weight, logits.shape[1])
                if aligned_pos_weight is not None:
                    loss = loss * aligned_pos_weight
                return loss

            return loss_fn

        # 回退BCE，手动应用pos_weight以便按需对齐维度
        if self.base_pos_weight is not None:
            logger.info("Using pos_weight for BCEWithLogitsLoss: %s", self.base_pos_weight.tolist())

        def bce_loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            aligned_pos_weight = self._align_pos_weight(self.base_pos_weight, logits.shape[1])
            if aligned_pos_weight is not None:
                loss = loss * aligned_pos_weight
            return loss

        return bce_loss_fn

    def _align_pos_weight(self, base_weight: Optional[torch.Tensor], target_dim: int) -> Optional[torch.Tensor]:
        """对齐pos_weight到当前logits维度，避免20→5或5→20时形状不一致"""
        if base_weight is None:
            return None

        weight = base_weight.to(self.device)
        if weight.numel() == target_dim:
            return weight

        # item级权重压缩到框架级
        if weight.numel() == self.item_loss_weights.numel() and target_dim == self.frame_loss_weights.numel():
            agg = self.model.frame_agg_matrix.to(weight.device)
            return torch.matmul(agg, weight)

        # 框架级权重扩展到题项级
        if weight.numel() == self.frame_loss_weights.numel() and target_dim == self.item_loss_weights.numel():
            agg = self.model.frame_agg_matrix.to(weight.device)
            return torch.matmul(agg.t(), weight)

        # 回退：截断或填充
        if weight.numel() > target_dim:
            logger.warning("pos_weight长度(%d)大于目标维度(%d)，已截断匹配", weight.numel(), target_dim)
            return weight[:target_dim]

        logger.warning("pos_weight长度(%d)小于目标维度(%d)，已使用1填充", weight.numel(), target_dim)
        padded = torch.ones(target_dim, device=weight.device)
        padded[:weight.numel()] = weight
        return padded

    def _build_frame_loss_weights(self) -> torch.Tensor:
        """为每个框架构建损失权重，默认提升moral/resp关注度"""
        base_weights = {
            'conflict': 1.0,
            'human': 1.0,
            'econ': 1.0,
            'moral': 1.6,
            'resp': 1.4
        }
        custom_weights = getattr(self.config, "frame_loss_weights", None) or {}
        frame_order = ['conflict', 'human', 'econ', 'moral', 'resp']
        weights = [custom_weights.get(name, base_weights[name]) for name in frame_order]
        logger.info("Frame loss weights applied: %s", weights)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _build_item_loss_weights(self, frame_weights: torch.Tensor) -> torch.Tensor:
        """题项权重：继承对应框架权重"""
        frame_weights = frame_weights.detach().cpu().numpy()
        item_weights = []
        for frame_name in ['conflict', 'human', 'econ', 'moral', 'resp']:
            frame_idx = ['conflict', 'human', 'econ', 'moral', 'resp'].index(frame_name)
            weight = frame_weights[frame_idx]
            if frame_name == 'conflict':
                count = 4
            elif frame_name == 'human':
                count = 5
            elif frame_name == 'econ':
                count = 3
            elif frame_name == 'moral':
                count = 3
            else:
                count = 5
            item_weights.extend([weight] * count)
        return torch.tensor(item_weights, dtype=torch.float32, device=self.device)

    def _compute_pos_weight(self, targets: np.ndarray) -> Optional[torch.Tensor]:
        """根据训练集标签计算pos_weight，缓解正负样本失衡"""
        if targets is None or len(targets) == 0:
            return None

        positive_counts = np.clip(targets.sum(axis=0), 1e-6, None)
        negative_counts = np.clip((1 - targets).sum(axis=0), 1e-6, None)
        pos_weight = negative_counts / positive_counts
        cap = getattr(self.config, "pos_weight_cap", None)
        if cap is not None:
            pos_weight = np.clip(pos_weight, 0.0, cap)
        return torch.tensor(pos_weight, dtype=torch.float32)

    def _build_epoch_indices(self, targets: np.ndarray) -> np.ndarray:
        """构造平衡采样索引，提升少数正例覆盖"""
        if not getattr(self.config, "balance_batches", False):
            return np.random.permutation(len(targets))

        label_strength = targets
        if label_strength.size == 0:
            return np.random.permutation(len(targets))

        pos_counts = np.clip(label_strength.sum(axis=0), 1e-6, None)
        weights = label_strength @ (1.0 / pos_counts)
        weights = weights + 1e-3  # 保证非零
        weights = weights / weights.sum()
        return np.random.choice(len(targets), size=len(targets), replace=True, p=weights)
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
    
    def train(self, data_loader: SV2000DataLoader, num_epochs: int = 10, 
              patience: int = 5, save_best: bool = True) -> Dict:
        """训练模型
        
        Args:
            data_loader: SV2000数据加载器
            num_epochs: 训练轮数
            patience: 早停耐心值
            save_best: 是否保存最佳模型
            
        Returns:
            训练历史字典
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # 获取训练和验证数据
        train_texts, train_targets = data_loader.get_training_data(
            mode=self.config.training_mode, split="train"
        )
        val_texts, val_targets = data_loader.get_training_data(
            mode=self.config.training_mode, split="val"
        )
        
        if len(train_texts) == 0:
            raise ValueError("No training data available")
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")

        # 根据训练集动态设置loss（包含pos_weight）
        pos_weight = self._compute_pos_weight(train_targets)
        self.criterion = self._setup_loss_function(pos_weight)
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_correlation': [],
            'learning_rate': []
        }
        
        # 训练循环
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # 训练阶段
            train_loss = self._train_epoch(train_texts, train_targets)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_texts:
                val_metrics = self.evaluate(val_texts, val_targets)
                val_loss = val_metrics['loss']
                val_correlation = val_metrics['avg_correlation']

                history['val_loss'].append(val_loss)
                history['val_correlation'].append(val_correlation)
                
                # 学习率调度
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
                history['learning_rate'].append(current_lr)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                           f"Val Correlation: {val_correlation:.4f}, LR: {current_lr:.6f}")
                self._log_prediction_stats(val_metrics, split="val")
                self._update_dynamic_weights(val_metrics)
                
                # 早停检查
                if val_correlation > self.best_val_correlation:
                    self.best_val_correlation = val_correlation
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self.best_epoch = epoch + 1
                    self.patience_counter = 0
                    
                    if save_best:
                        self._save_best_model()
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")
        
        logger.info("Training completed")
        logger.info(f"Best validation correlation: {self.best_val_correlation:.4f}")
        
        history['best_epoch'] = self.best_epoch
        history['best_val_loss'] = self.best_val_loss if self.best_val_loss != float('inf') else None
        history['best_val_correlation'] = self.best_val_correlation if self.best_val_correlation != -1.0 else None
        return history
    
    def _train_epoch(self, texts: List[str], targets: np.ndarray) -> float:
        """训练一个epoch"""
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0
        
        # 创建批次
        batch_size = self.config.batch_size
        indices = self._build_epoch_indices(targets)
        for start in tqdm(range(0, len(indices), batch_size), desc="Training"):
            batch_idx = indices[start:start + batch_size]
            batch_texts = [texts[j] for j in batch_idx]
            batch_targets = targets[batch_idx]

            embeddings = self.model._encode_texts(
                batch_texts, requires_grad=self.enable_encoder_grad
            )
            logits = self.model.frame_classifier(embeddings)
            frame_logits = self.model.aggregate_item_logits(logits) if self.model.uses_item_level else logits

            targets_tensor = torch.FloatTensor(batch_targets).to(self.device)
            loss = self._calculate_loss(logits, targets_tensor, frame_logits)

            self.optimizer.zero_grad()
            loss.backward()
            max_grad_norm = getattr(self.config, "max_grad_norm", None)
            if max_grad_norm is not None and max_grad_norm > 0:
                params_to_clip = list(self.model.frame_classifier.parameters())
                if self.enable_encoder_grad and self.model.encoder is not None:
                    params_to_clip += list(self.model.encoder.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item() if hasattr(loss, "item") else float(loss)
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, texts: List[str], targets: np.ndarray, return_outputs: bool = False):
        """评估模型性能并可选返回预测/标签"""
        if self.criterion is None:
            self.criterion = self._setup_loss_function()

        self.model.eval_mode()

        # 预测（同时返回logits以避免重复前向）
        predictions, raw_logits, frame_logits = self.model.predict_frames(
            texts, return_logits=True
        )
        frame_targets_for_metrics = np.asarray(targets)
        if self.model.uses_item_level and targets.shape[1] != self.frame_loss_weights.numel():
            frame_targets_for_metrics = self.model.aggregate_item_targets(
                torch.FloatTensor(targets)
            ).cpu().numpy()
        
        # 计算损失
        with torch.no_grad():
            loss = None
            if raw_logits is not None:
                targets_tensor = torch.FloatTensor(targets).to(self.device)
                loss = self._calculate_loss(raw_logits, targets_tensor, frame_logits).item()
            else:
                logger.warning("predict_frames 未返回logits，loss设置为None")
        
        # 计算评估指标
        metrics = self._compute_metrics(predictions, frame_targets_for_metrics)
        metrics['loss'] = loss if loss is not None else float('nan')

        # 诊断：预测均值和方差，用于检测塌缩到常数预测
        metrics['pred_stats'] = self._prediction_stats(predictions)
        metrics['target_stats'] = self._target_stats(frame_targets_for_metrics)

        if return_outputs:
            return metrics, predictions, frame_targets_for_metrics
        return metrics

    def _apply_dimension_weights(self, raw_loss: torch.Tensor, dim_size: int) -> torch.Tensor:
        """按维度权重聚合loss"""
        if raw_loss.dim() == 0:
            return raw_loss
        if raw_loss.dim() == 1:
            return raw_loss.mean()

        weight_vec = None
        if dim_size == self.frame_loss_weights.numel():
            weight_vec = self.frame_loss_weights
        elif dim_size == self.item_loss_weights.numel():
            weight_vec = self.item_loss_weights

        if weight_vec is not None:
            raw_loss = raw_loss * weight_vec
        return raw_loss.mean()

    def _calculate_loss(
        self,
        logits: torch.Tensor,
        targets_tensor: torch.Tensor,
        frame_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """应用正负样本权重、框架/题项加权与一致性约束后的loss"""
        has_item_targets = targets_tensor.shape[1] == self.item_loss_weights.numel()
        use_frame_targets = targets_tensor.shape[1] == self.frame_loss_weights.numel()

        # 若标签为框架级但当前仍输出题项logits，统一使用聚合后的frame logits计算主损失
        logits_for_loss = logits
        if logits.shape[1] != targets_tensor.shape[1]:
            if frame_logits is not None and frame_logits.shape[1] == targets_tensor.shape[1]:
                logits_for_loss = frame_logits
            elif self.model.uses_item_level and use_frame_targets:
                logits_for_loss = self.model.aggregate_item_logits(logits)
            else:
                raise ValueError(
                    f"Logits dim {logits.shape[1]} mismatches targets {targets_tensor.shape[1]} "
                    "and no compatible frame_logits available"
                )

        main_loss = self._apply_dimension_weights(
            self.criterion(logits_for_loss, targets_tensor), logits_for_loss.shape[1]
        )

        if (
            frame_logits is not None
            and self.model.uses_item_level
            and has_item_targets
            and logits.shape[1] == self.item_loss_weights.numel()
        ):
            consistency_weight = max(0.0, getattr(self.config, "item_consistency_weight", 0.0))
            if consistency_weight > 0:
                frame_targets = self.model.aggregate_item_targets(targets_tensor)
                frame_loss = self._apply_dimension_weights(
                    self._frame_level_loss(frame_logits, frame_targets),
                    frame_logits.shape[1]
                )
                main_loss = main_loss + consistency_weight * frame_loss
        return main_loss

    def _frame_level_loss(self, frame_logits: torch.Tensor, frame_targets: torch.Tensor) -> torch.Tensor:
        """题项一致性使用的框架级loss，避免pos_weight维度不匹配"""
        loss_type = getattr(self.config, "loss_type", "bce")
        if loss_type == "focal":
            return FocalLoss(gamma=getattr(self.config, "focal_gamma", 2.0), reduction="none")(
                frame_logits, frame_targets
            )
        return F.binary_cross_entropy_with_logits(frame_logits, frame_targets, reduction="none")

    def _update_dynamic_weights(self, val_metrics: Dict):
        """基于验证MAE动态调整框架权重"""
        if not getattr(self.config, "dynamic_frame_reweight", False):
            return

        frame_names = ['conflict', 'human', 'econ', 'moral', 'resp']
        errors = []
        for frame in frame_names:
            key = f"{frame}_mae"
            if key not in val_metrics:
                return
            errors.append(max(val_metrics[key], 1e-6))

        errors = np.array(errors)
        normalized = errors / (errors.mean() + 1e-6)
        cap = getattr(self.config, "dynamic_reweight_cap", 2.0)
        normalized = np.clip(normalized, 0.5, cap)

        self.frame_loss_weights = torch.tensor(normalized, dtype=torch.float32, device=self.device)
        self.item_loss_weights = self._build_item_loss_weights(self.frame_loss_weights)
        logger.info("动态框架权重更新为: %s (cap=%.2f)", normalized.tolist(), cap)
    
    def _compute_metrics(self, predictions: Dict[str, np.ndarray], 
                        targets: np.ndarray) -> Dict:
        """计算评估指标"""
        frame_names = ['conflict', 'human', 'econ', 'moral', 'resp']
        metrics = {}
        
        correlations = []
        maes = []
        
        # 逐框架计算指标
        for i, frame in enumerate(frame_names):
            pred_key = f'sv_{frame}_pred'
            if pred_key in predictions and i < targets.shape[1]:
                pred_scores = predictions[pred_key]
                true_scores = targets[:, i]
                
                # 相关性
                if len(pred_scores) > 1:
                    pearson_r = pearsonr(pred_scores, true_scores)[0]
                    spearman_r = spearmanr(pred_scores, true_scores)[0]
                else:
                    pearson_r = spearman_r = 0.0
                
                # 误差
                mae = mean_absolute_error(true_scores, pred_scores)
                rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
                
                metrics[f'{frame}_pearson'] = pearson_r
                metrics[f'{frame}_spearman'] = spearman_r
                metrics[f'{frame}_mae'] = mae
                metrics[f'{frame}_rmse'] = rmse
                
                correlations.append(pearson_r)
                maes.append(mae)
        
        # 整体指标
        if correlations:
            metrics['avg_correlation'] = np.mean(correlations)
            metrics['avg_mae'] = np.mean(maes)
        else:
            metrics['avg_correlation'] = 0.0
            metrics['avg_mae'] = float('inf')
        
        # 框架平均分数指标
        if 'sv_frame_avg_pred' in predictions:
            pred_avg = predictions['sv_frame_avg_pred']
            true_avg = np.mean(targets, axis=1)
            
            if len(pred_avg) > 1:
                avg_pearson = pearsonr(pred_avg, true_avg)[0]
                avg_mae = mean_absolute_error(true_avg, pred_avg)
                
                metrics['frame_avg_pearson'] = avg_pearson
                metrics['frame_avg_mae'] = avg_mae
        
        return metrics

    @staticmethod
    def _prediction_stats(predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """统计预测均值/方差，辅助诊断模型塌缩"""
        stats = {}
        for key, values in predictions.items():
            if isinstance(values, np.ndarray) and values.size > 0:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        return stats

    @staticmethod
    def _target_stats(targets: np.ndarray) -> Dict[str, Dict[str, float]]:
        """统计标签均值/方差，便于对比"""
        frame_names = ['conflict', 'human', 'econ', 'moral', 'resp']
        stats = {}
        if targets is None or len(targets) == 0:
            return stats

        for idx, frame in enumerate(frame_names):
            if idx < targets.shape[1]:
                col = targets[:, idx]
                stats[frame] = {
                    'mean': float(np.mean(col)),
                    'std': float(np.std(col))
                }
        # 框架平均
        stats['frame_avg'] = {
            'mean': float(np.mean(targets)),
            'std': float(np.std(targets))
        }
        return stats

    def _log_prediction_stats(self, metrics: Dict, split: str = "val"):
        """日志输出预测/标签分布统计，方便快速诊断"""
        pred_stats = metrics.get('pred_stats', {})
        target_stats = metrics.get('target_stats', {})
        if not pred_stats:
            return

        logger.info(f"[{split}] Prediction stats (mean/std):")
        for key, vals in pred_stats.items():
            logger.info(f"  {key}: mean={vals.get('mean', 0):.4f}, std={vals.get('std', 0):.4f}")
        if target_stats:
            logger.info(f"[{split}] Target stats (mean/std):")
        for key, vals in target_stats.items():
            logger.info(f"  {key}: mean={vals.get('mean', 0):.4f}, std={vals.get('std', 0):.4f}")
    
    def _save_best_model(self):
        """保存最佳模型"""
        save_dir = Path(self.config.model_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / "best_sv2000_model.pt"
        self.model.save_model(str(model_path))
        
        logger.info(f"Best model saved to {model_path}")
    
    def save_model(self, path: str):
        """保存模型到指定路径"""
        self.model.save_model(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """从指定路径加载模型"""
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")

    def calibrate_with_temperature(self, texts: List[str], targets: np.ndarray) -> Optional[Dict]:
        """使用验证集进行温度标定，缓解整体预测偏高"""
        if not texts:
            logger.warning("No texts provided for calibration")
            return None

        self.model.eval_mode()
        logits = self._collect_logits(texts)
        targets_tensor = torch.FloatTensor(targets)
        if self.model.uses_item_level and targets_tensor.shape[1] != self.frame_loss_weights.numel():
            targets_tensor = self.model.aggregate_item_targets(targets_tensor)
        targets_tensor = targets_tensor.to(self.device)

        temperature = torch.ones(logits.shape[1], device=self.device, requires_grad=True)
        use_bias = getattr(self.config, "calibration_use_bias", False)
        bias = torch.zeros_like(temperature, requires_grad=True) if use_bias else None
        bce = nn.BCEWithLogitsLoss(reduction='mean')
        lr = getattr(self.config, "calibration_lr", 0.01)
        max_iter = getattr(self.config, "calibration_max_iter", 50)
        params = [temperature] + ([bias] if bias is not None else [])
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / temperature.clamp(min=1e-3)
            if bias is not None:
                scaled_logits = scaled_logits + bias
            loss = bce(scaled_logits, targets_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)
        optimized_temp = temperature.detach().clamp(min=1e-3)
        self.model.set_temperature(optimized_temp)
        optimized_bias = None
        if bias is not None:
            optimized_bias = bias.detach()
            self.model.set_logit_bias(optimized_bias)

        with torch.no_grad():
            scaled_logits = logits / optimized_temp
            if optimized_bias is not None:
                scaled_logits = scaled_logits + optimized_bias
            calibrated_loss = bce(scaled_logits, targets_tensor).item()

        logger.info(
            "Calibration done. Temperature: %s, bias: %s, calibrated_loss: %.4f",
            optimized_temp.cpu().tolist(),
            optimized_bias.cpu().tolist() if optimized_bias is not None else "None",
            calibrated_loss
        )
        return {
            "temperature": optimized_temp.cpu().tolist(),
            "bias": optimized_bias.cpu().tolist() if optimized_bias is not None else None,
            "calibrated_loss": calibrated_loss,
            "optimizer": "LBFGS",
            "max_iter": max_iter
        }

    def _collect_logits(self, texts: List[str]) -> torch.Tensor:
        """获取未标定的logits供标定或分析使用"""
        embeddings = self.model._encode_texts(texts)
        with torch.no_grad():
            logits = self.model.frame_classifier(embeddings)
            if self.model.uses_item_level:
                logits = self.model.aggregate_item_logits(logits)
        return logits

class SV2000TrainingPipeline:
    """SV2000完整训练管道"""
    
    def __init__(self, config, data_path: str):
        self.config = config
        self.data_path = data_path
        # 统一管理保存路径，避免直接访问顶层AnalyzerConfig缺失字段
        if hasattr(config, "sv_framing") and hasattr(config.sv_framing, "model_save_path"):
            self.model_save_path = Path(config.sv_framing.model_save_path)
        else:
            # 向后兼容直接传入SVFramingConfig的场景
            self.model_save_path = Path(getattr(config, "model_save_path", "./sv2000_models"))
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.data_loader = SV2000DataLoader(data_path, config)
        self._adjust_training_mode_by_item_coverage()
        self.trainer = SVFramingTrainer(config)
        self.weight_optimizer = FusionWeightOptimizer(config.fusion)
    
    def _adjust_training_mode_by_item_coverage(self) -> None:
        """当item级标注缺失时回退到frame-level，避免维度不匹配"""
        cfg = getattr(self.config, "sv_framing", self.config)
        requested_mode = getattr(cfg, "training_mode", "frame_level")
        if requested_mode != "item_level":
            return

        coverage = getattr(self.data_loader, "item_coverage_stats", {}) or {}
        found = coverage.get("found", 0)
        total = coverage.get("total", 20)

        if found == 0:
            logger.warning(
                "检测到 item_level 训练模式但题项标签覆盖率为0，自动回退至 frame_level "
                "以避免 logits(20) 与标签(5) 维度不匹配"
            )
            cfg.training_mode = "frame_level"
            return

        logger.info("item-level 标签覆盖率: %.1f%% (%d/%d)", (found / total) * 100, found, total)
        
    def run_full_training(
        self, 
        num_epochs: int = 10, 
        validation_split: float = 0.2,
        test_split: float = 0.0,
        early_stopping_patience: int = 5,
        save_best_model: bool = True
    ) -> Dict:
        """运行完整训练管道"""
        logger.info("Starting full SV2000 training pipeline")
        
        # 1. 数据验证
        validation_results = self.data_loader.validate_annotation_format()
        if not validation_results['is_valid']:
            logger.error(f"Data validation failed: {validation_results['issues']}")
            raise ValueError("Invalid training data format")
        
        # 2. 数据分割（显式使用传入的验证比例，避免硬编码）
        self.data_loader.create_train_val_split(val_ratio=validation_split, test_ratio=test_split)
        
        # 2. 数据统计
        data_stats = self.data_loader.get_data_statistics()
        logger.info(f"Dataset statistics: {data_stats}")
        
        # 3. 训练SV2000模型
        training_history = self.trainer.train(
            self.data_loader, 
            num_epochs=num_epochs,
            save_best=save_best_model,
            patience=early_stopping_patience
        )

        # 3.1 统一加载最佳权重作为后续标定与评估的基线
        best_model_path = self.model_save_path / "best_sv2000_model.pt"
        if save_best_model and best_model_path.exists():
            self.trainer.load_model(str(best_model_path))
        
        # 4. 评估最终模型
        val_texts, val_targets = self.data_loader.get_training_data(
            mode=self.trainer.config.training_mode, split="val"
        )
        test_texts, test_targets = self.data_loader.get_training_data(
            mode=self.trainer.config.training_mode, split="test"
        )

        # 4.1 温度标定（优先使用验证集）
        calibration_report = None
        if val_texts:
            calibration_report = self.trainer.calibrate_with_temperature(val_texts, val_targets)
            if save_best_model:
                calibrated_path = self.model_save_path / "best_sv2000_model_calibrated.pt"
                self.trainer.save_model(str(calibrated_path))
            else:
                calibrated_path = None
        else:
            calibrated_path = None

        final_metrics = {}
        validation_predictions = None
        validation_ground_truth = None
        if val_texts:
            val_metrics, val_predictions, val_targets_for_metrics = self.trainer.evaluate(
                val_texts, val_targets, return_outputs=True
            )
            final_metrics['val'] = val_metrics
            validation_predictions = val_predictions
            validation_ground_truth = self._targets_to_frame_dict(val_targets_for_metrics)
            logger.info(f"Final validation metrics: {final_metrics['val']}")

        test_predictions = None
        test_ground_truth = None
        if test_texts:
            test_metrics, test_predictions, test_targets_for_metrics = self.trainer.evaluate(
                test_texts, test_targets, return_outputs=True
            )
            final_metrics['test'] = test_metrics
            test_ground_truth = self._targets_to_frame_dict(test_targets_for_metrics)
            logger.info(f"Test metrics: {final_metrics['test']}")
        
        # 5. 优化融合权重（如果有足够数据）
        optimized_weights = {}
        if len(val_texts) >= 20:  # 需要足够的验证数据
            try:
                # 准备融合权重优化数据
                fusion_data = self._prepare_fusion_data(val_texts, val_targets)
                optimized_weights = self.weight_optimizer.optimize_weights(fusion_data)
                logger.info(f"Optimized fusion weights: {optimized_weights}")
            except Exception as e:
                logger.warning(f"Fusion weight optimization failed: {e}")
        
        # 6. 生成训练报告
        training_report = {
            'data_statistics': data_stats,
            'training_history': training_history,
            'final_metrics': final_metrics,
            'optimized_weights': optimized_weights,
            'validation_results': validation_results,
            'best_epoch': self.trainer.best_epoch,
            'best_val_correlation': self.trainer.best_val_correlation if self.trainer.best_val_correlation != -1.0 else None,
            'best_val_loss': self.trainer.best_val_loss if self.trainer.best_val_loss != float('inf') else None,
            'best_val_metrics': self.trainer.best_val_metrics,
            'calibration': calibration_report,
            'calibrated_model_path': str(calibrated_path) if calibrated_path else None,
            'split_strategy': getattr(self.data_loader, "grouping_info", {}),
            'validation_predictions': validation_predictions,
            'validation_ground_truth': validation_ground_truth,
            'test_predictions': test_predictions,
            'test_ground_truth': test_ground_truth
        }
        
        logger.info("Full training pipeline completed")
        return training_report

    @staticmethod
    def _targets_to_frame_dict(targets: np.ndarray) -> Dict[str, np.ndarray]:
        """将数组标签转为评估期望的字典格式"""
        if targets is None:
            return {}
        arr = np.asarray(targets)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return {}

        frame_names = ["conflict", "human", "econ", "moral", "resp"]
        target_dict: Dict[str, np.ndarray] = {}
        for idx, name in enumerate(frame_names):
            if idx < arr.shape[1]:
                target_dict[f"y_{name}"] = arr[:, idx]
        target_dict["frame_avg"] = arr.mean(axis=1)
        return target_dict
    
    def _prepare_fusion_data(self, texts: List[str], targets: np.ndarray) -> List[Dict]:
        """准备融合权重优化数据"""
        if self.trainer.model.uses_item_level and targets.shape[1] != 5:
            targets = self.trainer.model.aggregate_item_targets(
                torch.FloatTensor(targets)
            ).cpu().numpy()
        # 获取SV2000预测
        sv_predictions = self.trainer.model.predict_frames(texts)
        
        # 计算目标强度（框架平均分数）
        target_intensities = np.mean(targets, axis=1)
        
        # 准备融合数据
        fusion_data = []
        for i in range(len(texts)):
            sample = {
                'sv_frame_avg_pred': sv_predictions['sv_frame_avg_pred'][i],
                'bias_score': 0.5,  # 占位符，实际应该从BiasTeacher获取
                'omission_score': 0.0,  # 占位符
                'relative_score': 0.0,  # 占位符
                'quote_score': 0.0,  # 占位符
                'ground_truth_intensity': target_intensities[i]
            }
            fusion_data.append(sample)
        
        return fusion_data

def create_training_config(data_path: str, **kwargs) -> 'AnalyzerConfig':
    """创建训练配置"""
    from .config import AnalyzerConfig, SVFramingConfig, FusionConfig
    
    config = AnalyzerConfig()
    
    # SV2000配置
    config.sv_framing = SVFramingConfig(
        enabled=True,
        training_mode=kwargs.get('training_mode', 'frame_level'),
        learning_rate=kwargs.get('learning_rate', 2e-5),
        batch_size=kwargs.get('batch_size', 16),
        dropout_rate=kwargs.get('dropout_rate', 0.1),
        **kwargs
    )
    
    # 融合配置
    config.fusion = FusionConfig(
        use_ridge_optimization=True,
        ridge_alpha=kwargs.get('ridge_alpha', 1.0)
    )
    
    return config
