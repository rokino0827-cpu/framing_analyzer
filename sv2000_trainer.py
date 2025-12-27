"""
SV2000 Trainer - SV2000框架预测模型训练器
"""

import torch
import torch.nn as nn
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

class SVFramingTrainer:
    """SV2000框架预测训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = self._setup_device()
        
        # 初始化模型
        self.model = SVFramingHead(config)
        
        # 初始化优化器和损失函数
        self.optimizer = self._setup_optimizer()
        self.criterion = self._setup_loss_function()
        self.scheduler = self._setup_scheduler()
        
        # 训练状态
        self.best_val_loss = float('inf')
        self.best_val_correlation = -1.0
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
        return optim.AdamW(
            self.model.frame_classifier.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
    
    def _setup_loss_function(self):
        """设置损失函数"""
        # 使用BCEWithLogitsLoss用于多标签分类
        return nn.BCEWithLogitsLoss()
    
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
                
                # 早停检查
                if val_correlation > self.best_val_correlation:
                    self.best_val_correlation = val_correlation
                    self.best_val_loss = val_loss
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
        
        return history
    
    def _train_epoch(self, texts: List[str], targets: np.ndarray) -> float:
        """训练一个epoch"""
        self.model.train_mode()
        total_loss = 0.0
        num_batches = 0
        
        # 创建批次
        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Training"):
            batch_texts = texts[i:i + batch_size]
            batch_targets = targets[i:i + batch_size]
            
            # 训练步骤
            loss = self.model.train_step(
                batch_texts, batch_targets, 
                self.optimizer, self.criterion
            )
            
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, texts: List[str], targets: np.ndarray) -> Dict:
        """评估模型性能"""
        self.model.eval_mode()
        
        # 预测
        predictions = self.model.predict_frames(texts)
        
        # 计算损失
        with torch.no_grad():
            embeddings = self.model._encode_texts(texts)
            logits = self.model.frame_classifier(embeddings)
            targets_tensor = torch.FloatTensor(targets).to(self.device)
            loss = self.criterion(logits, targets_tensor).item()
        
        # 计算评估指标
        metrics = self._compute_metrics(predictions, targets)
        metrics['loss'] = loss
        
        return metrics
    
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

class SV2000TrainingPipeline:
    """SV2000完整训练管道"""
    
    def __init__(self, config, data_path: str):
        self.config = config
        self.data_path = data_path
        
        # 初始化组件
        self.data_loader = SV2000DataLoader(data_path, config)
        self.trainer = SVFramingTrainer(config)
        self.weight_optimizer = FusionWeightOptimizer(config.fusion)
        
    def run_full_training(self, num_epochs: int = 10) -> Dict:
        """运行完整训练管道"""
        logger.info("Starting full SV2000 training pipeline")
        
        # 1. 数据验证
        validation_results = self.data_loader.validate_annotation_format()
        if not validation_results['is_valid']:
            logger.error(f"Data validation failed: {validation_results['issues']}")
            raise ValueError("Invalid training data format")
        
        # 2. 数据统计
        data_stats = self.data_loader.get_data_statistics()
        logger.info(f"Dataset statistics: {data_stats}")
        
        # 3. 训练SV2000模型
        training_history = self.trainer.train(
            self.data_loader, 
            num_epochs=num_epochs,
            save_best=True
        )
        
        # 4. 评估最终模型
        val_texts, val_targets = self.data_loader.get_training_data(split="val")
        if val_texts:
            final_metrics = self.trainer.evaluate(val_texts, val_targets)
            logger.info(f"Final validation metrics: {final_metrics}")
        else:
            final_metrics = {}
        
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
            'validation_results': validation_results
        }
        
        logger.info("Full training pipeline completed")
        return training_report
    
    def _prepare_fusion_data(self, texts: List[str], targets: np.ndarray) -> List[Dict]:
        """准备融合权重优化数据"""
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