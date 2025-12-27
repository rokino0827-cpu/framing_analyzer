"""
Fusion Scorer - 多组件融合评分器
将SV2000分数与现有的偏见检测、省略检测等辅助特征融合
"""

import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FusionResult:
    """融合评分结果"""
    final_intensity: float  # 最终融合分数
    component_scores: Dict[str, float]  # 各组件分数
    component_contributions: Dict[str, float]  # 各组件贡献度
    fusion_weights: Dict[str, float]  # 融合权重

class FusionScorer:
    """多组件融合评分器"""
    
    def __init__(self, config):
        self.config = config
        self.weights = self._initialize_weights()
        self.component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score', 
                               'relative_score', 'quote_score']
        
        logger.info("FusionScorer initialized with weights: %s", self.weights)
    
    def _initialize_weights(self) -> Dict[str, float]:
        """初始化融合权重"""
        weights = {
            'sv_frame_avg_pred': self.config.alpha,    # SV2000权重
            'bias_score': self.config.beta,            # 偏见检测权重
            'omission_score': self.config.gamma,       # 省略检测权重
            'relative_score': self.config.delta,       # 相对框架权重
            'quote_score': self.config.epsilon         # 引用分析权重
        }
        
        # 权重约束处理
        if self.config.enforce_positive_weights:
            weights = {k: max(0.0, v) for k, v in weights.items()}
        
        if self.config.normalize_weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def compute_fusion_score(self, features: Dict[str, float]) -> FusionResult:
        """计算融合分数
        
        Args:
            features: 包含各组件分数的字典
            
        Returns:
            融合评分结果
        """
        # 提取组件分数，缺失的设为0
        component_scores = {}
        component_contributions = {}
        
        for component in self.component_names:
            score = features.get(component, 0.0)
            weight = self.weights.get(component, 0.0)
            
            component_scores[component] = score
            component_contributions[component] = weight * score
        
        # 计算最终融合分数
        final_intensity = sum(component_contributions.values())
        
        # 确保在[0, 1]范围内
        final_intensity = np.clip(final_intensity, 0.0, 1.0)
        
        return FusionResult(
            final_intensity=final_intensity,
            component_scores=component_scores,
            component_contributions=component_contributions,
            fusion_weights=self.weights.copy()
        )
    
    def update_weights(self, new_weights: Dict[str, float]):
        """更新融合权重"""
        # 权重约束处理
        if self.config.enforce_positive_weights:
            new_weights = {k: max(0.0, v) for k, v in new_weights.items()}
        
        if self.config.normalize_weights:
            total_weight = sum(new_weights.values())
            if total_weight > 0:
                new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        self.weights.update(new_weights)
        logger.info("Fusion weights updated: %s", self.weights)
    
    def get_component_contributions(self, features: Dict[str, float]) -> Dict[str, float]:
        """获取各组件贡献度分析"""
        contributions = {}
        
        for component in self.component_names:
            score = features.get(component, 0.0)
            weight = self.weights.get(component, 0.0)
            contributions[component] = {
                'raw_score': score,
                'weight': weight,
                'contribution': weight * score,
                'relative_importance': weight / sum(self.weights.values()) if sum(self.weights.values()) > 0 else 0
            }
        
        return contributions

class FusionWeightOptimizer:
    """融合权重优化器 - 使用Ridge回归优化权重"""
    
    def __init__(self, config):
        self.config = config
        self.ridge_alpha = config.ridge_alpha
        self.cv_folds = config.cross_validation_folds
    
    def optimize_weights(self, training_data: List[Dict]) -> Dict[str, float]:
        """使用Ridge回归优化融合权重
        
        Args:
            training_data: 训练数据列表，每个元素包含组件分数和真实标签
            
        Returns:
            优化后的权重字典
        """
        if not training_data:
            logger.warning("No training data provided for weight optimization")
            return self._get_default_weights()
        
        try:
            # 准备特征矩阵和目标向量
            X, y = self._prepare_training_data(training_data)
            
            if X.shape[0] < 10:
                logger.warning("Insufficient training data for weight optimization")
                return self._get_default_weights()
            
            # Ridge回归优化
            ridge = Ridge(alpha=self.ridge_alpha, positive=self.config.enforce_positive_weights)
            ridge.fit(X, y)
            
            # 提取权重
            component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score', 
                             'relative_score', 'quote_score']
            
            optimized_weights = {}
            for i, component in enumerate(component_names):
                optimized_weights[component] = float(ridge.coef_[i])
            
            # 权重后处理
            optimized_weights = self._postprocess_weights(optimized_weights)
            
            # 交叉验证评估
            cv_scores = cross_val_score(ridge, X, y, cv=min(self.cv_folds, X.shape[0]))
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            logger.info(f"Weight optimization completed. CV score: {cv_mean:.3f} ± {cv_std:.3f}")
            logger.info(f"Optimized weights: {optimized_weights}")
            
            return optimized_weights
            
        except Exception as e:
            logger.error(f"Error in weight optimization: {e}")
            return self._get_default_weights()
    
    def _prepare_training_data(self, training_data: List[Dict]) -> tuple:
        """准备训练数据"""
        component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score', 
                          'relative_score', 'quote_score']
        
        X = []
        y = []
        
        for sample in training_data:
            # 提取特征
            features = []
            for component in component_names:
                features.append(sample.get(component, 0.0))
            
            # 提取目标值
            target = sample.get('ground_truth_intensity', sample.get('target_intensity', 0.0))
            
            X.append(features)
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def _postprocess_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """权重后处理"""
        # 确保非负（如果配置要求）
        if self.config.enforce_positive_weights:
            weights = {k: max(0.0, v) for k, v in weights.items()}
        
        # 归一化（如果配置要求）
        if self.config.normalize_weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认权重"""
        return {
            'sv_frame_avg_pred': 0.5,
            'bias_score': 0.2,
            'omission_score': 0.15,
            'relative_score': 0.1,
            'quote_score': 0.05
        }
    
    def cross_validate_weights(self, training_data: List[Dict], 
                              weight_candidates: List[Dict[str, float]]) -> Dict:
        """交叉验证不同权重设置的性能"""
        if not training_data or not weight_candidates:
            return {}
        
        X, y = self._prepare_training_data(training_data)
        results = {}
        
        for i, weights in enumerate(weight_candidates):
            # 使用当前权重计算预测
            predictions = []
            for sample in training_data:
                fusion_score = sum(
                    weights.get(comp, 0.0) * sample.get(comp, 0.0)
                    for comp in weights.keys()
                )
                predictions.append(np.clip(fusion_score, 0.0, 1.0))
            
            predictions = np.array(predictions)
            
            # 计算性能指标
            mse = np.mean((predictions - y) ** 2)
            mae = np.mean(np.abs(predictions - y))
            correlation = np.corrcoef(predictions, y)[0, 1] if len(y) > 1 else 0.0
            
            results[f'weights_{i}'] = {
                'weights': weights,
                'mse': mse,
                'mae': mae,
                'correlation': correlation,
                'score': correlation - 0.1 * mse  # 综合评分
            }
        
        return results

class ComponentAnalyzer:
    """组件分析器 - 分析各组件的贡献和重要性"""
    
    @staticmethod
    def analyze_component_correlations(results: List[Dict]) -> Dict:
        """分析组件间相关性"""
        component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score', 
                          'relative_score', 'quote_score']
        
        # 提取组件分数
        component_data = {comp: [] for comp in component_names}
        final_scores = []
        
        for result in results:
            final_scores.append(result.get('final_intensity', 0.0))
            for comp in component_names:
                component_data[comp].append(result.get(comp, 0.0))
        
        # 计算相关性
        correlations = {}
        for comp in component_names:
            if len(component_data[comp]) > 1:
                corr = np.corrcoef(component_data[comp], final_scores)[0, 1]
                correlations[comp] = {
                    'correlation_with_final': corr,
                    'mean_score': np.mean(component_data[comp]),
                    'std_score': np.std(component_data[comp]),
                    'contribution_strength': abs(corr) * np.std(component_data[comp])
                }
        
        return correlations
    
    @staticmethod
    def compare_fusion_vs_components(fusion_results: List[Dict], 
                                   ground_truth: List[float]) -> Dict:
        """比较融合性能与单组件性能"""
        if not fusion_results or not ground_truth:
            return {}
        
        component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score']
        
        # 提取分数
        fusion_scores = [r.get('final_intensity', 0.0) for r in fusion_results]
        
        performance = {}
        
        # 评估融合性能
        if len(fusion_scores) == len(ground_truth):
            fusion_corr = np.corrcoef(fusion_scores, ground_truth)[0, 1]
            fusion_mae = np.mean(np.abs(np.array(fusion_scores) - np.array(ground_truth)))
            
            performance['fusion'] = {
                'correlation': fusion_corr,
                'mae': fusion_mae
            }
        
        # 评估各组件单独性能
        for comp in component_names:
            comp_scores = [r.get(comp, 0.0) for r in fusion_results]
            if len(comp_scores) == len(ground_truth):
                comp_corr = np.corrcoef(comp_scores, ground_truth)[0, 1]
                comp_mae = np.mean(np.abs(np.array(comp_scores) - np.array(ground_truth)))
                
                performance[comp] = {
                    'correlation': comp_corr,
                    'mae': comp_mae
                }
        
        # 计算改进幅度
        if 'fusion' in performance:
            fusion_corr = performance['fusion']['correlation']
            improvements = {}
            
            for comp in component_names:
                if comp in performance:
                    comp_corr = performance[comp]['correlation']
                    improvements[f'fusion_vs_{comp}'] = fusion_corr - comp_corr
            
            performance['improvements'] = improvements
        
        return performance