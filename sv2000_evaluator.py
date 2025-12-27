"""
SV2000 Evaluator - SV2000框架对齐评估器
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, roc_curve, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class SV2000Evaluator:
    """SV2000框架对齐评估器"""
    
    def __init__(self, config=None):
        self.config = config
        self.frame_names = ['conflict', 'human', 'econ', 'moral', 'resp']
    
    def evaluate_frame_alignment(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """评估SV2000框架对齐性能
        
        Args:
            predictions: 预测结果字典
            ground_truth: 真实标签字典
            
        Returns:
            评估结果字典
        """
        logger.info("Evaluating SV2000 frame alignment")
        
        results = {
            'frame_correlations': {},
            'frame_average_alignment': {},
            'frame_presence_detection': {},
            'overall_alignment_score': 0.0
        }
        
        # 1. 逐框架相关性分析
        results['frame_correlations'] = self._evaluate_frame_correlations(predictions, ground_truth)
        
        # 2. 整体框架平均对齐
        results['frame_average_alignment'] = self._evaluate_frame_average_alignment(predictions, ground_truth)
        
        # 3. 框架存在检测（AUC分析）
        results['frame_presence_detection'] = self._evaluate_frame_presence_detection(predictions, ground_truth)
        
        # 4. 计算整体对齐分数
        results['overall_alignment_score'] = self._calculate_overall_alignment_score(results)
        
        logger.info(f"Frame alignment evaluation completed. Overall score: {results['overall_alignment_score']:.3f}")
        return results
    
    def _evaluate_frame_correlations(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """评估逐框架相关性"""
        correlations = {}
        
        for frame in self.frame_names:
            pred_key = f'sv_{frame}_pred'
            true_key = f'y_{frame}'
            
            if pred_key in predictions and true_key in ground_truth:
                pred_scores = np.array(predictions[pred_key])
                true_scores = np.array(ground_truth[true_key])
                
                if len(pred_scores) > 1 and len(true_scores) > 1:
                    # Pearson相关性（线性关系）
                    pearson_r, pearson_p = pearsonr(pred_scores, true_scores)
                    
                    # Spearman相关性（单调关系）
                    spearman_r, spearman_p = spearmanr(pred_scores, true_scores)
                    
                    # 误差指标
                    mae = mean_absolute_error(true_scores, pred_scores)
                    rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
                    
                    correlations[frame] = {
                        'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else 0.0,
                        'pearson_p': float(pearson_p) if not np.isnan(pearson_p) else 1.0,
                        'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else 0.0,
                        'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else 1.0,
                        'mae': float(mae),
                        'rmse': float(rmse),
                        'n_samples': len(pred_scores)
                    }
                else:
                    correlations[frame] = {
                        'pearson_r': 0.0, 'pearson_p': 1.0,
                        'spearman_r': 0.0, 'spearman_p': 1.0,
                        'mae': float('inf'), 'rmse': float('inf'),
                        'n_samples': 0
                    }
        
        return correlations
    
    def _evaluate_frame_average_alignment(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """评估整体框架平均对齐"""
        # 预测的框架平均
        if 'sv_frame_avg_pred' in predictions:
            pred_avg = np.array(predictions['sv_frame_avg_pred'])
        else:
            # 如果没有直接的平均分数，计算5个框架的平均
            frame_preds = []
            for frame in self.frame_names:
                pred_key = f'sv_{frame}_pred'
                if pred_key in predictions:
                    frame_preds.append(predictions[pred_key])
            
            if frame_preds:
                pred_avg = np.mean(frame_preds, axis=0)
            else:
                return {'error': 'No frame predictions available'}
        
        # 真实的框架平均
        frame_trues = []
        for frame in self.frame_names:
            true_key = f'y_{frame}'
            if true_key in ground_truth:
                frame_trues.append(ground_truth[true_key])
        
        if not frame_trues:
            return {'error': 'No ground truth frame scores available'}
        
        true_avg = np.mean(frame_trues, axis=0)
        
        if len(pred_avg) > 1 and len(true_avg) > 1:
            # 相关性指标
            pearson_r, _ = pearsonr(pred_avg, true_avg)
            spearman_r, _ = spearmanr(pred_avg, true_avg)
            
            # 误差指标
            mae = mean_absolute_error(true_avg, pred_avg)
            rmse = np.sqrt(mean_squared_error(true_avg, pred_avg))
            r2 = r2_score(true_avg, pred_avg)
            
            return {
                'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else 0.0,
                'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else 0.0,
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2) if not np.isnan(r2) else 0.0,
                'n_samples': len(pred_avg)
            }
        else:
            return {'error': 'Insufficient data for frame average alignment'}
    
    def _evaluate_frame_presence_detection(self, predictions: Dict, ground_truth: Dict, 
                                         threshold: float = 0.5) -> Dict:
        """评估框架存在检测（AUC分析）"""
        detection_results = {}
        
        for frame in self.frame_names:
            pred_key = f'sv_{frame}_pred'
            true_key = f'y_{frame}'
            
            if pred_key in predictions and true_key in ground_truth:
                pred_scores = np.array(predictions[pred_key])
                true_scores = np.array(ground_truth[true_key])
                
                if len(pred_scores) > 1 and len(true_scores) > 1:
                    # 二值化真实标签
                    true_binary = (true_scores >= threshold).astype(int)
                    
                    # 检查是否有正负样本
                    if len(np.unique(true_binary)) > 1:
                        # AUC-ROC
                        try:
                            auc_roc = roc_auc_score(true_binary, pred_scores)
                        except:
                            auc_roc = 0.5
                        
                        # AUC-PR
                        try:
                            auc_pr = average_precision_score(true_binary, pred_scores)
                        except:
                            auc_pr = np.mean(true_binary)
                        
                        # 最优阈值和F1分数
                        try:
                            fpr, tpr, thresholds = roc_curve(true_binary, pred_scores)
                            optimal_idx = np.argmax(tpr - fpr)
                            optimal_threshold = thresholds[optimal_idx]
                            
                            pred_binary = (pred_scores >= optimal_threshold).astype(int)
                            f1 = f1_score(true_binary, pred_binary)
                            precision = precision_score(true_binary, pred_binary, zero_division=0)
                            recall = recall_score(true_binary, pred_binary, zero_division=0)
                        except:
                            optimal_threshold = threshold
                            f1 = precision = recall = 0.0
                        
                        detection_results[frame] = {
                            'auc_roc': float(auc_roc),
                            'auc_pr': float(auc_pr),
                            'f1_score': float(f1),
                            'optimal_threshold': float(optimal_threshold),
                            'precision': float(precision),
                            'recall': float(recall),
                            'positive_rate': float(np.mean(true_binary))
                        }
                    else:
                        # 所有样本都是同一类别
                        detection_results[frame] = {
                            'auc_roc': 0.5, 'auc_pr': float(np.mean(true_binary)),
                            'f1_score': 0.0, 'optimal_threshold': threshold,
                            'precision': 0.0, 'recall': 0.0,
                            'positive_rate': float(np.mean(true_binary))
                        }
        
        return detection_results
    
    def _calculate_overall_alignment_score(self, results: Dict) -> float:
        """计算整体对齐分数"""
        scores = []
        
        # 框架相关性分数
        if 'frame_correlations' in results:
            correlations = []
            for frame_result in results['frame_correlations'].values():
                if 'pearson_r' in frame_result:
                    correlations.append(abs(frame_result['pearson_r']))
            
            if correlations:
                scores.append(np.mean(correlations))
        
        # 框架平均对齐分数
        if 'frame_average_alignment' in results and 'pearson_r' in results['frame_average_alignment']:
            scores.append(abs(results['frame_average_alignment']['pearson_r']))
        
        # AUC分数
        if 'frame_presence_detection' in results:
            aucs = []
            for frame_result in results['frame_presence_detection'].values():
                if 'auc_roc' in frame_result:
                    aucs.append(frame_result['auc_roc'])
            
            if aucs:
                scores.append(np.mean(aucs))
        
        return np.mean(scores) if scores else 0.0
    
    def evaluate_fusion_performance(self, fusion_results: List[Dict], 
                                  ground_truth: List[Dict]) -> Dict:
        """评估多组件融合性能"""
        logger.info("Evaluating fusion performance")
        
        if not fusion_results or not ground_truth:
            return {'error': 'No data provided for fusion evaluation'}
        
        # 组件贡献分析
        component_analysis = self._analyze_component_contributions(fusion_results)
        
        # 融合vs单组件性能比较
        performance_comparison = self._compare_fusion_vs_components(fusion_results, ground_truth)
        
        return {
            'component_analysis': component_analysis,
            'performance_comparison': performance_comparison
        }
    
    def _analyze_component_contributions(self, fusion_results: List[Dict]) -> Dict:
        """分析组件贡献度"""
        component_names = ['sv_frame_avg_pred', 'bias_score', 'omission_score', 
                          'relative_score', 'quote_score']
        
        # 提取组件分数和最终分数
        component_data = {comp: [] for comp in component_names}
        final_scores = []
        
        for result in fusion_results:
            final_scores.append(result.get('final_intensity', result.get('framing_intensity', 0.0)))
            for comp in component_names:
                component_data[comp].append(result.get(comp, 0.0))
        
        # 计算各组件与最终分数的相关性
        contributions = {}
        for comp in component_names:
            if len(component_data[comp]) > 1 and len(final_scores) > 1:
                try:
                    corr, _ = pearsonr(component_data[comp], final_scores)
                    contributions[comp] = {
                        'correlation_with_final': float(corr) if not np.isnan(corr) else 0.0,
                        'mean_score': float(np.mean(component_data[comp])),
                        'std_score': float(np.std(component_data[comp])),
                        'contribution_strength': float(abs(corr) * np.std(component_data[comp])) if not np.isnan(corr) else 0.0
                    }
                except:
                    contributions[comp] = {
                        'correlation_with_final': 0.0,
                        'mean_score': float(np.mean(component_data[comp])),
                        'std_score': float(np.std(component_data[comp])),
                        'contribution_strength': 0.0
                    }
        
        return contributions
    
    def _compare_fusion_vs_components(self, fusion_results: List[Dict], 
                                    ground_truth: List[Dict]) -> Dict:
        """比较融合性能与单组件性能"""
        # 提取分数
        fusion_scores = [r.get('final_intensity', r.get('framing_intensity', 0.0)) for r in fusion_results]
        sv_scores = [r.get('sv_frame_avg', r.get('sv_frame_avg_pred', 0.0)) for r in fusion_results]
        bias_scores = [r.get('bias_score', 0.0) for r in fusion_results]
        
        # 提取真实分数
        if isinstance(ground_truth[0], dict):
            true_scores = [gt.get('ground_truth_intensity', 0.0) for gt in ground_truth]
        else:
            true_scores = ground_truth
        
        # 确保长度一致
        min_len = min(len(fusion_scores), len(sv_scores), len(bias_scores), len(true_scores))
        fusion_scores = fusion_scores[:min_len]
        sv_scores = sv_scores[:min_len]
        bias_scores = bias_scores[:min_len]
        true_scores = true_scores[:min_len]
        
        if min_len < 2:
            return {'error': 'Insufficient data for comparison'}
        
        # 计算各方法的性能
        performance = {}
        
        approaches = {
            'fusion': fusion_scores,
            'sv2000_only': sv_scores,
            'bias_only': bias_scores
        }
        
        for name, scores in approaches.items():
            try:
                pearson_r, _ = pearsonr(scores, true_scores)
                spearman_r, _ = spearmanr(scores, true_scores)
                mae = mean_absolute_error(true_scores, scores)
                rmse = np.sqrt(mean_squared_error(true_scores, scores))
                
                performance[name] = {
                    'pearson_r': float(pearson_r) if not np.isnan(pearson_r) else 0.0,
                    'spearman_r': float(spearman_r) if not np.isnan(spearman_r) else 0.0,
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
            except:
                performance[name] = {
                    'pearson_r': 0.0, 'spearman_r': 0.0,
                    'mae': float('inf'), 'rmse': float('inf')
                }
        
        # 计算改进幅度
        if 'fusion' in performance and 'sv2000_only' in performance and 'bias_only' in performance:
            fusion_r = performance['fusion']['pearson_r']
            sv_r = performance['sv2000_only']['pearson_r']
            bias_r = performance['bias_only']['pearson_r']
            
            performance['improvements'] = {
                'fusion_vs_sv2000': fusion_r - sv_r,
                'fusion_vs_bias': fusion_r - bias_r,
                'sv2000_vs_bias': sv_r - bias_r
            }
        
        return performance
    
    def generate_evaluation_report(self, results: Dict, output_path: Optional[str] = None) -> str:
        """生成评估报告"""
        report_lines = []
        
        # 报告头部
        report_lines.append("# SV2000 Framing Alignment Evaluation Report")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 执行摘要
        if 'overall_alignment_score' in results:
            report_lines.append("## Executive Summary")
            report_lines.append(f"- Overall alignment score: {results['overall_alignment_score']:.3f}")
            report_lines.append("")
        
        # SV2000框架对齐
        if 'frame_correlations' in results:
            report_lines.append("## SV2000 Frame Alignment")
            report_lines.append("### Per-Frame Performance")
            report_lines.append("| Frame | Pearson r | Spearman r | MAE | RMSE |")
            report_lines.append("|-------|-----------|------------|-----|------|")
            
            for frame, metrics in results['frame_correlations'].items():
                report_lines.append(f"| {frame.capitalize()} | {metrics['pearson_r']:.3f} | "
                                  f"{metrics['spearman_r']:.3f} | {metrics['mae']:.3f} | "
                                  f"{metrics['rmse']:.3f} |")
            report_lines.append("")
        
        # 框架平均对齐
        if 'frame_average_alignment' in results and 'pearson_r' in results['frame_average_alignment']:
            fa = results['frame_average_alignment']
            report_lines.append("### Overall Frame Average")
            report_lines.append(f"- Pearson correlation: {fa['pearson_r']:.3f}")
            report_lines.append(f"- MAE: {fa['mae']:.3f}")
            report_lines.append(f"- R² score: {fa['r2_score']:.3f}")
            report_lines.append("")
        
        # 融合性能
        if 'component_analysis' in results:
            report_lines.append("## Fusion Performance")
            report_lines.append("### Component Contributions")
            report_lines.append("| Component | Correlation | Mean Score | Contribution Strength |")
            report_lines.append("|-----------|-------------|------------|----------------------|")
            
            for comp, metrics in results['component_analysis'].items():
                report_lines.append(f"| {comp} | {metrics['correlation_with_final']:.3f} | "
                                  f"{metrics['mean_score']:.3f} | {metrics['contribution_strength']:.3f} |")
            report_lines.append("")
        
        # 性能比较
        if 'performance_comparison' in results and 'fusion' in results['performance_comparison']:
            pc = results['performance_comparison']
            report_lines.append("### Performance Comparison")
            report_lines.append("| Method | Pearson r | MAE | RMSE |")
            report_lines.append("|--------|-----------|-----|------|")
            
            for method in ['fusion', 'sv2000_only', 'bias_only']:
                if method in pc:
                    metrics = pc[method]
                    report_lines.append(f"| {method} | {metrics['pearson_r']:.3f} | "
                                      f"{metrics['mae']:.3f} | {metrics['rmse']:.3f} |")
            
            if 'improvements' in pc:
                report_lines.append("")
                report_lines.append("### Improvements")
                for improvement, value in pc['improvements'].items():
                    report_lines.append(f"- {improvement}: {value:+.3f}")
            report_lines.append("")
        
        # 生成报告文本
        report_text = "\n".join(report_lines)
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report_text
    
    def create_visualization(self, results: Dict, output_dir: str = "./evaluation_plots"):
        """创建评估可视化图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 框架相关性热力图
        if 'frame_correlations' in results:
            self._plot_frame_correlations(results['frame_correlations'], 
                                        os.path.join(output_dir, 'frame_correlations.png'))
        
        # 2. 组件贡献雷达图
        if 'component_analysis' in results:
            self._plot_component_contributions(results['component_analysis'],
                                             os.path.join(output_dir, 'component_contributions.png'))
        
        logger.info(f"Evaluation plots saved to {output_dir}")
    
    def _plot_frame_correlations(self, correlations: Dict, output_path: str):
        """绘制框架相关性热力图"""
        frames = list(correlations.keys())
        metrics = ['pearson_r', 'spearman_r', 'mae', 'rmse']
        
        data = []
        for frame in frames:
            row = []
            for metric in metrics:
                value = correlations[frame].get(metric, 0.0)
                # 对于误差指标，使用负值以便可视化
                if metric in ['mae', 'rmse']:
                    value = -value
                row.append(value)
            data.append(row)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(data, 
                   xticklabels=metrics,
                   yticklabels=[f.capitalize() for f in frames],
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlBu_r')
        plt.title('SV2000 Frame Alignment Performance')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_component_contributions(self, contributions: Dict, output_path: str):
        """绘制组件贡献雷达图"""
        components = list(contributions.keys())
        values = [contributions[comp]['contribution_strength'] for comp in components]
        
        # 创建雷达图
        angles = np.linspace(0, 2 * np.pi, len(components), endpoint=False)
        values += values[:1]  # 闭合图形
        angles = np.concatenate((angles, [angles[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([comp.replace('_', ' ').title() for comp in components])
        ax.set_title('Component Contribution Strength', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()