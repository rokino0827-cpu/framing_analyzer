"""
SV2000 Data Loader - SV2000标注数据加载器
用于加载和处理Semetko & Valkenburg (2000)框架标注数据
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split
from pathlib import Path

logger = logging.getLogger(__name__)

class SV2000DataLoader:
    """SV2000标注数据加载器"""
    
    def __init__(self, csv_path: str, config):
        self.csv_path = csv_path
        self.config = config
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        self._load_and_validate_data()
        
    def _load_and_validate_data(self) -> pd.DataFrame:
        """加载并验证数据格式"""
        try:
            logger.info(f"Loading SV2000 data from: {self.csv_path}")
            
            if not Path(self.csv_path).exists():
                raise FileNotFoundError(f"SV2000 data file not found: {self.csv_path}")
            
            # 读取CSV数据
            self.df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self.df)} samples from SV2000 dataset")
            
            # 验证必需列
            required_columns = ['content']
            frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
            
            missing_required = [col for col in required_columns if col not in self.df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")
            
            missing_frame = [col for col in frame_columns if col not in self.df.columns]
            if missing_frame:
                logger.warning(f"Missing frame columns: {missing_frame}")
                # 为缺失的框架列创建默认值
                for col in missing_frame:
                    self.df[col] = 0.0
            
            # 验证数据质量
            self._validate_data_quality()
            
            logger.info("SV2000 data validation completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading SV2000 data: {e}")
            raise
    
    def _validate_data_quality(self):
        """验证数据质量"""
        # 检查空值
        content_nulls = self.df['content'].isnull().sum()
        if content_nulls > 0:
            logger.warning(f"Found {content_nulls} null content entries, removing them")
            self.df = self.df.dropna(subset=['content'])
        
        # 检查框架分数范围
        frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
        for col in frame_columns:
            if col in self.df.columns:
                # 确保分数在[0, 1]范围内
                self.df[col] = np.clip(self.df[col], 0.0, 1.0)
                
                # 检查异常值
                outliers = ((self.df[col] < 0) | (self.df[col] > 1)).sum()
                if outliers > 0:
                    logger.warning(f"Found {outliers} outliers in {col}, clipped to [0, 1]")
        
        # 检查文本长度
        text_lengths = self.df['content'].str.len()
        short_texts = (text_lengths < 50).sum()
        long_texts = (text_lengths > 10000).sum()
        
        if short_texts > 0:
            logger.warning(f"Found {short_texts} very short texts (< 50 chars)")
        if long_texts > 0:
            logger.warning(f"Found {long_texts} very long texts (> 10000 chars)")
        
        logger.info(f"Data quality validation completed. Final dataset size: {len(self.df)}")
    
    def get_training_data(self, mode: str = "frame_level", split: str = "train") -> Tuple[List[str], np.ndarray]:
        """获取训练数据
        
        Args:
            mode: "frame_level" 或 "item_level"
            split: "train", "val", 或 "test"
            
        Returns:
            (texts, targets) 元组
        """
        # 确保数据已分割
        if self.train_df is None:
            self.create_train_val_split()
        
        # 选择数据集
        if split == "train":
            df = self.train_df
        elif split == "val":
            df = self.val_df
        elif split == "test":
            df = self.test_df if self.test_df is not None else self.val_df
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if df is None or len(df) == 0:
            logger.warning(f"No data available for split: {split}")
            return [], np.array([])
        
        # 提取文本
        texts = df['content'].tolist()
        
        # 提取目标标签
        if mode == "frame_level":
            # 使用聚合的框架分数
            frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
            targets = df[frame_columns].values
        elif mode == "item_level":
            # 使用单独的问卷条目（如果可用）
            item_columns = [f'item_{i}' for i in range(1, 21)]
            available_items = [col for col in item_columns if col in df.columns]
            
            if not available_items:
                logger.warning("No item-level data available, falling back to frame-level")
                frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
                targets = df[frame_columns].values
            else:
                targets = df[available_items].values
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        logger.info(f"Retrieved {len(texts)} samples for {split} split in {mode} mode")
        return texts, targets
    
    def create_train_val_split(self, val_ratio: float = 0.2, test_ratio: float = 0.1, 
                              random_seed: int = 42):
        """创建训练/验证/测试数据分割"""
        if len(self.df) == 0:
            logger.error("No data available for splitting")
            return
        
        # 首先分离测试集
        if test_ratio > 0:
            train_val_df, self.test_df = train_test_split(
                self.df, 
                test_size=test_ratio, 
                random_state=random_seed,
                stratify=None  # 回归任务不需要分层
            )
        else:
            train_val_df = self.df
            self.test_df = None
        
        # 然后分离训练集和验证集
        if val_ratio > 0:
            self.train_df, self.val_df = train_test_split(
                train_val_df,
                test_size=val_ratio / (1 - test_ratio),  # 调整比例
                random_state=random_seed
            )
        else:
            self.train_df = train_val_df
            self.val_df = None
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {len(self.train_df) if self.train_df is not None else 0}")
        logger.info(f"  Val: {len(self.val_df) if self.val_df is not None else 0}")
        logger.info(f"  Test: {len(self.test_df) if self.test_df is not None else 0}")
    
    def get_data_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if self.df is None:
            return {}
        
        stats = {
            'total_samples': len(self.df),
            'columns': list(self.df.columns),
            'text_statistics': {},
            'frame_statistics': {}
        }
        
        # 文本统计
        if 'content' in self.df.columns:
            text_lengths = self.df['content'].str.len()
            stats['text_statistics'] = {
                'mean_length': text_lengths.mean(),
                'median_length': text_lengths.median(),
                'min_length': text_lengths.min(),
                'max_length': text_lengths.max(),
                'std_length': text_lengths.std()
            }
        
        # 框架分数统计
        frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
        for col in frame_columns:
            if col in self.df.columns:
                stats['frame_statistics'][col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'median': self.df[col].median()
                }
        
        # 数据分割统计
        if self.train_df is not None:
            stats['split_statistics'] = {
                'train_size': len(self.train_df),
                'val_size': len(self.val_df) if self.val_df is not None else 0,
                'test_size': len(self.test_df) if self.test_df is not None else 0
            }
        
        return stats
    
    def get_sample_data(self, n_samples: int = 5) -> List[Dict]:
        """获取样本数据用于检查"""
        if self.df is None or len(self.df) == 0:
            return []
        
        sample_df = self.df.head(n_samples)
        samples = []
        
        for _, row in sample_df.iterrows():
            sample = {
                'content': row.get('content', '')[:200] + '...',  # 截断显示
                'title': row.get('title', 'N/A'),
            }
            
            # 添加框架分数
            frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
            for col in frame_columns:
                if col in row:
                    sample[col] = row[col]
            
            samples.append(sample)
        
        return samples
    
    def validate_annotation_format(self) -> Dict:
        """验证标注格式"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'recommendations': []
        }
        
        if self.df is None:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No data loaded")
            return validation_results
        
        # 检查必需列
        required_columns = ['content']
        missing_required = [col for col in required_columns if col not in self.df.columns]
        if missing_required:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_required}")
        
        # 检查框架列
        frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
        missing_frames = [col for col in frame_columns if col not in self.df.columns]
        if missing_frames:
            validation_results['issues'].append(f"Missing frame columns: {missing_frames}")
            validation_results['recommendations'].append("Consider adding missing frame annotations")
        
        # 检查数据范围
        for col in frame_columns:
            if col in self.df.columns:
                out_of_range = ((self.df[col] < 0) | (self.df[col] > 1)).sum()
                if out_of_range > 0:
                    validation_results['issues'].append(f"{col} has {out_of_range} values outside [0,1] range")
        
        # 检查空值
        null_content = self.df['content'].isnull().sum()
        if null_content > 0:
            validation_results['issues'].append(f"{null_content} samples have null content")
        
        # 检查数据量
        if len(self.df) < 100:
            validation_results['recommendations'].append("Dataset is quite small, consider collecting more data")
        
        return validation_results

class SV2000DataProcessor:
    """SV2000数据预处理器"""
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """预处理文本"""
        if not isinstance(text, str):
            return ""
        
        # 基本清理
        text = text.strip()
        
        # 移除过多的空白字符
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符（保留基本标点）
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\"\']+', ' ', text)
        
        return text
    
    @staticmethod
    def augment_data(texts: List[str], targets: np.ndarray, 
                    augmentation_factor: float = 0.2) -> Tuple[List[str], np.ndarray]:
        """数据增强（简单的同义词替换等）"""
        # 这里可以实现更复杂的数据增强策略
        # 目前只是简单的复制
        n_augment = int(len(texts) * augmentation_factor)
        
        if n_augment == 0:
            return texts, targets
        
        # 随机选择样本进行增强
        indices = np.random.choice(len(texts), n_augment, replace=True)
        
        augmented_texts = []
        augmented_targets = []
        
        for idx in indices:
            # 简单的文本变换（可以扩展为更复杂的增强）
            original_text = texts[idx]
            augmented_text = original_text  # 暂时不做变换
            
            augmented_texts.append(augmented_text)
            augmented_targets.append(targets[idx])
        
        # 合并原始数据和增强数据
        all_texts = texts + augmented_texts
        all_targets = np.vstack([targets, np.array(augmented_targets)])
        
        return all_texts, all_targets
    
    @staticmethod
    def balance_dataset(texts: List[str], targets: np.ndarray, 
                       balance_threshold: float = 0.5) -> Tuple[List[str], np.ndarray]:
        """平衡数据集（基于框架平均分数）"""
        # 计算每个样本的平均框架分数
        avg_scores = np.mean(targets, axis=1)
        
        # 分为高分和低分组
        high_indices = np.where(avg_scores >= balance_threshold)[0]
        low_indices = np.where(avg_scores < balance_threshold)[0]
        
        # 平衡策略：使两组数量相近
        min_size = min(len(high_indices), len(low_indices))
        
        if min_size == 0:
            return texts, targets
        
        # 随机采样
        selected_high = np.random.choice(high_indices, min_size, replace=False)
        selected_low = np.random.choice(low_indices, min_size, replace=False)
        
        selected_indices = np.concatenate([selected_high, selected_low])
        np.random.shuffle(selected_indices)
        
        balanced_texts = [texts[i] for i in selected_indices]
        balanced_targets = targets[selected_indices]
        
        return balanced_texts, balanced_targets