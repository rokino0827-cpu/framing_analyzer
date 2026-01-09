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
        self.column_mapping = None  # 存储列映射信息
        
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
            
            # 智能检测和映射列
            self.column_mapping = self._detect_column_mapping()
            logger.info(f"Detected column mapping: {self.column_mapping}")
            
            # 应用列映射，创建标准化列
            self._apply_column_mapping()
            
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
    
    def _detect_column_mapping(self) -> Dict[str, str]:
        """智能检测列映射"""
        columns = list(self.df.columns)
        mapping = {}
        
        # 内容列检测
        # 优先精确匹配 content/text，再退化到包含 article 的列，避免误选 article_id
        content_priority = [col for col in columns if col.lower() in ['content', 'text']]
        if content_priority:
            mapping['content'] = content_priority[0]
            logger.info(f"Detected content column: {content_priority[0]}")
        else:
            content_candidates = [col for col in columns if any(keyword in col.lower() 
                                for keyword in ['content', 'text', 'article', 'body'])]
            if content_candidates:
                mapping['content'] = content_candidates[0]
                logger.info(f"Detected content column: {content_candidates[0]}")
        
        # 标题列检测
        title_candidates = [col for col in columns if any(keyword in col.lower() 
                          for keyword in ['title', 'headline', 'header'])]
        if title_candidates:
            mapping['title'] = title_candidates[0]
            logger.info(f"Detected title column: {title_candidates[0]}")
        
        # ID列检测
        id_candidates = [col for col in columns if any(keyword in col.lower() 
                        for keyword in ['id', 'article_id', 'identifier'])]
        if id_candidates:
            mapping['id'] = id_candidates[0]
            logger.info(f"Detected ID column: {id_candidates[0]}")
        
        # 框架分数列检测 - 支持用户的详细格式
        frame_mappings = self._detect_frame_columns(columns)
        mapping.update(frame_mappings)
        
        return mapping
    
    def _detect_frame_columns(self, columns: List[str]) -> Dict[str, str]:
        """检测框架分数列"""
        frame_mapping = {}
        
        # 检查是否有聚合的框架平均分数列
        avg_candidates = [col for col in columns if any(keyword in col.lower() 
                         for keyword in ['sv_frame_avg', 'frame_avg', 'avg_frame'])]
        if avg_candidates:
            frame_mapping['sv_frame_avg'] = avg_candidates[0]
            logger.info(f"Detected frame average column: {avg_candidates[0]}")
        
        # 检测各个框架的分数列
        frame_patterns = {
            'y_conflict': ['conflict', 'sv_conflict', 'disagreement', 'dispute'],
            'y_human': ['human', 'sv_human', 'interest', 'personal', 'emotion'],
            'y_econ': ['econ', 'sv_econ', 'economic', 'financial', 'cost'],
            'y_moral': ['moral', 'sv_moral', 'ethic', 'morality', 'religious'],
            'y_resp': ['resp', 'sv_resp', 'responsibility', 'responsible', 'government']
        }
        
        for frame_name, keywords in frame_patterns.items():
            # 首先查找聚合分数列
            candidates = [col for col in columns if any(keyword in col.lower() for keyword in keywords)
                         and not any(q in col.lower() for q in ['q1', 'q2', 'q3', 'q4', 'q5'])]
            
            if candidates:
                frame_mapping[frame_name] = candidates[0]
                logger.info(f"Detected {frame_name} column: {candidates[0]}")
            else:
                # 如果没有聚合列，查找问题级别的列并计算平均值
                question_cols = [col for col in columns if any(keyword in col.lower() for keyword in keywords)
                               and any(q in col.lower() for q in ['q1', 'q2', 'q3', 'q4', 'q5'])]
                
                if question_cols:
                    frame_mapping[f'{frame_name}_questions'] = question_cols
                    logger.info(f"Detected {frame_name} question columns: {question_cols}")
        
        return frame_mapping
    
    def _apply_column_mapping(self):
        """应用列映射，创建标准化列"""
        if not self.column_mapping:
            logger.warning("No column mapping detected, using original column names")
            return
        
        # 映射基本列
        for standard_col, source_col in self.column_mapping.items():
            if isinstance(source_col, str) and source_col in self.df.columns:
                if standard_col not in self.df.columns:
                    self.df[standard_col] = self.df[source_col].copy()
                    logger.info(f"Mapped {source_col} -> {standard_col}")
        
        # 处理框架分数列
        self._process_frame_scores()
        
        # 如果没有聚合的框架平均分数，计算一个
        if 'sv_frame_avg' not in self.df.columns:
            frame_cols = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
            available_frame_cols = [col for col in frame_cols if col in self.df.columns]
            
            if available_frame_cols:
                self.df['sv_frame_avg'] = self.df[available_frame_cols].mean(axis=1)
                logger.info(f"Computed sv_frame_avg from {available_frame_cols}")
    
    def _process_frame_scores(self):
        """处理框架分数列，支持问题级别聚合"""
        frame_names = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
        
        for frame_name in frame_names:
            # 检查是否有问题级别的列需要聚合
            question_key = f'{frame_name}_questions'
            if question_key in self.column_mapping:
                question_cols = self.column_mapping[question_key]
                
                # 验证问题列存在且为数值类型
                valid_question_cols = []
                for col in question_cols:
                    if col in self.df.columns:
                        # 转换为数值类型
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        valid_question_cols.append(col)
                
                if valid_question_cols:
                    # 计算问题级别的平均分数
                    self.df[frame_name] = self.df[valid_question_cols].mean(axis=1)
                    logger.info(f"Computed {frame_name} from questions: {valid_question_cols}")
                    
                    # 可选：保留原始问题列用于详细分析
                    for col in valid_question_cols:
                        new_col_name = f'{frame_name}_{col.split("_")[-1]}'  # 提取问题编号
                        if new_col_name not in self.df.columns:
                            self.df[new_col_name] = self.df[col].copy()
        
        # 确保所有框架分数在[0, 1]范围内
        for frame_name in frame_names:
            if frame_name in self.df.columns:
                self.df[frame_name] = pd.to_numeric(self.df[frame_name], errors='coerce')
                self.df[frame_name] = self.df[frame_name].fillna(0.0)
                self.df[frame_name] = np.clip(self.df[frame_name], 0.0, 1.0)
    
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
            'recommendations': [],
            'detected_format': 'unknown',
            'column_mapping': self.column_mapping or {}
        }
        
        if self.df is None:
            validation_results['is_valid'] = False
            validation_results['issues'].append("No data loaded")
            return validation_results
        
        # 检测数据格式类型
        if 'sv_frame_avg' in self.df.columns:
            validation_results['detected_format'] = 'user_machine_annotated'
        elif any('sv_' in col and '_q' in col for col in self.df.columns):
            validation_results['detected_format'] = 'detailed_questions'
        elif all(col in self.df.columns for col in ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']):
            validation_results['detected_format'] = 'standard_sv2000'
        else:
            validation_results['detected_format'] = 'custom'
        
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
            
            # 检查是否有可以用于计算的问题级别列
            question_cols = [col for col in self.df.columns if any(f'sv_{frame.split("_")[1]}_q' in col 
                           for frame in missing_frames)]
            if question_cols:
                validation_results['recommendations'].append(
                    f"Found question-level columns {question_cols[:3]}... that can be aggregated into frame scores"
                )
            else:
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
        total_samples = len(self.df)
        validation_results['total_samples'] = total_samples
        validation_results['valid_samples'] = total_samples - null_content
        
        if total_samples < 100:
            validation_results['recommendations'].append("Dataset is quite small, consider collecting more data")
        elif total_samples < 500:
            validation_results['recommendations'].append("Dataset size is adequate for initial training")
        elif total_samples < 2000:
            validation_results['recommendations'].append("Good dataset size for reliable training")
        else:
            validation_results['recommendations'].append("Excellent dataset size for robust training")
        
        # 检查用户特定格式的完整性
        if validation_results['detected_format'] == 'user_machine_annotated':
            self._validate_user_format(validation_results)
        
        return validation_results
    
    def _validate_user_format(self, validation_results: Dict):
        """验证用户机器标注数据的特定格式"""
        # 检查用户提供的关键列
        expected_user_columns = [
            'article_id', 'author', 'title', 'content', 'url', 'section', 'publication',
            'annotator_id', 'sv_frame_avg', 'avg_stratum'
        ]
        
        missing_user_cols = [col for col in expected_user_columns if col not in self.df.columns]
        if missing_user_cols:
            validation_results['issues'].append(f"Missing expected user columns: {missing_user_cols}")
        
        # 检查问题级别列的完整性
        frame_question_patterns = {
            'conflict': ['sv_conflict_q1_reflects_disagreement', 'sv_conflict_q2_refers_to_two_sides', 
                        'sv_conflict_q3_refers_to_winners_losers_optional', 'sv_conflict_q4_reproach_between_sides'],
            'human': ['sv_human_q1_human_example_or_face', 'sv_human_q2_adjectives_personal_vignettes',
                     'sv_human_q3_feelings_empathy', 'sv_human_q4_how_people_affected', 
                     'sv_human_q5_visual_information_optional'],
            'econ': ['sv_econ_q1_financial_losses_gains', 'sv_econ_q2_costs_degree_of_expense',
                    'sv_econ_q3_economic_consequences_pursue_or_not'],
            'moral': ['sv_moral_q1_moral_message', 'sv_moral_q2_morality_god_religious_tenets',
                     'sv_moral_q3_social_prescriptions'],
            'resp': ['sv_resp_q1_government_ability_solve', 'sv_resp_q2_individual_group_responsible',
                    'sv_resp_q3_government_responsible', 'sv_resp_q4_solution_proposed',
                    'sv_resp_q5_urgent_action_required_optional']
        }
        
        question_coverage = {}
        for frame, questions in frame_question_patterns.items():
            available_questions = [q for q in questions if q in self.df.columns]
            question_coverage[frame] = {
                'available': len(available_questions),
                'total': len(questions),
                'coverage': len(available_questions) / len(questions),
                'missing': [q for q in questions if q not in self.df.columns]
            }
        
        validation_results['question_coverage'] = question_coverage
        
        # 检查数据质量
        if 'sv_frame_avg' in self.df.columns:
            avg_scores = self.df['sv_frame_avg'].dropna()
            if len(avg_scores) > 0:
                validation_results['frame_avg_stats'] = {
                    'mean': float(avg_scores.mean()),
                    'std': float(avg_scores.std()),
                    'min': float(avg_scores.min()),
                    'max': float(avg_scores.max()),
                    'valid_count': len(avg_scores)
                }
                
                # 检查分数分布
                if avg_scores.std() < 0.1:
                    validation_results['recommendations'].append(
                        "Frame scores have low variance, consider more diverse samples"
                    )
        
        # 检查标注者信息
        if 'annotator_id' in self.df.columns:
            annotator_counts = self.df['annotator_id'].value_counts()
            validation_results['annotator_stats'] = {
                'unique_annotators': len(annotator_counts),
                'samples_per_annotator': annotator_counts.to_dict()
            }
            
            if len(annotator_counts) == 1:
                validation_results['recommendations'].append(
                    "Data from single annotator - consider inter-annotator reliability checks"
                )

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
