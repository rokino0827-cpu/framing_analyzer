"""
配置文件 - 框架偏见分析器配置
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path

from utils import find_hf_cache_model_path

logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """文本预处理配置"""
    # 噪声过滤关键词
    noise_keywords: List[str] = None
    
    # 句子切分配置
    chinese_sentence_endings: List[str] = None
    english_sentence_endings: List[str] = None
    
    # 结构区划分配置
    lede_sentence_count: int = 4  # 导语句子数量
    
    # 引号模式
    quote_patterns: List[str] = None
    
    def __post_init__(self):
        if self.noise_keywords is None:
            self.noise_keywords = [
                'advertisement', 'subscribe', 'sign up', 'all rights reserved',
                'cookie', 'privacy policy', 'terms of service', 'newsletter',
                'follow us', 'social media', 'share this', 'related articles',
                'trending now', 'breaking news alert', 'download our app'
            ]
        
        if self.chinese_sentence_endings is None:
            self.chinese_sentence_endings = ['。', '！', '？', '…', '；']
        
        if self.english_sentence_endings is None:
            self.english_sentence_endings = ['.', '!', '?', '...']
        
        if self.quote_patterns is None:
            self.quote_patterns = [
                r'“([^”]+)”',      # 弯双引号
                r'‘([^’]+)’',      # 弯单引号
                r'"([^"]+)"',      # 英文/直双引号
                r"'([^']+)'",      # 英文/直单引号
                r'&quot;([^&]+?)&quot;',  # HTML 实体双引号
                r'「([^」]+)」',    # 中文书名号/引号
                r'『([^』]+)』',    # 中文书名号（内外层）
            ]

@dataclass
class TeacherConfig:
    """Teacher模型配置"""
    model_name: str = "himel7/bias-detector"
    model_local_path: Optional[str] = "bias_detector_data"  # 优先使用本地模型以避免联网
    max_length: int = 512
    batch_size: int = 16
    device: str = "cuda"  # auto: 优先CUDA，失败自动降级CPU
    
    # 片段处理模式
    fragment_mode: str = "sentence"  # "sentence" 或 "chunk"
    
    # 滑窗配置（用于超长句子）
    sliding_window_size: int = 512
    sliding_window_stride: int = 256
    
    # 长句子分数聚合方式
    long_sentence_aggregation: str = "max"  # "max", "mean", "top2_mean"
    
    # Bias类别配置
    bias_class_index: Optional[int] = 1  # 直接指定0/1
    bias_class_name: Optional[str] = None   # 或指定label名称（如果模型有明确标签）

    def __post_init__(self):
        # 将相对路径解析为仓库内绝对路径，避免离线加载失败
        if self.model_local_path:
            path_obj = Path(self.model_local_path)
            if not path_obj.is_absolute():
                candidate = Path(__file__).resolve().parent / path_obj
                if candidate.exists():
                    self.model_local_path = str(candidate)

@dataclass
class ScoringConfig:
    """评分配置"""
    # 结构区权重
    headline_weight: float = 0.25
    lede_weight: float = 0.35
    narration_weight: float = 0.25
    quote_gap_weight: float = 0.10
    sparse_signal_weight: float = 0.05
    
    # TopK参数
    lede_topk: int = 2
    narration_topk: int = 3
    quotes_topk: int = 3
    global_topk: int = 3
    
    # 稀疏信号阈值
    high_bias_threshold: float = 0.8
    
    # 弱标签阈值
    positive_threshold_percentile: int = 80  # Q80
    negative_threshold_percentile: int = 20  # Q20
    
    # 证据片段数量
    evidence_count: int = 5

@dataclass
class OutputConfig:
    """输出配置"""
    # 输出格式
    output_format: str = "json"  # "json", "csv", "both"
    
    # 输出字段
    include_components: bool = True
    include_evidence: bool = True
    include_raw_scores: bool = False
    include_statistics: bool = True
    
    # 文件路径
    output_dir: str = "./results"
    results_filename: str = "framing_analysis_results"
    
    # 可视化
    generate_plots: bool = True
    plot_distribution: bool = True
    plot_evidence_examples: bool = True

@dataclass
class RelativeFramingConfig:
    """相对框架配置（可选功能）"""
    enabled: bool = False
    
    # 事件聚类配置
    similarity_threshold: float = 0.3
    time_window_days: int = 1
    min_cluster_size: int = 2
    
    # TF-IDF配置
    max_features: int = 1000
    ngram_range: tuple = (1, 2)

@dataclass
class OmissionConfig:
    """省略感知配置（新增OmiGraph功能）"""
    enabled: bool = False
    # 嵌入模型路径（支持本地目录或模型名称）
    embedding_model_name_or_path: str = "bge_m3"
    
    # 图构建配置
    similarity_threshold: float = 0.5  # 跨文边相似度阈值
    guidance_threshold: float = 0.3   # lede→narration引导边阈值
    
    # 省略检测配置
    key_topics_count: int = 15        # 关键主题数量
    min_topic_frequency: int = 2      # 主题最小频率
    omission_weight_headline: float = 0.4  # headline省略权重
    omission_weight_lede: float = 0.4      # lede省略权重
    omission_weight_full: float = 0.2      # 全文省略权重
    omission_effect_threshold: float = 0.35  # 最低生效的省略分数（低于则不计入融合/统计）
    
    # 证据提取配置
    max_evidence_count: int = 5       # 最大证据数量
    max_examples_per_evidence: int = 3 # 每个证据的最大例子数
    
    # 实体识别配置
    use_spacy: bool = True           # 是否使用spaCy进行实体识别
    entity_types: List[str] = None   # 关注的实体类型
    
    # 融合配置 - 新增
    fusion_weight: float = 0.2       # 省略分数在最终分数中的权重 (推荐0.1-0.3)
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ['PERSON', 'ORG', 'GPE', 'EVENT']

@dataclass
class SVFramingConfig:
    """SV2000框架预测配置"""
    enabled: bool = False  # 是否启用SV2000模式
    encoder_name: str = "bge_m3"
    encoder_local_path: Optional[str] = None
    hidden_size: int = 1024
    dropout_rate: float = 0.1
    # 优化与训练
    learning_rate: float = 2e-5  # 向后兼容：作为默认编码器学习率
    encoder_learning_rate: float = 2e-5
    # 运行实证最优：若未显式指定，则回落到max(learning_rate, 1e-3)
    head_learning_rate: Optional[float] = None
    min_head_learning_rate: float = 1e-3
    training_mode: str = "frame_level"  # "frame_level" 或 "item_level"
    selection_metric: str = "frame_avg_pearson"  # 用于早停/最佳模型选择的验证指标
    accumulation_steps: int = 1  # 梯度累积步数，>1 可在显存不变的情况下放大等效batch
    # 切分与覆盖约束
    split_retry_limit: int = 5  # 切分失败时的最大重试次数（满足最小正例覆盖）
    min_pos_per_frame_val: int = 1  # 验证集每个框架最少正例数，0表示不做约束
    min_pos_per_frame_test: int = 1  # 测试集每个框架最少正例数，0表示不做约束
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512
    # 长文本分片
    chunk_stride: int = 128
    max_chunks_per_text: int = 4
    fine_tune_encoder: bool = False
    # 轻量微调：仅解冻编码器末尾若干层，0 表示保持冻结
    trainable_encoder_layers: int = 0
    max_grad_norm: float = 1.0
    # 类别不平衡与标定
    # 默认提升 moral/resp 权重以拉高稀缺框架的梯度
    frame_loss_weights: Optional[Dict[str, float]] = None
    calibration_max_iter: int = 50
    calibration_lr: float = 0.01
    calibration_use_bias: bool = True  # 是否在温度缩放时同时学习logit偏置，修正整体高估
    pos_weight_cap: float = 8.0  # 控制pos_weight上限，避免极端不平衡时过度推高正例概率
    # 评价对齐与均值约束
    corr_loss_weight: float = 0.1  # 相关性辅助损失权重，直接优化Pearson
    frame_avg_loss_weight: float = 0.1  # 框架平均分Huber损失权重，防止整体偏移
    save_encoder_state: bool = True  # 训练后是否一并保存编码器权重，便于微调持久化

    # 数据切分与采样
    use_group_split: bool = True  # 默认启用事件/媒体分组切分，减少泄漏
    group_column: Optional[str] = "event_cluster"
    fallback_group_columns: Optional[list] = None  # 额外候选列，如 ["cluster_id", "topic_cluster"]
    publication_column: Optional[str] = "publication"
    time_column: Optional[str] = "published_at"
    time_freq: str = "W"  # 按周分组 publication+时间 窗口
    balance_batches: bool = True  # 基于标签稀缺度的平衡采样

    # 损失函数与动态加权
    loss_type: str = "focal"  # bce 或 focal
    focal_gamma: float = 2.0
    dynamic_frame_reweight: bool = True  # 根据验证误差调整框架权重
    dynamic_reweight_cap: float = 2.5  # 防止权重大幅波动，同时允许对弱类进一步倾斜

    # 题项级训练
    item_consistency_weight: float = 0.3  # 题项→框架一致性loss权重
    return_item_predictions: bool = False  # 推理时是否返回item logits/probs

    # 模型保存路径
    model_save_path: str = "./sv2000_models"
    pretrained_model_path: Optional[str] = None

    def __post_init__(self):
        """
        优先使用本地sentence-transformers副本，避免离线环境拉取失败；
        同时将相对路径解析为仓库内绝对路径。
        """
        encoder_name_lower = str(self.encoder_name).lower()

        if not self.encoder_local_path:
            preferred_local_dirs = ["bge_m3", "bge-m3"]
            for dir_name in preferred_local_dirs:
                local_dir = Path(__file__).resolve().parent / dir_name
                if local_dir.exists():
                    self.encoder_local_path = str(local_dir)
                    break
            if not self.encoder_local_path:
                if "all-minilm" in encoder_name_lower:
                    legacy_dir = Path(__file__).resolve().parent / "all-MiniLM-L6-v2"
                    if legacy_dir.exists():
                        self.encoder_local_path = str(legacy_dir)
            if not self.encoder_local_path:
                cache_hit = find_hf_cache_model_path(self.encoder_name)
                if not cache_hit and "bge" in encoder_name_lower and "m3" in encoder_name_lower:
                    cache_hit = find_hf_cache_model_path("BAAI/bge-m3")
                if cache_hit:
                    self.encoder_local_path = cache_hit
        else:
            path_obj = Path(self.encoder_local_path)
            if not path_obj.is_absolute():
                candidate = Path(__file__).resolve().parent / path_obj
                if candidate.exists():
                    self.encoder_local_path = str(candidate)

        # 如果仍未找到bge-m3，本地回退到内置的MiniLM以确保离线可运行
        fallback_dir = Path(__file__).resolve().parent / "all-MiniLM-L6-v2"
        if (not self.encoder_local_path or not Path(self.encoder_local_path).exists()) and fallback_dir.exists():
            logger.warning(
                "未找到本地编码器 %s，已回退到内置的 all-MiniLM-L6-v2，避免离线下载失败",
                self.encoder_name,
            )
            self.encoder_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.encoder_local_path = str(fallback_dir)

        # 自动绑定最新评测约70%相关性的校准模型
        if self.pretrained_model_path:
            path_obj = Path(self.pretrained_model_path)
            if not path_obj.is_absolute():
                candidate = Path(__file__).resolve().parent / path_obj
                if candidate.exists():
                    self.pretrained_model_path = str(candidate)
        else:
            best_calibrated = Path(__file__).resolve().parent / "sv2000_training_results/best_sv2000_model_calibrated.pt"
            if best_calibrated.exists():
                self.pretrained_model_path = str(best_calibrated)
            else:
                legacy_calibrated = Path(__file__).resolve().parent / "outputs/run4_bias_calib/best_sv2000_model_calibrated.pt"
                if legacy_calibrated.exists():
                    logger.warning("最新校准模型缺失，回退至legacy: %s", legacy_calibrated)
                    self.pretrained_model_path = str(legacy_calibrated)

        # 确保学习率配置兼容旧字段
        if getattr(self, "encoder_learning_rate", None) is None:
            self.encoder_learning_rate = self.learning_rate
        # 头部默认更大学习率以便快速收敛（恢复run3配置：1e-3基准）
        if getattr(self, "head_learning_rate", None) is None:
            self.head_learning_rate = max(self.learning_rate, self.min_head_learning_rate)
        else:
            self.head_learning_rate = max(self.head_learning_rate, self.min_head_learning_rate)

        # 默认对 moral/resp 增权，提升梯度信号；允许用户显式覆盖
        if self.frame_loss_weights is None:
            self.frame_loss_weights = {
                "conflict": 1.0,
                "human": 1.0,
                "econ": 1.0,
                "moral": 1.8,
                "resp": 1.6,
            }
        # 防止负权重意外导致loss为负
        self.corr_loss_weight = max(0.0, self.corr_loss_weight)
        self.frame_avg_loss_weight = max(0.0, self.frame_avg_loss_weight)

@dataclass
class FusionConfig:
    """多组件融合配置"""
    # 默认融合权重
    alpha: float = 0.5    # SV2000权重
    beta: float = 0.2     # Bias-detector权重
    gamma: float = 0.15   # Omission权重
    delta: float = 0.1    # Relative framing权重
    epsilon: float = 0.05 # Quote analysis权重
    
    # 权重约束
    enforce_positive_weights: bool = True
    normalize_weights: bool = True
    
    # 优化设置
    use_ridge_optimization: bool = True
    ridge_alpha: float = 1.0
    cross_validation_folds: int = 5

@dataclass
class AnalyzerConfig:
    """主配置类"""
    processing: ProcessingConfig = None
    teacher: TeacherConfig = None
    scoring: ScoringConfig = None
    output: OutputConfig = None
    relative_framing: RelativeFramingConfig = None
    omission: OmissionConfig = None  # 省略感知配置
    sv_framing: SVFramingConfig = None  # 新增SV2000框架配置
    fusion: FusionConfig = None  # 新增融合配置
    
    # 全局配置
    random_seed: int = 42
    verbose: bool = True
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.processing is None:
            self.processing = ProcessingConfig()
        if self.teacher is None:
            self.teacher = TeacherConfig()
        if self.scoring is None:
            self.scoring = ScoringConfig()
        if self.output is None:
            self.output = OutputConfig()
        if self.relative_framing is None:
            self.relative_framing = RelativeFramingConfig()
        if self.omission is None:
            self.omission = OmissionConfig()
        if self.sv_framing is None:
            self.sv_framing = SVFramingConfig()
        if self.fusion is None:
            self.fusion = FusionConfig()
    
    def enable_sv2000_mode(self):
        """启用SV2000框架分析模式"""
        self.sv_framing.enabled = True
        
    def disable_sv2000_mode(self):
        """禁用SV2000框架分析模式（传统模式）"""
        self.sv_framing.enabled = False

# 默认配置实例
default_config = AnalyzerConfig()

def load_config(config_path: str) -> AnalyzerConfig:
    """从文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return AnalyzerConfig(
        processing=ProcessingConfig(**config_dict.get('processing', {})),
        teacher=TeacherConfig(**config_dict.get('teacher', {})),
        scoring=ScoringConfig(**config_dict.get('scoring', {})),
        output=OutputConfig(**config_dict.get('output', {})),
        relative_framing=RelativeFramingConfig(**config_dict.get('relative_framing', {})),
        omission=OmissionConfig(**config_dict.get('omission', {})),
        sv_framing=SVFramingConfig(**config_dict.get('sv_framing', {})),
        fusion=FusionConfig(**config_dict.get('fusion', {})),
        **{k: v for k, v in config_dict.items() 
           if k not in ['processing', 'teacher', 'scoring', 'output', 'relative_framing', 'omission', 'sv_framing', 'fusion']}
    )

def save_config(config: AnalyzerConfig, config_path: str):
    """保存配置到文件"""
    from dataclasses import asdict
    
    config_dict = {
        'processing': asdict(config.processing),
        'teacher': asdict(config.teacher),
        'scoring': asdict(config.scoring),
        'output': asdict(config.output),
        'relative_framing': asdict(config.relative_framing),
        'omission': asdict(config.omission),
        'sv_framing': asdict(config.sv_framing),
        'fusion': asdict(config.fusion),
        'random_seed': config.random_seed,
        'verbose': config.verbose,
        'log_level': config.log_level
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

def create_quick_config(
    lede_count: int = 4,
    evidence_count: int = 5,
    batch_size: int = 16,
    output_dir: str = "./results"
) -> AnalyzerConfig:
    """创建快速配置"""
    config = AnalyzerConfig()
    config.processing.lede_sentence_count = lede_count
    config.scoring.evidence_count = evidence_count
    config.teacher.batch_size = batch_size
    config.output.output_dir = output_dir
    return config

def create_high_precision_config() -> AnalyzerConfig:
    """创建高精度配置（更严格的阈值）"""
    config = AnalyzerConfig()
    config.scoring.positive_threshold_percentile = 90  # Q90
    config.scoring.negative_threshold_percentile = 10  # Q10
    config.scoring.high_bias_threshold = 0.85
    return config

def create_fast_config() -> AnalyzerConfig:
    """创建快速处理配置"""
    config = AnalyzerConfig()
    config.teacher.fragment_mode = "chunk"  # 使用块模式而非句子模式
    config.teacher.batch_size = 32
    config.scoring.evidence_count = 3
    config.output.include_raw_scores = False
    return config

def create_sv2000_config() -> AnalyzerConfig:
    """创建SV2000模式配置"""
    config = AnalyzerConfig()
    config.enable_sv2000_mode()
    config.fusion.use_ridge_optimization = True
    # 使用最新训练的校准模型，并锁定融合权重为实证最优（只依赖SV2000主模型）
    latest_calibrated = Path(__file__).resolve().parent / "sv2000_training_results/best_sv2000_model_calibrated.pt"
    if not latest_calibrated.exists():
        logger.warning("未找到最新校准模型，回退至旧路径: %s", latest_calibrated)
        latest_calibrated = Path(__file__).resolve().parent / "outputs/run4_bias_calib/best_sv2000_model_calibrated.pt"

    config.sv_framing.pretrained_model_path = str(latest_calibrated)

    # 按最新验证结果的优化权重：仅使用SV2000预测，关闭辅助项噪声
    config.fusion.alpha = 1.0
    config.fusion.beta = 0.0
    config.fusion.gamma = 0.0
    config.fusion.delta = 0.0
    config.fusion.epsilon = 0.0
    return config
