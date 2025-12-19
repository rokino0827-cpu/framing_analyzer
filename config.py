"""
配置文件 - 框架偏见分析器配置
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json

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
                r'"([^"]+)"',      # 英文双引号
                r"'([^']+)'",      # 英文单引号
                r'"([^"]+)"',      # 中文双引号
                r'‘([^’]+)’',      # 中文单引号
            ]

@dataclass
class TeacherConfig:
    """Teacher模型配置"""
    model_name: str = "himel7/bias-detector"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"
    
    # 片段处理模式
    fragment_mode: str = "sentence"  # "sentence" 或 "chunk"
    
    # 滑窗配置（用于超长句子）
    sliding_window_size: int = 512
    sliding_window_stride: int = 256
    
    # 长句子分数聚合方式
    long_sentence_aggregation: str = "max"  # "max", "mean", "top2_mean"

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
    
    # 图构建配置
    similarity_threshold: float = 0.5  # 跨文边相似度阈值
    guidance_threshold: float = 0.3   # lede→narration引导边阈值
    
    # 省略检测配置
    key_topics_count: int = 15        # 关键主题数量
    min_topic_frequency: int = 2      # 主题最小频率
    omission_weight_headline: float = 0.4  # headline省略权重
    omission_weight_lede: float = 0.4      # lede省略权重
    omission_weight_full: float = 0.2      # 全文省略权重
    
    # 证据提取配置
    max_evidence_count: int = 5       # 最大证据数量
    max_examples_per_evidence: int = 3 # 每个证据的最大例子数
    
    # 实体识别配置
    use_spacy: bool = True           # 是否使用spaCy进行实体识别
    entity_types: List[str] = None   # 关注的实体类型
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = ['PERSON', 'ORG', 'GPE', 'EVENT']

@dataclass
class AnalyzerConfig:
    """主配置类"""
    processing: ProcessingConfig = None
    teacher: TeacherConfig = None
    scoring: ScoringConfig = None
    output: OutputConfig = None
    relative_framing: RelativeFramingConfig = None
    omission: OmissionConfig = None  # 新增省略感知配置
    
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
        **{k: v for k, v in config_dict.items() 
           if k not in ['processing', 'teacher', 'scoring', 'output', 'relative_framing', 'omission']}
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