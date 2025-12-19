"""
框架偏见分析器 (Framing Bias Analyzer)
基于bias_detector的无训练框架偏见检测系统

使用原版bias_detector (512 token) 作为teacher模型，
通过结构化分析生成文章级framing强度分数和证据片段。
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# 核心组件导入
from .config import AnalyzerConfig, ProcessingConfig, TeacherConfig, ScoringConfig, OutputConfig, OmissionConfig
from .text_processor import TextProcessor, StructureZoneExtractor, FragmentGenerator
from .bias_teacher import BiasTeacher, TeacherInference
from .framing_scorer import FramingScorer, PseudoLabelGenerator, EvidenceExtractor, FramingAnalysisEngine
from .analyzer import FramingAnalyzer
from .relative_framing import RelativeFramingAnalyzer
from .omission_detector import OmissionDetector, OmissionResult
from .omission_graph import OmissionGraph, GraphNode, GraphEdge, OmissionAwareGraphBuilder
from .utils import setup_environment, validate_input_data, save_results

# 便捷函数
def create_analyzer(config=None, enable_omission=False):
    """创建框架偏见分析器实例
    
    Args:
        config: 分析器配置，如果为None则使用默认配置
        enable_omission: 是否启用省略检测功能
    """
    if config is None:
        config = AnalyzerConfig()
        if enable_omission and hasattr(config, 'omission'):
            config.omission.enabled = True
    return FramingAnalyzer(config)

def quick_analyze(text, title="", enable_omission=False):
    """快速分析单篇文章"""
    analyzer = create_analyzer(enable_omission=enable_omission)
    return analyzer.analyze_article(text, title)

def batch_analyze(articles, output_path=None, enable_omission=False):
    """批量分析文章"""
    analyzer = create_analyzer(enable_omission=enable_omission)
    return analyzer.analyze_batch(articles, output_path)

def analyze_csv(csv_path, output_path=None, content_column='content', title_column='title', enable_omission=False):
    """从CSV文件分析文章"""
    analyzer = create_analyzer(enable_omission=enable_omission)
    return analyzer.analyze_from_csv(csv_path, content_column, title_column, output_path=output_path)

# 配置创建函数
def create_quick_config(**kwargs):
    """创建快速配置"""
    from .config import create_quick_config
    return create_quick_config(**kwargs)

def create_high_precision_config():
    """创建高精度配置"""
    from .config import create_high_precision_config
    return create_high_precision_config()

def create_fast_config():
    """创建快速处理配置"""
    from .config import create_fast_config
    return create_fast_config()

def create_omission_enabled_config():
    """创建启用省略检测的配置"""
    config = AnalyzerConfig()
    if hasattr(config, 'omission'):
        config.omission.enabled = True
    return config

# 导出的公共API
__all__ = [
    # 配置
    'AnalyzerConfig', 'ProcessingConfig', 'TeacherConfig', 'ScoringConfig', 'OutputConfig', 'OmissionConfig',
    
    # 核心组件
    'TextProcessor', 'StructureZoneExtractor', 'FragmentGenerator',
    'BiasTeacher', 'TeacherInference', 
    'FramingScorer', 'PseudoLabelGenerator', 'EvidenceExtractor', 'FramingAnalysisEngine',
    'FramingAnalyzer', 'RelativeFramingAnalyzer',
    
    # 省略检测组件
    'OmissionDetector', 'OmissionResult', 'OmissionGraph', 'GraphNode', 'GraphEdge', 'OmissionAwareGraphBuilder',
    
    # 工具函数
    'setup_environment', 'validate_input_data', 'save_results',
    
    # 便捷函数
    'create_analyzer', 'quick_analyze', 'batch_analyze', 'analyze_csv',
    'create_quick_config', 'create_high_precision_config', 'create_fast_config', 'create_omission_enabled_config'
]