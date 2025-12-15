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
from .config import AnalyzerConfig, ProcessingConfig, TeacherConfig, ScoringConfig, OutputConfig
from .text_processor import TextProcessor, StructureZoneExtractor, FragmentGenerator
from .bias_teacher import BiasTeacher, TeacherInference
from .framing_scorer import FramingScorer, PseudoLabelGenerator, EvidenceExtractor, FramingAnalysisEngine
from .analyzer import FramingAnalyzer
from .relative_framing import RelativeFramingAnalyzer
from .utils import setup_environment, validate_input_data, save_results

# 便捷函数
def create_analyzer(config=None):
    """创建框架偏见分析器实例"""
    if config is None:
        config = AnalyzerConfig()
    return FramingAnalyzer(config)

def quick_analyze(text, title=""):
    """快速分析单篇文章"""
    analyzer = create_analyzer()
    return analyzer.analyze_article(text, title)

def batch_analyze(articles, output_path=None):
    """批量分析文章"""
    analyzer = create_analyzer()
    return analyzer.analyze_batch(articles, output_path)

def analyze_csv(csv_path, output_path=None, content_column='content', title_column='title'):
    """从CSV文件分析文章"""
    analyzer = create_analyzer()
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

# 导出的公共API
__all__ = [
    # 配置
    'AnalyzerConfig', 'ProcessingConfig', 'TeacherConfig', 'ScoringConfig', 'OutputConfig',
    
    # 核心组件
    'TextProcessor', 'StructureZoneExtractor', 'FragmentGenerator',
    'BiasTeacher', 'TeacherInference', 
    'FramingScorer', 'PseudoLabelGenerator', 'EvidenceExtractor', 'FramingAnalysisEngine',
    'FramingAnalyzer', 'RelativeFramingAnalyzer',
    
    # 工具函数
    'setup_environment', 'validate_input_data', 'save_results',
    
    # 便捷函数
    'create_analyzer', 'quick_analyze', 'batch_analyze', 'analyze_csv',
    'create_quick_config', 'create_high_precision_config', 'create_fast_config'
]