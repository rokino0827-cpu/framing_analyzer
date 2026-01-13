# SV2000框架对齐使用指南

## 概述

本指南介绍如何使用新的SV2000框架对齐功能，该功能基于Semetko & Valkenburg (2000)的学术框架定义，提供更准确和标准化的新闻框架分析。

## 核心特性

### 🎯 SV2000框架预测
- **5个标准框架**：冲突、人情、经济、道德、责任
- **学术标准**：基于Semetko & Valkenburg (2000)定义
- **量化输出**：每个框架的0-1分数

### 🔄 多组件融合
- **主要信号**：SV2000框架预测
- **辅助特征**：偏见检测、省略检测、相对框架、引用分析
- **智能融合**：Ridge回归优化权重

### 📊 向后兼容
- **双模式运行**：SV2000模式 vs 传统模式
- **渐进升级**：可选择性启用新功能
- **现有接口**：保持API兼容性

## 快速开始

### 1. 基础使用

```python
from framing_analyzer import FramingAnalyzer
from framing_analyzer.config import create_sv2000_config

# 创建SV2000配置
config = create_sv2000_config()

# 初始化分析器
analyzer = FramingAnalyzer(config)

# 分析单篇文章
article_text = "Your news article content here..."
result = analyzer.analyze_article(article_text, title="Article Title")

# 查看SV2000框架分数
print(f"冲突框架: {result.sv_conflict:.3f}")
print(f"人情框架: {result.sv_human:.3f}")
print(f"经济框架: {result.sv_econ:.3f}")
print(f"道德框架: {result.sv_moral:.3f}")
print(f"责任框架: {result.sv_resp:.3f}")
print(f"框架平均: {result.sv_frame_avg:.3f}")
print(f"最终强度: {result.framing_intensity:.3f}")
```

### 2. 批量分析

```python
# 准备文章数据
articles = [
    {
        'content': 'Article 1 content...',
        'title': 'Article 1 Title',
        'id': 'article_1'
    },
    {
        'content': 'Article 2 content...',
        'title': 'Article 2 Title', 
        'id': 'article_2'
    }
]

# 批量分析
results = analyzer.analyze_batch(articles, output_path="sv2000_results.json")

# 查看结果
for result in results['results']:
    print(f"文章: {result['title']}")
    print(f"SV2000分数: {result.get('sv_frame_avg', 'N/A')}")
    print(f"融合强度: {result['framing_intensity']}")
    print("---")
```

## 配置选项

### SV2000框架配置

```python
from framing_analyzer.config import AnalyzerConfig, SVFramingConfig, FusionConfig

config = AnalyzerConfig()

# SV2000框架配置
config.sv_framing = SVFramingConfig(
    enabled=True,                    # 启用SV2000模式
    encoder_name="bge_m3",           # 编码器模型（默认BGE M3）
    encoder_local_path=None,         # 本地模型路径（可选）
    hidden_size=1024,                # 隐藏层大小
    dropout_rate=0.1,                # Dropout率
    learning_rate=2e-5,              # 学习率
    training_mode="frame_level",     # 训练模式：frame_level 或 item_level
    device="auto",                   # 计算设备：auto, cpu, cuda
    batch_size=16,                   # 批处理大小
    max_length=512,                  # 最大序列长度
    model_save_path="./sv2000_models",  # 模型保存路径
    pretrained_model_path=None       # 预训练模型路径（可选）
)

# 融合配置
config.fusion = FusionConfig(
    alpha=0.5,                       # SV2000权重
    beta=0.2,                        # 偏见检测权重
    gamma=0.15,                      # 省略检测权重
    delta=0.1,                       # 相对框架权重
    epsilon=0.05,                    # 引用分析权重
    enforce_positive_weights=True,   # 强制非负权重
    normalize_weights=True,          # 权重归一化
    use_ridge_optimization=True,     # 使用Ridge回归优化
    ridge_alpha=1.0,                 # Ridge正则化参数
    cross_validation_folds=5         # 交叉验证折数
)
```

### 模式切换

```python
# 启用SV2000模式
config.enable_sv2000_mode()

# 禁用SV2000模式（传统模式）
config.disable_sv2000_mode()

# 检查当前模式
is_sv2000_enabled = config.sv_framing.enabled
```

## 训练自定义模型

### 1. 准备训练数据

训练数据应为CSV格式，包含以下列：
- `content`: 文章内容
- `y_conflict`: 冲突框架分数 (0-1)
- `y_human`: 人情框架分数 (0-1)
- `y_econ`: 经济框架分数 (0-1)
- `y_moral`: 道德框架分数 (0-1)
- `y_resp`: 责任框架分数 (0-1)

可选列：
- `title`: 文章标题
- `item_1` 到 `item_20`: 单独的问卷条目分数

### 2. 训练模型

```python
from framing_analyzer.sv2000_trainer import SV2000TrainingPipeline
from framing_analyzer.config import create_sv2000_config

# 创建训练配置
config = create_sv2000_config()
config.sv_framing.learning_rate = 2e-5
config.sv_framing.batch_size = 16
config.sv_framing.training_mode = "frame_level"

# 初始化训练管道
trainer = SV2000TrainingPipeline(config, "path/to/training_data.csv")

# 运行完整训练
training_report = trainer.run_full_training(num_epochs=10)

print("训练完成！")
print(f"最终验证相关性: {training_report['final_metrics'].get('avg_correlation', 'N/A')}")
print(f"优化后的融合权重: {training_report['optimized_weights']}")
```

### 3. 使用训练好的模型

```python
# 加载训练好的模型
config.sv_framing.pretrained_model_path = "./sv2000_models/best_sv2000_model.pt"

# 使用自定义模型进行分析
analyzer = FramingAnalyzer(config)
result = analyzer.analyze_article("Your article content...")
```

## 评估和验证

### 1. 模型性能评估

```python
from framing_analyzer.sv2000_evaluator import SV2000Evaluator

# 初始化评估器
evaluator = SV2000Evaluator()

# 准备预测和真实标签
predictions = {
    'sv_conflict_pred': [0.3, 0.7, 0.2],
    'sv_human_pred': [0.5, 0.4, 0.8],
    'sv_econ_pred': [0.2, 0.9, 0.1],
    'sv_moral_pred': [0.6, 0.3, 0.7],
    'sv_resp_pred': [0.4, 0.6, 0.5],
    'sv_frame_avg_pred': [0.4, 0.58, 0.46]
}

ground_truth = {
    'y_conflict': [0.2, 0.8, 0.1],
    'y_human': [0.6, 0.3, 0.9],
    'y_econ': [0.1, 0.9, 0.2],
    'y_moral': [0.7, 0.2, 0.8],
    'y_resp': [0.3, 0.7, 0.4]
}

# 评估框架对齐
alignment_results = evaluator.evaluate_frame_alignment(predictions, ground_truth)

print(f"整体对齐分数: {alignment_results['overall_alignment_score']:.3f}")

# 生成评估报告
report = evaluator.generate_evaluation_report(alignment_results, "evaluation_report.md")
print(report)
```

### 2. 融合性能分析

```python
# 准备融合结果数据
fusion_results = [
    {
        'final_intensity': 0.6,
        'sv_frame_avg_pred': 0.5,
        'bias_score': 0.4,
        'omission_score': 0.2,
        'relative_score': 0.1,
        'quote_score': 0.3
    },
    # ... 更多结果
]

ground_truth_intensity = [0.65, 0.45, 0.75]  # 真实强度分数

# 评估融合性能
fusion_performance = evaluator.evaluate_fusion_performance(fusion_results, ground_truth_intensity)

print("组件贡献分析:")
for component, metrics in fusion_performance['component_analysis'].items():
    print(f"  {component}: 相关性 {metrics['correlation_with_final']:.3f}")

print("性能比较:")
for method, metrics in fusion_performance['performance_comparison'].items():
    if isinstance(metrics, dict) and 'pearson_r' in metrics:
        print(f"  {method}: 相关性 {metrics['pearson_r']:.3f}, MAE {metrics['mae']:.3f}")
```

## 输出格式

### SV2000模式输出

```json
{
  "id": "article_1",
  "title": "Article Title",
  "framing_intensity": 0.65,
  "pseudo_label": "positive",
  
  // SV2000框架分数
  "sv_conflict": 0.3,
  "sv_human": 0.7,
  "sv_econ": 0.2,
  "sv_moral": 0.8,
  "sv_resp": 0.5,
  "sv_frame_avg": 0.5,
  
  // 融合信息
  "fusion_weights": {
    "sv_frame_avg_pred": 0.5,
    "bias_score": 0.2,
    "omission_score": 0.15,
    "relative_score": 0.1,
    "quote_score": 0.05
  },
  
  "component_contributions": {
    "sv_frame_avg_pred": 0.25,
    "bias_score": 0.08,
    "omission_score": 0.03,
    "relative_score": 0.02,
    "quote_score": 0.01
  },
  
  // 传统字段（保持兼容性）
  "components": {
    "headline": 0.4,
    "lede": 0.6,
    "narration": 0.5,
    "quotes": 0.3
  },
  
  "evidence": [
    {
      "rank": 1,
      "text": "Evidence sentence...",
      "bias_score": 0.8,
      "zone": "headline"
    }
  ],
  
  "statistics": {
    "sv2000_mode": true,
    "fusion_applied": true,
    "total_fragments": 15,
    "zones_with_content": 4
  }
}
```

## 最佳实践

### 1. 数据准备
- **质量优先**：确保训练数据质量高于数量
- **平衡分布**：各框架分数分布相对平衡
- **文本长度**：避免过短（<50字符）或过长（>10000字符）的文本

### 2. 模型训练
- **渐进学习**：从较小的学习率开始
- **早停机制**：使用验证集防止过拟合
- **权重优化**：使用足够的验证数据优化融合权重

### 3. 性能优化
- **批处理**：使用适当的批处理大小
- **设备选择**：GPU加速训练，CPU推理也可接受
- **内存管理**：大批量数据时注意内存使用

### 4. 评估验证
- **多指标评估**：不仅看相关性，也要看误差指标
- **交叉验证**：使用交叉验证确保结果稳定性
- **对比分析**：与传统方法对比验证改进效果

## 故障排除

### 常见问题

1. **模型加载失败**
   ```python
   # 检查模型路径
   import os
   print(os.path.exists(config.sv_framing.encoder_local_path))
   
   # 使用在线模型
   config.sv_framing.encoder_local_path = None
   ```

2. **CUDA内存不足**
   ```python
   # 减小批处理大小
   config.sv_framing.batch_size = 8
   
   # 使用CPU
   config.sv_framing.device = "cpu"
   ```

3. **训练数据格式错误**
   ```python
   # 验证数据格式
   from framing_analyzer.sv2000_data_loader import SV2000DataLoader
   
   loader = SV2000DataLoader("data.csv", config)
   validation_results = loader.validate_annotation_format()
   print(validation_results)
   ```

4. **融合权重异常**
   ```python
   # 检查融合配置
   print(config.fusion.enforce_positive_weights)
   print(config.fusion.normalize_weights)
   
   # 手动设置权重
   config.fusion.alpha = 0.6  # 增加SV2000权重
   ```

### 性能调优

1. **提高准确性**
   - 增加训练数据量
   - 调整学习率和训练轮数
   - 优化融合权重

2. **提高速度**
   - 使用更小的模型
   - 增加批处理大小
   - 使用GPU加速

3. **减少内存使用**
   - 减小批处理大小
   - 使用梯度累积
   - 清理不必要的中间结果

## 更新日志

### v1.0.0 (2024-12)
- ✅ 初始SV2000框架对齐实现
- ✅ 多组件融合评分器
- ✅ 训练和评估框架
- ✅ 向后兼容性支持

### 计划功能
- 🔄 更多预训练模型
- 🔄 增强的数据增强
- 🔄 实时模型更新
- 🔄 可视化分析工具

## 支持和反馈

如有问题或建议，请：
1. 查看本指南的故障排除部分
2. 检查配置和数据格式
3. 提交issue或联系开发团队

---

*本指南将随着功能更新持续完善。*
