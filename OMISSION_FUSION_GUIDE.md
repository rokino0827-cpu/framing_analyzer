# 省略检测融合功能使用指南

## 概述

本指南详细说明如何启用和使用省略检测功能，以及如何将省略分数融合到最终的framing_intensity评分中。

## 两步实施方法

### 步骤1：启用省略检测

确保省略检测功能正确启用并产生预期字段。

#### 配置设置

```python
from framing_analyzer import AnalyzerConfig, create_analyzer

# 创建配置
config = AnalyzerConfig()

# 基础配置
config.teacher.bias_class_index = 1  # 根据verify_bias_class.py的结果设置
config.teacher.model_local_path = "bias_detector_data"

# 启用省略检测
config.omission.enabled = True
config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"  # 本地模型路径
config.omission.key_topics_count = 10
config.omission.similarity_threshold = 0.5
config.omission.guidance_threshold = 0.3
config.omission.min_topic_frequency = 2
config.omission.fusion_weight = 0.2  # 融合权重（步骤2）

# 创建分析器
analyzer = create_analyzer(config)
```

#### 验证省略检测启用

运行验证脚本：

```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/test_omission_enabled.py
```

预期输出应包含：
- ✅ Omission detector initialized
- ✅ Omission fields present in results
- 每篇文章结果中包含 `omission_score` 和 `omission_evidence` 字段

### 步骤2：线性融合实现

将省略分数融合到最终的framing_intensity中。

#### 融合公式

```
final_intensity = (1 - α) * framing_intensity + α * omission_intensity
```

其中：
- `α` 是融合权重 (`config.omission.fusion_weight`)
- `framing_intensity` 是基础框架强度分数
- `omission_intensity` 是省略检测分数

#### 融合权重推荐

| 权重范围 | 适用场景 | 说明 |
|---------|---------|------|
| 0.1-0.15 | 轻微增强 | 省略检测提供微妙但有价值的信号 |
| 0.15-0.25 | 平衡融合 | 推荐的默认范围，平衡两种信号 |
| 0.25-0.35 | 重视省略 | 省略检测对该数据集特别有价值 |
| >0.35 | 需要检查 | 权重过高，建议检查省略检测配置 |

## 测试和验证

### 1. 基础功能测试

```bash
# 测试省略检测融合功能
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/test_omission_fusion.py
```

### 2. 全面测试

```bash
# 启用省略检测的全面测试
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py --sample 25 --enable-omission
```

### 3. 融合权重优化

```bash
# 优化融合权重
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/optimize_fusion_weight.py --sample 25 --weight-range 0.1,0.3,0.05
```

### 4. 运行所有测试

```bash
# 运行完整测试套件
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/run_all_tests.py --sample 20
```

## 结果验证

### 预期输出字段

启用省略检测后，每篇文章的分析结果应包含：

```json
{
  "id": "article_id",
  "framing_intensity": 0.456,  // 融合后的最终分数
  "omission_score": 0.234,     // 省略检测分数
  "omission_evidence": [       // 省略证据
    {
      "missing_topic": "healthcare funding",
      "evidence_articles": ["article_2", "article_3"],
      "coverage_score": 0.8
    }
  ],
  "statistics": {
    "omission_score": 0.234,
    "key_topics_missing_count": 2,
    "key_topics_covered_count": 8,
    "omission_locations_count": 3
  }
}
```

### 融合效果验证

1. **分数变化**：比较启用/禁用省略检测时的framing_intensity分数
2. **相关性**：检查与ground truth的相关性是否提升
3. **分布质量**：确保分数分布合理，不过于集中或分散

## 性能考虑

### 计算开销

- 省略检测会增加约50-100%的计算时间
- 主要开销来自：
  - 事件聚类（TF-IDF + 聚类算法）
  - 嵌入模型推理（sentence-transformers）
  - 图构建和分析

### 优化建议

1. **批量处理**：一次处理多篇文章以摊销聚类成本
2. **缓存嵌入**：对重复文本缓存嵌入向量
3. **参数调优**：
   - 减少 `key_topics_count` 以降低计算复杂度
   - 调整 `similarity_threshold` 以控制图的密度

## 故障排除

### 常见问题

1. **省略字段缺失**
   - 检查 `config.omission.enabled = True`
   - 确认嵌入模型路径正确
   - 验证文章数量足够进行聚类（至少3-5篇）

2. **融合权重无效果**
   - 检查 `compute_framing_intensity` 方法是否接收省略配置
   - 验证省略分数不为None
   - 确认融合公式实现正确

3. **性能问题**
   - 减少样本数量进行测试
   - 检查GPU内存使用情况
   - 考虑使用更小的嵌入模型

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查省略检测器状态
if hasattr(analyzer, 'omission_detector') and analyzer.omission_detector:
    print("✅ Omission detector initialized")
else:
    print("❌ Omission detector not initialized")

# 验证配置
print(f"Omission enabled: {config.omission.enabled}")
print(f"Fusion weight: {config.omission.fusion_weight}")
print(f"Embedding model: {config.omission.embedding_model_name_or_path}")
```

## 最佳实践

1. **渐进式启用**：先在小样本上测试，确认功能正常后再扩展
2. **权重调优**：使用优化脚本找到最适合数据集的融合权重
3. **结果验证**：定期检查省略检测结果的合理性
4. **性能监控**：跟踪计算时间和资源使用情况

## 配置模板

### 生产环境配置

```python
config = AnalyzerConfig()

# 基础配置
config.teacher.bias_class_index = 1  # 根据实际模型调整
config.teacher.model_local_path = "bias_detector_data"
config.teacher.batch_size = 16

# 省略检测配置
config.omission.enabled = True
config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
config.omission.fusion_weight = 0.2  # 根据优化结果调整
config.omission.key_topics_count = 10
config.omission.similarity_threshold = 0.5
config.omission.guidance_threshold = 0.3
config.omission.min_topic_frequency = 2

# 输出配置
config.output.include_evidence = True
config.output.include_statistics = True
config.output.generate_plots = False  # 生产环境关闭图表生成
```

### 开发测试配置

```python
config = AnalyzerConfig()

# 基础配置
config.teacher.bias_class_index = 1
config.teacher.model_local_path = "bias_detector_data"
config.teacher.batch_size = 8  # 较小批次用于调试

# 省略检测配置（调试友好）
config.omission.enabled = True
config.omission.embedding_model_name_or_path = "all-MiniLM-L6-v2"
config.omission.fusion_weight = 0.2
config.omission.key_topics_count = 8  # 减少计算量
config.omission.similarity_threshold = 0.4  # 稍微宽松
config.omission.max_evidence_count = 3  # 减少输出量

# 输出配置（详细调试）
config.output.include_evidence = True
config.output.include_statistics = True
config.output.include_raw_scores = True  # 调试时启用
config.output.generate_plots = True
config.verbose = True
config.log_level = "DEBUG"
```

## 总结

省略检测融合功能通过两步实施：
1. **启用省略检测**：确保功能正确初始化并产生省略相关字段
2. **线性融合**：将省略分数按配置权重融合到最终framing_intensity中

正确实施后，系统将能够检测文章中的信息省略，并将这种省略模式作为框架偏见的重要指标融入最终评分。