# Framing Analyzer Test Suite

这个目录包含了完整的测试套件，用于验证和评估framing_analyzer的各项功能。

## 测试脚本概览

### 🚀 快速测试
**`quick_test.py`** - 日常开发和调试用的快速测试
```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/quick_test.py
```
- ✅ 验证基本功能
- ✅ 测试少量文章（3-5篇）
- ✅ 快速反馈（<30秒）
- ✅ 内置测试数据

### 🧪 全面测试
**`comprehensive_test.py`** - 完整功能测试
```bash
# 基础测试（50篇文章）
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py

# 启用所有功能
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py \
    --sample 100 \
    --enable-omission \
    --enable-relative \
    --config-bias-index 1

# 全数据集测试（谨慎使用）
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py --full
```

**功能特性：**
- ✅ 基础框架偏见分析
- ✅ 省略检测功能（可选）
- ✅ 相对框架分析（可选）
- ✅ Ground truth对比评估
- ✅ 详细性能统计
- ✅ 错误处理和报告

**参数说明：**
- `--sample N`: 测试前N篇文章（默认50）
- `--full`: 测试全部数据（可能很慢）
- `--enable-omission`: 启用省略检测
- `--enable-relative`: 启用相对框架分析
- `--output-dir DIR`: 输出目录
- `--config-bias-index N`: 设置bias_class_index

### 📊 性能基准测试
**`benchmark_test.py`** - 性能评估和优化
```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/benchmark_test.py
```
- ✅ 多种配置性能对比
- ✅ 不同数据量测试（1, 5, 10, 20, 50篇）
- ✅ 吞吐量统计（articles/second）
- ✅ 配置优化建议

### 🔧 配置验证工具
**`verify_bias_class.py`** - 验证bias_class_index
```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/verify_bias_class.py
```

**`determine_bias_class.py`** - 详细的bias类别分析
```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/determine_bias_class.py
```

## 数据要求

### 主要数据文件
```
data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv
```

**必需字段：**
- `title`: 文章标题
- `content`: 文章内容

**可选字段（用于评估）：**
- `bias_label`: Ground truth偏见标签
- `bias_probability`: Ground truth偏见概率
- `publication`: 媒体来源
- `date`: 发布日期

### 数据格式示例
```csv
title,content,publication,date,bias_label,bias_probability
"Economic Policy Update","The government announced...","Reuters","2025-01-01",0,0.2
"Breaking News Alert","Sources report that...","CNN","2025-01-02",1,0.8
```

## 输出结果

### 测试结果目录结构
```
results/
├── quick_test/                 # 快速测试结果
├── comprehensive_test/         # 全面测试结果
│   ├── comprehensive_test_results.json
│   ├── framing_analysis_results.json
│   └── plots/                  # 可视化图表
└── benchmark/                  # 性能基准结果
    └── benchmark_results.json
```

### 结果文件说明

**`comprehensive_test_results.json`** - 测试摘要
```json
{
  "data_stats": {
    "total_articles": 50,
    "avg_content_length": 1250.5,
    "has_ground_truth": 45
  },
  "basic_analysis": {
    "success": true,
    "analysis_time": 12.34,
    "framing_intensity_stats": {
      "mean": 0.456,
      "std": 0.123
    }
  },
  "evaluation": {
    "correlation": 0.678
  }
}
```

**分析结果字段说明：**
- `framing_intensity`: 框架偏见强度 (0.0-1.0)
- `pseudo_label`: 伪标签 ("positive", "negative", "uncertain")
- `components`: 各组件分数 (headline, lede, narration, quotes)
- `evidence`: 证据片段列表
- `statistics`: 统计信息
- `omission_score`: 省略分数（如果启用）
- `omission_evidence`: 省略证据（如果启用）

**`benchmark_results.json`** - 性能数据
```json
{
  "configs": {
    "fast": {
      "results": {
        "50": {
          "total_time": 8.5,
          "articles_per_second": 5.9,
          "avg_framing_intensity": 0.456
        }
      }
    }
  }
}
```

## 调试工具

### 🔍 结果结构调试
**`debug_result_structure.py`** - 检查返回值结构
```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/debug_result_structure.py
```
- ✅ 显示返回值的完整结构
- ✅ 保存结果到JSON文件
- ✅ 帮助理解字段名和数据类型

## 使用建议

### 开发阶段
1. **日常开发**: 使用 `quick_test.py` 验证基本功能
2. **功能测试**: 使用 `comprehensive_test.py --sample 20` 测试新功能
3. **性能优化**: 使用 `benchmark_test.py` 评估性能改进

### 部署前验证
1. **完整测试**: `comprehensive_test.py --sample 100 --enable-omission`
2. **性能验证**: `benchmark_test.py` 确保性能满足要求
3. **配置验证**: `verify_bias_class.py` 确认模型配置

### 生产环境监控
1. **定期测试**: 定期运行 `comprehensive_test.py` 监控系统健康
2. **性能监控**: 使用 `benchmark_test.py` 监控性能退化
3. **数据质量**: 检查ground truth相关性是否稳定

## 故障排除

### 常见问题

**1. 数据文件不存在**
```
FileNotFoundError: 数据文件不存在
```
- 检查数据文件路径
- 使用内置测试数据（quick_test.py会自动处理）

**2. 模型加载失败**
```
Failed to load model: bias_detector_data
```
- 确认模型文件存在
- 检查 `bias_detector_data` 目录
- 运行 `verify_bias_class.py` 验证模型

**3. CUDA内存不足**
```
RuntimeError: CUDA out of memory
```
- 减少batch_size: `--config-bias-index 1` 并修改配置
- 减少测试样本: `--sample 10`
- 使用CPU: 修改配置中的device设置

**4. 省略检测失败**
```
Omission analysis failed
```
- 确认依赖安装: `sentence-transformers`, `spacy`
- 检查网络连接（首次下载模型）
- 暂时禁用: 不使用 `--enable-omission`

### 性能优化建议

**提高速度:**
- 使用更大的batch_size（如果内存允许）
- 关闭不必要的功能（plots, evidence详情）
- 使用GPU加速

**提高准确性:**
- 增加evidence_count
- 启用省略检测
- 使用更多测试样本进行阈值拟合

**平衡配置:**
- batch_size=16, evidence_count=5
- 启用基础功能，按需启用高级功能
- 监控内存和时间消耗

## 扩展测试

### 添加新测试
1. 在相应脚本中添加测试函数
2. 更新配置选项
3. 添加结果统计
4. 更新文档

### 自定义数据集
1. 准备CSV文件（符合格式要求）
2. 修改脚本中的数据路径
3. 调整采样策略
4. 验证结果格式

### 集成CI/CD
```bash
# 在CI管道中运行
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/quick_test.py
if [ $? -eq 0 ]; then
    echo "✅ Tests passed"
else
    echo "❌ Tests failed"
    exit 1
fi
```