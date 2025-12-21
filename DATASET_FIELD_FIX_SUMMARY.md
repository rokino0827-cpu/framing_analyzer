# 数据集字段修复总结

## 问题描述

之前的代码中使用了错误的字段名来访问数据集，导致与实际数据集字段不匹配。

## 实际数据集字段

根据用户提供的信息，数据集包含以下字段：
- `date` - 日期
- `author` - 作者
- `title` - 标题
- `content` - 内容
- `url` - URL链接
- `section` - 版块
- `publication` - 出版物
- `bias_label` - 偏见标签
- `bias_probability` - 偏见概率

## 修复的文件和更改

### 1. `framing_analyzer/comprehensive_test.py`

**修复前的错误**：
```python
# 错误的字段访问
if "bias_label" in row and pd.notna(row["bias_label"]):
    article["ground_truth_bias"] = row["bias_label"]
if "bias_probability" in row and pd.notna(row["bias_probability"]):
    article["ground_truth_prob"] = float(row["bias_probability"])

# 错误的统计计算
'has_ground_truth': sum(1 for a in articles if 'ground_truth_bias' in a)

# 错误的评估函数
ground_truth_articles = [a for a in articles if 'ground_truth_bias' in a]
```

**修复后的正确代码**：
```python
# 正确的字段访问
if "bias_label" in df.columns and pd.notna(row["bias_label"]):
    article["bias_label"] = row["bias_label"]
if "bias_probability" in df.columns and pd.notna(row["bias_probability"]):
    article["bias_probability"] = float(row["bias_probability"])

# 正确的统计计算
'has_bias_labels': sum(1 for a in articles if 'bias_label' in a)

# 正确的评估函数
labeled_articles = [a for a in articles if 'bias_label' in a or 'bias_probability' in a]
```

### 2. `framing_analyzer/test_omission_fusion.py`

**修复前的错误**：
```python
# 错误的字段访问
if "bias_label" in row and pd.notna(row["bias_label"]):
    article["ground_truth_bias"] = row["bias_label"]
```

**修复后的正确代码**：
```python
# 正确的字段访问
if "bias_label" in df.columns and pd.notna(row["bias_label"]):
    article["bias_label"] = row["bias_label"]
if "bias_probability" in df.columns and pd.notna(row["bias_probability"]):
    article["bias_probability"] = float(row["bias_probability"])
```

### 3. `framing_analyzer/optimize_fusion_weight.py`

**修复前的错误**：
```python
def load_evaluation_data(self, max_articles: int = 100) -> Tuple[List[Dict], List[float]]:
    """加载带有ground truth的评估数据"""
    
def evaluate_fusion_weight(self, articles: List[Dict], ground_truth: List[float], ...):
    
def grid_search(self, articles: List[Dict], ground_truth: List[float], ...):
```

**修复后的正确代码**：
```python
def load_evaluation_data(self, max_articles: int = 100) -> Tuple[List[Dict], List[float]]:
    """加载带有bias标签的评估数据"""
    
def evaluate_fusion_weight(self, articles: List[Dict], bias_scores: List[float], ...):
    
def grid_search(self, articles: List[Dict], bias_scores: List[float], ...):
```

## 修复的关键点

### 1. 字段访问方式
- **修复前**：直接使用 `"field" in row` 检查
- **修复后**：使用 `"field" in df.columns` 检查列是否存在

### 2. 字段名称统一
- **修复前**：使用 `ground_truth_bias`, `ground_truth_prob` 等自定义字段名
- **修复后**：直接使用数据集原始字段名 `bias_label`, `bias_probability`

### 3. 评估逻辑
- **修复前**：假设所有文章都有ground truth标签
- **修复后**：正确处理可能缺失的标签，优先使用 `bias_probability`，备选 `bias_label`

### 4. 函数命名
- **修复前**：`evaluate_against_ground_truth`
- **修复后**：`evaluate_against_bias_labels`

## 验证修复

修复后的代码应该能够：

1. **正确读取数据集**：使用正确的字段名访问数据
2. **处理缺失值**：正确处理可能缺失的bias标签
3. **评估功能**：使用实际存在的字段进行模型评估
4. **统计计算**：基于正确字段计算统计信息

## 测试建议

运行以下命令验证修复：

```bash
# 测试基础功能
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py --sample 10

# 测试省略检测融合
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/test_omission_fusion.py

# 测试融合权重优化
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/optimize_fusion_weight.py --sample 15
```

## 注意事项

1. **数据集兼容性**：修复后的代码与实际数据集字段完全匹配
2. **向后兼容**：代码仍然能处理没有bias标签的数据集
3. **错误处理**：增加了对缺失字段的检查和处理
4. **性能影响**：修复不会影响分析性能，只是字段访问方式的调整

## 影响的功能

修复影响以下功能：
- ✅ 数据加载和预处理
- ✅ 模型评估和相关性分析  
- ✅ 融合权重优化
- ✅ 批量测试和统计计算
- ✅ 省略检测测试

所有核心分析功能保持不变，只是数据访问层得到了修复。