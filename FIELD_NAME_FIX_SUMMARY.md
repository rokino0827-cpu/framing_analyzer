# 字段名修复总结

## 问题描述

测试脚本中出现错误：
```
'dict' object has no attribute 'framing_score'
```

## 根本原因

1. **返回值类型**: 分析器返回的是字典格式，不是对象
2. **字段名不匹配**: 实际字段名是 `framing_intensity`，不是 `framing_score`
3. **数据结构**: 结果在 `results['results']` 数组中，每个元素是字典

## 正确的字段名

### 主要字段
- ✅ `framing_intensity` - 框架偏见强度 (0.0-1.0)
- ✅ `pseudo_label` - 伪标签 ("positive", "negative", "uncertain")
- ✅ `components` - 各组件分数字典
- ✅ `evidence` - 证据片段列表
- ✅ `statistics` - 统计信息字典

### 省略检测字段（可选）
- ✅ `omission_score` - 省略分数
- ✅ `omission_evidence` - 省略证据列表

### 错误的字段名（已修复）
- ❌ `framing_score` → ✅ `framing_intensity`
- ❌ `bias_intensity` → ✅ `pseudo_label`
- ❌ `article_id` → ✅ `id`

## 修复的文件

### 1. `comprehensive_test.py`
- ✅ 修复 `load_data()` - 添加所有CSV字段支持
- ✅ 修复 `test_basic_analysis()` - 正确访问字典字段
- ✅ 修复 `test_omission_analysis()` - 正确检查省略结果
- ✅ 修复 `evaluate_against_ground_truth()` - 正确映射结果
- ✅ 修复 `print_summary()` - 显示正确的统计信息

### 2. `quick_test.py`
- ✅ 修复结果显示逻辑
- ✅ 修复省略检测测试
- ✅ 添加兼容性检查（字典 vs 对象）

### 3. `benchmark_test.py`
- ✅ 修复性能统计计算
- ✅ 更新字段名引用

### 4. `README_TEST_SUITE.md`
- ✅ 更新文档中的字段名
- ✅ 添加正确的JSON示例
- ✅ 添加调试工具说明

## 新增调试工具

### `debug_result_structure.py`
- 🔍 检查实际返回值结构
- 💾 保存完整结果到JSON文件
- 📊 显示字段名和数据类型
- 🛠️ 帮助开发者理解数据格式

## 数据集字段支持

现在支持完整的CSV字段：
```csv
date,author,title,content,url,section,publication,bias_label,bias_probability
```

### 必需字段
- `title` - 文章标题
- `content` - 文章内容

### 可选字段
- `date` - 发布日期
- `author` - 作者
- `url` - 文章链接（用作ID）
- `section` - 版块
- `publication` - 媒体来源
- `bias_label` - Ground truth偏见标签
- `bias_probability` - Ground truth偏见概率

## 兼容性处理

测试脚本现在同时支持：
1. **字典格式** (当前实际格式)
2. **对象格式** (向后兼容)

```python
# 兼容性访问示例
if isinstance(result, dict):
    intensity = result.get('framing_intensity', 0.0)
else:
    intensity = getattr(result, 'framing_intensity', 0.0)
```

## 验证方法

运行以下命令验证修复：

```bash
# 1. 调试结果结构
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/debug_result_structure.py

# 2. 快速测试
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/quick_test.py

# 3. 全面测试
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/comprehensive_test.py --sample 10

# 4. 性能测试
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/benchmark_test.py
```

## 预期结果

修复后应该看到：
- ✅ 无字段访问错误
- ✅ 正确的统计信息显示
- ✅ 完整的测试摘要
- ✅ Ground truth相关性计算

## 未来建议

1. **类型注解**: 为返回值添加明确的类型注解
2. **文档同步**: 保持代码和文档中字段名的一致性
3. **单元测试**: 添加返回值结构的单元测试
4. **版本控制**: 如果更改返回值结构，考虑版本兼容性