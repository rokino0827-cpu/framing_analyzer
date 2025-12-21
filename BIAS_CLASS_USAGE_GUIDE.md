# Bias Class Index Configuration Guide

## 问题说明

当你看到这个警告时：
```
Could not determine bias class index, using default index 1
```

这表示模型使用了通用标签名（如 `LABEL_0`, `LABEL_1`），系统无法自动推断哪个索引对应bias类。

## 解决方案

### 方法1：使用现有验证脚本（推荐）

运行现有的验证脚本：

```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/verify_bias_class.py
```

这个脚本会：
1. 加载你的bias_detector模型
2. 使用中性和偏见文本进行测试
3. 分析哪个索引对应bias类
4. 给出配置建议

### 方法2：使用新的确定脚本

运行新创建的脚本：

```bash
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/determine_bias_class.py
```

这个脚本提供更详细的分析和多个测试文本。

### 方法3：使用framing_analyzer内置函数

在Python代码中：

```python
from framing_analyzer import verify_bias_class_index

result = verify_bias_class_index(
    model_name_or_path="himel7/bias-detector",
    test_texts=[
        "This is a neutral factual report.",
        "Those people are absolutely disgusting."
    ]
)

print(result)
```

## 配置bias_class_index

一旦确定了正确的索引（比如是1），有几种配置方法：

### 方法1：在代码中配置

```python
from framing_analyzer import AnalyzerConfig, create_analyzer

# 创建配置
config = AnalyzerConfig()
config.teacher.bias_class_index = 1  # 使用验证得到的索引

# 创建分析器
analyzer = create_analyzer(config)
```

### 方法2：使用配置文件

创建一个配置文件 `config.json`：

```json
{
  "teacher": {
    "bias_class_index": 1,
    "model_name": "himel7/bias-detector",
    "model_local_path": "bias_detector_data"
  }
}
```

### 方法3：使用现有的配置示例

参考 `framing_analyzer/config_with_bias_class.py` 文件中的示例。

## 验证配置是否生效

配置后重新运行测试，警告应该消失：

```bash
PYTHONPATH="/root/autodl-tmp" python tests/sample_run.py
```

如果配置正确，你应该看到：
- 没有 "Could not determine bias class index" 警告
- 正常的分析结果

## 常见问题

### Q: 为什么需要手动配置？
A: 因为模型使用了通用标签名（LABEL_0, LABEL_1），无法从标签名推断语义。

### Q: 如何知道配置是否正确？
A: 运行验证脚本，看偏见文本在你配置的索引上是否有更高的概率。

### Q: 配置错了会怎样？
A: 系统会把中性内容识别为偏见，或把偏见内容识别为中性，影响分析准确性。

### Q: 可以不配置吗？
A: 可以，系统会使用默认值（通常是1），但可能不准确。建议验证后明确配置。

## 技术原理

bias_detector是一个二分类模型：
- 输出两个概率：P(LABEL_0) 和 P(LABEL_1)
- 其中一个对应"中性/无偏见"，另一个对应"偏见"
- 我们需要确定哪个索引对应偏见类，以便正确解释模型输出

通过对比中性文本和偏见文本的预测概率，可以确定哪个索引在偏见文本上有更高的激活。