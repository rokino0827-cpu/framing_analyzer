# Bias Class Configuration Tools

这个目录包含了用于配置和验证bias_class_index的工具和文档。

## 文件说明

### 🔧 工具脚本
- **`verify_bias_class.py`** - 验证脚本（中文界面，简洁）
- **`determine_bias_class.py`** - 确定脚本（英文界面，详细分析）
- **`config_with_bias_class.py`** - 配置示例代码

### 📚 文档
- **`BIAS_CLASS_USAGE_GUIDE.md`** - 详细使用指南
- **`BIAS_CLASS_FIX_SUMMARY.md`** - 问题修复总结

## 快速使用

### 1. 验证bias_class_index

```bash
# 方法1：简洁版本
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/verify_bias_class.py

# 方法2：详细版本
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/determine_bias_class.py
```

### 2. 配置到代码中

```python
from framing_analyzer import AnalyzerConfig, create_analyzer

config = AnalyzerConfig()
config.teacher.bias_class_index = 1  # 使用验证得到的索引
analyzer = create_analyzer(config)
```

### 3. 测试配置

```bash
# 运行示例代码
PYTHONPATH="/root/autodl-tmp" python framing_analyzer/config_with_bias_class.py
```

## 问题解决

如果看到警告：
```
Could not determine bias class index, using default index 1
```

1. 运行验证脚本确定正确索引
2. 在配置中设置 `bias_class_index`
3. 重新运行，警告消失

详细说明请参考 `BIAS_CLASS_USAGE_GUIDE.md`。