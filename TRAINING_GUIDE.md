# SV2000模型训练和权重优化指南

本指南详细介绍如何训练自定义的SV2000框架预测模型和优化融合权重。

## 📋 目录

1. [快速开始](#快速开始)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [权重优化](#权重优化)
5. [高级配置](#高级配置)
6. [故障排除](#故障排除)

## 🚀 快速开始

### 一键式体验

如果你想快速体验完整的训练流程，可以使用快速开始脚本：

```bash
# 运行完整的快速开始流程
python framing_analyzer/quick_start.py

# 自定义参数
python framing_analyzer/quick_start.py \
    --work_dir ./my_sv2000_project \
    --num_samples 500 \
    --epochs 10 \
    --device cuda
```

这将自动完成：
- 生成示例训练数据
- 训练SV2000模型
- 优化融合权重
- 测试系统功能
- 生成完整报告

## 📊 数据准备

### 数据格式要求

训练数据应为CSV格式，包含以下列：

**必需列：**
- `content`: 文章内容（字符串）
- `y_conflict`: 冲突框架分数 (0-1)
- `y_human`: 人情框架分数 (0-1)
- `y_econ`: 经济框架分数 (0-1)
- `y_moral`: 道德框架分数 (0-1)
- `y_resp`: 责任框架分数 (0-1)

**可选列：**
- `title`: 文章标题
- `id`: 文章唯一标识
- `item_1` 到 `item_20`: 单独的问卷条目分数

### 生成示例数据

```bash
# 生成100个示例样本
python framing_analyzer/prepare_training_data.py \
    --generate_sample \
    --num_samples 100 \
    --output_dir ./data
```

### 验证数据格式

```bash
# 验证数据格式
python framing_analyzer/prepare_training_data.py \
    --input_path your_data.csv \
    --validate_only
```

### 数据清理和预处理

```bash
# 完整的数据预处理
python framing_analyzer/prepare_training_data.py \
    --input_path raw_data.csv \
    --output_dir ./processed_data \
    --clean \
    --augment \
    --split \
    --min_content_length 50 \
    --max_content_length 5000
```

## 🎯 模型训练

### 基础训练

```bash
# 基础训练命令
python framing_analyzer/train_sv2000_model.py \
    --data_path ./data/training_data.csv \
    --output_dir ./models \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5
```

### 高级训练配置

```bash
# 高级训练配置
python framing_analyzer/train_sv2000_model.py \
    --data_path ./data/training_data.csv \
    --output_dir ./models \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-5 \
    --dropout_rate 0.2 \
    --validation_split 0.2 \
    --early_stopping_patience 5 \
    --device cuda \
    --optimize_weights \
    --evaluate
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 16 | 批处理大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--dropout_rate` | 0.1 | Dropout率 |
| `--validation_split` | 0.2 | 验证集比例 |
| `--early_stopping_patience` | 3 | 早停耐心值 |
| `--device` | auto | 计算设备 |
| `--optimize_weights` | False | 是否优化融合权重 |

### 使用自定义编码器

```bash
# 使用不同的编码器
python framing_analyzer/train_sv2000_model.py \
    --data_path ./data/training_data.csv \
    --encoder_name "sentence-transformers/all-mpnet-base-v2" \
    --output_dir ./models
```

## ⚖️ 权重优化

### 基础权重优化

```bash
# 基础权重优化
python framing_analyzer/optimize_fusion_weights.py \
    --data_path ./data/training_data.csv \
    --output_dir ./optimization
```

### 高级权重优化

```bash
# 高级权重优化配置
python framing_analyzer/optimize_fusion_weights.py \
    --data_path ./data/training_data.csv \
    --output_dir ./optimization \
    --ridge_alpha 0.5 \
    --cv_folds 10 \
    --max_samples 1000
```

### 权重优化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ridge_alpha` | 1.0 | Ridge回归正则化参数 |
| `--cv_folds` | 5 | 交叉验证折数 |
| `--max_samples` | 1000 | 最大样本数（加速用） |
| `--skip_ridge` | False | 跳过Ridge回归优化 |
| `--skip_cv` | False | 跳过交叉验证 |

### 优化结果解读

权重优化会生成以下文件：
- `fusion_optimization_results.json`: 详细的优化结果
- `fusion_optimization_results_report.md`: 可读的优化报告

优化报告包含：
- Ridge回归优化的权重
- 交叉验证结果对比
- 组件重要性分析
- 使用建议

## 🔧 高级配置

### 自定义配置文件

创建自定义配置：

```python
from framing_analyzer.config import AnalyzerConfig, SVFramingConfig, FusionConfig

# 创建配置
config = AnalyzerConfig()

# SV2000配置
config.sv_framing = SVFramingConfig(
    enabled=True,
    encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    hidden_size=384,
    dropout_rate=0.1,
    learning_rate=2e-5,
    batch_size=16,
    device="cuda",
    model_save_path="./custom_models"
)

# 融合配置
config.fusion = FusionConfig(
    alpha=0.6,      # SV2000权重
    beta=0.15,      # 偏见检测权重
    gamma=0.1,      # 省略检测权重
    delta=0.1,      # 相对框架权重
    epsilon=0.05,   # 引用分析权重
    use_ridge_optimization=True,
    ridge_alpha=1.0
)
```

### 多GPU训练

```bash
# 使用特定GPU
CUDA_VISIBLE_DEVICES=0 python framing_analyzer/train_sv2000_model.py \
    --data_path ./data/training_data.csv \
    --device cuda \
    --batch_size 32
```

### 内存优化

对于内存受限的环境：

```bash
# 小批处理大小
python framing_analyzer/train_sv2000_model.py \
    --data_path ./data/training_data.csv \
    --batch_size 4 \
    --device cpu
```

## 📈 性能监控

### 训练监控

训练过程中会输出：
- 每轮的训练损失和验证损失
- 框架预测的相关性指标
- 早停信息
- 最佳模型保存信息

### 评估指标

主要评估指标：
- **Pearson相关性**: 预测与真实标签的线性相关性
- **MAE**: 平均绝对误差
- **框架对齐分数**: 整体框架预测质量

### 结果可视化

训练完成后会生成：
- 训练曲线图
- 框架相关性热图
- 权重分布图

## 🔍 故障排除

### 常见问题

#### 1. CUDA内存不足

```bash
# 解决方案：减小批处理大小
python framing_analyzer/train_sv2000_model.py \
    --batch_size 4 \
    --device cuda
```

#### 2. 模型加载失败

```bash
# 检查模型路径
ls -la ./models/best_sv2000_model.pt

# 使用CPU加载
python framing_analyzer/train_sv2000_model.py \
    --device cpu
```

#### 3. 数据格式错误

```bash
# 验证数据格式
python framing_analyzer/prepare_training_data.py \
    --input_path your_data.csv \
    --validate_only
```

#### 4. 训练收敛慢

```bash
# 调整学习率
python framing_analyzer/train_sv2000_model.py \
    --learning_rate 5e-5 \
    --epochs 20
```

### 性能优化建议

1. **数据质量**：
   - 确保标注数据质量高
   - 移除异常值和噪声数据
   - 平衡各框架的分布

2. **模型配置**：
   - 根据数据量调整模型复杂度
   - 使用适当的正则化
   - 调整学习率和批处理大小

3. **硬件优化**：
   - 使用GPU加速训练
   - 增加内存以支持更大批处理
   - 使用SSD存储数据

### 调试技巧

1. **启用详细日志**：
   ```bash
   python framing_analyzer/train_sv2000_model.py \
       --verbose \
       --data_path ./data/training_data.csv
   ```

2. **小数据集测试**：
   ```bash
   # 使用小数据集快速测试
   python framing_analyzer/prepare_training_data.py \
       --generate_sample \
       --num_samples 50
   ```

3. **检查中间结果**：
   - 查看训练报告JSON文件
   - 检查验证预测结果
   - 分析组件贡献度

## 📚 进阶使用

### 集成到现有系统

```python
from framing_analyzer import FramingAnalyzer
from framing_analyzer.config import create_sv2000_config

# 加载训练好的模型
config = create_sv2000_config()
config.sv_framing.pretrained_model_path = "./models/best_sv2000_model.pt"

# 使用优化后的权重
config.fusion.alpha = 0.6    # 从优化结果中获取
config.fusion.beta = 0.15
# ... 其他权重

analyzer = FramingAnalyzer(config)
```

### 批量处理

```python
# 批量分析大量文章
articles = load_articles_from_database()
results = analyzer.analyze_batch(articles, output_path="results.json")
```

### API服务部署

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
analyzer = FramingAnalyzer(config)

@app.route('/analyze', methods=['POST'])
def analyze_article():
    data = request.json
    result = analyzer.analyze_article(data['content'], data.get('title', ''))
    return jsonify({
        'framing_intensity': result.framing_intensity,
        'sv_frames': {
            'conflict': result.sv_conflict,
            'human': result.sv_human,
            'economic': result.sv_econ,
            'moral': result.sv_moral,
            'responsibility': result.sv_resp
        }
    })
```

## 📞 支持和反馈

如果遇到问题或有改进建议：

1. 查看本指南的故障排除部分
2. 检查日志输出中的错误信息
3. 验证数据格式和配置参数
4. 尝试使用示例数据重现问题

---

*本指南将随着功能更新持续完善。*