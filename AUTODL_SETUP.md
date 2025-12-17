# AutoDL环境配置指南

## 🚀 快速开始

### 1. 选择AutoDL实例
推荐配置：
- **GPU**: RTX 3090/4090, V100, A100 (显存 ≥ 16GB)
- **内存**: 32GB+ (最低16GB)
- **磁盘**: 50GB+ 可用空间
- **镜像**: PyTorch 2.0+ (CUDA 11.8)

### 2. 连接VSCode
```bash
# 在AutoDL控制台获取SSH连接信息
# 在VSCode中使用Remote-SSH插件连接
```

### 3. 上传项目文件
```bash
# 方法1: 使用VSCode直接拖拽上传整个framing_analyzer文件夹
# 方法2: 使用git clone (如果已上传到GitHub)
git clone <your-repo-url>
cd framing_analyzer
```

## 🔧 环境配置

### 方法1: 自动配置 (推荐)
```bash
# 检查环境
python check_autodl_env.py

# 自动配置
python setup_autodl.py

# 或使用bash脚本
chmod +x setup_autodl.sh
./setup_autodl.sh
```

### 方法2: 手动配置
```bash
# 更新pip
pip install --upgrade pip

# 安装核心依赖
pip install -r requirements_autodl.txt

# 或安装完整依赖
pip install -r requirements.txt
```

## 📋 环境验证

### 检查安装
```bash
python check_autodl_env.py
```

### 快速测试
```bash
python quick_test.py
```

## 🎯 开始使用

### 准备数据
```bash
# 上传数据文件到data目录
mkdir -p data
# 将CSV文件放入data/目录
```

### 运行分析
```bash
# 查看帮助
python main.py --help

# 快速测试 (200篇文章)
python main.py --input data/sample.csv --preset quick --output ./test_results

# 大规模处理 (30k文章)
python main.py --input data/full_dataset.csv --preset fast --output ./full_results
```

## 🛠️ 常见问题

### 1. CUDA版本不匹配
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装对应版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 内存不足
```bash
# 减少batch_size
python main.py --input data.csv --config-override '{"teacher": {"batch_size": 8}}'

# 使用chunk模式而非sentence模式
python main.py --input data.csv --preset fast
```

### 3. 磁盘空间不足
```bash
# 清理缓存
rm -rf cache/*
rm -rf ~/.cache/huggingface/

# 使用外部存储
ln -s /path/to/external/storage ./outputs
```

### 4. 网络问题 (模型下载)
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型到本地
# 然后修改config中的model_name为本地路径
```

## 📊 性能优化

### AutoDL环境优化
```python
# 在config中设置
{
    "teacher": {
        "device": "cuda",
        "batch_size": 16,  # 根据显存调整
        "max_length": 512
    },
    "processing": {
        "fragment_mode": "chunk"  # 更快的处理模式
    }
}
```

### 监控资源使用
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控内存使用
htop

# 在Python中监控
python -c "
import GPUtil
import psutil
print(f'GPU使用率: {GPUtil.getGPUs()[0].load*100:.1f}%')
print(f'内存使用率: {psutil.virtual_memory().percent:.1f}%')
"
```

## 🔄 数据处理流程

### 1. 小规模测试
```bash
# 先用200篇测试
head -n 201 data/full_dataset.csv > data/sample_200.csv
python main.py --input data/sample_200.csv --preset quick
```

### 2. 检查结果
```bash
# 查看输出
ls -la outputs/
cat outputs/analysis_summary.json
```

### 3. 大规模处理
```bash
# 确认无误后处理全量数据
python main.py --input data/full_dataset.csv --preset fast --output ./full_results
```

## 💡 最佳实践

### 1. 数据准备
- CSV格式，包含title和content列
- 确保文本编码为UTF-8
- 预处理去除明显的噪声数据

### 2. 配置选择
- **快速验证**: `--preset quick`
- **大规模处理**: `--preset fast` 
- **高质量结果**: `--preset high_precision`

### 3. 结果保存
- 定期保存中间结果
- 使用时间戳命名输出目录
- 备份重要的分析结果

### 4. 错误处理
- 检查日志文件 `logs/analyzer.log`
- 使用 `--debug` 模式获取详细信息
- 分批处理大数据集避免内存问题

## 📞 技术支持

如果遇到问题：
1. 运行 `python check_autodl_env.py` 检查环境
2. 查看 `logs/analyzer.log` 日志文件
3. 使用 `--debug` 模式获取详细错误信息
4. 检查AutoDL实例的资源使用情况