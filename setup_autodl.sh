#!/bin/bash
# AutoDL环境配置脚本 - 框架偏见分析器

echo "🚀 开始配置AutoDL环境..."

# 检查Python版本
echo "📋 检查Python版本..."
python --version

# 检查CUDA版本
echo "📋 检查CUDA版本..."
nvidia-smi

# 更新pip
echo "📦 更新pip..."
pip install --upgrade pip

# 安装核心依赖（分批安装，避免内存问题）
echo "📦 安装PyTorch生态系统..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "📦 安装Transformers生态系统..."
pip install transformers tokenizers accelerate datasets

echo "📦 安装科学计算包..."
pip install numpy pandas scipy scikit-learn

echo "📦 安装文本处理包..."
pip install regex nltk spacy

echo "📦 安装可视化包..."
pip install matplotlib seaborn plotly

echo "📦 安装工具包..."
pip install tqdm rich psutil GPUtil nvidia-ml-py3

echo "📦 安装配置和数据格式支持..."
pip install pyyaml toml omegaconf openpyxl xlsxwriter pyarrow fastparquet

echo "📦 安装开发工具..."
pip install ipython jupyter notebook requests urllib3 typing-extensions

# 下载必要的NLTK数据
echo "📦 下载NLTK数据..."
python -c "
import nltk
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    print('✅ NLTK数据下载完成')
except:
    print('⚠️ NLTK数据下载失败，请手动下载')
"

# 验证安装
echo "🔍 验证安装..."
python -c "
import torch
import transformers
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import seaborn
import tqdm
import regex
print('✅ 所有核心包导入成功')
print(f'PyTorch版本: {torch.__version__}')
print(f'Transformers版本: {transformers.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    print(f'当前CUDA设备: {torch.cuda.get_device_name()}')
"

# 创建必要的目录
echo "📁 创建工作目录..."
mkdir -p logs
mkdir -p outputs
mkdir -p cache
mkdir -p data

echo "✅ AutoDL环境配置完成！"
echo ""
echo "🎯 下一步操作："
echo "1. 上传你的数据文件到 data/ 目录"
echo "2. 运行测试: python quick_test.py"
echo "3. 开始分析: python main.py --help"
echo ""
echo "💡 如果遇到问题，请检查："
echo "- CUDA版本是否匹配 (推荐CUDA 11.8)"
echo "- 内存是否足够 (推荐16GB+)"
echo "- 磁盘空间是否足够 (推荐50GB+)"