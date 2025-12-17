#!/usr/bin/env python3
"""
AutoDL环境配置脚本 - Python版本
适用于框架偏见分析器
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """运行命令并处理错误"""
    print(f"📦 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - 成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - 失败")
        print(f"错误信息: {e.stderr}")
        return False

def check_environment():
    """检查环境信息"""
    print("🔍 检查环境信息...")
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # CUDA检查
    try:
        result = subprocess.run("nvidia-smi", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU可用")
        else:
            print("⚠️ 未检测到NVIDIA GPU")
    except:
        print("⚠️ nvidia-smi命令不可用")

def install_packages():
    """安装依赖包"""
    print("📦 开始安装依赖包...")
    
    # 更新pip
    run_command("pip install --upgrade pip", "更新pip")
    
    # 核心包列表（分组安装避免内存问题）
    package_groups = [
        {
            "name": "PyTorch生态系统",
            "packages": [
                "torch>=2.0.0,<2.3.0",
                "torchvision>=0.15.0,<0.18.0", 
                "torchaudio>=2.0.0,<2.3.0",
                "--index-url https://download.pytorch.org/whl/cu118"
            ]
        },
        {
            "name": "Transformers生态系统",
            "packages": [
                "transformers>=4.30.0,<5.0.0",
                "tokenizers>=0.13.0,<1.0.0",
                "accelerate>=0.20.0",
                "datasets>=2.10.0"
            ]
        },
        {
            "name": "科学计算核心",
            "packages": [
                "numpy>=1.21.0,<2.0.0",
                "pandas>=1.3.0,<3.0.0",
                "scipy>=1.7.0",
                "scikit-learn>=1.0.0,<2.0.0"
            ]
        },
        {
            "name": "文本处理",
            "packages": [
                "regex>=2022.0.0",
                "nltk>=3.8.0"
            ]
        },
        {
            "name": "可视化工具",
            "packages": [
                "matplotlib>=3.5.0,<4.0.0",
                "seaborn>=0.11.0,<1.0.0",
                "plotly>=5.0.0"
            ]
        },
        {
            "name": "系统工具",
            "packages": [
                "tqdm>=4.60.0",
                "rich>=13.0.0",
                "psutil>=5.8.0",
                "GPUtil>=1.4.0"
            ]
        },
        {
            "name": "配置和数据格式",
            "packages": [
                "pyyaml>=6.0.0",
                "toml>=0.10.0",
                "openpyxl>=3.0.0",
                "pyarrow>=10.0.0"
            ]
        },
        {
            "name": "开发工具",
            "packages": [
                "ipython>=8.0.0",
                "jupyter>=1.0.0",
                "requests>=2.28.0",
                "typing-extensions>=4.0.0"
            ]
        }
    ]
    
    # 逐组安装
    for group in package_groups:
        packages_str = " ".join(group["packages"])
        cmd = f"pip install {packages_str}"
        run_command(cmd, f"安装{group['name']}")

def download_nltk_data():
    """下载NLTK数据"""
    print("📦 下载NLTK数据...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK数据下载完成")
    except Exception as e:
        print(f"⚠️ NLTK数据下载失败: {e}")

def verify_installation():
    """验证安装"""
    print("🔍 验证安装...")
    
    verification_code = """
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
print(f'NumPy版本: {np.__version__}')
print(f'Pandas版本: {pandas.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'设备{i}: {torch.cuda.get_device_name(i)}')
        print(f'显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f}GB')
else:
    print('⚠️ CUDA不可用，将使用CPU模式')
"""
    
    try:
        exec(verification_code)
        return True
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def create_directories():
    """创建必要的目录"""
    print("📁 创建工作目录...")
    
    directories = ['logs', 'outputs', 'cache', 'data', 'models']
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_name}")

def main():
    """主函数"""
    print("🚀 AutoDL环境配置开始...")
    print("=" * 50)
    
    # 检查环境
    check_environment()
    print()
    
    # 安装包
    install_packages()
    print()
    
    # 下载NLTK数据
    download_nltk_data()
    print()
    
    # 创建目录
    create_directories()
    print()
    
    # 验证安装
    if verify_installation():
        print()
        print("✅ AutoDL环境配置完成！")
        print()
        print("🎯 下一步操作:")
        print("1. 上传数据文件到 data/ 目录")
        print("2. 运行测试: python quick_test.py")
        print("3. 开始分析: python main.py --help")
        print()
        print("💡 推荐配置:")
        print("- GPU: RTX 3090/4090 或 V100/A100")
        print("- 内存: 16GB+ (推荐32GB)")
        print("- 磁盘: 50GB+ 可用空间")
        print("- CUDA: 11.8 (与PyTorch版本匹配)")
    else:
        print("❌ 环境配置失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()