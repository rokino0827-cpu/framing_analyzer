#!/usr/bin/env python3
"""
AutoDL环境检查脚本
检查环境是否满足框架偏见分析器的运行要求
"""

import sys
import subprocess
import importlib
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    print(f"   当前版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python版本满足要求 (>=3.8)")
        return True
    else:
        print("   ❌ Python版本过低，需要3.8+")
        return False

def check_gpu():
    """检查GPU环境"""
    print("🎮 检查GPU环境...")
    
    try:
        result = subprocess.run("nvidia-smi", shell=True, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ NVIDIA GPU可用")
            # 解析GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'Tesla' in line or 'Quadro' in line or 'RTX' in line:
                    gpu_info = line.strip()
                    print(f"   GPU: {gpu_info}")
            return True
        else:
            print("   ⚠️ nvidia-smi命令失败")
            return False
    except:
        print("   ❌ 未检测到NVIDIA GPU")
        return False

def check_cuda():
    """检查CUDA"""
    print("🔥 检查CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"   ✅ CUDA可用，版本: {cuda_version}")
            print(f"   GPU数量: {device_count}")
            
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   设备{i}: {name} ({memory:.1f}GB)")
            return True
        else:
            print("   ❌ CUDA不可用")
            return False
    except ImportError:
        print("   ❌ PyTorch未安装")
        return False

def check_package(package_name, import_name=None):
    """检查单个包"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"   ❌ {package_name}: 未安装")
        return False

def check_packages():
    """检查必要的包"""
    print("📦 检查Python包...")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('regex', 'regex'),
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            missing_packages.append(package_name)
    
    return missing_packages

def check_memory():
    """检查内存"""
    print("💾 检查系统内存...")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        
        print(f"   总内存: {total_gb:.1f}GB")
        print(f"   可用内存: {available_gb:.1f}GB")
        
        if total_gb >= 16:
            print("   ✅ 内存充足 (>=16GB)")
            return True
        elif total_gb >= 8:
            print("   ⚠️ 内存较少 (8-16GB)，可能影响大数据集处理")
            return True
        else:
            print("   ❌ 内存不足 (<8GB)")
            return False
    except ImportError:
        print("   ⚠️ 无法检查内存 (psutil未安装)")
        return True

def check_disk_space():
    """检查磁盘空间"""
    print("💿 检查磁盘空间...")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free / 1024**3
        
        print(f"   可用空间: {free_gb:.1f}GB")
        
        if free_gb >= 50:
            print("   ✅ 磁盘空间充足 (>=50GB)")
            return True
        elif free_gb >= 20:
            print("   ⚠️ 磁盘空间较少 (20-50GB)")
            return True
        else:
            print("   ❌ 磁盘空间不足 (<20GB)")
            return False
    except:
        print("   ⚠️ 无法检查磁盘空间")
        return True

def check_directories():
    """检查必要目录"""
    print("📁 检查工作目录...")
    
    required_dirs = ['logs', 'outputs', 'cache', 'data']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"   ✅ {dir_name}/")
        else:
            print(f"   ⚠️ {dir_name}/ (不存在，将自动创建)")
            missing_dirs.append(dir_name)
    
    return missing_dirs

def main():
    """主检查函数"""
    print("🔍 AutoDL环境检查")
    print("=" * 50)
    
    checks = []
    
    # 基础环境检查
    checks.append(("Python版本", check_python_version()))
    checks.append(("GPU环境", check_gpu()))
    checks.append(("CUDA支持", check_cuda()))
    
    # 包检查
    missing_packages = check_packages()
    checks.append(("Python包", len(missing_packages) == 0))
    
    # 系统资源检查
    checks.append(("系统内存", check_memory()))
    checks.append(("磁盘空间", check_disk_space()))
    
    # 目录检查
    missing_dirs = check_directories()
    
    print("\n" + "=" * 50)
    print("📋 检查总结:")
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"   {status} {check_name}")
        if not passed:
            all_passed = False
    
    print()
    
    if missing_packages:
        print("❌ 缺少以下包，请安装:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n安装命令:")
        print("   pip install " + " ".join(missing_packages))
        print("   或运行: python setup_autodl.py")
    
    if missing_dirs:
        print("📁 将创建缺少的目录:")
        for dir_name in missing_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            print(f"   ✅ 创建 {dir_name}/")
    
    print()
    
    if all_passed and not missing_packages:
        print("🎉 环境检查通过！可以开始使用框架偏见分析器")
        print("\n🚀 快速开始:")
        print("   python quick_test.py          # 运行快速测试")
        print("   python main.py --help        # 查看使用说明")
    else:
        print("⚠️ 环境存在问题，请根据上述提示进行修复")
        
        if not missing_packages:
            print("\n💡 如果只是警告，通常仍可正常使用")
    
    print("\n📚 更多帮助:")
    print("   - 查看README.md了解详细使用方法")
    print("   - 运行setup_autodl.py自动配置环境")

if __name__ == "__main__":
    main()