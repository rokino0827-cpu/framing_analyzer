#!/usr/bin/env python3
"""
测试套件总览脚本
按顺序运行所有测试，提供完整的系统验证

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/run_all_tests.py [options]

选项：
    --quick-only        只运行快速测试
    --skip-optimization 跳过融合权重优化（耗时较长）
    --sample N          测试样本数量（默认：20）
"""

import argparse
import sys
import time
import subprocess
from pathlib import Path

# 设置路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_command(cmd: str, description: str, timeout: int = 300) -> bool:
    """运行命令并返回是否成功"""
    
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=Path(__file__).parent.parent
        )
        
        duration = time.time() - start_time
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} - 成功 ({duration:.1f}s)")
            return True
        else:
            print(f"❌ {description} - 失败 (返回码: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - 超时 ({timeout}s)")
        return False
    except Exception as e:
        print(f"💥 {description} - 异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="运行完整测试套件")
    parser.add_argument("--quick-only", action="store_true", help="只运行快速测试")
    parser.add_argument("--skip-optimization", action="store_true", help="跳过融合权重优化")
    parser.add_argument("--sample", type=int, default=20, help="测试样本数量")
    
    args = parser.parse_args()
    
    print("🚀 启动完整测试套件...")
    print(f"📊 测试样本数量: {args.sample}")
    
    results = {}
    
    # 1. 验证bias_class_index
    results['bias_verification'] = run_command(
        "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/verify_bias_class.py",
        "验证bias_class_index配置",
        timeout=60
    )
    
    # 2. 快速测试
    results['quick_test'] = run_command(
        "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/quick_test.py",
        "快速功能测试",
        timeout=120
    )
    
    if args.quick_only:
        print_summary(results)
        return
    
    # 3. 省略检测验证
    results['omission_test'] = run_command(
        "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/test_omission_enabled.py",
        "省略检测启用验证",
        timeout=180
    )
    
    # 4. 全面测试（基础）
    results['comprehensive_basic'] = run_command(
        f"PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/comprehensive_test.py --sample {args.sample}",
        f"全面测试 - 基础功能 ({args.sample}篇文章)",
        timeout=300
    )
    
    # 5. 全面测试（启用省略检测）
    results['comprehensive_omission'] = run_command(
        f"PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/comprehensive_test.py --sample {args.sample} --enable-omission",
        f"全面测试 - 省略检测 ({args.sample}篇文章)",
        timeout=600
    )
    
    # 6. 性能基准测试
    results['benchmark'] = run_command(
        "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/benchmark_test.py",
        "性能基准测试",
        timeout=300
    )
    
    # 7. 融合权重优化（可选）
    if not args.skip_optimization:
        results['optimization'] = run_command(
            "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/optimize_fusion_weight.py",
            "融合权重优化",
            timeout=900
        )
    
    # 8. 配置示例测试
    results['config_example'] = run_command(
        "PYTHONPATH=\"/root/autodl-tmp\" python framing_analyzer/config_with_bias_class.py",
        "配置示例测试",
        timeout=120
    )
    
    print_summary(results)

def print_summary(results: dict):
    """打印测试总结"""
    
    print("\n" + "="*60)
    print("📊 测试套件总结")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for success in results.values() if success)
    
    print(f"总测试数: {total_tests}")
    print(f"通过测试: {passed_tests}")
    print(f"失败测试: {total_tests - passed_tests}")
    print(f"成功率: {passed_tests/total_tests*100:.1f}%")
    
    print("\n详细结果:")
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name:20} {status}")
    
    print("\n" + "="*60)
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！系统运行正常。")
    elif passed_tests >= total_tests * 0.8:
        print("⚠️  大部分测试通过，但有少数失败。请检查失败的测试。")
    else:
        print("❌ 多个测试失败。请检查系统配置和依赖。")
    
    print("="*60)
    
    # 推荐下一步
    print("\n💡 推荐下一步:")
    if results.get('bias_verification', False):
        print("  ✅ bias_class_index 配置正确")
    else:
        print("  ⚠️  请先运行 verify_bias_class.py 确定正确的 bias_class_index")
    
    if results.get('omission_test', False):
        print("  ✅ 省略检测功能正常")
    else:
        print("  ⚠️  省略检测可能有问题，检查依赖和配置")
    
    if results.get('comprehensive_omission', False):
        print("  ✅ 可以在生产环境中使用省略检测")
    else:
        print("  ⚠️  建议先修复问题再启用省略检测")

if __name__ == "__main__":
    main()