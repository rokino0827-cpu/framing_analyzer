#!/usr/bin/env python3
"""
测试结果验证脚本
验证测试输出是否符合预期格式和范围

用法：
    PYTHONPATH="/root/autodl-tmp" python framing_analyzer/validate_test_results.py [result_file]
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

def validate_analysis_result(result: Dict) -> List[str]:
    """验证单个分析结果"""
    
    errors = []
    
    # 必需字段检查
    required_fields = ['id', 'framing_intensity', 'pseudo_label']
    for field in required_fields:
        if field not in result:
            errors.append(f"Missing required field: {field}")
    
    # 数值范围检查
    if 'framing_intensity' in result:
        intensity = result['framing_intensity']
        if not isinstance(intensity, (int, float)):
            errors.append(f"framing_intensity must be numeric, got {type(intensity)}")
        elif not (0.0 <= intensity <= 1.0):
            errors.append(f"framing_intensity must be in [0,1], got {intensity}")
    
    # 伪标签检查
    if 'pseudo_label' in result:
        label = result['pseudo_label']
        valid_labels = ['positive', 'negative', 'uncertain']
        if label not in valid_labels:
            errors.append(f"pseudo_label must be one of {valid_labels}, got '{label}'")
    
    # 省略检测字段检查（如果存在）
    if 'omission_score' in result:
        omission_score = result['omission_score']
        if omission_score is not None:
            if not isinstance(omission_score, (int, float)):
                errors.append(f"omission_score must be numeric or None, got {type(omission_score)}")
            elif not (0.0 <= omission_score <= 1.0):
                errors.append(f"omission_score must be in [0,1], got {omission_score}")
    
    # 组件分数检查
    if 'components' in result:
        components = result['components']
        if isinstance(components, dict):
            expected_components = ['headline', 'lede', 'narration', 'quotes']
            for comp in expected_components:
                if comp in components:
                    score = components[comp]
                    if not isinstance(score, (int, float)):
                        errors.append(f"Component {comp} must be numeric, got {type(score)}")
                    elif not (0.0 <= score <= 1.0):
                        errors.append(f"Component {comp} must be in [0,1], got {score}")
    
    return errors

def validate_batch_results(results: Dict) -> Dict[str, Any]:
    """验证批量分析结果"""
    
    validation_report = {
        'total_articles': 0,
        'valid_articles': 0,
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    # 检查顶层结构
    if 'results' not in results:
        validation_report['errors'].append("Missing 'results' field in batch results")
        return validation_report
    
    article_results = results['results']
    validation_report['total_articles'] = len(article_results)
    
    # 验证每篇文章
    article_errors = []
    valid_count = 0
    
    for i, result in enumerate(article_results):
        errors = validate_analysis_result(result)
        if errors:
            article_errors.append(f"Article {i} ({result.get('id', 'unknown')}): {'; '.join(errors)}")
        else:
            valid_count += 1
    
    validation_report['valid_articles'] = valid_count
    validation_report['errors'].extend(article_errors)
    
    # 统计信息验证
    if 'statistics' in results:
        stats = results['statistics']
        
        # 检查省略检测统计
        if 'omission_detection' in stats:
            omission_stats = stats['omission_detection']
            required_omission_fields = ['mean', 'std', 'articles_with_omissions', 'omission_rate']
            
            for field in required_omission_fields:
                if field not in omission_stats:
                    validation_report['warnings'].append(f"Missing omission statistic: {field}")
    
    # 生成统计摘要
    if valid_count > 0:
        intensities = [r.get('framing_intensity', 0) for r in article_results if 'framing_intensity' in r]
        omission_scores = [r.get('omission_score') for r in article_results if r.get('omission_score') is not None]
        
        validation_report['statistics'] = {
            'framing_intensity': {
                'mean': sum(intensities) / len(intensities) if intensities else 0,
                'min': min(intensities) if intensities else 0,
                'max': max(intensities) if intensities else 0,
                'count': len(intensities)
            },
            'omission_detection': {
                'articles_with_omission': len(omission_scores),
                'omission_rate': len(omission_scores) / len(article_results) if article_results else 0,
                'avg_omission_score': sum(omission_scores) / len(omission_scores) if omission_scores else 0
            }
        }
    
    return validation_report

def validate_test_file(file_path: Path) -> Dict[str, Any]:
    """验证测试结果文件"""
    
    print(f"🔍 Validating test results: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查文件类型
        if 'results' in data and isinstance(data['results'], list):
            # 批量分析结果
            return validate_batch_results(data)
        elif isinstance(data, dict) and 'test_results' in data:
            # 测试套件结果
            return validate_test_suite_results(data)
        else:
            return {
                'errors': ['Unknown file format'],
                'warnings': [],
                'statistics': {}
            }
    
    except json.JSONDecodeError as e:
        return {
            'errors': [f'Invalid JSON: {e}'],
            'warnings': [],
            'statistics': {}
        }
    except Exception as e:
        return {
            'errors': [f'Error reading file: {e}'],
            'warnings': [],
            'statistics': {}
        }

def validate_test_suite_results(data: Dict) -> Dict[str, Any]:
    """验证测试套件结果"""
    
    validation_report = {
        'errors': [],
        'warnings': [],
        'statistics': {}
    }
    
    if 'test_results' in data:
        test_results = data['test_results']
        
        # 检查基础分析结果
        if 'basic_analysis' in test_results:
            basic = test_results['basic_analysis']
            if not basic.get('success', False):
                validation_report['errors'].append("Basic analysis failed")
            
            if 'framing_intensity_stats' in basic:
                stats = basic['framing_intensity_stats']
                mean_intensity = stats.get('mean', 0)
                if not (0.0 <= mean_intensity <= 1.0):
                    validation_report['warnings'].append(f"Unusual mean framing intensity: {mean_intensity}")
        
        # 检查省略检测结果
        if 'omission_analysis' in test_results:
            omission = test_results['omission_analysis']
            if omission.get('success', False):
                omission_rate = omission.get('omission_rate', 0)
                if omission_rate == 0:
                    validation_report['warnings'].append("No omissions detected - check if this is expected")
                elif omission_rate > 0.8:
                    validation_report['warnings'].append(f"Very high omission rate: {omission_rate:.2f}")
    
    return validation_report

def print_validation_report(report: Dict[str, Any], file_path: Path):
    """打印验证报告"""
    
    print(f"\n📊 Validation Report for: {file_path.name}")
    print("="*60)
    
    # 总体状态
    if report['errors']:
        print("❌ VALIDATION FAILED")
        print(f"Errors: {len(report['errors'])}")
        for error in report['errors']:
            print(f"  - {error}")
    else:
        print("✅ VALIDATION PASSED")
    
    # 警告
    if report['warnings']:
        print(f"\n⚠️  Warnings: {len(report['warnings'])}")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    # 统计信息
    if 'statistics' in report and report['statistics']:
        print(f"\n📈 Statistics:")
        stats = report['statistics']
        
        if 'total_articles' in report:
            print(f"  Total articles: {report['total_articles']}")
            print(f"  Valid articles: {report['valid_articles']}")
            print(f"  Success rate: {report['valid_articles']/report['total_articles']*100:.1f}%")
        
        if 'framing_intensity' in stats:
            fi_stats = stats['framing_intensity']
            print(f"  Framing intensity - Mean: {fi_stats['mean']:.3f}, Range: [{fi_stats['min']:.3f}, {fi_stats['max']:.3f}]")
        
        if 'omission_detection' in stats:
            om_stats = stats['omission_detection']
            print(f"  Omission detection - Rate: {om_stats['omission_rate']:.3f}, Avg score: {om_stats['avg_omission_score']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="验证测试结果文件")
    parser.add_argument("file", nargs="?", help="测试结果文件路径")
    parser.add_argument("--auto-find", action="store_true", help="自动查找最新的测试结果")
    
    args = parser.parse_args()
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        
        report = validate_test_file(file_path)
        print_validation_report(report, file_path)
        
    elif args.auto_find:
        # 自动查找最新的测试结果
        results_dirs = [
            Path("results/comprehensive_test"),
            Path("results/quick_test"),
            Path("results/omission_test"),
            Path("results/benchmark")
        ]
        
        latest_files = []
        for results_dir in results_dirs:
            if results_dir.exists():
                json_files = list(results_dir.glob("*.json"))
                if json_files:
                    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
                    latest_files.append(latest_file)
        
        if not latest_files:
            print("❌ No test result files found")
            sys.exit(1)
        
        print(f"🔍 Found {len(latest_files)} test result files")
        
        for file_path in latest_files:
            report = validate_test_file(file_path)
            print_validation_report(report, file_path)
            print()
    
    else:
        print("Usage: python validate_test_results.py <file> or --auto-find")
        sys.exit(1)

if __name__ == "__main__":
    main()