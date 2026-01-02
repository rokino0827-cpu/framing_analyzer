#!/usr/bin/env python3
"""
现有数据验证和适配工具
专门用于验证和适配已有的机器标注数据
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from framing_analyzer.sv2000_data_loader import SV2000DataLoader
from framing_analyzer.config import create_sv2000_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_data_structure(csv_path: str) -> Dict[str, Any]:
    """检查数据结构"""
    logger.info(f"检查数据文件: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    # 读取数据
    df = pd.read_csv(csv_path)
    
    inspection_results = {
        'file_path': csv_path,
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'sample_data': {},
        'data_types': {},
        'missing_values': {},
        'value_ranges': {},
        'potential_mappings': {}
    }
    
    # 数据类型和缺失值
    for col in df.columns:
        inspection_results['data_types'][col] = str(df[col].dtype)
        inspection_results['missing_values'][col] = df[col].isna().sum()
        
        # 采样数据
        sample_values = df[col].dropna().head(3).tolist()
        inspection_results['sample_data'][col] = sample_values
        
        # 数值列的范围
        if df[col].dtype in ['int64', 'float64']:
            inspection_results['value_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    # 智能列映射建议
    inspection_results['potential_mappings'] = suggest_column_mappings(df.columns)
    
    return inspection_results

def suggest_column_mappings(columns: List[str]) -> Dict[str, List[str]]:
    """智能建议列映射"""
    mappings = {
        'content': [],
        'title': [],
        'id': [],
        'y_conflict': [],
        'y_human': [],
        'y_econ': [],
        'y_moral': [],
        'y_resp': [],
        'sv_frame_avg': []
    }
    
    # 内容列候选
    content_keywords = ['content', 'text', 'article', 'body', 'news', 'story']
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in content_keywords):
            mappings['content'].append(col)
    
    # 标题列候选
    title_keywords = ['title', 'headline', 'header', 'subject']
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in title_keywords):
            mappings['title'].append(col)
    
    # ID列候选
    id_keywords = ['id', 'index', 'key', 'identifier', 'article_id']
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in id_keywords):
            mappings['id'].append(col)
    
    # 框架平均分数列候选
    avg_keywords = ['sv_frame_avg', 'frame_avg', 'avg_frame', 'average']
    for col in columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in avg_keywords):
            mappings['sv_frame_avg'].append(col)
    
    # 框架分数列候选 - 支持用户的详细问题格式
    frame_mappings = {
        'y_conflict': ['conflict', '争议', '冲突', 'dispute', 'tension', 'disagreement', 'sv_conflict'],
        'y_human': ['human', 'interest', '人情', '人文', 'personal', 'emotion', 'sv_human'],
        'y_econ': ['econ', 'economic', '经济', 'financial', 'money', 'cost', 'sv_econ'],
        'y_moral': ['moral', 'ethic', '道德', '伦理', 'value', 'principle', 'sv_moral', 'religious'],
        'y_resp': ['resp', 'responsibility', '责任', 'account', 'blame', 'sv_resp', 'government']
    }
    
    for frame, keywords in frame_mappings.items():
        for col in columns:
            col_lower = col.lower()
            # 查找聚合分数列（不包含问题编号）
            if any(keyword in col_lower for keyword in keywords) and not any(q in col_lower for q in ['q1', 'q2', 'q3', 'q4', 'q5']):
                mappings[frame].append(col)
    
    # 检测用户的详细问题格式
    user_question_patterns = {
        'conflict_questions': [col for col in columns if 'sv_conflict_q' in col.lower()],
        'human_questions': [col for col in columns if 'sv_human_q' in col.lower()],
        'econ_questions': [col for col in columns if 'sv_econ_q' in col.lower()],
        'moral_questions': [col for col in columns if 'sv_moral_q' in col.lower()],
        'resp_questions': [col for col in columns if 'sv_resp_q' in col.lower()]
    }
    
    # 如果发现问题级别的列，添加到映射中
    for pattern_name, question_cols in user_question_patterns.items():
        if question_cols:
            frame_name = pattern_name.replace('_questions', '')
            mappings[f'y_{frame_name}_questions'] = question_cols
    
    # 移除空列表
    mappings = {k: v for k, v in mappings.items() if v}
    
    return mappings

def print_inspection_report(results: Dict[str, Any]):
    """打印检查报告"""
    print("\n" + "=" * 80)
    print("数据文件检查报告")
    print("=" * 80)
    
    print(f"文件路径: {results['file_path']}")
    print(f"数据行数: {results['total_rows']:,}")
    print(f"列数: {results['total_columns']}")
    
    # 检测用户的特定格式
    user_format_detected = any('sv_' in col and '_q' in col for col in results['columns'])
    if user_format_detected:
        print(f"\n✅ 检测到用户机器标注数据格式 (包含详细问题级别列)")
        
        # 统计问题级别列
        question_cols = [col for col in results['columns'] if 'sv_' in col and '_q' in col]
        frame_types = set()
        for col in question_cols:
            if 'conflict' in col:
                frame_types.add('冲突框架')
            elif 'human' in col:
                frame_types.add('人情框架')
            elif 'econ' in col:
                frame_types.add('经济框架')
            elif 'moral' in col:
                frame_types.add('道德框架')
            elif 'resp' in col:
                frame_types.add('责任框架')
        
        print(f"   问题级别列数: {len(question_cols)}")
        print(f"   涵盖框架类型: {', '.join(sorted(frame_types))}")
        
        # 检查是否有聚合分数列
        if 'sv_frame_avg' in results['columns']:
            print(f"   ✅ 发现框架平均分数列: sv_frame_avg")
        else:
            print(f"   ⚠️  未发现框架平均分数列，将从问题级别列计算")
    
    print(f"\n列信息:")
    print("-" * 60)
    
    # 分类显示列信息
    basic_cols = ['article_id', 'author', 'title', 'content', 'url', 'section', 'publication']
    annotation_cols = ['annotator_id', 'sv_frame_avg', 'avg_stratum', 'sv_notes']
    question_cols = [col for col in results['columns'] if 'sv_' in col and '_q' in col]
    other_cols = [col for col in results['columns'] 
                  if col not in basic_cols + annotation_cols + question_cols]
    
    def print_column_group(title, cols):
        if cols:
            print(f"\n{title}:")
            for col in cols:
                if col in results['columns']:
                    idx = results['columns'].index(col) + 1
                    dtype = results['data_types'][col]
                    missing = results['missing_values'][col]
                    sample = results['sample_data'][col]
                    
                    print(f"  {idx:2d}. {col}")
                    print(f"      类型: {dtype}, 缺失: {missing}, 示例: {sample}")
                    
                    if col in results['value_ranges']:
                        ranges = results['value_ranges'][col]
                        print(f"      范围: {ranges['min']:.3f} - {ranges['max']:.3f} (均值: {ranges['mean']:.3f})")
    
    print_column_group("基本信息列", [col for col in basic_cols if col in results['columns']])
    print_column_group("标注信息列", [col for col in annotation_cols if col in results['columns']])
    
    if question_cols:
        print(f"\n问题级别列 ({len(question_cols)} 个):")
        # 按框架类型分组显示
        frame_groups = {}
        for col in question_cols:
            for frame in ['conflict', 'human', 'econ', 'moral', 'resp']:
                if frame in col:
                    if frame not in frame_groups:
                        frame_groups[frame] = []
                    frame_groups[frame].append(col)
                    break
        
        for frame, cols in frame_groups.items():
            print(f"  {frame.upper()} ({len(cols)} 个问题):")
            for col in sorted(cols):
                idx = results['columns'].index(col) + 1
                dtype = results['data_types'][col]
                missing = results['missing_values'][col]
                print(f"    {idx:2d}. {col} ({dtype}, 缺失: {missing})")
    
    print_column_group("其他列", other_cols)
    
    print("\n智能列映射建议:")
    print("-" * 60)
    if results['potential_mappings']:
        for target, candidates in results['potential_mappings'].items():
            if 'questions' in target:
                print(f"{target:20s}: {len(candidates)} 个问题列")
            else:
                print(f"{target:20s}: {', '.join(candidates)}")
    else:
        print("未找到明显的列映射模式，需要手动指定")
    
    # 数据质量评估
    print(f"\n数据质量评估:")
    print("-" * 60)
    
    # 检查关键列的完整性
    key_completeness = {}
    for col in ['content', 'title', 'sv_frame_avg']:
        if col in results['columns']:
            missing_pct = (results['missing_values'][col] / results['total_rows']) * 100
            key_completeness[col] = 100 - missing_pct
    
    if key_completeness:
        print("关键列完整性:")
        for col, completeness in key_completeness.items():
            status = "✅" if completeness > 95 else "⚠️" if completeness > 80 else "❌"
            print(f"  {status} {col}: {completeness:.1f}%")
    
    # 数据规模评估
    total_rows = results['total_rows']
    if total_rows < 300:
        scale_assessment = "❌ 数据量较小，建议增加到300+样本"
    elif total_rows < 1000:
        scale_assessment = "⚠️ 数据量适中，可用于初步训练"
    elif total_rows < 5000:
        scale_assessment = "✅ 数据量良好，适合模型训练"
    else:
        scale_assessment = "✅ 数据量充足，可进行高质量训练"
    
    print(f"数据规模: {scale_assessment}")
    
    print("\n" + "=" * 80)

def create_column_mapping_interactive(columns: List[str], suggestions: Dict[str, List[str]]) -> Dict[str, str]:
    """交互式创建列映射"""
    print("\n请指定列映射 (输入列号或列名，回车跳过):")
    print("-" * 50)
    
    mapping = {}
    
    required_fields = {
        'content': '文章内容列',
        'y_conflict': '冲突框架分数列',
        'y_human': '人情框架分数列', 
        'y_econ': '经济框架分数列',
        'y_moral': '道德框架分数列',
        'y_resp': '责任框架分数列'
    }
    
    optional_fields = {
        'title': '文章标题列',
        'id': '文章ID列'
    }
    
    # 显示列列表
    print("可用列:")
    for i, col in enumerate(columns):
        print(f"  {i+1:2d}. {col}")
    print()
    
    # 必需字段
    for field, description in required_fields.items():
        suggestion = suggestions.get(field, [])
        prompt = f"{description}"
        if suggestion:
            prompt += f" (建议: {', '.join(suggestion)})"
        prompt += ": "
        
        while True:
            user_input = input(prompt).strip()
            if not user_input:
                print(f"  ❌ {field} 是必需字段，不能跳过")
                continue
            
            # 尝试解析输入
            selected_col = None
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(columns):
                    selected_col = columns[idx]
            elif user_input in columns:
                selected_col = user_input
            
            if selected_col:
                mapping[field] = selected_col
                print(f"  ✅ {field} -> {selected_col}")
                break
            else:
                print(f"  ❌ 无效输入，请输入1-{len(columns)}的数字或有效列名")
    
    # 可选字段
    for field, description in optional_fields.items():
        suggestion = suggestions.get(field, [])
        prompt = f"{description}"
        if suggestion:
            prompt += f" (建议: {', '.join(suggestion)})"
        prompt += " [可选]: "
        
        user_input = input(prompt).strip()
        if user_input:
            selected_col = None
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(columns):
                    selected_col = columns[idx]
            elif user_input in columns:
                selected_col = user_input
            
            if selected_col:
                mapping[field] = selected_col
                print(f"  ✅ {field} -> {selected_col}")
            else:
                print(f"  ❌ 无效输入，跳过 {field}")
    
    return mapping

def create_column_mapping_auto(columns: List[str], suggestions: Dict[str, List[str]]) -> Dict[str, str]:
    """自动创建列映射"""
    mapping = {}
    
    required_fields = ['content', 'y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
    optional_fields = ['title', 'id']
    
    # 自动映射
    for field in required_fields + optional_fields:
        if field in suggestions and suggestions[field]:
            # 选择第一个建议
            mapping[field] = suggestions[field][0]
            logger.info(f"自动映射: {field} -> {suggestions[field][0]}")
    
    # 检查必需字段
    missing_required = [f for f in required_fields if f not in mapping]
    if missing_required:
        logger.error(f"缺少必需字段的映射: {missing_required}")
        logger.error("请使用交互模式 (--interactive) 手动指定映射")
        return None
    
    return mapping

def adapt_data_format(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """适配数据格式"""
    logger.info("适配数据格式...")
    
    # 创建新的DataFrame
    adapted_df = pd.DataFrame()
    
    # 映射列
    for target_col, source_col in column_mapping.items():
        if source_col in df.columns:
            adapted_df[target_col] = df[source_col].copy()
            logger.info(f"映射列: {source_col} -> {target_col}")
    
    # 数据清理和验证
    original_count = len(adapted_df)
    
    # 移除空内容
    if 'content' in adapted_df.columns:
        adapted_df = adapted_df.dropna(subset=['content'])
        adapted_df = adapted_df[adapted_df['content'].str.strip() != '']
        logger.info(f"移除空内容后: {len(adapted_df)} 行 (原: {original_count})")
    
    # 验证和清理框架分数
    frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
    
    for col in frame_columns:
        if col in adapted_df.columns:
            # 转换为数值类型
            adapted_df[col] = pd.to_numeric(adapted_df[col], errors='coerce')
            
            # 移除缺失值
            before_count = len(adapted_df)
            adapted_df = adapted_df.dropna(subset=[col])
            if len(adapted_df) < before_count:
                logger.info(f"移除{col}缺失值后: {len(adapted_df)} 行")
            
            # 检查值范围
            out_of_range = ((adapted_df[col] < 0) | (adapted_df[col] > 1)).sum()
            if out_of_range > 0:
                logger.warning(f"{col} 有 {out_of_range} 个值超出[0,1]范围")
                # 可选：限制到[0,1]范围
                # adapted_df[col] = adapted_df[col].clip(0, 1)
    
    # 添加缺失的可选列
    if 'title' not in adapted_df.columns:
        adapted_df['title'] = 'Untitled'
    
    if 'id' not in adapted_df.columns:
        adapted_df['id'] = [f'article_{i}' for i in range(len(adapted_df))]
    
    logger.info(f"数据适配完成: {original_count} -> {len(adapted_df)} 行")
    return adapted_df

def validate_adapted_data(df: pd.DataFrame) -> Dict[str, Any]:
    """验证适配后的数据"""
    logger.info("验证适配后的数据...")
    
    # 创建临时CSV文件进行验证
    temp_path = "temp_validation.csv"
    df.to_csv(temp_path, index=False)
    
    try:
        # 使用数据加载器验证
        config = create_sv2000_config()
        data_loader = SV2000DataLoader(temp_path, config)
        validation_results = data_loader.validate_annotation_format()
        
        # 添加额外的统计信息
        validation_results.update({
            'total_samples': len(df),
            'valid_samples': len(df.dropna(subset=['content'])),
            'adapted_columns': list(df.columns),
            'data_loader_mapping': data_loader.column_mapping
        })
        
        # 删除临时文件
        os.remove(temp_path)
        
        return validation_results
    
    except Exception as e:
        # 删除临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 返回基本验证结果
        logger.error(f"Data loader validation failed: {e}, using basic validation")
        return {
            'total_samples': len(df),
            'valid_samples': len(df.dropna(subset=['content'])),
            'adapted_columns': list(df.columns),
            'validation_error': str(e)
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="现有数据验证和适配工具")
    
    # 输入输出参数
    parser.add_argument("--input_path", type=str, required=True,
                       help="输入CSV文件路径")
    parser.add_argument("--output_path", type=str,
                       help="输出适配后的CSV文件路径")
    
    # 操作模式
    parser.add_argument("--inspect_only", action="store_true",
                       help="仅检查数据结构，不进行适配")
    parser.add_argument("--interactive", action="store_true",
                       help="交互式指定列映射")
    parser.add_argument("--auto_adapt", action="store_true",
                       help="自动适配数据格式")
    
    # 列映射参数（手动指定）
    parser.add_argument("--content_col", type=str,
                       help="内容列名")
    parser.add_argument("--title_col", type=str,
                       help="标题列名")
    parser.add_argument("--id_col", type=str,
                       help="ID列名")
    parser.add_argument("--conflict_col", type=str,
                       help="冲突框架分数列名")
    parser.add_argument("--human_col", type=str,
                       help="人情框架分数列名")
    parser.add_argument("--econ_col", type=str,
                       help="经济框架分数列名")
    parser.add_argument("--moral_col", type=str,
                       help="道德框架分数列名")
    parser.add_argument("--resp_col", type=str,
                       help="责任框架分数列名")
    
    # 其他参数
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 检查数据结构
        inspection_results = inspect_data_structure(args.input_path)
        print_inspection_report(inspection_results)
        
        # 如果只是检查，则退出
        if args.inspect_only:
            return
        
        # 读取数据
        df = pd.read_csv(args.input_path)
        
        # 确定列映射
        column_mapping = None
        
        if args.interactive:
            # 交互式映射
            column_mapping = create_column_mapping_interactive(
                inspection_results['columns'],
                inspection_results['potential_mappings']
            )
        
        elif args.auto_adapt:
            # 自动映射
            column_mapping = create_column_mapping_auto(
                inspection_results['columns'],
                inspection_results['potential_mappings']
            )
        
        else:
            # 手动指定映射
            manual_mapping = {
                'content': args.content_col,
                'title': args.title_col,
                'id': args.id_col,
                'y_conflict': args.conflict_col,
                'y_human': args.human_col,
                'y_econ': args.econ_col,
                'y_moral': args.moral_col,
                'y_resp': args.resp_col
            }
            
            # 移除None值
            column_mapping = {k: v for k, v in manual_mapping.items() if v is not None}
            
            # 检查必需字段
            required_fields = ['content', 'y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
            missing_required = [f for f in required_fields if f not in column_mapping]
            
            if missing_required:
                logger.error(f"缺少必需字段: {missing_required}")
                logger.error("请使用 --interactive 或 --auto_adapt 模式，或手动指定所有必需列")
                return
        
        if not column_mapping:
            logger.error("无法创建列映射")
            return
        
        # 适配数据格式
        adapted_df = adapt_data_format(df, column_mapping)
        
        # 验证适配后的数据
        validation_results = validate_adapted_data(adapted_df)
        
        logger.info("适配后数据验证结果:")
        logger.info(f"  总样本数: {validation_results['total_samples']}")
        logger.info(f"  有效样本数: {validation_results['valid_samples']}")
        logger.info(f"  有效率: {validation_results['valid_samples']/validation_results['total_samples']*100:.1f}%")
        
        if validation_results['missing_fields']:
            logger.warning(f"  缺失字段: {validation_results['missing_fields']}")
        
        # 保存适配后的数据
        if args.output_path:
            adapted_df.to_csv(args.output_path, index=False, encoding='utf-8')
            logger.info(f"适配后的数据已保存: {args.output_path}")
        else:
            # 默认输出路径
            input_path = Path(args.input_path)
            output_path = input_path.parent / f"{input_path.stem}_adapted.csv"
            adapted_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"适配后的数据已保存: {output_path}")
        
        logger.info("数据适配完成！现在可以用于训练SV2000模型")
        
    except Exception as e:
        logger.error(f"数据验证和适配过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()