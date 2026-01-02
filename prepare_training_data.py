#!/usr/bin/env python3
"""
训练数据准备工具
用于准备和验证SV2000训练数据格式
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
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

def validate_csv_format(csv_path: str) -> Dict[str, Any]:
    """验证CSV文件格式"""
    logger.info(f"验证CSV文件格式: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在: {csv_path}")
    
    # 读取CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"无法读取CSV文件: {e}")
    
    # 检查必需列
    required_columns = ['content']
    frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
    optional_columns = ['title', 'id']
    
    validation_results = {
        'total_rows': len(df),
        'columns': list(df.columns),
        'missing_required': [],
        'missing_frame': [],
        'has_optional': {},
        'data_quality': {}
    }
    
    # 检查必需列
    for col in required_columns:
        if col not in df.columns:
            validation_results['missing_required'].append(col)
    
    # 检查框架列
    for col in frame_columns:
        if col not in df.columns:
            validation_results['missing_frame'].append(col)
    
    # 检查可选列
    for col in optional_columns:
        validation_results['has_optional'][col] = col in df.columns
    
    # 数据质量检查
    if 'content' in df.columns:
        content_stats = {
            'empty_content': df['content'].isna().sum(),
            'avg_length': df['content'].str.len().mean() if not df['content'].isna().all() else 0,
            'min_length': df['content'].str.len().min() if not df['content'].isna().all() else 0,
            'max_length': df['content'].str.len().max() if not df['content'].isna().all() else 0
        }
        validation_results['data_quality']['content'] = content_stats
    
    # 框架分数质量检查
    frame_stats = {}
    for col in frame_columns:
        if col in df.columns:
            frame_stats[col] = {
                'missing': df[col].isna().sum(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'out_of_range': ((df[col] < 0) | (df[col] > 1)).sum()
            }
    validation_results['data_quality']['frames'] = frame_stats
    
    return validation_results

def print_validation_report(validation_results: Dict[str, Any]):
    """打印验证报告"""
    print("\n" + "=" * 60)
    print("数据验证报告")
    print("=" * 60)
    
    print(f"总行数: {validation_results['total_rows']}")
    print(f"列数: {len(validation_results['columns'])}")
    print(f"列名: {', '.join(validation_results['columns'])}")
    
    # 必需列检查
    if validation_results['missing_required']:
        print(f"\n❌ 缺失必需列: {', '.join(validation_results['missing_required'])}")
    else:
        print(f"\n✅ 所有必需列都存在")
    
    # 框架列检查
    if validation_results['missing_frame']:
        print(f"❌ 缺失框架列: {', '.join(validation_results['missing_frame'])}")
    else:
        print(f"✅ 所有框架列都存在")
    
    # 可选列检查
    print(f"\n可选列状态:")
    for col, exists in validation_results['has_optional'].items():
        status = "✅" if exists else "❌"
        print(f"  {status} {col}: {'存在' if exists else '不存在'}")
    
    # 数据质量报告
    print(f"\n数据质量:")
    
    if 'content' in validation_results['data_quality']:
        content_stats = validation_results['data_quality']['content']
        print(f"  内容统计:")
        print(f"    空内容: {content_stats['empty_content']}")
        print(f"    平均长度: {content_stats['avg_length']:.1f} 字符")
        print(f"    长度范围: {content_stats['min_length']} - {content_stats['max_length']}")
    
    if 'frames' in validation_results['data_quality']:
        print(f"  框架分数统计:")
        for frame, stats in validation_results['data_quality']['frames'].items():
            print(f"    {frame}:")
            print(f"      缺失值: {stats['missing']}")
            print(f"      均值: {stats['mean']:.3f}")
            print(f"      标准差: {stats['std']:.3f}")
            print(f"      范围: {stats['min']:.3f} - {stats['max']:.3f}")
            if stats['out_of_range'] > 0:
                print(f"      ❌ 超出[0,1]范围: {stats['out_of_range']} 个")

def clean_data(df: pd.DataFrame, args) -> pd.DataFrame:
    """清理数据"""
    logger.info("清理数据...")
    
    original_count = len(df)
    
    # 移除空内容
    if 'content' in df.columns:
        df = df.dropna(subset=['content'])
        df = df[df['content'].str.strip() != '']
        logger.info(f"移除空内容后: {len(df)} 行 (原: {original_count})")
    
    # 移除过短或过长的内容
    if args.min_content_length or args.max_content_length:
        if 'content' in df.columns:
            content_lengths = df['content'].str.len()
            
            if args.min_content_length:
                df = df[content_lengths >= args.min_content_length]
                logger.info(f"移除过短内容后: {len(df)} 行")
            
            if args.max_content_length:
                df = df[content_lengths <= args.max_content_length]
                logger.info(f"移除过长内容后: {len(df)} 行")
    
    # 清理框架分数
    frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
    
    for col in frame_columns:
        if col in df.columns:
            # 移除缺失值
            before_count = len(df)
            df = df.dropna(subset=[col])
            if len(df) < before_count:
                logger.info(f"移除{col}缺失值后: {len(df)} 行")
            
            # 限制到[0,1]范围
            df[col] = df[col].clip(0, 1)
    
    # 移除所有框架分数都缺失的行
    existing_frame_cols = [col for col in frame_columns if col in df.columns]
    if existing_frame_cols:
        df = df.dropna(subset=existing_frame_cols, how='all')
        logger.info(f"移除所有框架分数缺失的行后: {len(df)} 行")
    
    logger.info(f"数据清理完成: {original_count} -> {len(df)} 行")
    return df

def augment_data(df: pd.DataFrame, args) -> pd.DataFrame:
    """数据增强"""
    if not args.augment:
        return df
    
    logger.info("进行数据增强...")
    
    # 简单的数据增强：添加噪声到框架分数
    frame_columns = ['y_conflict', 'y_human', 'y_econ', 'y_moral', 'y_resp']
    
    augmented_rows = []
    
    for _, row in df.iterrows():
        # 原始行
        augmented_rows.append(row.copy())
        
        # 增强行（添加小量噪声）
        if args.augment_noise > 0:
            augmented_row = row.copy()
            
            for col in frame_columns:
                if col in df.columns and not pd.isna(row[col]):
                    noise = np.random.normal(0, args.augment_noise)
                    augmented_row[col] = np.clip(row[col] + noise, 0, 1)
            
            # 修改ID以避免重复
            if 'id' in augmented_row:
                augmented_row['id'] = str(augmented_row['id']) + '_aug'
            
            augmented_rows.append(augmented_row)
    
    augmented_df = pd.DataFrame(augmented_rows)
    logger.info(f"数据增强完成: {len(df)} -> {len(augmented_df)} 行")
    
    return augmented_df

def split_data(df: pd.DataFrame, args) -> tuple:
    """分割数据"""
    if not args.split:
        return df, None, None
    
    logger.info(f"分割数据 (训练:{args.train_ratio}, 验证:{args.val_ratio}, 测试:{args.test_ratio})")
    
    # 随机打乱
    df = df.sample(frac=1, random_state=args.random_seed).reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end] if args.val_ratio > 0 else None
    test_df = df[val_end:] if args.test_ratio > 0 else None
    
    logger.info(f"数据分割结果:")
    logger.info(f"  训练集: {len(train_df)} 行")
    if val_df is not None:
        logger.info(f"  验证集: {len(val_df)} 行")
    if test_df is not None:
        logger.info(f"  测试集: {len(test_df)} 行")
    
    return train_df, val_df, test_df

def save_processed_data(train_df: pd.DataFrame, val_df: Optional[pd.DataFrame], 
                       test_df: Optional[pd.DataFrame], output_dir: str, base_name: str):
    """保存处理后的数据"""
    logger.info(f"保存处理后的数据到: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存训练集
    train_path = os.path.join(output_dir, f"{base_name}_train.csv")
    train_df.to_csv(train_path, index=False, encoding='utf-8')
    logger.info(f"训练集已保存: {train_path}")
    
    # 保存验证集
    if val_df is not None:
        val_path = os.path.join(output_dir, f"{base_name}_val.csv")
        val_df.to_csv(val_path, index=False, encoding='utf-8')
        logger.info(f"验证集已保存: {val_path}")
    
    # 保存测试集
    if test_df is not None:
        test_path = os.path.join(output_dir, f"{base_name}_test.csv")
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        logger.info(f"测试集已保存: {test_path}")

def generate_sample_data(output_path: str, num_samples: int = 100):
    """生成示例数据"""
    logger.info(f"生成示例数据: {num_samples} 个样本")
    
    np.random.seed(42)
    
    # 示例文章内容模板
    content_templates = [
        "The conflict between {entity1} and {entity2} has escalated, with both sides reporting casualties. Military officials confirmed the dispute centers around {issue}.",
        "Local community members came together to support {entity1} whose {property} was destroyed. Volunteers organized {action}, demonstrating human compassion.",
        "The new economic policy is expected to create {number} jobs while reducing the deficit by {percentage}%. Critics argue the tax increases will burden {group}.",
        "The moral implications of {action} have sparked debate among {group}. Religious leaders and ethicists are calling for {response}.",
        "Government officials are taking responsibility for {issue}. The {position} announced new measures to address {problem} and restore public confidence."
    ]
    
    entities = ["Nation A", "Nation B", "the government", "protesters", "citizens", "officials"]
    issues = ["territorial claims", "trade disputes", "environmental concerns", "human rights", "economic policies"]
    actions = ["food drives", "fundraising", "protests", "reforms", "investigations"]
    
    samples = []
    
    for i in range(num_samples):
        # 随机选择模板和填充内容
        template = np.random.choice(content_templates)
        
        content = template.format(
            entity1=np.random.choice(entities),
            entity2=np.random.choice(entities),
            issue=np.random.choice(issues),
            property=np.random.choice(["home", "business", "farm", "school"]),
            action=np.random.choice(actions),
            number=np.random.randint(100, 10000),
            percentage=np.random.randint(5, 25),
            group=np.random.choice(["families", "businesses", "students", "workers"]),
            position=np.random.choice(["minister", "mayor", "director", "spokesperson"]),
            problem=np.random.choice(issues),
            response=np.random.choice(["dialogue", "reform", "investigation", "action"])
        )
        
        # 生成框架分数（基于内容类型）
        if "conflict" in content.lower() or "dispute" in content.lower():
            y_conflict = np.random.beta(3, 2)  # 偏向高冲突
            y_human = np.random.beta(2, 3)     # 偏向低人情
        elif "community" in content.lower() or "support" in content.lower():
            y_conflict = np.random.beta(2, 3)  # 偏向低冲突
            y_human = np.random.beta(3, 2)     # 偏向高人情
        elif "economic" in content.lower() or "job" in content.lower():
            y_econ = np.random.beta(3, 2)      # 偏向高经济
        else:
            y_conflict = np.random.beta(2, 2)  # 均匀分布
            y_human = np.random.beta(2, 2)
        
        # 其他框架分数
        y_econ = np.random.beta(2, 2) if 'y_econ' not in locals() else y_econ
        y_moral = np.random.beta(2, 2)
        y_resp = np.random.beta(2, 2)
        
        sample = {
            'id': f'sample_{i:04d}',
            'title': f'Sample Article {i+1}',
            'content': content,
            'y_conflict': round(y_conflict, 3),
            'y_human': round(y_human, 3),
            'y_econ': round(y_econ, 3),
            'y_moral': round(y_moral, 3),
            'y_resp': round(y_resp, 3)
        }
        
        samples.append(sample)
        
        # 重置局部变量
        if 'y_econ' in locals():
            del y_econ
    
    # 保存到CSV
    df = pd.DataFrame(samples)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"示例数据已生成: {output_path}")
    return df

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="训练数据准备工具")
    
    # 输入输出参数
    parser.add_argument("--input_path", type=str,
                       help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default="./prepared_data",
                       help="输出目录")
    parser.add_argument("--output_name", type=str, default="sv2000_data",
                       help="输出文件基础名称")
    
    # 数据清理参数
    parser.add_argument("--clean", action="store_true",
                       help="清理数据")
    parser.add_argument("--min_content_length", type=int, default=50,
                       help="最小内容长度")
    parser.add_argument("--max_content_length", type=int, default=10000,
                       help="最大内容长度")
    
    # 数据增强参数
    parser.add_argument("--augment", action="store_true",
                       help="进行数据增强")
    parser.add_argument("--augment_noise", type=float, default=0.05,
                       help="增强噪声标准差")
    
    # 数据分割参数
    parser.add_argument("--split", action="store_true",
                       help="分割数据")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                       help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.15,
                       help="测试集比例")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="随机种子")
    
    # 示例数据生成
    parser.add_argument("--generate_sample", action="store_true",
                       help="生成示例数据")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="示例数据样本数")
    
    # 其他参数
    parser.add_argument("--validate_only", action="store_true",
                       help="仅验证数据格式")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 生成示例数据
        if args.generate_sample:
            sample_path = os.path.join(args.output_dir, f"{args.output_name}_sample.csv")
            os.makedirs(args.output_dir, exist_ok=True)
            generate_sample_data(sample_path, args.num_samples)
            return
        
        # 检查输入文件
        if not args.input_path:
            raise ValueError("必须指定输入文件路径 (--input_path)")
        
        # 验证数据格式
        validation_results = validate_csv_format(args.input_path)
        print_validation_report(validation_results)
        
        # 如果只是验证，则退出
        if args.validate_only:
            return
        
        # 读取数据
        df = pd.read_csv(args.input_path)
        logger.info(f"读取数据: {len(df)} 行")
        
        # 清理数据
        if args.clean:
            df = clean_data(df, args)
        
        # 数据增强
        if args.augment:
            df = augment_data(df, args)
        
        # 分割数据
        train_df, val_df, test_df = split_data(df, args)
        
        # 保存处理后的数据
        save_processed_data(train_df, val_df, test_df, args.output_dir, args.output_name)
        
        logger.info("数据准备完成！")
        
    except Exception as e:
        logger.error(f"数据准备过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()