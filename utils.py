"""
工具函数模块
"""

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def find_hf_cache_model_path(model_id: str) -> Optional[str]:
    """
    在 Hugging Face 缓存中查找已下载的模型快照路径。
    返回值指向具体快照目录，找不到则返回 None。
    """
    normalized_ids = [model_id]
    hyphenated = model_id.replace("_", "-")
    if hyphenated not in normalized_ids:
        normalized_ids.append(hyphenated)

    candidate_roots: List[Path] = []

    # 优先使用huggingface_hub提供的缓存路径
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        candidate_roots.append(Path(HF_HUB_CACHE))
    except Exception:
        pass

    # 环境变量覆盖
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        candidate_roots.append(Path(hf_home) / "hub")

    # 仓库/安装环境的本地虚拟环境缓存（如 /root/autodl-tmp/.venv/hf_cache/hub）
    repo_root = Path(__file__).resolve().parent
    checked_paths = set()
    for base in [repo_root, *repo_root.parents]:
        venv_cache = base / ".venv" / "hf_cache" / "hub"
        if venv_cache.exists() and str(venv_cache) not in checked_paths:
            candidate_roots.append(venv_cache)
            checked_paths.add(str(venv_cache))
        if base.name == ".venv":
            direct_cache = base / "hf_cache" / "hub"
            if direct_cache.exists() and str(direct_cache) not in checked_paths:
                candidate_roots.append(direct_cache)
                checked_paths.add(str(direct_cache))

    # 默认用户缓存
    candidate_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    for cache_root in candidate_roots:
        if not cache_root.exists():
            continue

        for candidate in normalized_ids:
            repo_dir = cache_root / f"models--{candidate.replace('/', '--')}"
            if not repo_dir.exists():
                continue

            snapshot_path: Optional[Path] = None
            refs_main = repo_dir / "refs" / "main"
            if refs_main.exists():
                commit_hash = refs_main.read_text().strip()
                if commit_hash:
                    commit_dir = repo_dir / "snapshots" / commit_hash
                    if commit_dir.exists():
                        snapshot_path = commit_dir

            if snapshot_path is None:
                snapshots_dir = repo_dir / "snapshots"
                if snapshots_dir.exists():
                    snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
                    if snapshots:
                        snapshot_path = snapshots[-1]

            if snapshot_path and snapshot_path.exists():
                logging.info("Resolved cached HF model %s to %s", model_id, snapshot_path)
                return str(snapshot_path)

            logging.info("Using HF cached repo directory for %s at %s", model_id, repo_dir)
            return str(repo_dir)

    return None

def setup_environment():
    """设置运行环境"""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 设置matplotlib中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )

def validate_input_data(articles: List[Dict]) -> List[Dict]:
    """验证输入数据格式"""
    validated = []
    
    for i, article in enumerate(articles):
        if not isinstance(article, dict):
            logging.warning(f"Article {i} is not a dictionary, skipping")
            continue
        
        if 'content' not in article or not article['content']:
            logging.warning(f"Article {i} has no content, skipping")
            continue
        
        # 确保必要字段存在
        validated_article = {
            'content': str(article['content']),
            'title': str(article.get('title', '')),
            'id': str(article.get('id', f'article_{i}'))
        }
        
        # 保留其他字段
        for key, value in article.items():
            if key not in ['content', 'title', 'id']:
                validated_article[key] = value
        
        validated.append(validated_article)
    
    logging.info(f"Validated {len(validated)} articles out of {len(articles)}")
    return validated

def save_results(results: Dict, output_path: str, output_config):
    """保存分析结果"""
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON格式
    if output_config.output_format in ["json", "both"]:
        json_path = output_path if output_path.endswith('.json') else f"{output_path}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"Results saved to JSON: {json_path}")
    
    # 保存CSV格式
    if output_config.output_format in ["csv", "both"]:
        csv_path = output_path.replace('.json', '.csv') if output_path.endswith('.json') else f"{output_path}.csv"
        
        # 展平结果为DataFrame
        df_data = []
        for result in results['results']:
            row = {
                'id': result['id'],
                'title': result['title'],
                'framing_intensity': result['framing_intensity'],
                'pseudo_label': result['pseudo_label']
            }
            
            # 添加组件分数
            if 'components' in result:
                for comp, score in result['components'].items():
                    row[f'component_{comp}'] = score
            
            # 添加统计信息
            if 'statistics' in result:
                for stat, value in result['statistics'].items():
                    row[f'stat_{stat}'] = value
            
            # 添加相对框架信息
            if 'relative_framing' in result:
                for key, value in result['relative_framing'].items():
                    row[f'relative_{key}'] = value
            
            # 添加元数据
            for key, value in result.items():
                if key.startswith('meta_'):
                    row[key] = value
            
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logging.info(f"Results saved to CSV: {csv_path}")
    
    # 生成可视化图表
    if output_config.generate_plots:
        plot_dir = os.path.join(output_dir, 'plots')
        generate_analysis_plots(results, plot_dir, output_config)

def generate_analysis_plots(results: Dict, plot_dir: str, output_config):
    """生成分析图表"""
    
    os.makedirs(plot_dir, exist_ok=True)
    
    # 提取有效结果
    valid_results = [r for r in results['results'] if not r.get('error')]
    if not valid_results:
        logging.warning("No valid results for plotting")
        return
    
    # 1. Framing强度分布图
    if output_config.plot_distribution:
        plot_framing_distribution(valid_results, plot_dir)
    
    # 2. 组件分数对比图
    if output_config.include_components and 'components' in valid_results[0]:
        plot_component_scores(valid_results, plot_dir)
    
    # 3. 伪标签分布图
    plot_pseudo_label_distribution(valid_results, plot_dir)
    
    # 4. 证据片段示例
    if output_config.plot_evidence_examples and output_config.include_evidence:
        plot_evidence_examples(valid_results, plot_dir)
    
    logging.info(f"Plots saved to {plot_dir}")

def plot_framing_distribution(results: List[Dict], plot_dir: str):
    """绘制framing强度分布图"""
    
    scores = [r['framing_intensity'] for r in results]
    
    plt.figure(figsize=(10, 6))
    
    # 直方图
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Framing Intensity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Framing Intensity Scores')
    plt.grid(True, alpha=0.3)
    
    # 箱线图
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=True)
    plt.ylabel('Framing Intensity')
    plt.title('Framing Intensity Box Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'framing_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_component_scores(results: List[Dict], plot_dir: str):
    """绘制组件分数对比图"""
    
    components = ['headline', 'lede', 'narration', 'quotes']
    component_data = {comp: [] for comp in components}
    
    for result in results:
        if 'components' in result:
            for comp in components:
                component_data[comp].append(result['components'].get(comp, 0))
    
    # 箱线图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    data_to_plot = [component_data[comp] for comp in components]
    plt.boxplot(data_to_plot, labels=components)
    plt.ylabel('Bias Score')
    plt.title('Component Bias Scores Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 平均分数条形图
    plt.subplot(1, 2, 2)
    means = [np.mean(component_data[comp]) for comp in components]
    plt.bar(components, means, color=['red', 'orange', 'green', 'blue'], alpha=0.7)
    plt.ylabel('Average Bias Score')
    plt.title('Average Component Bias Scores')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'component_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pseudo_label_distribution(results: List[Dict], plot_dir: str):
    """绘制伪标签分布图"""
    
    labels = [r['pseudo_label'] for r in results]
    label_counts = pd.Series(labels).value_counts()
    
    plt.figure(figsize=(10, 6))
    
    # 饼图
    plt.subplot(1, 2, 1)
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
    plt.pie(label_counts.values, labels=label_counts.index, autopct='%1.1f%%', 
            colors=colors[:len(label_counts)])
    plt.title('Pseudo Label Distribution')
    
    # 条形图
    plt.subplot(1, 2, 2)
    plt.bar(label_counts.index, label_counts.values, 
            color=colors[:len(label_counts)], alpha=0.7)
    plt.xlabel('Pseudo Label')
    plt.ylabel('Count')
    plt.title('Pseudo Label Counts')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'pseudo_label_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_evidence_examples(results: List[Dict], plot_dir: str, max_examples: int = 10):
    """绘制证据片段示例"""
    
    # 选择高分文章的证据
    high_score_results = sorted(results, key=lambda x: x['framing_intensity'], reverse=True)[:max_examples]
    
    fig, axes = plt.subplots(max_examples, 1, figsize=(15, 3*max_examples))
    if max_examples == 1:
        axes = [axes]
    
    for i, result in enumerate(high_score_results):
        if 'evidence' in result and result['evidence']:
            evidence = result['evidence'][:3]  # 取前3个证据
            
            zones = [e['zone'] for e in evidence]
            scores = [e['bias_score'] for e in evidence]
            
            axes[i].barh(range(len(evidence)), scores, color=['red', 'orange', 'yellow'][:len(evidence)])
            axes[i].set_yticks(range(len(evidence)))
            axes[i].set_yticklabels([f"{zones[j]}: {evidence[j]['text'][:50]}..." 
                                   for j in range(len(evidence))])
            axes[i].set_xlabel('Bias Score')
            axes[i].set_title(f"Evidence for: {result['title'][:60]}... (F={result['framing_intensity']:.3f})")
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'evidence_examples.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_report(results: Dict, output_path: str):
    """创建总结报告"""
    
    report_lines = []
    
    # 标题
    report_lines.append("# 框架偏见分析报告")
    report_lines.append("")
    
    # 基本信息
    metadata = results['metadata']
    report_lines.append("## 基本信息")
    report_lines.append(f"- 分析时间: {metadata['timestamp']}")
    report_lines.append(f"- 总文章数: {metadata['total_articles']}")
    report_lines.append(f"- 成功分析: {metadata['successful_analyses']}")
    report_lines.append(f"- 失败分析: {metadata['failed_analyses']}")
    report_lines.append("")
    
    # 阈值信息
    if 'threshold_info' in results:
        thresholds = results['threshold_info']['thresholds']
        report_lines.append("## 伪标签阈值")
        report_lines.append(f"- 正面阈值: {thresholds['positive']:.3f}")
        report_lines.append(f"- 负面阈值: {thresholds['negative']:.3f}")
        report_lines.append("")
    
    # 统计信息
    if 'batch_statistics' in results:
        stats = results['batch_statistics']
        
        report_lines.append("## 框架强度统计")
        framing_stats = stats['framing_intensity']
        report_lines.append(f"- 平均值: {framing_stats['mean']:.3f}")
        report_lines.append(f"- 标准差: {framing_stats['std']:.3f}")
        report_lines.append(f"- 中位数: {framing_stats['median']:.3f}")
        report_lines.append(f"- 范围: {framing_stats['min']:.3f} - {framing_stats['max']:.3f}")
        report_lines.append("")
        
        report_lines.append("## 伪标签分布")
        label_dist = stats['pseudo_label_percentages']
        for label, percentage in label_dist.items():
            report_lines.append(f"- {label}: {percentage:.1f}%")
        report_lines.append("")
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logging.info(f"Summary report saved to {output_path}")

def load_config_from_file(config_path: str):
    """从文件加载配置"""
    from .config import load_config
    return load_config(config_path)

def get_device_info():
    """获取设备信息"""
    import torch
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name()
        info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
        info['memory_allocated'] = torch.cuda.memory_allocated()
    
    return info
