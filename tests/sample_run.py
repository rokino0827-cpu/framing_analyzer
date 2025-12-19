"""
示例小样本批量分析脚本。

用法（在仓库根目录）：
    python tests/sample_run.py

需求：
- 数据集位于 data/all-the-news-2-1_sample.csv_bias_scored.csv
- 结果输出至 results/test_sample
"""

import json
from pathlib import Path

import pandas as pd

from framing_analyzer.analyzer import FramingAnalyzer
from framing_analyzer.config import default_config


def load_sample_articles(path: Path, max_rows: int = 3):
    """加载小样本数据集，返回标准化的文章列表。"""
    df = pd.read_csv(path, encoding="latin-1")
    df = df[df["content"].notna() & df["title"].notna()]
    df = df.head(max_rows)

    articles = []
    for idx, row in df.iterrows():
        articles.append(
            {
                "id": row.get("url") or row.get("publication") or f"article_{idx}",
                "title": str(row["title"]),
                "content": str(row["content"]),
                "bias_label": row.get("bias_label"),
                "bias_probability": row.get("bias_probability"),
            }
        )
    return articles


def main():
    data_path = Path("data/all-the-news-2-1_2025-window_bias_scored_balanced_500_clean.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {data_path}")

    # 使用深拷贝避免修改全局单例
    import copy
    cfg = copy.deepcopy(default_config)
    cfg.output.output_dir = "results/test_sample"

    articles = load_sample_articles(data_path, max_rows=3)
    analyzer = FramingAnalyzer(cfg)
    results = analyzer.analyze_batch(articles, output_path="results/test_sample/run.json")

    print("分析完成，概要：")
    print(json.dumps(results["metadata"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
