"""可视化模块

生成趋势图：Hallucination Rate / Self-Consistency / Self-Confidence by Jaccard Bucket
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.use("Agg")  # 非交互模式
logger = logging.getLogger(__name__)


def plot_bucket_trends(
    bucket_csv: str,
    output_dir: str,
):
    """根据 bucket_stats.csv 生成趋势图。

    Args:
        bucket_csv: 桶统计 CSV 路径
        output_dir: 图片输出目录
    """
    logger.info(f"📈 读取桶统计: {bucket_csv}")
    df = pd.read_csv(bucket_csv)

    if df.empty:
        logger.error("❌ 桶统计为空，跳过画图")
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    buckets = df["bucket"].tolist()
    x = range(len(buckets))

    # 图表样式
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.figsize": (8, 5),
    })

    # ── 图 1: Hallucination Rate ──
    fig, ax = plt.subplots()
    ax.bar(x, df["hallucination_rate"], color="#e74c3c", alpha=0.8, edgecolor="black")
    ax.plot(x, df["hallucination_rate"], "o-", color="#c0392b", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_xlabel("Jaccard Bucket (T1=highest co-occurrence)")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Hallucination Rate by Entity Co-occurrence Bucket")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(df["hallucination_rate"]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.tight_layout()
    path1 = out / "hallucination_rate_by_bucket.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    logger.info(f"   ✅ 已保存: {path1}")

    # ── 图 2: Self-Consistency ──
    if "mean_self_consistency" in df.columns:
        fig, ax = plt.subplots()
        ax.bar(x, df["mean_self_consistency"], color="#3498db", alpha=0.8, edgecolor="black")
        ax.plot(x, df["mean_self_consistency"], "o-", color="#2980b9", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("Jaccard Bucket (T1=highest co-occurrence)")
        ax.set_ylabel("Self-Consistency")
        ax.set_title("Self-Consistency by Entity Co-occurrence Bucket")
        ax.set_ylim(0, 1.05)
        for i, v in enumerate(df["mean_self_consistency"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        fig.tight_layout()
        path2 = out / "self_consistency_by_bucket.png"
        fig.savefig(path2, dpi=150)
        plt.close(fig)
        logger.info(f"   ✅ 已保存: {path2}")

    # ── 图 3: Self-Confidence ──
    if "mean_confidence" in df.columns:
        fig, ax = plt.subplots()
        ax.bar(x, df["mean_confidence"], color="#2ecc71", alpha=0.8, edgecolor="black")
        ax.plot(x, df["mean_confidence"], "o-", color="#27ae60", linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(buckets)
        ax.set_xlabel("Jaccard Bucket (T1=highest co-occurrence)")
        ax.set_ylabel("Self-Confidence (avg token probability)")
        ax.set_title("Self-Confidence by Entity Co-occurrence Bucket")
        for i, v in enumerate(df["mean_confidence"]):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=10)
        fig.tight_layout()
        path3 = out / "self_confidence_by_bucket.png"
        fig.savefig(path3, dpi=150)
        plt.close(fig)
        logger.info(f"   ✅ 已保存: {path3}")

    logger.info("📈 所有图表生成完毕")
