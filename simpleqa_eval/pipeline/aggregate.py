"""聚合统计模块

读取 sample_results.csv → 按 Jaccard 分 5 桶 → 输出 bucket_stats.csv
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def aggregate_buckets(
    sample_csv: str,
    output_csv: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """按 Jaccard 分桶并统计各桶指标。

    Args:
        sample_csv: 样本级结果 CSV 路径
        output_csv: 桶统计输出 CSV 路径
        n_buckets: 桶数量（默认 5）

    Returns:
        桶统计 DataFrame
    """
    logger.info(f"📊 读取样本结果: {sample_csv}")
    df = pd.read_csv(sample_csv)
    df["jaccard_proxy"] = pd.to_numeric(df["jaccard_proxy"], errors="coerce")

    total = len(df)
    valid = df["jaccard_proxy"].notna().sum()
    dropped = total - valid
    logger.info(f"   总样本: {total}, 有 Jaccard 值: {valid}, 缺失: {dropped}")
    if dropped > 0:
        logger.warning(f"   ⚠️  {dropped} 条样本无 Jaccard 值 (NaN)，不参与分桶但会记录")

    # 只对有 Jaccard 值的样本分桶
    df_valid = df[df["jaccard_proxy"].notna()].copy()

    if len(df_valid) < n_buckets:
        logger.error(f"   ❌ 有效样本({len(df_valid)})少于桶数({n_buckets})，无法分桶")
        return pd.DataFrame()

    # 分位数分桶
    try:
        df_valid["bucket_num"] = pd.qcut(
            df_valid["jaccard_proxy"],
            q=n_buckets,
            labels=False,
            duplicates="drop",
        )
    except ValueError as e:
        logger.warning(f"   ⚠️  qcut 失败: {e}，改用均匀分桶")
        df_valid["bucket_num"] = pd.cut(
            df_valid["jaccard_proxy"],
            bins=n_buckets,
            labels=False,
        )

    df_valid = df_valid[df_valid["bucket_num"].notna()].copy()
    if df_valid.empty:
        logger.error("   ❌ 分桶后无有效桶（可能 Jaccard 值完全相同），无法统计")
        return pd.DataFrame()

    # T1 = 最高 Jaccard, T5 = 最低
    actual_buckets = df_valid["bucket_num"].nunique()
    if actual_buckets < n_buckets:
        logger.warning(f"   ⚠️  实际只形成 {actual_buckets} 个桶（目标 {n_buckets}）")
    max_bucket = df_valid["bucket_num"].max()
    df_valid["bucket"] = df_valid["bucket_num"].apply(
        lambda x: f"T{int(max_bucket - x) + 1}"
    )

    # 按桶统计
    stats = []
    for bucket_name in sorted(df_valid["bucket"].unique()):
        bucket_df = df_valid[df_valid["bucket"] == bucket_name]
        stat = {
            "bucket": bucket_name,
            "n_samples": len(bucket_df),
            "mean_jaccard": bucket_df["jaccard_proxy"].mean(),
            "min_jaccard": bucket_df["jaccard_proxy"].min(),
            "max_jaccard": bucket_df["jaccard_proxy"].max(),
            "hallucination_rate": bucket_df["is_hallucination"].mean(),
        }

        if "self_consistency" in bucket_df.columns:
            stat["mean_self_consistency"] = bucket_df["self_consistency"].mean()

        if "confidence" in bucket_df.columns:
            stat["mean_confidence"] = bucket_df["confidence"].mean()

        if "wiki_failed" in bucket_df.columns:
            stat["wiki_failure_rate"] = bucket_df["wiki_failed"].mean()

        stats.append(stat)

    stats_df = pd.DataFrame(stats)

    # 输出
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_csv, index=False)
    logger.info(f"💾 桶统计已保存: {output_csv}")

    # 打印表格
    logger.info("\n" + "=" * 70)
    logger.info("  📊 分桶统计结果")
    logger.info("=" * 70)
    for _, row in stats_df.iterrows():
        line = (
            f"  {row['bucket']}  |  "
            f"n={int(row['n_samples']):>4d}  |  "
            f"Jaccard={row['mean_jaccard']:.4f}  |  "
            f"HalluRate={row['hallucination_rate']:.3f}"
        )
        if "mean_self_consistency" in row and pd.notna(row.get("mean_self_consistency")):
            line += f"  |  SelfConsis={row['mean_self_consistency']:.3f}"
        if "mean_confidence" in row and pd.notna(row.get("mean_confidence")):
            line += f"  |  Confidence={row['mean_confidence']:.4f}"
        if "wiki_failure_rate" in row and pd.notna(row.get("wiki_failure_rate")):
            line += f"  |  WikiFail={row['wiki_failure_rate']:.3f}"
        logger.info(line)
    logger.info("=" * 70)

    # 也保存分桶后的样本级结果
    df_valid.to_csv(
        str(Path(output_csv).parent / "sample_results_bucketed.csv"),
        index=False,
    )

    return stats_df
