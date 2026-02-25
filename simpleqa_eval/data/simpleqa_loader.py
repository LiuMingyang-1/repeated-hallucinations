"""SimpleQA 数据加载器

从 HuggingFace openai/simple-evals 加载 SimpleQA 数据集。
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def load_simpleqa(first_n: int = -1, cache_path: Optional[str] = None) -> List[Dict]:
    """加载 SimpleQA 数据集。

    优先从本地缓存加载，否则从 HuggingFace 下载。

    Args:
        first_n: 取前 N 条，-1 表示全部
        cache_path: 本地缓存文件路径

    Returns:
        List[dict]，每条包含 qid, question, ground_truth
    """
    from simpleqa_eval.config import CACHE_DIR

    if cache_path is None:
        cache_path = str(CACHE_DIR / "simpleqa_dataset.json")

    cache_file = Path(cache_path)

    # 尝试从缓存加载
    if cache_file.exists():
        logger.info(f"📂 从缓存加载 SimpleQA: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"   缓存中共 {len(data)} 条样本")
        if first_n > 0:
            data = data[:first_n]
            logger.info(f"   截取前 {first_n} 条")
        return data

    # 从 HuggingFace 下载
    logger.info("🌐 从 HuggingFace 下载 SimpleQA 数据集...")
    try:
        from datasets import load_dataset
        ds = load_dataset("basicv8vc/SimpleQA", split="test")
    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.info("💡 尝试备用方式：直接下载 CSV...")
        return _download_csv_fallback(first_n, cache_file)

    data = []
    for idx, row in enumerate(ds):
        item = {
            "qid": idx,
            "question": row.get("problem", ""),
            "ground_truth": row.get("answer", ""),
        }
        data.append(item)

    logger.info(f"✅ 加载完成，共 {len(data)} 条样本")

    # 保存缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 已缓存到 {cache_file}")

    if first_n > 0:
        data = data[:first_n]
        logger.info(f"   截取前 {first_n} 条")

    return data


def _download_csv_fallback(first_n: int, cache_file: Path) -> List[Dict]:
    """备用下载方式：用 requests 直接下载 CSV"""
    import requests

    url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    logger.info(f"   下载 URL: {url}")

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    lines = resp.text.strip().split("\n")
    reader = csv.DictReader(lines)

    data = []
    for idx, row in enumerate(reader):
        item = {
            "qid": idx,
            "question": row.get("problem", ""),
            "ground_truth": row.get("answer", ""),
        }
        data.append(item)

    logger.info(f"✅ CSV 下载完成，共 {len(data)} 条样本")

    # 保存缓存
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 已缓存到 {cache_file}")

    if first_n > 0:
        data = data[:first_n]

    return data
