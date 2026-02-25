"""Jaccard 相似度计算模块"""

from typing import Set


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """计算两个集合的 Jaccard 相似度。

    J(A, B) = |A ∩ B| / |A ∪ B|

    当两个集合均为空时返回 0.0。
    """
    if not set_a and not set_b:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union
