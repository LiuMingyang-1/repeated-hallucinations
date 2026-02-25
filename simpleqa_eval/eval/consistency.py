"""答案一致性模块

计算 self-consistency（多次采样的共识答案和一致性比例）。
使用 DeepSeek LLM 做语义聚类分组。
"""

import re
import time
import logging
from typing import List, Tuple, Optional
from collections import Counter
from openai import OpenAI

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """文本归一化：小写、去标点、去冠词、去首尾空白"""
    text = text.lower().strip()
    # 去冠词
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # 去标点
    text = re.sub(r'[^\w\s]', ' ', text)
    # 合并空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text


CLUSTER_PROMPT = """You are grouping answers that are semantically equivalent.

Given a list of answers (numbered), group them by semantic equivalence.
Two answers are equivalent if they refer to the same entity, fact, or value.

Answers:
{answers_list}

Output format: List the group number for each answer, separated by commas.
For example, if answers 1,3,5 are equivalent, and 2,4 are equivalent:
1,1,2,1,2

Output ONLY the comma-separated group numbers, nothing else."""


def compute_consensus_simple(
    answers: List[str],
) -> Tuple[str, int, float]:
    """简单模式：纯文本归一化后取众数。

    Returns:
        (consensus_answer, consensus_count, self_consistency)
    """
    if not answers:
        return "", 0, 0.0

    normalized = [normalize_text(a) for a in answers]
    counter = Counter(normalized)
    mode_norm, mode_count = counter.most_common(1)[0]

    # 找到原始答案中第一个匹配的
    consensus_answer = answers[0]
    for orig, norm in zip(answers, normalized):
        if norm == mode_norm:
            consensus_answer = orig
            break

    self_consistency = mode_count / len(answers)
    return consensus_answer, mode_count, self_consistency


def compute_consensus_llm(
    answers: List[str],
    api_key: str,
    base_url: str,
    model: str,
    retry_times: int = 3,
    retry_delay: float = 2.0,
) -> Tuple[str, int, float]:
    """LLM 语义聚类模式：通过 DeepSeek 将语义相同的答案分组。

    Returns:
        (consensus_answer, consensus_count, self_consistency)
    """
    if not answers:
        return "", 0, 0.0

    if len(answers) == 1:
        return answers[0], 1, 1.0

    # 先尝试简单归一化（如果结果很明确就不用 API）
    normalized = [normalize_text(a) for a in answers]
    counter = Counter(normalized)
    mode_norm, mode_count = counter.most_common(1)[0]

    # 如果归一化后大多数一致（≥60%），直接用简单模式
    if mode_count >= len(answers) * 0.6:
        consensus_answer = answers[0]
        for orig, norm in zip(answers, normalized):
            if norm == mode_norm:
                consensus_answer = orig
                break
        self_consistency = mode_count / len(answers)
        return consensus_answer, mode_count, self_consistency

    # 否则用 LLM 聚类
    answers_list = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = CLUSTER_PROMPT.format(answers_list=answers_list)

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(retry_times):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            groups = _parse_groups(result, len(answers))

            if groups is not None:
                # 按组统计
                group_counter = Counter(groups)
                mode_group, mode_count = group_counter.most_common(1)[0]

                # 找到该组的第一个原始答案
                consensus_answer = answers[0]
                for i, g in enumerate(groups):
                    if g == mode_group:
                        consensus_answer = answers[i]
                        break

                self_consistency = mode_count / len(answers)
                return consensus_answer, mode_count, self_consistency

        except Exception as e:
            logger.warning(f"   ⚠️  聚类 API 调用失败 (attempt {attempt+1}): {e}")
            if attempt < retry_times - 1:
                time.sleep(retry_delay * (attempt + 1))

    # fallback 到简单模式
    logger.warning("   ⚠️  LLM 聚类失败，回退到简单归一化模式")
    return compute_consensus_simple(answers)


def _parse_groups(result: str, expected_len: int) -> Optional[List[int]]:
    """解析 LLM 返回的分组结果"""
    try:
        # 提取数字
        numbers = re.findall(r'\d+', result)
        groups = [int(n) for n in numbers]
        if len(groups) == expected_len:
            return groups
        # 如果数量不匹配，尝试只取前 expected_len 个
        if len(groups) >= expected_len:
            return groups[:expected_len]
    except Exception:
        pass
    return None
