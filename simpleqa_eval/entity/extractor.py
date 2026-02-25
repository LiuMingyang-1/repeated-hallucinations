"""实体抽取模块

调用 DeepSeek API 从文本中抽取可在 Wikipedia 找到的实体。
"""

import time
import re
import logging
from typing import List
from openai import OpenAI

logger = logging.getLogger(__name__)


ENTITY_PROMPT = """Extract named entities from the following text that could be found as Wikipedia articles.
Include: person names, place names, organization names, event names, dates, etc.

Text: {text}

Output ONLY a JSON list of entity strings, e.g.: ["Entity1", "Entity2"]
If no entities found, output: []"""


def extract_entities(
    text: str,
    api_key: str,
    base_url: str,
    model: str,
    retry_times: int = 3,
    retry_delay: float = 2.0,
) -> List[str]:
    """从文本中抽取实体列表。

    Args:
        text: 输入文本（问题或答案）
        api_key, base_url, model: DeepSeek API 配置

    Returns:
        去重后的实体列表
    """
    if not text or not text.strip():
        return []

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = ENTITY_PROMPT.format(text=text)

    for attempt in range(retry_times):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0,
            )
            result = response.choices[0].message.content.strip()
            entities = _parse_entity_list(result)
            return entities

        except Exception as e:
            logger.warning(f"   ⚠️  实体抽取 API 失败 (attempt {attempt+1}): {e}")
            if attempt < retry_times - 1:
                time.sleep(retry_delay * (attempt + 1))

    logger.error(f"   ❌ 实体抽取彻底失败，返回空列表")
    return []


def _parse_entity_list(result: str) -> List[str]:
    """解析 LLM 返回的实体 JSON 列表"""
    import json

    # 尝试直接 JSON 解析
    try:
        # 找到 JSON 数组部分
        match = re.search(r'\[.*?\]', result, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            if isinstance(entities, list):
                return list(dict.fromkeys(str(e).strip() for e in entities if e))
    except (json.JSONDecodeError, ValueError):
        pass

    # fallback：按行/逗号分割
    entities = []
    for line in result.replace(",", "\n").split("\n"):
        line = line.strip().strip('"').strip("'").strip("-").strip("•").strip()
        if line and len(line) > 1 and not line.startswith("[") and not line.startswith("]"):
            entities.append(line)

    return list(dict.fromkeys(entities))
