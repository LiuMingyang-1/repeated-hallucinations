"""答案判错模块

调用 DeepSeek API 判断模型预测与 ground truth 是否语义等价。
"""

import time
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


def _create_client(api_key: str, base_url: str) -> OpenAI:
    """创建 OpenAI 兼容客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)


JUDGE_PROMPT = """You are a strict answer correctness judge. 
Given a ground truth answer and a model's prediction, determine if they are semantically equivalent.

Rules:
- The prediction must convey the same core fact as the ground truth
- Minor differences in phrasing, formatting, or extra details are OK
- If the ground truth is a name, the prediction must refer to the same entity
- If the ground truth is a number/date, the prediction must match exactly
- Partial answers count as INCORRECT

Ground Truth Answer: {ground_truth}
Model Prediction: {prediction}

Output ONLY one word: CORRECT or INCORRECT"""


def llm_judge_correct(
    prediction: str,
    ground_truth: str,
    api_key: str,
    base_url: str,
    model: str,
    retry_times: int = 3,
    retry_delay: float = 2.0,
) -> bool:
    """调用 LLM 判断预测是否正确。

    Args:
        prediction: 模型预测答案
        ground_truth: 标准答案
        api_key: DeepSeek API key
        base_url: API base URL
        model: 模型名称
        retry_times: 重试次数
        retry_delay: 重试间隔秒数

    Returns:
        True 表示答案正确，False 表示错误
    """
    if not prediction or not prediction.strip():
        return False

    client = _create_client(api_key, base_url)
    prompt = JUDGE_PROMPT.format(ground_truth=ground_truth, prediction=prediction)

    for attempt in range(retry_times):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            result = response.choices[0].message.content.strip().upper()

            if "CORRECT" in result and "INCORRECT" not in result:
                return True
            elif "INCORRECT" in result:
                return False
            else:
                logger.warning(f"   ⚠️  LLM 判错返回异常: '{result}'，视为 INCORRECT")
                return False

        except Exception as e:
            logger.warning(f"   ⚠️  API 调用失败 (attempt {attempt+1}/{retry_times}): {e}")
            if attempt < retry_times - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                logger.error(f"   ❌ API 调用彻底失败，默认判为 INCORRECT")
                return False

    return False
