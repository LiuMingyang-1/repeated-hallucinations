"""统一配置模块"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.resolve()
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CACHE_DIR = PROJECT_ROOT / "cache"


@dataclass
class Config:
    """SimpleQA 评测链路全局配置"""

    # ── 本地模型 ──
    model_name: str = "Qwen/Qwen2.5-0.5B"
    device: str = "auto"  # "auto" / "cuda" / "cpu" / "mps"

    # ── DeepSeek API（用于判错、实体抽取、答案聚类） ──
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.siliconflow.cn/v1"
    deepseek_model: str = "Pro/deepseek-ai/DeepSeek-V3"

    # ── 采样参数 ──
    n_samples: int = 10          # 每题采样次数
    temperature: float = 0.7     # 采样温度
    max_new_tokens: int = 128    # 单次生成最大 token 数

    # ── 数据 ──
    first_n: int = 200           # 取前 N 条 SimpleQA 样本，-1 = 全部

    # ── Wikipedia 缓存 ──
    wiki_cache_db: str = str(CACHE_DIR / "wiki_cache.db")
    wiki_search_limit: int = 50  # 每个实体最多取多少篇关联文章

    # ── API 调用控制 ──
    api_retry_times: int = 3     # API 失败重试次数
    api_retry_delay: float = 2.0 # 重试间隔（秒）
    api_call_delay: float = 0.3  # 连续调用间隔（秒），防 rate-limit

    def __post_init__(self):
        """环境变量优先覆盖"""
        env_key = os.environ.get("DEEPSEEK_API_KEY")
        if env_key:
            self.deepseek_api_key = env_key

        env_url = os.environ.get("DEEPSEEK_BASE_URL")
        if env_url:
            self.deepseek_base_url = env_url

        env_model = os.environ.get("DEEPSEEK_MODEL")
        if env_model:
            self.deepseek_model = env_model

    def resolve_device(self) -> str:
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def ensure_dirs(self):
        """创建所有必要的输出目录"""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
