"""Wikipedia 文章集合代理

通过 Wikipedia API 获取与实体相关的文章标题集合，
并用 SQLite 做持久化缓存。
"""

import json
import time
import sqlite3
import logging
import requests
from typing import Set, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# ── 全局 Session：设置 User-Agent（Wikipedia 强制要求）+ 代理 ──
WIKI_SESSION = requests.Session()
WIKI_SESSION.headers.update({
    "User-Agent": "SimpleQA-Eval/1.0 (Academic Research; Python/requests)"
})
WIKI_SESSION.proxies = {
    "http": "http://127.0.0.1:1280",
    "https": "http://127.0.0.1:1280",
}


class WikiCache:
    """SQLite 缓存：entity -> article_set"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS wiki_cache (
                entity TEXT PRIMARY KEY,
                articles TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get(self, entity: str) -> Optional[Set[str]]:
        cur = self.conn.execute(
            "SELECT articles FROM wiki_cache WHERE entity = ?", (entity,)
        )
        row = cur.fetchone()
        if row:
            articles = set(json.loads(row[0]))
            # 历史上失败请求可能缓存为空集；空集视为无效缓存并触发重查
            if not articles:
                self.conn.execute("DELETE FROM wiki_cache WHERE entity = ?", (entity,))
                self.conn.commit()
                return None
            return articles
        return None

    def put(self, entity: str, articles: Set[str]):
        if not articles:
            return
        self.conn.execute(
            "INSERT OR REPLACE INTO wiki_cache (entity, articles) VALUES (?, ?)",
            (entity, json.dumps(list(articles))),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def get_article_set(
    entity: str,
    cache: Optional[WikiCache] = None,
    search_limit: int = 50,
) -> Tuple[Set[str], bool]:
    """获取与实体相关的 Wikipedia 文章标题集合。

    策略：
    1. 先搜索实体对应的文章
    2. 获取该文章的内链（links）作为 article set

    Args:
        entity: 实体名称
        cache: WikiCache 实例
        search_limit: 最多取多少篇关联文章

    Returns:
        (相关文章标题集合, 是否请求失败)
    """
    # 检查缓存
    if cache is not None:
        cached = cache.get(entity)
        if cached is not None:
            return cached, False

    articles = set()
    request_failed = False

    try:
        # Step 1: 搜索实体对应的页面
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": entity,
            "srlimit": 3,
            "format": "json",
        }
        resp = WIKI_SESSION.get(WIKI_API_URL, params=search_params, timeout=15)
        resp.raise_for_status()
        search_data = resp.json()

        search_results = search_data.get("query", {}).get("search", [])
        if not search_results:
            return articles, False

        # 取第一个搜索结果的标题
        page_title = search_results[0]["title"]
        articles.add(page_title)

        # Step 2: 获取该页面的内链
        links_params = {
            "action": "query",
            "titles": page_title,
            "prop": "links",
            "pllimit": str(min(search_limit, 500)),
            "format": "json",
        }
        resp = WIKI_SESSION.get(WIKI_API_URL, params=links_params, timeout=15)
        resp.raise_for_status()
        links_data = resp.json()

        pages = links_data.get("query", {}).get("pages", {})
        for page_id, page_info in pages.items():
            for link in page_info.get("links", []):
                title = link.get("title", "")
                if title and not title.startswith("Wikipedia:") and not title.startswith("Help:"):
                    articles.add(title)

        # Step 3: 还可以取 backlinks（哪些页面链接到这个实体）
        backlinks_params = {
            "action": "query",
            "list": "backlinks",
            "bltitle": page_title,
            "bllimit": str(min(search_limit, 500)),
            "format": "json",
        }
        resp = WIKI_SESSION.get(WIKI_API_URL, params=backlinks_params, timeout=15)
        resp.raise_for_status()
        bl_data = resp.json()

        for bl in bl_data.get("query", {}).get("backlinks", []):
            articles.add(bl.get("title", ""))

    except Exception as e:
        request_failed = True
        logger.warning(f"   ⚠️  Wikipedia API 查询失败 (entity='{entity}'): {e}")

    # 清理空字符串
    articles.discard("")

    # 仅缓存成功请求（避免把临时错误固化成永久空结果）
    if cache is not None and not request_failed:
        cache.put(entity, articles)

    return articles, request_failed


def get_article_set_for_entities(
    entities: List[str],
    cache: Optional[WikiCache] = None,
    search_limit: int = 50,
    call_delay: float = 0.2,
) -> Tuple[Set[str], int, int]:
    """获取多个实体的文章集合并集。

    Args:
        entities: 实体列表
        cache: WikiCache 实例
        search_limit: 每个实体最多取多少篇
        call_delay: API 调用间隔秒数

    Returns:
        (所有实体关联文章并集, 失败实体数, 实体总数)
    """
    all_articles = set()
    failed_entities = 0
    for entity in entities:
        arts, failed = get_article_set(entity, cache=cache, search_limit=search_limit)
        all_articles.update(arts)
        if failed:
            failed_entities += 1
        if call_delay > 0:
            time.sleep(call_delay)
    return all_articles, failed_entities, len(entities)
