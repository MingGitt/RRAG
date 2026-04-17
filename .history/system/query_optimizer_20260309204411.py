import json
import os
import re
import threading
from typing import Dict, List, Tuple, Optional

from qwen_client import QwenClient


class QueryOptimizer:
    """
    查询优化器：
    - simple: 普通查询，直接检索
    - fuzzy: 模糊查询，改写成多个平行查询
    - complex: 复杂查询，拆成多个子查询

    带本地 JSON 缓存，避免重复调用 LLM。
    """

    CACHE_FILE = "./query_optimizer_cache.json"
    _cache_lock = threading.Lock()
    _cache: Optional[Dict] = None

    @classmethod
    def _ensure_cache_loaded(cls):
        if cls._cache is not None:
            return

        with cls._cache_lock:
            if cls._cache is not None:
                return

            if os.path.exists(cls.CACHE_FILE):
                try:
                    with open(cls.CACHE_FILE, "r", encoding="utf-8") as f:
                        cls._cache = json.load(f)
                except Exception:
                    cls._cache = {}
            else:
                cls._cache = {}

            cls._cache.setdefault("classify", {})
            cls._cache.setdefault("rewrite", {})
            cls._cache.setdefault("decompose", {})

    @classmethod
    def _save_cache(cls):
        cls._ensure_cache_loaded()
        with cls._cache_lock:
            tmp_file = cls.CACHE_FILE + ".tmp"
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(cls._cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_file, cls.CACHE_FILE)

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.sub(r"\s+", " ", query.strip())

    @staticmethod
    def _extract_json(text: str):
        """
        尝试从 LLM 返回中提取 JSON
        兼容：
        1. 纯 JSON
        2. ```json ... ```
        3. 前后带解释文本
        """
        if not text:
            return None

        candidates = [text.strip()]

        code_block_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```",
            text,
            flags=re.DOTALL
        )
        if code_block_match:
            candidates.append(code_block_match.group(1).strip())

        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            candidates.append(json_match.group(0).strip())

        for item in candidates:
            try:
                return json.loads(item)
            except Exception:
                continue

        return None

    @staticmethod
    def _parse_lines(text: str) -> List[str]:
        """
        将 LLM 返回解析为按行列表，兼容：
        1. xxx
        2) xxx
        - xxx
        """
        results = []
        if not text:
            return results

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip()

            if line:
                results.append(line)

        return results

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        dedup = []
        seen = set()

        for item in items:
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                dedup.append(item.strip())

        return dedup

    @classmethod
    def classify_query(cls, query: str) -> str:
        """
        用 LLM 判断查询类型：
        - simple
        - fuzzy
        - complex
        """
        cls._ensure_cache_loaded()
        norm_query = cls._normalize_query(query)

        cached = cls._cache["classify"].get(norm_query)
        if cached in {"simple", "fuzzy", "complex"}:
            return cached

        prompt = f"""
你是一个 RAG 检索系统中的“查询路由分类器”。

请将下面的用户查询严格分类为以下三类之一：

1. simple
含义：
- 查询表达清晰
- 信息需求单一
- 可以直接拿去检索
- 不需要改写，也不需要拆分

2. fuzzy
含义：
- 查询表达模糊、不完整、口语化、指代不清，或者检索词不够好
- 更适合改写成多个更清晰但语义等价的查询，再分别检索

3. complex
含义：
- 查询包含多个子问题、多条件、比较、分步骤、多个信息维度，或需要拆成多个子查询分别检索

请只输出 JSON，不要输出任何解释，不要加代码块。

输出格式：
{{"type":"simple|fuzzy|complex","reason":"简短原因"}}

用户查询：
{norm_query}
""".strip()

        resp = QwenClient.call(
            [{"role": "user", "content": prompt}],
            max_tokens=128
        )

        data = cls._extract_json(resp)
        if data and isinstance(data, dict):
            qtype = str(data.get("type", "")).strip().lower()
            if qtype in {"simple", "fuzzy", "complex"}:
                cls._cache["classify"][norm_query] = qtype
                cls._save_cache()
                return qtype

        # 兜底规则
        q = norm_query.lower()
        if len(norm_query) > 40:
            qtype = "complex"
        elif any(x in q for x in ["compare", "vs", "difference", "比较", "对比", "区别", "分别", "以及", "并且"]):
            qtype = "complex"
        elif len(norm_query) < 12 or any(x in q for x in ["这个", "那个", "它", "this", "that", "it"]):
            qtype = "fuzzy"
        else:
            qtype = "simple"

        cls._cache["classify"][norm_query] = qtype
        cls._save_cache()
        return qtype

    @classmethod
    def rewrite_query(cls, query: str, max_rewrites: int = 3) -> List[str]:
        """
        模糊查询 -> 多个平行改写查询
        """
        cls._ensure_cache_loaded()
        norm_query = cls._normalize_query(query)

        cache_key = f"{norm_query}||{max_rewrites}"
        cached = cls._cache["rewrite"].get(cache_key)
        if isinstance(cached, list) and cached:
            return cached

        prompt = f"""
你是一个检索查询优化助手。

请把下面这个“模糊、不完整或表达不够清晰”的查询，改写成 {max_rewrites} 个更适合检索的查询。

要求：
1. 保持原始意图不变
2. 每个改写查询都应该更清晰、更利于检索
3. 可以使用更明确的表达、近义词、相关术语
4. 每行只输出一个查询
5. 不要编号
6. 不要解释
7. 不要输出多余内容

原查询：
{norm_query}
""".strip()

        resp = QwenClient.call(
            [{"role": "user", "content": prompt}],
            max_tokens=256
        )

        rewrites = cls._parse_lines(resp)
        rewrites = [
            q for q in rewrites
            if q and q.strip() and q.strip() != norm_query
        ]

        final_queries = [norm_query] + rewrites[:max_rewrites]
        final_queries = cls._dedup_keep_order(final_queries)

        if not final_queries:
            final_queries = [norm_query]

        cls._cache["rewrite"][cache_key] = final_queries
        cls._save_cache()
        return final_queries

    @classmethod
    def decompose_query(cls, query: str, max_subqueries: int = 4) -> List[str]:
        """
        复杂查询 -> 拆成多个子查询
        """
        cls._ensure_cache_loaded()
        norm_query = cls._normalize_query(query)

        cache_key = f"{norm_query}||{max_subqueries}"
        cached = cls._cache["decompose"].get(cache_key)
        if isinstance(cached, list) and cached:
            return cached

        prompt = f"""
你是一个检索查询拆分助手。

请把下面这个“复杂查询”拆成最多 {max_subqueries} 个适合分别检索的子查询。

要求：
1. 每个子查询只表达一个原子信息需求
2. 所有子查询合起来应尽量覆盖原查询
3. 子查询要简洁、清晰、适合检索
4. 每行只输出一个子查询
5. 不要编号
6. 不要解释
7. 不要输出多余内容

原查询：
{norm_query}
""".strip()

        resp = QwenClient.call(
            [{"role": "user", "content": prompt}],
            max_tokens=256
        )

        subqueries = cls._parse_lines(resp)
        subqueries = [q for q in subqueries if q and q.strip()]
        subqueries = cls._dedup_keep_order(subqueries[:max_subqueries])

        if not subqueries:
            subqueries = [norm_query]

        cls._cache["decompose"][cache_key] = subqueries
        cls._save_cache()
        return subqueries

    @classmethod
    def expand(cls, query: str) -> Tuple[str, List[str]]:
        """
        返回:
        qtype, expanded_queries
        """
        qtype = cls.classify_query(query)

        if qtype == "simple":
            return qtype, [cls._normalize_query(query)]

        if qtype == "fuzzy":
            return qtype, cls.rewrite_query(query)

        if qtype == "complex":
            return qtype, cls.decompose_query(query)

        return "simple", [cls._normalize_query(query)]