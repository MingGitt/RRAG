# query_optimizer.py

import json
import re
from typing import List, Tuple

from qwen_client import QwenClient


class QueryOptimizer:
    """
    两阶段设计：

    第一次 LLM 调用：
      classify(query) -> "NO" / "simple" / "fuzzy" / "complex"

    第二次 LLM 调用：
      - fuzzy  -> rewrite_query(query)
      - complex -> decompose_query(query)

    兼容旧接口：
      expand(query) -> (qtype, expanded_queries)
    """

    @staticmethod
    def _extract_json(text: str):
        if not text:
            return {}

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return {}

    @staticmethod
    def _extract_list(text: str) -> List[str]:
        """
        尝试从模型输出中抽取字符串列表
        """
        if not text:
            return []

        text = text.strip()

        # 先试 JSON
        parsed = QueryOptimizer._extract_json(text)
        if isinstance(parsed, dict):
            for key in ["queries", "expanded_queries", "subqueries", "rewrites"]:
                value = parsed.get(key)
                if isinstance(value, list):
                    return [str(x).strip() for x in value if str(x).strip()]

        # 再试直接 JSON list
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass

        # 再试逐行抽取
        lines = []
        for line in text.splitlines():
            line = line.strip()
            line = re.sub(r"^\d+[\.\)\-、]\s*", "", line)
            line = re.sub(r"^[\-\*\•]\s*", "", line)
            if line:
                lines.append(line)

        return lines

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = str(label or "").strip().lower()
        if label == "no":
            return "NO"
        if label in {"simple", "fuzzy", "complex"}:
            return label
        return "simple"

    @staticmethod
    def classify(query: str) -> str:
        """
        第一次 LLM 调用：
        只输出一个标签：
        NO / simple / fuzzy / complex
        """
        prompt = f"""
你是一个查询路由分类器。

请判断下面这个用户问题属于哪一类，只能输出以下四个标签之一，不要输出任何其他内容：

- NO：不需要检索，直接回答即可
- simple：需要检索，但问题单一明确，不需要改写或拆分
- fuzzy：需要检索，问题表述模糊、口语化、同义表达较多，适合改写查询
- complex：需要检索，问题包含多个子问题、多个条件或组合需求，适合拆分成多个子查询

用户问题：
{query}
""".strip()

        raw = QwenClient.call(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=16,
        )

        return QueryOptimizer._normalize_label(raw)

    @staticmethod
    def rewrite_query(query: str) -> List[str]:
        """
        第二次 LLM 调用（仅 fuzzy）
        生成多个改写查询
        """
        prompt = f"""
你是一个查询改写助手。

请把下面这个用户问题改写成 2 到 4 条更适合检索的平行查询。
要求：
1. 保持原意不变
2. 改写后更明确、更利于文档检索
3. 各查询之间尽量有一定表达差异
4. 只输出 JSON

输出格式：
{{
  "queries": ["改写1", "改写2", "改写3"]
}}

用户问题：
{query}
""".strip()

        raw = QwenClient.call(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )

        queries = QueryOptimizer._extract_list(raw)
        queries = [q for q in queries if q]

        if not queries:
            return [query]

        dedup = []
        seen = set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                dedup.append(q)

        return dedup[:5] or [query]

    @staticmethod
    def decompose_query(query: str) -> List[str]:
        """
        第二次 LLM 调用（仅 complex）
        拆分成多个子查询
        """
        prompt = f"""
你是一个复杂问题拆分助手。

请把下面这个复杂问题拆分为 2 到 4 条子查询。
要求：
1. 每个子查询都应是可以独立检索的明确问题
2. 合起来应覆盖原问题的主要信息需求
3. 不要遗漏关键约束
4. 只输出 JSON

输出格式：
{{
  "queries": ["子查询1", "子查询2", "子查询3"]
}}

用户问题：
{query}
""".strip()

        raw = QwenClient.call(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=256,
        )

        queries = QueryOptimizer._extract_list(raw)
        queries = [q for q in queries if q]

        if not queries:
            return [query]

        dedup = []
        seen = set()
        for q in queries:
            if q not in seen:
                seen.add(q)
                dedup.append(q)

        return dedup[:5] or [query]

    @staticmethod
    def expand(query: str) -> Tuple[str, List[str]]:
        """
        兼容旧接口：
        返回 (qtype, expanded_queries)

        注意：
        - 如果 classify = NO，这里返回 ("simple", [query])
          因为 NO 分支应由上层 qa_service 先处理。
        """
        label = QueryOptimizer.classify(query)

        if label == "NO":
            return "simple", [query]
        if label == "simple":
            return "simple", [query]
        if label == "fuzzy":
            return "fuzzy", QueryOptimizer.rewrite_query(query)
        if label == "complex":
            return "complex", QueryOptimizer.decompose_query(query)

        return "simple", [query]