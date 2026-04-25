# chunk_critique_judge.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from config import QWEN_RELEVANCE_MODEL, RELEVANCE_MAX_WORKERS
from qwen_client import QwenClient


class ChunkCritiqueJudge:
    """
    一次 LLM 调用同时判断：
    1. relevance: Relevant / Irrelevant
    2. usefulness: Useful / Somewhat Useful / Not Useful
    3. score: 0.0 ~ 1.0

    用途：
    - 直接用于 generation chunk 选择
    """

    @staticmethod
    def judge_one(query: str, chunk_text: str) -> Dict:
        prompt = f"""
You are a chunk critique judge for a document-grounded QA system.

User query:
{query}

Evidence chunk:
{chunk_text}

Task:
Judge this chunk from two perspectives:
1. Relevance: whether the chunk is relevant to answering the query
2. Usefulness: whether the chunk is actually useful as evidence for generating the final answer

Return JSON only:
{{
  "relevance": "Relevant" or "Irrelevant",
  "usefulness": "Useful" or "Somewhat Useful" or "Not Useful",
  "score": 0.0,
  "reason": "short reason"
}}

Scoring rule:
- Use a high score (0.8-1.0) if the chunk directly states the answer or key evidence
- Use a medium score (0.4-0.7) if the chunk is related but only partially useful
- Use a low score (0.0-0.3) if the chunk is irrelevant or mostly background
""".strip()

        result = QwenClient.call_json(
            [{"role": "user", "content": prompt}],
            model=QWEN_RELEVANCE_MODEL,
            temperature=0.0,
            max_tokens=160,
            default={
                "relevance": "Irrelevant",
                "usefulness": "Not Useful",
                "score": 0.0,
                "reason": "parse_failed"
            },
        )

        relevance = str(result.get("relevance", "Irrelevant")).strip()
        usefulness = str(result.get("usefulness", "Not Useful")).strip()

        if relevance not in {"Relevant", "Irrelevant"}:
            relevance = "Irrelevant"

        if usefulness not in {"Useful", "Somewhat Useful", "Not Useful"}:
            usefulness = "Not Useful"

        try:
            score = float(result.get("score", 0.0))
        except Exception:
            score = 0.0

        score = max(0.0, min(score, 1.0))

        return {
            "relevance": relevance,
            "usefulness": usefulness,
            "score": score,
            "reason": str(result.get("reason", "")).strip(),
        }

    @staticmethod
    def judge_chunks(query: str, chunks: List[Dict]) -> List[Dict]:
        if not chunks:
            return []

        results = []

        with ThreadPoolExecutor(max_workers=RELEVANCE_MAX_WORKERS) as executor:
            future_map = {
                executor.submit(
                    ChunkCritiqueJudge.judge_one,
                    query,
                    item.get("text", "") or ""
                ): item
                for item in chunks
            }

            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    judge_result = future.result()
                except Exception as e:
                    judge_result = {
                        "relevance": "Irrelevant",
                        "usefulness": "Not Useful",
                        "score": 0.0,
                        "reason": f"judge_error: {repr(e)}"
                    }

                new_item = dict(item)
                new_item["critique_relevance"] = judge_result["relevance"]
                new_item["critique_usefulness"] = judge_result["usefulness"]
                new_item["critique_score"] = float(judge_result["score"])
                new_item["critique_reason"] = judge_result["reason"]
                results.append(new_item)

        # 先按 critique_score，再按原 rerank score
        results.sort(
            key=lambda x: (
                float(x.get("critique_score", 0.0)),
                float(x.get("score", 0.0))
            ),
            reverse=True
        )
        return results

    @staticmethod
    def select_generation_chunks(
        chunks: List[Dict],
        top_n: int = 5,
        max_per_doc: int = 3,
        min_score: float = 0.35,
    ) -> List[Dict]:
        if not chunks:
            return []

        selected = []
        per_doc_count = {}

        # 只优先选 Relevant 且 score 达标的块
        filtered = [
            x for x in chunks
            if x.get("critique_relevance") == "Relevant"
            and float(x.get("critique_score", 0.0)) >= min_score
        ]

        if not filtered:
            # 没有达标的就回退到前1~3个最高 critique score 的块
            filtered = chunks[:max(1, min(3, len(chunks)))]

        for item in filtered:
            doc_id = item.get("doc_id", "")
            if per_doc_count.get(doc_id, 0) >= max_per_doc:
                continue
            selected.append(item)
            per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
            if len(selected) >= top_n:
                break

        return selected