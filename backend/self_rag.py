import json
from typing import Dict, List

from qwen_client import QwenClient


class EvidenceCritiqueScorer:
    """
    在 rerank 之后，对候选 chunk 做一次 LLM-based evidence critique：
    - relevance_score: 相关性
    - support_score: 支持性
    - usefulness_score: 有用性
    然后按加权分数排序，供生成器使用。
    """

    def __init__(self, w_rel: float = 0.4, w_sup: float = 0.4, w_use: float = 0.2):
        self.w_rel = w_rel
        self.w_sup = w_sup
        self.w_use = w_use

    @staticmethod
    def _safe_parse_json(text: str) -> Dict:
        if not text:
            return {
                "relevance_score": 0.0,
                "support_score": 0.0,
                "usefulness_score": 0.0,
                "reason": "empty_response",
            }

        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except Exception:
            pass

        # 尝试提取 ```json ... ```
        if "```" in text:
            try:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(text[start:end + 1])
            except Exception:
                pass

        return {
            "relevance_score": 0.0,
            "support_score": 0.0,
            "usefulness_score": 0.0,
            "reason": "json_parse_failed",
        }

    def _build_prompt(self, query: str, chunk_text: str) -> str:
        return f"""你是文档智能检索系统中的证据评估器。

请根据“用户问题”和“文档片段”打三个分数：
1. relevance_score：与问题的相关性，范围 0~1
2. support_score：这段内容对回答问题的直接支持程度，范围 0~1
3. usefulness_score：这段内容对生成高质量最终答案的帮助程度，范围 0~1

评分标准：
- relevance_score 高：片段内容与问题主题直接匹配
- support_score 高：片段中包含可以直接支撑答案的事实、定义、结论、说明
- usefulness_score 高：片段对组织最终答案有明显帮助，即使它不一定单独足够回答全部问题

要求：
- 只依据问题和片段内容评分
- 输出严格为 JSON
- 不要输出任何额外解释
- 分数请用小数，如 0.82

用户问题：
{query}

文档片段：
{chunk_text}

请严格输出：
{{
  "relevance_score": 0.0,
  "support_score": 0.0,
  "usefulness_score": 0.0,
  "reason": "简短原因"
}}"""

    def score_chunk(self, query: str, chunk: Dict) -> Dict:
        prompt = self._build_prompt(query, chunk.get("text", ""))

        try:
            raw = QwenClient.call(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
                max_retry=3
            )
            data = self._safe_parse_json(raw)
        except Exception as e:
            data = {
                "relevance_score": 0.0,
                "support_score": 0.0,
                "usefulness_score": 0.0,
                "reason": f"llm_call_failed: {e}",
            }

        rel = float(data.get("relevance_score", 0.0))
        sup = float(data.get("support_score", 0.0))
        use = float(data.get("usefulness_score", 0.0))
        final = self.w_rel * rel + self.w_sup * sup + self.w_use * use

        new_chunk = dict(chunk)
        new_chunk["critique"] = {
            "relevance_score": rel,
            "support_score": sup,
            "usefulness_score": use,
            "final_score": final,
            "reason": str(data.get("reason", "")),
        }
        return new_chunk

    def score_chunks(self, query: str, ranked_chunks: List[Dict]) -> List[Dict]:
        scored = [self.score_chunk(query, item) for item in ranked_chunks]
        scored.sort(
            key=lambda x: x.get("critique", {}).get("final_score", 0.0),
            reverse=True
        )
        return scored


class EvidenceSelector:
    """
    按 critique 分数筛选最终送给生成器的 chunk。
    """

    def __init__(self, min_score: float = 0.55):
        self.min_score = min_score

    def select_generation_chunks(
        self,
        ranked_chunks: List[Dict],
        top_n: int = 5,
        max_per_doc: int = 3
    ) -> List[Dict]:
        selected = []
        per_doc_count = {}

        for item in ranked_chunks:
            critique = item.get("critique", {})
            final_score = float(critique.get("final_score", 0.0))

            if final_score < self.min_score:
                continue

            doc_id = item["doc_id"]
            if per_doc_count.get(doc_id, 0) >= max_per_doc:
                continue

            selected.append({
                "doc_id": item["doc_id"],
                "text": item["text"],
                "score": item["score"],
                "meta": item["meta"],
                "critique": critique,
            })
            per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1

            if len(selected) >= top_n:
                break

        return selected

    @staticmethod
    def fallback_select(
        ranked_chunks: List[Dict],
        top_n: int = 5,
        max_per_doc: int = 3
    ) -> List[Dict]:
        """
        如果 critique 后没有块达到阈值，就退回到原 rerank 结果做兜底。
        """
        selected = []
        per_doc_count = {}

        for item in ranked_chunks:
            doc_id = item["doc_id"]
            if per_doc_count.get(doc_id, 0) >= max_per_doc:
                continue

            selected.append({
                "doc_id": item["doc_id"],
                "text": item["text"],
                "score": item["score"],
                "meta": item["meta"],
                "critique": item.get("critique", {}),
            })
            per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1

            if len(selected) >= top_n:
                break

        return selected