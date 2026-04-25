# answer_support_judge.py
from typing import Dict

from config import QWEN_SUPPORT_MODEL
from qwen_client import QwenClient


class AnswerSupportJudge:
    """
    多候选答案打分器：
    对每个 candidate answer 判断：
    1. 是否被证据支持
    2. 支持强度
    3. 回答是否完整
    4. 最终综合分
    """

    @staticmethod
    def judge(query: str, answer: str, evidence_text: str) -> Dict:
        prompt = f"""
You are an answer support judge.

Claim / user query:
{query}

Candidate answer:
{answer}

Evidence:
{evidence_text}

Task:
Judge whether the candidate answer is supported by the evidence, and how good this answer is.

Return JSON only:
{{
  "support": "Fully" or "Partially" or "No Support",
  "support_score": 0.0,
  "completeness_score": 0.0,
  "overall_score": 0.0,
  "reason": "short reason"
}}

Scoring guide:
- support_score: whether the answer is grounded in evidence
- completeness_score: whether the answer sufficiently addresses the query
- overall_score: your final combined quality score from 0.0 to 1.0
""".strip()

        result = QwenClient.call_json(
            [{"role": "user", "content": prompt}],
            model=QWEN_SUPPORT_MODEL,
            temperature=0.0,
            max_tokens=180,
            default={
                "support": "No Support",
                "support_score": 0.0,
                "completeness_score": 0.0,
                "overall_score": 0.0,
                "reason": "parse_failed",
            },
        )

        support = str(result.get("support", "No Support")).strip()
        if support not in {"Fully", "Partially", "No Support"}:
            support = "No Support"

        def _safe_float(v, default=0.0):
            try:
                x = float(v)
                return max(0.0, min(x, 1.0))
            except Exception:
                return default

        support_score = _safe_float(result.get("support_score", 0.0), 0.0)
        completeness_score = _safe_float(result.get("completeness_score", 0.0), 0.0)
        overall_score = _safe_float(result.get("overall_score", 0.0), 0.0)

        # 如果 overall_score 没给好，做一个兜底映射
        if overall_score == 0.0:
            if support == "Fully":
                overall_score = max(overall_score, 0.85)
            elif support == "Partially":
                overall_score = max(overall_score, 0.55)
            else:
                overall_score = max(overall_score, 0.10)

        return {
            "support": support,
            "support_score": support_score,
            "completeness_score": completeness_score,
            "overall_score": overall_score,
            "reason": str(result.get("reason", "")).strip(),
        }