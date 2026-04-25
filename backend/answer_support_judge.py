import json
import re
from typing import Dict

from qwen_client import QwenClient


class AnswerSupportJudge:
    """
    保留原有输出结构：
    - support
    - support_score
    - completeness_score
    - overall_score
    - reason

    但任务定义改成：
    “评审候选答案是否真正回答了用户问题，且是否与证据一致”
    而不是 claim verification。
    """

    @staticmethod
    def _extract_json(text: str) -> Dict:
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
    def _clip01(value, default=0.0):
        try:
            v = float(value)
        except Exception:
            v = default
        return max(0.0, min(1.0, v))

    @staticmethod
    def _normalize_support(s: str) -> str:
        s = str(s or "").strip()
        if s in {"Fully", "Partially", "No Support"}:
            return s
        return "No Support"

    @staticmethod
    def _fallback(answer: str) -> Dict:
        if answer and len(answer.strip()) > 0:
            return {
                "support": "Partially",
                "support_score": 0.55,
                "completeness_score": 0.50,
                "overall_score": 0.52,
                "reason": "fallback_used",
            }
        return {
            "support": "No Support",
            "support_score": 0.10,
            "completeness_score": 0.10,
            "overall_score": 0.10,
            "reason": "empty_answer",
        }

    @staticmethod
    def judge(query: str, answer: str, evidence_text: str) -> Dict:
        prompt = f"""
你是一个问答质量评审助手，支持中文、英文以及中英文混合场景。

请根据“用户问题、候选答案、证据材料”，评估这个候选答案的质量。

用户问题：
{query}

候选答案：
{answer}

证据材料：
{evidence_text}

评审要求：
1. 重点判断这个候选答案是否真正回答了用户问题，而不是去判定一个 claim 是否成立。
2. 重点判断答案的核心内容是否与证据一致，是否有明显脱离证据的内容。
3. 如果答案大体正确但有概括、推断或不够直接，可评为 Partially。
4. 如果答案与证据明显不一致，或证据无法支持答案核心结论，则评为 No Support。
5. 如果答案既准确又直接回答了问题，并且和证据高度一致，则评为 Fully。
6. 回答语言可以与用户问题语言一致，不需要输出英文说明。
7. 只输出 JSON，不要输出任何额外解释。

输出 JSON 格式固定为：
{{
  "support": "Fully" 或 "Partially" 或 "No Support",
  "support_score": 0.0,
  "completeness_score": 0.0,
  "overall_score": 0.0,
  "reason": "一句简短中文说明"
}}

字段含义：
- support:
  - Fully = 候选答案核心内容被证据充分支持，且确实回答了问题
  - Partially = 候选答案部分被支持，或回答不够完整，或包含适度推断
  - No Support = 候选答案核心结论无法从证据得到支持，或明显偏离问题
- support_score:
  证据支持度，0 到 1
- completeness_score:
  回答完整度，0 到 1
  重点看是否真正回答了用户问题、是否抓住了重点
- overall_score:
  综合质量分，0 到 1
  综合考虑支持度、完整度、表达自然度
- reason:
  一句简短说明，例如“回答基本符合证据，但对作者体会进行了推断”
""".strip()

        raw = QwenClient.call(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=220,
        )

        parsed = AnswerSupportJudge._extract_json(raw)
        if not parsed:
            return AnswerSupportJudge._fallback(answer)

        support = AnswerSupportJudge._normalize_support(parsed.get("support", "No Support"))
        support_score = AnswerSupportJudge._clip01(parsed.get("support_score", 0.0), 0.0)
        completeness_score = AnswerSupportJudge._clip01(parsed.get("completeness_score", 0.0), 0.0)
        overall_score = AnswerSupportJudge._clip01(parsed.get("overall_score", 0.0), 0.0)
        reason = str(parsed.get("reason", "") or "").strip()

        # 如果 overall_score 没给，按 support 给一个兜底
        if overall_score == 0.0:
            if support == "Fully":
                overall_score = max(overall_score, 0.85)
            elif support == "Partially":
                overall_score = max(overall_score, 0.55)
            else:
                overall_score = max(overall_score, 0.10)

        # 如果 support_score 没给，也做一个兜底
        if support_score == 0.0:
            if support == "Fully":
                support_score = 0.90
            elif support == "Partially":
                support_score = 0.60
            else:
                support_score = 0.15

        # completeness_score 没给时兜底
        if completeness_score == 0.0:
            if answer and len(answer.strip()) > 0:
                completeness_score = 0.50
            else:
                completeness_score = 0.10

        return {
            "support": support,
            "support_score": support_score,
            "completeness_score": completeness_score,
            "overall_score": overall_score,
            "reason": reason or "ok",
        }