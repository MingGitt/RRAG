import json
import re
from typing import List, Dict

from qwen_client import QwenClient


class CandidateGenerator:
    """
    基于 query + evidence_text 生成多个候选答案。
    兼容中英文混合场景，输出结构尽量与当前 generator.py 的后续逻辑保持一致。

    每个 candidate 至少包含：
    - answer
    - retrieve
    - isrel
    - issup
    - isuse
    """

    @staticmethod
    def _build_prompts(query: str, evidence_text: str) -> List[str]:
        """
        设计 3 种不同风格的 prompt，提高候选多样性：
        1. 直接问答型
        2. 摘要归纳型
        3. 谨慎证据型
        """
        common_rules = f"""
你是一个基于证据回答问题的助手，支持中文、英文以及中英文混合问题。

用户问题：
{query}

证据材料：
{evidence_text}

要求：
1. 优先直接回答用户问题，不要写成“the claim is supported / not supported”这种审稿式表达。
2. 如果问题是中文，就优先用中文回答；如果问题是英文，就优先用英文回答；如果问题是中英文混合，可以自然使用中英文混合，但整体表达要流畅。
3. 回答必须基于证据，不要编造。
4. 如果证据不足，请明确说明“从当前证据来看，无法直接确定……”，但仍尽量给出基于现有证据的最合理总结。
5. 输出必须是 JSON，不要输出 JSON 以外的任何内容。
6. JSON 格式固定为：
{{
  "answer": "...",
  "retrieve": "Yes" 或 "No",
  "isrel": "Relevant" 或 "Irrelevant",
  "issup": "Fully" 或 "Partially" 或 "No Support",
  "isuse": 1-5
}}
"""

        prompt_1 = f"""
{common_rules}

风格要求：
- 直接问答型
- 先给结论，再补一句依据
- 回答尽量简洁自然
""".strip()

        prompt_2 = f"""
{common_rules}

风格要求：
- 摘要归纳型
- 适合“总结、体会、主要内容、核心观点”这类问题
- 回答可以稍微完整一点，但不要过长
""".strip()

        prompt_3 = f"""
{common_rules}

风格要求：
- 谨慎证据型
- 如果证据不完全充分，要明确指出“当前证据没有直接说明……”
- 但不要写成 claim verification 风格
- 仍然要尽量回答用户真正想问的内容
""".strip()

        return [prompt_1, prompt_2, prompt_3]

    @staticmethod
    def _extract_json(text: str) -> Dict:
        """
        尽量从模型输出中提取 JSON。
        """
        if not text:
            return {}

        text = text.strip()

        # 先直接 parse
        try:
            return json.loads(text)
        except Exception:
            pass

        # 再尝试提取 {...}
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                pass

        return {}

    @staticmethod
    def _normalize_candidate(obj: Dict) -> Dict:
        answer = str(obj.get("answer", "") or "").strip()

        retrieve = str(obj.get("retrieve", "No") or "No").strip()
        if retrieve not in {"Yes", "No"}:
            retrieve = "No"

        isrel = str(obj.get("isrel", "Relevant") or "Relevant").strip()
        if isrel not in {"Relevant", "Irrelevant"}:
            isrel = "Relevant"

        issup = str(obj.get("issup", "Partially") or "Partially").strip()
        if issup not in {"Fully", "Partially", "No Support"}:
            issup = "Partially"

        try:
            isuse = int(obj.get("isuse", 3))
        except Exception:
            isuse = 3
        isuse = max(1, min(5, isuse))

        return {
            "answer": answer,
            "retrieve": retrieve,
            "isrel": isrel,
            "issup": issup,
            "isuse": isuse,
        }

    @staticmethod
    def _fallback_candidate(query: str, evidence_text: str) -> Dict:
        """
        当模型没按 JSON 返回时，给一个兜底候选。
        """
        preview = re.sub(r"\s+", " ", evidence_text or "").strip()
        preview = preview[:220] + ("..." if len(preview) > 220 else "")

        return {
            "answer": f"从当前证据来看，可以初步回答该问题，但模型未稳定输出结构化结果。相关证据片段为：{preview}",
            "retrieve": "Yes",
            "isrel": "Relevant",
            "issup": "Partially",
            "isuse": 2,
        }

    @staticmethod
    def generate_candidates(query: str, evidence_text: str, num_candidates: int = 3) -> List[Dict]:
        prompts = CandidateGenerator._build_prompts(query, evidence_text)
        prompts = prompts[:max(1, num_candidates)]

        results: List[Dict] = []

        for prompt in prompts:
            raw = QwenClient.call(
                [{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=512,
            )

            parsed = CandidateGenerator._extract_json(raw)
            if parsed:
                results.append(CandidateGenerator._normalize_candidate(parsed))
            else:
                results.append(CandidateGenerator._fallback_candidate(query, evidence_text))

        # 去重：按 answer 去重
        dedup = []
        seen = set()
        for item in results:
            key = item["answer"].strip()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            dedup.append(item)

        if not dedup:
            dedup.append(CandidateGenerator._fallback_candidate(query, evidence_text))

        return dedup