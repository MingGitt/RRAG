# selfrag_llm.py
import re
from typing import Optional

import requests

from config import (
    SELF_RAG_EAS_URL,
    SELF_RAG_EAS_TOKEN,
    SELF_RAG_MODEL_NAME,
    SELF_RAG_MAX_NEW_TOKENS,
    SELF_RAG_TEMPERATURE,
    SELF_RAG_TOP_P,
    SELF_RAG_REQUEST_TIMEOUT,
)


class SelfRagLLM:
    """
    Self-RAG 调用封装（面向 BEIR claim verification 任务）
    目标输出风格：
      Verdict: Supported / Refuted / Not Enough Evidence
      Reason: ...
      Relevance: Relevant / Irrelevant
      Support: Fully / Partially / No Support
      Utility: 1-5
    """

    def __init__(self):
        if not SELF_RAG_EAS_URL:
            raise ValueError("缺少 SELF_RAG_EAS_URL，请在配置中设置")
        if not SELF_RAG_EAS_TOKEN:
            raise ValueError("缺少 SELF_RAG_EAS_TOKEN，请在配置中设置")

    @staticmethod
    def _wrap_instruction(instruction: str) -> str:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def _call_eas(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        if max_new_tokens is None:
            max_new_tokens = SELF_RAG_MAX_NEW_TOKENS
        if temperature is None:
            temperature = SELF_RAG_TEMPERATURE
        if top_p is None:
            top_p = SELF_RAG_TOP_P

        headers = {
            "Authorization": SELF_RAG_EAS_TOKEN,
            "Content-Type": "application/json",
        }

        payload = {
            "model": SELF_RAG_MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }

        url = SELF_RAG_EAS_URL.rstrip("/") + "/completions"

        resp = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=SELF_RAG_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        try:
            text = data["choices"][0]["text"]
            if isinstance(text, str):
                return text.strip()
        except Exception:
            pass

        raise RuntimeError(f"EAS 返回格式异常: {data}")

    def generate(
        self,
        instruction: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        prompt = self._wrap_instruction(instruction)
        return self._call_eas(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_with_evidence(
        self,
        query: str,
        evidence_text: str,
        history_text: str = "",
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        instruction = f"""
You are a scientific claim verification assistant.

Task:
Given a claim and evidence, decide whether the claim is supported by the evidence.

Claim:
{query}

Evidence:
{evidence_text}

Conversation History:
{history_text if history_text else "None"}

Rules:
1. Do NOT repeat the claim word-for-word unless necessary.
2. Do NOT ask follow-up questions.
3. Do NOT output prompt instructions or meta commentary.
4. Base your judgment only on the provided evidence.
5. If evidence is insufficient, say so clearly.

Output format:
Verdict: Supported / Refuted / Not Enough Evidence
Reason: one or two short evidence-grounded sentences
Relevance: Relevant / Irrelevant
Support: Fully / Partially / No Support
Utility: 1-5
""".strip()

        return self.generate(
            instruction=instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    @staticmethod
    def extract_first_tag(text: str, pattern: str, default: str = "") -> str:
        m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            return default
        return m.group(1).strip()

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        text = text or ""
        text = text.replace("\r", "\n")

        # 去掉粘连中的明显控制词边界问题
        text = re.sub(r'(?<!\d)(\d+)\.', r'\1. ', text)
        text = re.sub(r'([.:;!?])([A-Za-z])', r'\1 \2', text)

        # 压缩空白
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def clean_answer(raw_text: str) -> str:
        if not raw_text:
            return ""

        answer = raw_text

        # 清控制 token / 特殊标记
        patterns = [
            r"\[Retrieve=.*?\]",
            r"\[Utility:\d+\]",
            r"\[Relevant\]",
            r"\[Irrelevant\]",
            r"\[Fully supported\]",
            r"\[Partially supported\]",
            r"\[No support / Contradictory\]",
            r"\[Retrieval\]",
            r"\[No Retrieval\]",
            r"\[Continue to Use Evidence\]",
            r"</s>",
            r"</?paragraph>",
            r"<paragraph>",
        ]
        for p in patterns:
            answer = re.sub(p, " ", answer, flags=re.IGNORECASE | re.DOTALL)

        # 清结构行
        answer = re.sub(r"(?im)^\s*(Verdict|Reason|Relevance|Support|Utility)\s*:\s*", "", answer)

        answer = SelfRagLLM._normalize_spacing(answer)
        return answer.strip()

    @staticmethod
    def _extract_field(text: str, field_name: str) -> str:
        """
        从形如 'Verdict: xxx' 中提取字段
        """
        pattern = rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _extract_last_occurrence_label(text: str, labels: list[str], default: str = "Unknown") -> str:
        last_pos = -1
        last_label = default
        lower_text = text.lower()

        for label in labels:
            pos = lower_text.rfind(label.lower())
            if pos > last_pos:
                last_pos = pos
                last_label = label

        return last_label

    @staticmethod
    def parse_structured_output(raw_text: str) -> dict:
        raw_text = raw_text or ""

        # ===== 优先解析新格式 =====
        verdict_line = SelfRagLLM._extract_field(raw_text, "Verdict")
        reason_line = SelfRagLLM._extract_field(raw_text, "Reason")
        relevance_line = SelfRagLLM._extract_field(raw_text, "Relevance")
        support_line = SelfRagLLM._extract_field(raw_text, "Support")
        utility_line = SelfRagLLM._extract_field(raw_text, "Utility")

        verdict_norm = verdict_line.lower().strip()
        if "supported" == verdict_norm:
            verdict = "Supported"
        elif "refuted" == verdict_norm:
            verdict = "Refuted"
        elif "not enough evidence" in verdict_norm:
            verdict = "Not Enough Evidence"
        else:
            verdict = "Unknown"

        relevance_norm = relevance_line.lower().strip()
        if relevance_norm == "relevant":
            isrel = "Relevant"
        elif relevance_norm == "irrelevant":
            isrel = "Irrelevant"
        else:
            # fallback 到旧 token
            has_relevant = "[Relevant]" in raw_text
            has_irrelevant = "[Irrelevant]" in raw_text
            if has_relevant and not has_irrelevant:
                isrel = "Relevant"
            elif has_irrelevant and not has_relevant:
                isrel = "Irrelevant"
            else:
                last_rel = SelfRagLLM._extract_last_occurrence_label(
                    raw_text,
                    ["[Relevant]", "[Irrelevant]"],
                    default="Unknown"
                )
                if last_rel == "[Relevant]":
                    isrel = "Relevant"
                elif last_rel == "[Irrelevant]":
                    isrel = "Irrelevant"
                else:
                    isrel = "Unknown"

        support_norm = support_line.lower().strip()
        if support_norm == "fully":
            issup = "Fully"
        elif support_norm == "partially":
            issup = "Partially"
        elif support_norm == "no support":
            issup = "No Support"
        else:
            # fallback 到旧 token
            last_support = SelfRagLLM._extract_last_occurrence_label(
                raw_text,
                ["[Fully supported]", "[Partially supported]", "[No support / Contradictory]"],
                default="Unknown"
            )
            if last_support == "[Fully supported]":
                issup = "Fully"
            elif last_support == "[Partially supported]":
                issup = "Partially"
            elif last_support == "[No support / Contradictory]":
                issup = "No Support"
            else:
                issup = "Unknown"

        if utility_line.isdigit():
            isuse = int(utility_line)
        else:
            utility = SelfRagLLM.extract_first_tag(raw_text, r"\[Utility:(\d+)\]", default="1")
            isuse = int(utility) if str(utility).isdigit() else 1

        # retrieve：新模板下默认 Yes；若出现旧 token 再细化
        retrieve_yes_markers = ["[Retrieve=Yes]", "[Retrieval]", "[Continue to Use Evidence]"]
        retrieve_no_markers = ["[Retrieve=No]", "[No Retrieval]"]

        yes_found = any(m.lower() in raw_text.lower() for m in retrieve_yes_markers)
        no_found = any(m.lower() in raw_text.lower() for m in retrieve_no_markers)

        if yes_found and not no_found:
            retrieve = "Yes"
        elif no_found and not yes_found:
            retrieve = "No"
        else:
            retrieve = "Yes" if verdict != "Unknown" else "Unknown"

        # answer 组装
        if verdict != "Unknown" and reason_line:
            answer = f"{verdict}. {reason_line}"
        elif verdict != "Unknown":
            answer = verdict
        else:
            answer = SelfRagLLM.clean_answer(raw_text)

        answer = SelfRagLLM._normalize_spacing(answer)

        return {
            "retrieve": retrieve,
            "isrel": isrel,
            "issup": issup,
            "isuse": isuse,
            "followup_query": "None",
            "verdict": verdict,
            "reason": reason_line.strip(),
            "answer": answer,
            "raw_text": raw_text,
        }