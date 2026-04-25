# candidate_generator.py
import re
from typing import Dict, List

from config import QWEN_ANSWER_MODEL

try:
    from config import QWEN_CANDIDATE_MODEL
except Exception:
    QWEN_CANDIDATE_MODEL = QWEN_ANSWER_MODEL

from qwen_client import QwenClient


class CandidateGenerator:
    """
    多候选答案生成器（Qwen版）
    基于同一份 evidence，生成多个候选答案，供后续逐个打分排序。
    """

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _looks_bad(answer: str) -> bool:
        text = (answer or "").strip().lower()
        if not text:
            return True

        bad_patterns = [
            r"^output[: ]",
            r"^response[: ]",
            r"^the answer is[: ]",
            r"^i cannot",
            r"^i can't",
            r"^\W+$",
        ]
        for p in bad_patterns:
            if re.search(p, text):
                return True

        if len(text) < 8:
            return True

        return False

    @staticmethod
    def _build_prompts(query: str, evidence_text: str) -> List[str]:
        """
        三种不同风格，提升候选多样性
        """
        prompt1 = f"""
You are a document-grounded scientific assistant.

User query / claim:
{query}

Evidence:
{evidence_text}

Task:
Answer the user query based only on the evidence.

Rules:
1. Use only the provided evidence.
2. If the evidence is insufficient, say so explicitly.
3. Be concise and factual.
4. Do not ask follow-up questions.
5. Do not output control tokens, JSON, or bullet lists.

Answer:
""".strip()

        prompt2 = f"""
You are a scientific evidence assistant.

Question / claim:
{query}

Evidence:
{evidence_text}

Please produce:
- a short direct answer
- followed by one brief supporting sentence from the evidence

Requirements:
- stay grounded in the evidence
- do not repeat the full query word-for-word
- do not output JSON
""".strip()

        prompt3 = f"""
You are verifying a claim with retrieved evidence.

Claim:
{query}

Evidence:
{evidence_text}

Write a careful answer that:
- states whether the claim is supported, contradicted, or not fully supported
- briefly explains why using the evidence
- stays concise
""".strip()

        return [prompt1, prompt2, prompt3]

    @staticmethod
    def generate_candidates(
        query: str,
        evidence_text: str,
        num_candidates: int = 3,
    ) -> List[Dict]:
        prompts = CandidateGenerator._build_prompts(query, evidence_text)

        # 不同 prompt + 不同 temperature，增加候选差异
        temperatures = [0.2, 0.4, 0.6, 0.3, 0.5]

        candidates: List[Dict] = []
        seen = set()

        trial_idx = 0
        while len(candidates) < num_candidates and trial_idx < max(num_candidates * 4, 8):
            prompt = prompts[trial_idx % len(prompts)]
            temperature = temperatures[trial_idx % len(temperatures)]
            trial_idx += 1

            answer = QwenClient.call(
                [{"role": "user", "content": prompt}],
                model=QWEN_CANDIDATE_MODEL,
                temperature=temperature,
                max_tokens=256,
            ).strip()

            if CandidateGenerator._looks_bad(answer):
                continue

            norm = CandidateGenerator._normalize_text(answer)
            if norm in seen:
                continue
            seen.add(norm)

            candidates.append({
                "answer": answer,
                "raw_text": answer,
            })

        # 兜底：至少保留一个
        if not candidates:
            fallback_prompt = prompts[0]
            answer = QwenClient.call(
                [{"role": "user", "content": fallback_prompt}],
                model=QWEN_CANDIDATE_MODEL,
                temperature=0.2,
                max_tokens=256,
            ).strip()

            if answer:
                candidates.append({
                    "answer": answer,
                    "raw_text": answer,
                })

        return candidates[:num_candidates]