# candidate_generator.py
import re
from typing import Dict, List, Optional

from selfrag_llm import SelfRagLLM


class CandidateGenerator:
    """
    面向 BEIR claim verification 的候选生成器

    目标：
    1. 生成 verdict + reason 风格的候选
    2. 过滤 query 复读
    3. 过滤问题列表、模板回显、提示词回显
    4. 过滤明显跑题的候选
    """

    def __init__(self, llm: Optional[SelfRagLLM] = None):
        self.llm = llm or SelfRagLLM()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _normalize_compact(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text)

    @staticmethod
    def _token_overlap_ratio(a: str, b: str) -> float:
        a_tokens = set(CandidateGenerator._tokenize(a))
        b_tokens = set(CandidateGenerator._tokenize(b))
        if not a_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens), 1)

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        a_set = set(CandidateGenerator._normalize_compact(a))
        b_set = set(CandidateGenerator._normalize_compact(b))
        if not a_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set | b_set), 1)

    @staticmethod
    def _looks_like_question_list(answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True

        numbered_questions = re.findall(r"\b\d+\.\s*[A-Za-z].*?\?", text)
        if len(numbered_questions) >= 2:
            return True

        if text.count("?") >= 2:
            return True

        normalized = text.lower().strip()
        bad_starts = [
            "does ",
            "do ",
            "is ",
            "are ",
            "can ",
            "could ",
            "would ",
            "should ",
            "what ",
            "why ",
            "how ",
            "whether ",
            "is the following sentence factually correct",
        ]
        if any(normalized.startswith(s) for s in bad_starts):
            return True

        return False

    @staticmethod
    def _looks_like_prompt_echo(answer: str) -> bool:
        text = (answer or "").lower()
        bad_signals = [
            "answer strictly based on the evidence",
            "if the evidence is insufficient",
            "do not fabricate facts",
            "output a direct answer",
            "task:",
            "rules:",
            "output format:",
        ]
        return any(sig in text for sig in bad_signals)

    @staticmethod
    def _looks_like_meta_or_garbage(answer: str) -> bool:
        text = (answer or "").strip().lower()
        if not text:
            return True

        if len(text) < 8:
            return True

        bad_patterns = [
            r"^here('?s| is) ",
            r"^the answer is[: ]",
            r"^output[: ]",
            r"^response[: ]",
            r"^\W+$",
        ]
        for p in bad_patterns:
            if re.search(p, text):
                return True

        return False

    @staticmethod
    def _looks_like_query_copy(answer: str, query: str) -> bool:
        """
        过滤几乎只是复述 claim 的答案
        """
        a_norm = CandidateGenerator._normalize_compact(answer)
        q_norm = CandidateGenerator._normalize_compact(query)

        if not a_norm or not q_norm:
            return False

        if a_norm == q_norm:
            return True

        if q_norm in a_norm and len(a_norm) <= int(len(q_norm) * 1.25):
            return True

        jacc = CandidateGenerator._char_jaccard(answer, query)
        return jacc >= 0.85

    @staticmethod
    def _has_valid_verdict(parsed: Dict) -> bool:
        verdict = (parsed.get("verdict") or "").strip()
        return verdict in {"Supported", "Refuted", "Not Enough Evidence"}

    @staticmethod
    def _is_query_relevant(answer: str, query: str, min_ratio: float = 0.08) -> bool:
        ratio = CandidateGenerator._token_overlap_ratio(answer, query)
        return ratio >= min_ratio

    @staticmethod
    def _is_evidence_relevant(answer: str, evidence_text: str, min_ratio: float = 0.08) -> bool:
        ratio = CandidateGenerator._token_overlap_ratio(answer, evidence_text)
        return ratio >= min_ratio

    def _is_valid_candidate(
        self,
        candidate: Dict,
        query: str,
        evidence_text: str,
    ) -> bool:
        answer = (candidate.get("answer") or "").strip()

        if self._looks_like_question_list(answer):
            return False
        if self._looks_like_prompt_echo(answer):
            return False
        if self._looks_like_meta_or_garbage(answer):
            return False
        if self._looks_like_query_copy(answer, query):
            return False

        # 希望有明确 verdict
        if not self._has_valid_verdict(candidate):
            return False

        # 与 query / evidence 至少要有基本相关性
        if not self._is_query_relevant(answer, query):
            return False
        if not self._is_evidence_relevant(answer, evidence_text):
            return False

        return True

    def _generate_once(
        self,
        query: str,
        evidence_text: str,
        history_text: str,
        temperature: float,
        top_p: float,
    ) -> Dict:
        raw_text = self.llm.generate_with_evidence(
            query=query,
            evidence_text=evidence_text,
            history_text=history_text,
            temperature=temperature,
            top_p=top_p,
        )
        parsed = self.llm.parse_structured_output(raw_text)

        return {
            "answer": (parsed.get("answer") or "").strip(),
            "raw_text": raw_text,
            "retrieve": parsed.get("retrieve", "Unknown"),
            "isrel": parsed.get("isrel", "Unknown"),
            "issup": parsed.get("issup", "Unknown"),
            "isuse": parsed.get("isuse", 1),
            "followup_query": parsed.get("followup_query", "None"),
            "verdict": parsed.get("verdict", "Unknown"),
            "reason": parsed.get("reason", ""),
        }

    def generate_candidates(
        self,
        query: str,
        evidence_text: str,
        history_text: str = "",
        num_candidates: int = 3,
    ) -> List[Dict]:
        candidates: List[Dict] = []
        seen_answers = set()

        # 低温度优先，避免乱飘；中温补多样性
        temperatures = [0.15, 0.25, 0.4, 0.55, 0.7]
        top_ps = [0.8, 0.85, 0.9, 0.92, 0.95]
        max_trials = max(num_candidates * 5, 10)

        trial_idx = 0
        while len(candidates) < num_candidates and trial_idx < max_trials:
            temp = temperatures[trial_idx % len(temperatures)]
            top_p = top_ps[trial_idx % len(top_ps)]
            trial_idx += 1

            cand = self._generate_once(
                query=query,
                evidence_text=evidence_text,
                history_text=history_text,
                temperature=temp,
                top_p=top_p,
            )

            answer = (cand.get("answer") or "").strip()
            if not answer:
                continue

            norm = self._normalize_text(answer)
            if norm in seen_answers:
                continue

            if not self._is_valid_candidate(cand, query, evidence_text):
                continue

            seen_answers.add(norm)
            candidates.append(cand)

        # 兜底：如果严格过滤后没有候选，至少保留一个“有 verdict”的
        if not candidates:
            fallback = self._generate_once(
                query=query,
                evidence_text=evidence_text,
                history_text=history_text,
                temperature=0.2,
                top_p=0.85,
            )
            answer = (fallback.get("answer") or "").strip()
            if answer:
                candidates.append(fallback)

        return candidates[:num_candidates]