# evidence_scorer.py
import math
import re
from typing import Dict, List, Optional

import numpy as np

from config import RERANK_MODEL_LOCAL_DIR, RERANK_MODEL_NAME
from model_utils import resolve_or_download_model


class EvidenceScorer:
    """
    面向 claim verification 的证据打分器

    分数来源：
    1. semantic_score: CrossEncoder(answer, evidence_chunk)
    2. lexical_score: token overlap
    3. support_bonus: 多证据支持加分
    4. reflection bonus: Self-RAG 的 isrel / issup / isuse
    5. penalties:
       - query copy penalty
       - prompt/template penalty
       - bad format penalty
    """

    _semantic_model = None

    def __init__(self):
        self.model = self._load_semantic_model()

    @classmethod
    def _load_semantic_model(cls):
        if cls._semantic_model is not None:
            return cls._semantic_model

        try:
            from sentence_transformers import CrossEncoder

            model_path = resolve_or_download_model(
                repo_id=RERANK_MODEL_NAME,
                local_dir=RERANK_MODEL_LOCAL_DIR,
                cache_dir=None,
                offline=False,
            )
            cls._semantic_model = CrossEncoder(model_path, device="cpu")
        except Exception as e:
            print(f"[EvidenceScorer][WARN] semantic model load failed: {repr(e)}")
            cls._semantic_model = None

        return cls._semantic_model

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        return re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text)

    @staticmethod
    def _token_overlap_ratio(a: str, b: str) -> float:
        a_tokens = set(EvidenceScorer._tokenize(a))
        b_tokens = set(EvidenceScorer._tokenize(b))
        if not a_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / max(len(a_tokens), 1)

    @staticmethod
    def _normalize_compact(text: str) -> str:
        text = (text or "").lower()
        text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
        return text

    @staticmethod
    def _char_jaccard(a: str, b: str) -> float:
        a_set = set(EvidenceScorer._normalize_compact(a))
        b_set = set(EvidenceScorer._normalize_compact(b))
        if not a_set:
            return 0.0
        return len(a_set & b_set) / max(len(a_set | b_set), 1)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _looks_like_query_copy(answer: str, query: str) -> bool:
        a = EvidenceScorer._normalize_compact(answer)
        q = EvidenceScorer._normalize_compact(query)
        if not a or not q:
            return False
        if a == q:
            return True
        if q in a and len(a) <= int(len(q) * 1.25):
            return True
        return EvidenceScorer._char_jaccard(answer, query) >= 0.85

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
    def _extract_verdict(answer: str, candidate: Dict) -> str:
        verdict = (candidate.get("verdict") or "").strip()
        if verdict in {"Supported", "Refuted", "Not Enough Evidence"}:
            return verdict

        lower = (answer or "").lower()
        if lower.startswith("supported"):
            return "Supported"
        if lower.startswith("refuted"):
            return "Refuted"
        if lower.startswith("not enough evidence"):
            return "Not Enough Evidence"
        return "Unknown"

    def _semantic_chunk_scores(self, answer: str, chunks: List[Dict]) -> List[float]:
        if not self.model or not chunks:
            return [0.0 for _ in chunks]

        pairs = [(answer, item.get("text", "") or "") for item in chunks]
        try:
            raw_scores = self.model.predict(pairs)
            raw_scores = [float(x) for x in raw_scores]
            # 归一化到 0-1
            return [self._sigmoid(x) for x in raw_scores]
        except Exception as e:
            print(f"[EvidenceScorer][WARN] semantic scoring failed: {repr(e)}")
            return [0.0 for _ in chunks]

    def score_one_candidate(self, candidate: Dict, chunks: List[Dict], query: str) -> Dict:
        answer = candidate.get("answer", "")
        verdict = self._extract_verdict(answer, candidate)

        semantic_scores = self._semantic_chunk_scores(answer, chunks)
        lexical_scores = []

        for chunk in chunks:
            chunk_text = chunk.get("text", "") or ""
            lexical = self._token_overlap_ratio(answer, chunk_text)
            lexical_scores.append(lexical)

        fused = []
        for i, chunk in enumerate(chunks):
            rerank_score = float(chunk.get("score", 0.0))
            rerank_weight = max(0.0, min(1.0, 0.5 + 0.05 * rerank_score))

            semantic = semantic_scores[i] if i < len(semantic_scores) else 0.0
            lexical = lexical_scores[i] if i < len(lexical_scores) else 0.0

            chunk_score = (
                0.65 * semantic +
                0.25 * lexical +
                0.10 * rerank_weight
            )

            chunk_id = str(chunk.get("chunk_idx", chunk.get("doc_id", i)))
            fused.append((chunk_id, chunk_score))

        fused.sort(key=lambda x: x[1], reverse=True)

        top1 = fused[0][1] if fused else 0.0
        support_cnt = sum(1 for _, s in fused if s >= 0.45)
        used_chunk_ids = [cid for cid, s in fused[:3] if s >= 0.35]

        evidence_score = min(top1 + min(support_cnt * 0.06, 0.18), 1.0)

        # ===== reflection bonus =====
        reflect_bonus = 0.0
        if candidate.get("isrel") == "Relevant":
            reflect_bonus += 0.03
        if candidate.get("issup") == "Fully":
            reflect_bonus += 0.06
        elif candidate.get("issup") == "Partially":
            reflect_bonus += 0.03
        if int(candidate.get("isuse", 1)) >= 4:
            reflect_bonus += 0.02

        # ===== verdict bonus =====
        verdict_bonus = 0.0
        if verdict in {"Supported", "Refuted", "Not Enough Evidence"}:
            verdict_bonus += 0.04

        # ===== penalties =====
        penalty = 0.0
        if self._looks_like_query_copy(answer, query):
            penalty += 0.18
        if self._looks_like_prompt_echo(answer):
            penalty += 0.20
        if len(answer.strip()) < 15:
            penalty += 0.08

        # 判断题模板惩罚
        lower = answer.lower()
        if "is the following sentence factually correct" in lower:
            penalty += 0.18
        if "options:" in lower:
            penalty += 0.10

        length_bonus = min(math.log(len(answer) + 1) / 25.0, 0.04)

        final_score = max(
            0.0,
            min(evidence_score + reflect_bonus + verdict_bonus + length_bonus - penalty, 1.0)
        )

        new_item = dict(candidate)
        new_item["verdict"] = verdict
        new_item["evidence_score"] = round(evidence_score, 4)
        new_item["final_score"] = round(final_score, 4)
        new_item["used_chunk_ids"] = used_chunk_ids
        return new_item

    def score_candidates(self, candidates: List[Dict], chunks: List[Dict], query: str) -> List[Dict]:
        scored = [self.score_one_candidate(c, chunks, query) for c in candidates]
        scored.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)
        return scored