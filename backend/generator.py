# generator.py
import asyncio
import re
from typing import Dict, List

from config import MAX_CONCURRENCY, QWEN_ANSWER_MODEL

try:
    from config import GEN_CANDIDATE_COUNT
except Exception:
    GEN_CANDIDATE_COUNT = 3

from qwen_client import QwenClient
from answer_support_judge import AnswerSupportJudge
from risk_controller import RiskController
from candidate_generator import CandidateGenerator


class Generator:
    """
    新版生成流程：
    - generation_chunks 已由 pipeline_executor 通过 critique 选好
    - 这里负责：
      1. 多候选答案生成
      2. 多候选逐个打分
      3. 选择最优答案
      4. 风险评估
    """

    @staticmethod
    def _format_source(meta, doc_id):
        title = meta.get("title", "") if meta else ""
        page_no = int(meta.get("page_no", 0)) if meta else 0
        source = title if title else doc_id
        if page_no > 0:
            source = f"{source} | page {page_no}"
        return source

    @staticmethod
    def _clean_excerpt(text, max_len=220):
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    @staticmethod
    def build_citations(chunks, max_citations=5, excerpt_len=220):
        citations = []
        for i, item in enumerate(chunks[:max_citations], start=1):
            meta = item.get("meta", {})
            source = Generator._format_source(meta, item.get("doc_id", "unknown_doc"))
            full_text = item.get("text", "") or ""
            excerpt = Generator._clean_excerpt(full_text, max_len=excerpt_len)

            citations.append({
                "index": i,
                "source": source,
                "excerpt": excerpt,
                "doc_id": item.get("doc_id", ""),
                "page_no": int(meta.get("page_no", 0) or 0),
                "text": full_text,
                "score": float(item.get("score", 0.0)),
                "critique_relevance": item.get("critique_relevance", "Unknown"),
                "critique_usefulness": item.get("critique_usefulness", "Unknown"),
                "critique_score": float(item.get("critique_score", 0.0)),
                "critique_reason": item.get("critique_reason", ""),
            })
        return citations

    @staticmethod
    def build_context_from_citations(citations):
        if not citations:
            return "No evidence."

        blocks = []
        for item in citations:
            idx = item["index"]
            source = item["source"]
            text = item["text"]
            blocks.append(f"[{idx}] Source: {source}\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def generate_direct_answer(query: str) -> Dict:
        prompt = f"""
你是一个智能问答助手。

用户问题：
{query}

请直接给出简洁、准确的回答。
如果问题不明确，请明确说明。
""".strip()

        answer = QwenClient.call(
            [{"role": "user", "content": prompt}],
            model=QWEN_ANSWER_MODEL,
            temperature=0.2,
            max_tokens=256,
        )

        return {
            "answer": answer.strip(),
            "raw_text": answer.strip(),
            "reflections": {
                "retrieve": "No",
                "isrel": "Unknown",
                "issup": "Unknown",
                "isuse": 3,
                "followup_query": "None",
            },
            "chunks": [],
            "citations": [],
            "candidates": [],
            "generation_risk": {
                "risk_score": 0.3,
                "risk_level": "medium",
                "confidence": 0.7,
                "reasons": ["未经过文档证据支持校验的直接回答"],
            },
        }

    @staticmethod
    def _score_candidate(query: str, candidate: Dict, evidence_text: str, chunks: List[Dict]) -> Dict:
        judge_result = AnswerSupportJudge.judge(
            query=query,
            answer=candidate["answer"],
            evidence_text=evidence_text,
        )

        support = judge_result["support"]
        support_score = float(judge_result["support_score"])
        completeness_score = float(judge_result["completeness_score"])
        overall_score = float(judge_result["overall_score"])

        # 映射到风险评估里使用的字段
        evidence_score = support_score if support_score > 0 else (
            1.0 if support == "Fully" else (0.6 if support == "Partially" else 0.1)
        )

        final_score = overall_score if overall_score > 0 else (
            1.0 if support == "Fully" else (0.6 if support == "Partially" else 0.1)
        )

        scored = dict(candidate)
        scored["issup"] = support
        scored["support_reason"] = judge_result["reason"]
        scored["support_score"] = support_score
        scored["completeness_score"] = completeness_score
        scored["evidence_score"] = evidence_score
        scored["final_score"] = final_score
        scored["retrieve"] = "Yes"
        scored["isrel"] = "Relevant"
        scored["isuse"] = 4 if support in {"Fully", "Partially"} else 2

        # 使用真实 chunk id，优先取 chunk_idx，没有就回退到 doc_id
        scored["used_chunk_ids"] = [
            str(item.get("chunk_idx", item.get("doc_id", "")))
            for item in chunks[:3]
        ]

        return scored

    @staticmethod
    async def run_single(qid, query, chunks):
        citations = Generator.build_citations(chunks, max_citations=5, excerpt_len=220)
        evidence_text = Generator.build_context_from_citations(citations)

        raw_candidates = await asyncio.to_thread(
            CandidateGenerator.generate_candidates,
            query,
            evidence_text,
            GEN_CANDIDATE_COUNT,
        )

        if not raw_candidates:
            raw_candidates = [{"answer": "根据当前证据，暂时无法确定答案。", "raw_text": ""}]

        scored_candidates = []
        for cand in raw_candidates:
            scored = await asyncio.to_thread(
                Generator._score_candidate,
                query,
                cand,
                evidence_text,
                chunks,
            )
            scored_candidates.append(scored)

        scored_candidates.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
        best_candidate = scored_candidates[0]

        final_answer = best_candidate["answer"]
        if best_candidate.get("issup") == "No Support":
            final_answer = (
                "根据当前检索到的文档，现有证据不足以支持非常确定的结论。"
                "请结合原文进一步核实。"
            )

        generation_risk = RiskController.assess_generation_risk(
            scored_candidates,
            best_candidate,
        )

        relevance_summary = "Relevant" if any(
            x.get("critique_relevance") == "Relevant" for x in chunks
        ) else "Irrelevant"

        return qid, {
            "answer": final_answer,
            "raw_text": best_candidate.get("raw_text", ""),
            "reflections": {
                "retrieve": "Yes",
                "isrel": relevance_summary,
                "issup": best_candidate.get("issup", "Unknown"),
                "isuse": best_candidate.get("isuse", 2),
                "followup_query": "None",
            },
            "chunks": chunks,
            "citations": citations,
            "candidates": scored_candidates,
            "generation_risk": generation_risk,
        }

    @staticmethod
    async def run(queries, generation_chunks, max_samples=1, max_concurrency=None):
        if max_concurrency is None:
            max_concurrency = MAX_CONCURRENCY

        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_run(qid, query, chunks):
            async with semaphore:
                return await Generator.run_single(qid, query, chunks)

        selected_items = list(queries.items())[:max_samples]
        tasks = []

        for qid, query in selected_items:
            chunks = generation_chunks.get(qid, [])
            if not chunks:
                continue
            tasks.append(limited_run(qid, query, chunks))

        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in outputs:
            if isinstance(output, Exception):
                print(f"[GENERATOR][ERROR] {repr(output)}")
                continue
            qid, result = output
            results[qid] = result

        return results