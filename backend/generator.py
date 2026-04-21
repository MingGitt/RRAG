# generator.py
import asyncio
import re
from typing import Dict, List

from config import MAX_CONCURRENCY
from candidate_generator import CandidateGenerator
from evidence_scorer import EvidenceScorer
from risk_controller import RiskController
from selfrag_llm import SelfRagLLM


class Generator:
    """
    模块化生成调度器：
    - CandidateGenerator：多候选生成
    - EvidenceScorer：证据一致性打分
    - RiskController：生成风险评估
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
            })
        return citations

    @staticmethod
    def build_context_from_citations(citations):
        blocks = []
        for item in citations:
            idx = item["index"]
            source = item["source"]
            text = item["text"]
            blocks.append(f"[{idx}] 来源: {source}\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    async def run_single(qid, query, chunks):
        llm = SelfRagLLM()
        candidate_generator = CandidateGenerator(llm)
        evidence_scorer = EvidenceScorer()
        risk_controller = RiskController()

        citations = Generator.build_citations(chunks, max_citations=5, excerpt_len=220)
        evidence_text = Generator.build_context_from_citations(citations)

        raw_candidates = await asyncio.to_thread(
            candidate_generator.generate_candidates,
            query,
            evidence_text,
            "",
            3,
        )

        scored_candidates = evidence_scorer.score_candidates(raw_candidates, chunks, query)

        best_candidate = scored_candidates[0] if scored_candidates else {
            "answer": "根据当前检索到的文档，暂时无法确定答案。",
            "raw_text": "",
            "retrieve": "Unknown",
            "isrel": "Unknown",
            "issup": "Unknown",
            "isuse": 1,
            "followup_query": "None",
            "evidence_score": 0.0,
            "final_score": 0.0,
            "used_chunk_ids": [],
        }

        generation_risk = risk_controller.assess_generation_risk(
            scored_candidates,
            best_candidate,
        )

        answer = best_candidate.get("answer", "") or "根据当前检索到的文档，暂时无法确定答案。"

        if generation_risk["risk_level"] == "high":
            answer = (
                "根据当前检索到的文档，现有证据不足以支持非常确定的结论。"
                "下面给出基于已有证据的谨慎回答：\n" + answer
            )

        return qid, {
            "answer": answer,
            "raw_text": best_candidate.get("raw_text", ""),
            "reflections": {
                "retrieve": best_candidate.get("retrieve", "Unknown"),
                "isrel": best_candidate.get("isrel", "Unknown"),
                "issup": best_candidate.get("issup", "Unknown"),
                "isuse": best_candidate.get("isuse", 1),
                "followup_query": best_candidate.get("followup_query", "None"),
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
        task_qids = []
        for qid, query in selected_items:
            chunks = generation_chunks.get(qid, [])
            if not chunks:
                print(f"[GENERATOR][SKIP] qid={qid} has no generation_chunks")
                continue
            tasks.append(limited_run(qid, query, chunks))
            task_qids.append(qid)

        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for qid, output in zip(task_qids, outputs):
            if isinstance(output, Exception):
                print(f"[GENERATOR][ERROR] qid={qid}: {repr(output)}")
                results[qid] = {
                    "answer": "生成失败。",
                    "raw_text": "",
                    "reflections": {},
                    "chunks": generation_chunks.get(qid, []),
                    "citations": [],
                    "candidates": [],
                    "generation_risk": {
                        "risk_score": 1.0,
                        "risk_level": "high",
                        "confidence": 0.0,
                        "reasons": [f"生成阶段异常: {repr(output)}"],
                    },
                }
                continue

            out_qid, result = output
            results[out_qid] = result

        return results