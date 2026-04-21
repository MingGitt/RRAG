from typing import Dict, List, Optional

from config import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    SELF_RAG_MAX_HISTORY_TURNS,
    SELF_RAG_MAX_ROUNDS,
    SELF_RAG_RETRIEVAL_TOP_K,
    TOP_K_CHUNKS,
)
from retrieval_judge import RetrievalJudge
from selfrag_llm import SelfRagLLM


class SelfRAGEngine:
    """
    当前方案：
    - RetrievalJudge（原来的 LLM）负责是否检索
    - SelfRagLLM（EAS 上的 selfrag-7b）负责证据后的生成与反思 token
    """

    def __init__(self, retriever, reranker, reranker_lock):
        self.retriever = retriever
        self.reranker = reranker
        self.reranker_lock = reranker_lock
        self.gen_llm = SelfRagLLM()

    @staticmethod
    def _format_history(history: Optional[List[Dict]]) -> str:
        if not history:
            return ""

        history = history[-SELF_RAG_MAX_HISTORY_TURNS:]
        lines = []
        for item in history:
            role = item.get("role", "user")
            content = (item.get("content") or "").strip()
            if not content:
                continue
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    @staticmethod
    def _format_source(meta: dict, doc_id: str) -> str:
        title = meta.get("title", "") if meta else ""
        page_no = int(meta.get("page_no", 0) or 0) if meta else 0
        source = title if title else doc_id
        if page_no > 0:
            source = f"{source} | page {page_no}"
        return source

    @staticmethod
    def _clean_excerpt(text: str, max_len: int = 220) -> str:
        text = (text or "").strip().replace("\n", " ")
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    def _retrieve_and_rerank(self, query: str) -> List[Dict]:
        retrieved = self.retriever.retrieve(
            query,
            top_k_chunks=TOP_K_CHUNKS,
            dense_k=DENSE_CANDIDATE_K,
            bm25_k=BM25_CANDIDATE_K,
        )

        if not retrieved:
            return []

        texts = []
        candidate_ids = []
        candidate_doc_map = {}
        candidate_meta_map = {}

        for idx, text, doc_id, score, meta in retrieved:
            texts.append(text)
            candidate_ids.append(idx)
            candidate_doc_map[idx] = doc_id
            candidate_meta_map[idx] = meta or {}

        metas = [candidate_meta_map[idx] for idx in candidate_ids]

        with self.reranker_lock:
            ranked = self.reranker.rerank_texts(
                query=query,
                texts=texts,
                ids=candidate_ids,
                metas=metas,
                top_k=SELF_RAG_RETRIEVAL_TOP_K,
            )

        results = []
        for chunk_idx, text, score, meta in ranked:
            results.append({
                "chunk_idx": chunk_idx,
                "text": text,
                "doc_id": candidate_doc_map.get(chunk_idx, ""),
                "score": float(score),
                "meta": meta or {},
            })
        return results

    def _build_citations(self, ranked_chunks: List[Dict]) -> List[Dict]:
        citations = []
        for i, item in enumerate(ranked_chunks[:SELF_RAG_RETRIEVAL_TOP_K], start=1):
            source = self._format_source(item.get("meta", {}), item.get("doc_id", "unknown_doc"))
            excerpt = self._clean_excerpt(item.get("text", ""), max_len=220)
            citations.append({
                "index": i,
                "source": source,
                "excerpt": excerpt,
                "doc_id": item.get("doc_id", ""),
                "page_no": int(item.get("meta", {}).get("page_no", 0) or 0),
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
            })
        return citations

    @staticmethod
    def _build_evidence_text(citations: List[Dict]) -> str:
        if not citations:
            return "No evidence."

        blocks = []
        for item in citations:
            blocks.append(
                f"[{item['index']}] Source: {item['source']}\n"
                f"{item['text']}"
            )
        return "\n\n".join(blocks)

    def answer(self, user_query: str, history: Optional[List[Dict]] = None) -> Dict:
        history_text = self._format_history(history)

        trace: List[Dict] = []
        reflections: List[Dict] = []
        last_citations: List[Dict] = []
        last_ranked_chunks: List[Dict] = []
        working_query = user_query
        final_answer = ""

        initial_decision = RetrievalJudge.decide_retrieval(
            user_query=user_query,
            history_text=history_text,
        )

        trace.append({
            "stage": "initial_decision",
            "query": user_query,
            "retrieve": initial_decision,
        })

        if initial_decision == "No":
            raw = self.gen_llm.generate_without_evidence(
                user_query=user_query,
                history_text=history_text,
            )
            parsed = self.gen_llm.parse_structured_output(raw)

            reflections.append({
                "round": 0,
                "retrieve": parsed["retrieve"],
                "isrel": parsed["isrel"],
                "issup": parsed["issup"],
                "isuse": parsed["isuse"],
                "followup_query": parsed["followup_query"],
                "raw_text": parsed["raw_text"],
            })

            final_answer = parsed["answer"]
            return {
                "answer": final_answer,
                "citations": [],
                "reflections": reflections,
                "trace": trace,
                "ranked_chunks": [],
            }

        for round_idx in range(1, SELF_RAG_MAX_ROUNDS + 1):
            ranked_chunks = self._retrieve_and_rerank(working_query)
            citations = self._build_citations(ranked_chunks)
            evidence_text = self._build_evidence_text(citations)

            trace.append({
                "stage": "retrieve",
                "round": round_idx,
                "query": working_query,
                "retrieved_count": len(ranked_chunks),
                "top_sources": [c["source"] for c in citations],
            })

            raw = self.gen_llm.generate_with_evidence(
                user_query=user_query,
                history_text=history_text,
                evidence_text=evidence_text,
            )
            parsed = self.gen_llm.parse_structured_output(raw)

            reflections.append({
                "round": round_idx,
                "retrieve": parsed["retrieve"],
                "isrel": parsed["isrel"],
                "issup": parsed["issup"],
                "isuse": parsed["isuse"],
                "followup_query": parsed["followup_query"],
                "raw_text": parsed["raw_text"],
            })

            last_citations = citations
            last_ranked_chunks = ranked_chunks
            final_answer = parsed["answer"]

            # 当前版本先不依赖 Self-RAG 自动 follow-up query
            break

        return {
            "answer": final_answer,
            "citations": last_citations,
            "reflections": reflections,
            "trace": trace,
            "ranked_chunks": last_ranked_chunks,
        }