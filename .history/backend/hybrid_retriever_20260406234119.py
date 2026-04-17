import re

from rank_bm25 import BM25Okapi


class HybridRetriever:
    def __init__(self, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, alpha=0.6, rrf_k=60):
        self.vectordb = vectordb
        self.chunk_texts = chunk_texts
        self.chunk_doc_ids = chunk_doc_ids
        self.chunk_ids = chunk_ids
        self.chunk_metas = chunk_metas
        self.alpha = alpha
        self.rrf_k = rrf_k

        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids) if cid is not None}

        try:
            tokenized = [self._tokenize(c) for c in chunk_texts]
            self.bm25 = BM25Okapi(tokenized)
        except Exception:
            self.bm25 = None

    def _rrf(self, rank):
        return 1.0 / (self.rrf_k + rank)

    def _tokenize(self, text):
        text = (text or "").lower()
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split() if text else []

    def retrieve(self, query, top_k_chunks=50, dense_k=200, bm25_k=200):
        dense_hits = self.vectordb.similarity_search_with_score(query, k=dense_k)

        dense_rank = {}
        for rank, (doc, _) in enumerate(dense_hits, start=1):
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id is None:
                continue
            idx = self.chunk_id_to_idx.get(chunk_id)
            if idx is not None:
                dense_rank[idx] = rank

        bm25_rank = {}
        if self.bm25 is not None:
            query_tokens = self._tokenize(query)
            bm25_scores = self.bm25.get_scores(query_tokens) if query_tokens else [0.0] * len(self.chunk_texts)
            bm25_top = sorted(
                enumerate(bm25_scores),
                key=lambda x: x[1],
                reverse=True
            )[:bm25_k]
            bm25_rank = {idx: rank for rank, (idx, _) in enumerate(bm25_top, start=1)}

        candidate_ids = set(dense_rank.keys()) | set(bm25_rank.keys())
        if not candidate_ids:
            return []

        fused = []
        for idx in candidate_ids:
            score = 0.0
            if idx in dense_rank:
                score += self.alpha * self._rrf(dense_rank[idx])
            if idx in bm25_rank:
                score += (1 - self.alpha) * self._rrf(bm25_rank[idx])

            candidate_key = self.chunk_ids[idx] if idx < len(self.chunk_ids) and self.chunk_ids[idx] else f"idx_{idx}"

            fused.append((
                candidate_key,              # 全局唯一 key，避免多库时 idx 冲突
                self.chunk_texts[idx],
                self.chunk_doc_ids[idx],
                score,
                self.chunk_metas[idx],
            ))

        fused.sort(key=lambda x: x[3], reverse=True)
        return fused[:top_k_chunks]


class MultiVectorRetriever:
    """
    对多个文件库做联合检索
    """
    def __init__(self, file_indexes: dict, alpha=0.6, rrf_k=60):
        self.file_indexes = file_indexes
        self.alpha = alpha
        self.rrf_k = rrf_k

    def retrieve(self, query, top_k_chunks=50, dense_k=50, bm25_k=50):
        all_results = []

        for file_id, index_obj in self.file_indexes.items():
            retriever = HybridRetriever(
                vectordb=index_obj["vectordb"],
                chunk_texts=index_obj["chunk_texts"],
                chunk_doc_ids=index_obj["chunk_doc_ids"],
                chunk_ids=index_obj["chunk_ids"],
                chunk_metas=index_obj["chunk_metas"],
                alpha=self.alpha,
                rrf_k=self.rrf_k,
            )

            results = retriever.retrieve(
                query,
                top_k_chunks=top_k_chunks,
                dense_k=dense_k,
                bm25_k=bm25_k
            )

            all_results.extend(results)

        all_results.sort(key=lambda x: x[3], reverse=True)
        return all_results[:top_k_chunks]