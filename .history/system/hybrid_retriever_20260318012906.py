from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, alpha=0.6, rrf_k=60):
        self.vectordb = vectordb
        self.chunk_texts = chunk_texts
        self.chunk_doc_ids = chunk_doc_ids
        self.chunk_ids = chunk_ids
        self.alpha = alpha
        self.rrf_k = rrf_k

        self.chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids) if cid is not None}

        tokenized = [c.split() for c in chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _rrf(self, rank):
        return 1.0 / (self.rrf_k + rank)

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

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top = sorted(
            enumerate(bm25_scores),
            key=lambda x: x[1],
            reverse=True
        )[:bm25_k]
        bm25_rank = {idx: rank for rank, (idx, _) in enumerate(bm25_top, start=1)}

        candidate_ids = set(dense_rank.keys()) | set(bm25_rank.keys())

        fused = []
        for idx in candidate_ids:
            score = 0.0
            if idx in dense_rank:
                score += self.alpha * self._rrf(dense_rank[idx])
            if idx in bm25_rank:
                score += (1 - self.alpha) * self._rrf(bm25_rank[idx])

            fused.append((
                idx,
                self.chunk_texts[idx],
                self.chunk_doc_ids[idx],
                score
            ))

        fused.sort(key=lambda x: x[3], reverse=True)
        #print("Hybrid alpha:", self.alpha)
        return fused[:top_k_chunks]