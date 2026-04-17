from sentence_transformers import CrossEncoder

from config import RERANK_MODEL_LOCAL_DIR, RERANK_MODEL_NAME
from model_utils import resolve_or_download_model


class Reranker:
    def __init__(self):
        model_path = resolve_or_download_model(
            repo_id=RERANK_MODEL_NAME,
            local_dir=RERANK_MODEL_LOCAL_DIR,
            cache_dir=None,
            offline=False,
        )
        self.model = CrossEncoder(model_path, device="cpu")

    def rerank_texts(self, query, texts, ids, metas=None, top_k=None):
        """
        query: str
        texts: List[str]
        ids:   List[idx]
        metas: List[dict]
        return:
            [(idx, text, score, meta), ...]
        """
        if not texts:
            return []

        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)

        results = []
        if metas is None:
            metas = [{} for _ in texts]

        for idx, text, score, meta in zip(ids, texts, scores, metas):
            results.append((idx, text, float(score), meta))

        results.sort(key=lambda x: x[2], reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results

    @staticmethod
    def select_generation_chunks(ranked_chunks, top_n=5, max_per_doc=3):
        """
        ranked_chunks: List[{
            "doc_id": str,
            "text": str,
            "score": float,
            "meta": dict
        }]
        """
        selected = []
        per_doc_count = {}

        for item in ranked_chunks:
            doc_id = item["doc_id"]
            if per_doc_count.get(doc_id, 0) >= max_per_doc:
                continue
            selected.append(item)
            per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1
            if len(selected) >= top_n:
                break

        return selected