from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self, model_name):
        self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query, doc_ids, corpus_texts, top_k=5):
        pairs = [(query, corpus_texts[d]) for d in doc_ids]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)

        return (
            [d for d, _ in ranked[:top_k]],
            {d: float(s) for d, s in ranked[:top_k]}
        )