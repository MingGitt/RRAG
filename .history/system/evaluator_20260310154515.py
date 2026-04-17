from collections import defaultdict
from beir.retrieval.evaluation import EvaluateRetrieval

class RetrievalEvaluator:

    @staticmethod
    def evaluate(qrels, results):
        evaluator = EvaluateRetrieval()
        return evaluator.evaluate(qrels, results, k_values=[1, 3, 5])
    
    @staticmethod
    def aggregate_doc_scores(ranked_chunks, mode="top2_mean"):
        """
        ranked_chunks: [(doc_id, text, score), ...]
        返回:
            {doc_id: aggregated_score}
        """
        doc_to_scores = defaultdict(list)

        for doc_id, text, score in ranked_chunks:
            doc_to_scores[str(doc_id)].append(float(score))

        doc_scores = {}
        for doc_id, scores in doc_to_scores.items():
            scores = sorted(scores, reverse=True)

            if mode == "max":
                doc_scores[doc_id] = scores[0]
            elif mode == "top2_mean":
                doc_scores[doc_id] = sum(scores[:2]) / min(2, len(scores))
            elif mode == "top3_sum":
                doc_scores[doc_id] = sum(scores[:3])
            else:
                doc_scores[doc_id] = scores[0]

        return doc_scores        
