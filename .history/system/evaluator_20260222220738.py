from beir.retrieval.evaluation import EvaluateRetrieval

class RetrievalEvaluator:

    @staticmethod
    def evaluate(qrels, results):
        evaluator = EvaluateRetrieval()
        return evaluator.evaluate(qrels, results, k_values=[1, 3, 5])