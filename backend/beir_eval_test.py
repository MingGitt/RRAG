import threading
import time

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET,
    DATASET_PATH,
    EMBED_MODEL_NAME,
    PERSIST_DIR,
    QUERY_WORKERS,
    SUBQUERY_WORKERS,
    TARGET_QID,
)
from hybrid_retriever import HybridRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from reranker import Reranker
from vector_store import VectorStoreBuilder


def load_beir_data():
    corpus, queries, qrels = GenericDataLoader(DATASET_PATH).load(split="test")

    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME,
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load(
        corpus,
        PERSIST_DIR,
        collection_name,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels


def evaluate_results(qrels, results):
    evaluator = EvaluateRetrieval()
    return evaluator.evaluate(qrels, results, k_values=[1, 3, 5, 10])


def run_beir_evaluation():
    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels = load_beir_data()

    retriever = HybridRetriever(
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        chunk_metas,
        ALPHA,
    )

    reranker = Reranker()
    reranker_lock = threading.Lock()

    retrieve_start = time.time()
    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
        subquery_workers=SUBQUERY_WORKERS,
    )
    retrieve_end = time.time()

    print(f"检索和重排序用时: {retrieve_end - retrieve_start:.2f}s")
    print("\n========== Query Routing Stats ==========")
    print(query_type_stats)

    ndcg, _map, recall, precision = evaluate_results(qrels, rerank_results_with_scores)

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    if TARGET_QID in ranked_chunks_map:
        print(f"\n========== Target Query: {TARGET_QID} ==========")
        print(queries.get(TARGET_QID, ""))

        for i, item in enumerate(ranked_chunks_map[TARGET_QID][:5], start=1):
            print(f"\n[Top {i}]")
            print(f"doc_id={item['doc_id']} | score={item['score']:.6f}")
            print(item["text"][:1000])


if __name__ == "__main__":
    run_beir_evaluation()