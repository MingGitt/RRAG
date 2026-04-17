import time
import asyncio
import threading

from beir.datasets.data_loader import GenericDataLoader
from config import *

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from evaluator import RetrievalEvaluator
from generator import Generator
from pipeline_executor import run_parallel_retrieval_pipeline


def main():
    start_time = time.time()
    print("===== System Start =====")

    load_start = time.time()
    corpus, queries, qrels = GenericDataLoader(DATASET_PATH).load(split="test")

    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids = VectorStoreBuilder.build_or_load(
        corpus,
        PERSIST_DIR,
        collection_name,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    load_end = time.time()
    print(f"数据加载和嵌入用时: {load_end - load_start:.2f}s")

    retriever = HybridRetriever(
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        ALPHA
    )

    reranker = Reranker()
    reranker_lock = threading.Lock()

    query_workers = globals().get("QUERY_WORKERS", 4)
    subquery_workers = globals().get("SUBQUERY_WORKERS", 8)

    retrieve_start = time.time()

    rerank_results_with_scores, generation_chunks, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=query_workers,
        subquery_workers=subquery_workers,
    )

    retrieve_end = time.time()
    retrieval_time = retrieve_end - retrieve_start
    print(f"检索和重排序用时: {retrieval_time:.2f}s")

    ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
        qrels,
        rerank_results_with_scores
    )

    print("\n========== Query Routing Stats ==========")
    print(query_type_stats)

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    gen_start = time.time()
    gen_scores = asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=MAX_SAMPLES,
            max_concurrency=MAX_CONCURRENCY
        )
    )
    gen_end = time.time()

    print("\n========== Generation Quality ==========")
    if gen_scores:
        print("Avg score:", sum(gen_scores) / len(gen_scores))
        print({i: gen_scores.count(i) for i in range(1, 6)})
    else:
        print("Avg score: N/A")
        print("No valid generation scores.")

    print(f"生成用时: {gen_end - gen_start:.2f}s")

    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n========== Runtime ==========")
    print(f"总用时: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()