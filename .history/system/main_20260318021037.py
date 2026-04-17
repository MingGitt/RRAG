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


SHOW_TOP_K = 5
TARGET_QUERY_INDEX = 1   # 0=第一条，1=第二条


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

    # 注意：这里是 4 个返回值
    rerank_results_with_scores, generation_chunks, query_expansions, query_type_stats = run_parallel_retrieval_pipeline(
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

    query_items = list(queries.items())
    if len(query_items) <= TARGET_QUERY_INDEX:
        print(f"\n[WARN] 查询数量不足，无法获取第 {TARGET_QUERY_INDEX + 1} 条查询")
        return

    target_qid, target_query = query_items[TARGET_QUERY_INDEX]
    target_expanded_queries = query_expansions.get(target_qid, [target_query])
    target_gen_chunks = generation_chunks.get(target_qid, [])[:SHOW_TOP_K]

    print("\n========== Second Query ==========")
    print(f"QID: {target_qid}")
    print(f"Query: {target_query}")

    print("\n========== Expanded / Optimized Queries ==========")
    for i, q in enumerate(target_expanded_queries, start=1):
        print(f"[{i}] {q}")

    print(f"\n========== Top-{SHOW_TOP_K} Generation Chunks ==========")
    if not target_gen_chunks:
        print("No generation chunks.")
    else:
        for i, chunk in enumerate(target_gen_chunks, start=1):
            print(f"\n[Chunk {i}]")
            print(chunk[:1500])

    gen_outputs = asyncio.run(
        Generator.run(
            {target_qid: target_query},
            {target_qid: target_gen_chunks},
            max_samples=1,
            max_concurrency=1
        )
    )

    print("\n========== Generated Answer ==========")
    target_output = gen_outputs.get(target_qid)
    if not target_output:
        print("No generation output.")
    else:
        print(target_output["answer"])

    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n========== Runtime ==========")
    print(f"总用时: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()