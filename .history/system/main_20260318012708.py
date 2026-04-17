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


TARGET_QID = "35"
SHOW_TOP_K = 10


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

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
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

    # 只生成 qid=35
    if TARGET_QID not in queries:
        print(f"\n[WARN] qid={TARGET_QID} 不存在")
        return

    target_queries = {
        TARGET_QID: queries[TARGET_QID]
    }

    gen_outputs = asyncio.run(
        Generator.run(
            target_queries,
            generation_chunks,
            max_samples=1,
            max_concurrency=1
        )
    )

    print("\n========== Target Query ==========")
    print(f"QID: {TARGET_QID}")
    print(f"Query: {queries[TARGET_QID]}")

    print(f"\n========== Top-{SHOW_TOP_K} Retrieved Chunks ==========")
    ranked_chunks = ranked_chunks_map.get(TARGET_QID, [])
    if not ranked_chunks:
        print("No ranked chunks.")
    else:
        for rank, (chunk_id, text, doc_id, score) in enumerate(ranked_chunks[:SHOW_TOP_K], start=1):
            print(f"\n[Rank {rank}] chunk_id={chunk_id} doc_id={doc_id} score={score:.6f}")
            print(text[:1000])

    print("\n========== Generation Chunks ==========")
    gen_chunks = generation_chunks.get(TARGET_QID, [])
    if not gen_chunks:
        print("No generation chunks.")
    else:
        for i, chunk in enumerate(gen_chunks, start=1):
            print(f"\n[Gen Chunk {i}]")
            print(chunk[:1000])

    print("\n========== Generation Result ==========")
    target_output = gen_outputs.get(TARGET_QID)
    if not target_output:
        print("No generation output.")
    else:
        print(target_output["answer"])

        print("\n========== Judge ==========")
        print(target_output["judge"])

    end_time = time.time()
    total_time = end_time - start_time

    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n========== Runtime ==========")
    print(f"总用时: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()