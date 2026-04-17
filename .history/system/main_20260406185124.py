import asyncio
import os
import threading
import time

from beir.datasets.data_loader import GenericDataLoader

from config import *
from data_loader import MultiFormatLoader
from evaluator import RetrievalEvaluator
from generator import Generator
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
        EMBED_MODEL_NAME
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load(
        corpus,
        PERSIST_DIR,
        collection_name,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )
    target_qid = TARGET_QID
    return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid


def load_local_docs_data():
    loader = MultiFormatLoader(max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    docs = loader.load_directory(LOCAL_DOCS_DIR)

    all_chunks = []
    for doc in docs.values():
        all_chunks.extend(loader.chunk_document(doc))

    collection_name = VectorStoreBuilder.make_collection_name(
        "localdocs",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load_from_chunks(
        all_chunks,
        PERSIST_DIR,
        collection_name,
    )

    queries = {
        LOCAL_QUERY_ID: LOCAL_TEST_QUERY
    }
    qrels = None
    target_qid = LOCAL_QUERY_ID
    return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid


def format_source(meta: dict, doc_id: str):
    title = meta.get("title", "") if meta else ""
    page_no = int(meta.get("page_no", 0)) if meta else 0
    source = title if title else doc_id
    if page_no > 0:
        source = f"{source} | page {page_no}"
    return source


def main():
    start_time = time.time()
    print("===== System Start =====")
    print(f"[INFO] DATA_SOURCE = {DATA_SOURCE}")

    load_start = time.time()
    if DATA_SOURCE == "beir":
        vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid = load_beir_data()
    elif DATA_SOURCE == "local_docs":
        vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid = load_local_docs_data()
    else:
        raise ValueError(f"Unsupported DATA_SOURCE: {DATA_SOURCE}")

    load_end = time.time()
    print(f"数据加载和嵌入用时: {load_end - load_start:.2f}s")
    print(f"chunk 总数: {len(chunk_texts)}")
    print(f"query 总数: {len(queries)}")

    retriever = HybridRetriever(
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        chunk_metas,
        ALPHA
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
    retrieval_time = retrieve_end - retrieve_start
    print(f"检索和重排序用时: {retrieval_time:.2f}s")

    print("\n========== Query Routing Stats ==========")
    print(query_type_stats)

    if qrels is not None:
        ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
            qrels,
            rerank_results_with_scores
        )

        print("\n========== Retrieval Quality ==========")
        print("NDCG:", ndcg)
        print("MAP:", _map)
        print("Recall:", recall)
        print("Precision:", precision)
    else:
        print("\n[INFO] 当前是本地文档模式，没有 qrels，跳过检索指标评测。")

    if target_qid not in queries:
        print(f"\n[WARN] qid={target_qid} 不存在")
        return

    target_queries = {
        target_qid: queries[target_qid]
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
    print(f"QID: {target_qid}")
    print(f"Query: {queries[target_qid]}")

    print(f"\n========== Top-{SHOW_TOP_K} Retrieved Chunks ==========")
    ranked_chunks = ranked_chunks_map.get(target_qid, [])
    if not ranked_chunks:
        print("No ranked chunks.")
    else:
        for rank, item in enumerate(ranked_chunks[:SHOW_TOP_K], start=1):
            chunk_idx = item["chunk_idx"]
            text = item["text"]
            doc_id = item["doc_id"]
            score = item["score"]
            meta = item.get("meta", {})
            source = format_source(meta, doc_id)

            print(f"\n[Rank {rank}] chunk_idx={chunk_idx} doc_id={doc_id} score={score:.6f}")
            print(f"Source: {source}")
            print(text[:1000])

    print("\n========== Generation Chunks ==========")
    gen_chunks = generation_chunks.get(target_qid, [])
    if not gen_chunks:
        print("No generation chunks.")
    else:
        for i, item in enumerate(gen_chunks, start=1):
            source = format_source(item.get("meta", {}), item.get("doc_id", "unknown_doc"))
            print(f"\n[Gen Chunk {i}]")
            print(f"Source: {source}")
            print(item["text"][:1000])

    print("\n========== Generation Result ==========")
    target_output = gen_outputs.get(target_qid)
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