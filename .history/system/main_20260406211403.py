import threading
import time

from beir.datasets.data_loader import GenericDataLoader

from config import *
from data_loader import MultiFormatLoader
from evaluator import RetrievalEvaluator
from hybrid_retriever import HybridRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from qa_service import interactive_local_qa
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

    return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas


def run_beir_mode(vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid):
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

    ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
        qrels,
        rerank_results_with_scores
    )

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    if target_qid in ranked_chunks_map:
        print(f"\n========== Target Query Preview: {target_qid} ==========")
        ranked_chunks = ranked_chunks_map[target_qid][:SHOW_TOP_K]
        for rank, item in enumerate(ranked_chunks, start=1):
            meta = item.get("meta", {})
            title = meta.get("title", "")
            page_no = int(meta.get("page_no", 0) or 0)
            source = title if title else item["doc_id"]
            if page_no > 0:
                source = f"{source} | page {page_no}"

            print(f"\n[Rank {rank}] score={item['score']:.6f}")
            print(f"Source: {source}")
            print(item["text"][:1000])


def run_local_docs_mode(vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas):
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

    interactive_local_qa(retriever, reranker, reranker_lock)


def main():
    start_time = time.time()
    print("===== System Start =====")
    print(f"[INFO] DATA_SOURCE = {DATA_SOURCE}")

    load_start = time.time()

    if DATA_SOURCE == "beir":
        vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas, queries, qrels, target_qid = load_beir_data()
    elif DATA_SOURCE == "local_docs":
        vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = load_local_docs_data()
        queries, qrels, target_qid = None, None, None
    else:
        raise ValueError(f"Unsupported DATA_SOURCE: {DATA_SOURCE}")

    load_end = time.time()
    print(f"数据加载和嵌入用时: {load_end - load_start:.2f}s")
    print(f"chunk 总数: {len(chunk_texts)}")

    if DATA_SOURCE == "beir":
        print(f"query 总数: {len(queries)}")
        run_beir_mode(
            vectordb,
            chunk_texts,
            chunk_doc_ids,
            chunk_ids,
            chunk_metas,
            queries,
            qrels,
            target_qid
        )
    else:
        run_local_docs_mode(
            vectordb,
            chunk_texts,
            chunk_doc_ids,
            chunk_ids,
            chunk_metas
        )

    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60

    print("\n========== Runtime ==========")
    print(f"总用时: {hours}h {minutes}m {seconds:.2f}s")


if __name__ == "__main__":
    main()