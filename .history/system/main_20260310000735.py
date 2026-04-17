import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from beir.datasets.data_loader import GenericDataLoader
from config import *

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from evaluator import RetrievalEvaluator
from generator import Generator
from query_optimizer import QueryOptimizer


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


def select_generation_chunks(ranked_chunks, top_n=5, max_per_doc=2):
    """
    ranked_chunks: [(doc_id, text, score), ...]
    选择用于生成的 chunks：
    - 去重文本
    - 每个 doc 最多取 max_per_doc 个 chunk
    """
    selected = []
    per_doc = defaultdict(int)
    seen_text = set()

    for doc_id, text, score in ranked_chunks:
        if text in seen_text:
            continue
        if per_doc[doc_id] >= max_per_doc:
            continue

        selected.append(text)
        seen_text.add(text)
        per_doc[doc_id] += 1

        if len(selected) >= top_n:
            break

    return selected


def merge_retrieved_chunks(all_chunks, top_k=50):
    """
    all_chunks: [(idx, text, doc_id, score), ...]
    对多路查询召回结果做去重融合：
    - 相同 (doc_id, text) 视为同一个 chunk
    - 保留最高分
    """
    best = {}

    for idx, text, doc_id, score in all_chunks:
        key = (str(doc_id), text)
        if key not in best or score > best[key][3]:
            best[key] = (idx, text, doc_id, score)

    merged = list(best.values())
    merged.sort(key=lambda x: x[3], reverse=True)
    return merged[:top_k]


def retrieve_single_subquery(retriever, sub_q):
    return retriever.retrieve(
        sub_q,
        top_k_chunks=TOP_K_CHUNKS,
        dense_k=DENSE_CANDIDATE_K,
        bm25_k=BM25_CANDIDATE_K
    )


def retrieve_expanded_queries_parallel(retriever, expanded_queries, subquery_executor):
    """
    第二层并发：同一个 query 下的多个 expanded_queries 并发检索
    """
    if not expanded_queries:
        return []

    if len(expanded_queries) == 1:
        return retrieve_single_subquery(retriever, expanded_queries[0])

    futures = [
        subquery_executor.submit(retrieve_single_subquery, retriever, sub_q)
        for sub_q in expanded_queries
    ]

    all_retrieved = []
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                all_retrieved.extend(result)
        except Exception:
            continue

    return all_retrieved


def process_single_query(
    qid,
    query,
    retriever,
    reranker,
    reranker_lock,
    subquery_executor,
):
    """
    第一层并发：多个原始 query 并发处理
    返回：
        {
            "qid": ...,
            "qtype": ...,
            "doc_scores": ... or None,
            "generation_chunks": ... or None
        }
    """
    # 查询优化
    #opt_start = time.time()
    qtype, expanded_queries = QueryOptimizer.expand(query)
    #opt_end = time.time()
    #opt_time = opt_end - opt_start
    

    all_retrieved = retrieve_expanded_queries_parallel(
        retriever,
        expanded_queries,
        subquery_executor
    )

    retrieved = merge_retrieved_chunks(
        all_retrieved,
        top_k=TOP_K_CHUNKS
    )

    if not retrieved:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
        }

    texts = []
    doc_ids = []

    for _, text, doc_id, _ in retrieved:
        texts.append(text)
        doc_ids.append(doc_id)

    # reranker 通常不保证线程安全，这里加锁
    with reranker_lock:
        ranked_chunks = reranker.rerank_texts(
            query,
            texts,
            doc_ids,
            top_k=TOP_K_CHUNKS
        )

    if not ranked_chunks:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
        }

    doc_scores = aggregate_doc_scores(
        ranked_chunks,
        mode=DOC_AGG_MODE
    )

    if not doc_scores:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
        }

    gen_chunks = select_generation_chunks(
        ranked_chunks,
        top_n=5,
        max_per_doc=3
    )
   

    return {
        "qid": qid,
        "qtype": qtype,
        "doc_scores": doc_scores,
        "generation_chunks": gen_chunks,
    }


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

    rerank_results_with_scores = {}
    generation_chunks = {}
    query_type_stats = defaultdict(int)
    retrieval_time = 0

    # 默认并发参数；如果你在 config.py 里定义了同名变量，会优先用 config 的值
    query_workers = globals().get("QUERY_WORKERS", 4)
    subquery_workers = globals().get("SUBQUERY_WORKERS", 8)
    # 检索 + 重排序
    retrieve_start = time.time()

    with ThreadPoolExecutor(max_workers=subquery_workers) as subquery_executor:
        with ThreadPoolExecutor(max_workers=query_workers) as query_executor:
            future_to_qid = {
                query_executor.submit(
                    process_single_query,
                    qid,
                    query,
                    retriever,
                    reranker,
                    reranker_lock,
                    subquery_executor,
                ): qid
                for qid, query in queries.items()
            }

            for future in as_completed(future_to_qid):
                try:
                    result = future.result()
                except Exception:
                    continue

                qid = result["qid"]
                qtype = result["qtype"]
                doc_scores = result["doc_scores"]
                gen_chunks = result["generation_chunks"]

                query_opt_time += result["opt_time"]
                retrieval_time += result["retrieval_time"]
                query_type_stats[qtype] += 1

                if doc_scores:
                    rerank_results_with_scores[qid] = doc_scores
                if gen_chunks:
                    generation_chunks[qid] = gen_chunks


    retrieve_end = time.time()
    retrieval_time = retrieve_end - retrieve_start
    print(f"检索和重排序用时: {retrieval_time:.2f}s")    

    ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
        qrels,
        rerank_results_with_scores
    )

    print("\n========== Query Routing Stats ==========")
    print(dict(query_type_stats))

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    gen_start = time.time()
    gen_scores = Generator.run(queries, generation_chunks)
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