from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    DOC_AGG_MODE,
    TOP_K_CHUNKS,
)
from evaluator import RetrievalEvaluator
from query_optimizer import QueryOptimizer
from reranker import Reranker


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
            "generation_chunks": ... or None,
            "ranked_chunks": ... or None
        }
    """
    qtype, expanded_queries = QueryOptimizer.expand(query)

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
            "ranked_chunks": None,
        }

    texts = []
    candidate_indices = []
    candidate_doc_map = {}

    for idx, text, doc_id, _ in retrieved:
        texts.append(text)
        candidate_indices.append(idx)
        candidate_doc_map[idx] = doc_id

    with reranker_lock:
        ranked_chunks_raw = reranker.rerank_texts(
            query,
            texts,
            candidate_indices,
            top_k=TOP_K_CHUNKS
        )

    if not ranked_chunks_raw:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
            "ranked_chunks": None,
        }

    # 补回 doc_id，形成 chunk 级重排序结果：
    # [(chunk_idx, text, doc_id, score), ...]
    ranked_chunks = []
    for candidate_idx, text, score in ranked_chunks_raw:
        ranked_chunks.append(
            (candidate_idx, text, candidate_doc_map.get(candidate_idx), float(score))
        )

    # 保留原有检索评测逻辑：仍然聚合成 doc_scores 给 evaluate 用
    doc_level_for_eval = [
        (doc_id, text, score)
        for _, text, doc_id, score in ranked_chunks
    ]

    doc_scores = RetrievalEvaluator.aggregate_doc_scores(
        doc_level_for_eval,
        mode=DOC_AGG_MODE
    )

    if not doc_scores:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
            "ranked_chunks": ranked_chunks,
        }

    # 给生成器用的 chunk
    gen_chunks_input = [
        (doc_id, text, score)
        for _, text, doc_id, score in ranked_chunks
    ]

    gen_chunks = Reranker.select_generation_chunks(
        gen_chunks_input,
        top_n=5,
        max_per_doc=3
    )

    return {
        "qid": qid,
        "qtype": qtype,
        "doc_scores": doc_scores,
        "generation_chunks": gen_chunks,
        "ranked_chunks": ranked_chunks,
    }


def run_parallel_retrieval_pipeline(
    queries,
    retriever,
    reranker,
    reranker_lock,
    query_workers=4,
    subquery_workers=8,
):
    rerank_results_with_scores = {}
    generation_chunks = {}
    ranked_chunks_map = {}
    query_type_stats = {}

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
                ranked_chunks = result["ranked_chunks"]

                query_type_stats[qtype] = query_type_stats.get(qtype, 0) + 1

                if doc_scores:
                    rerank_results_with_scores[qid] = doc_scores
                if gen_chunks:
                    generation_chunks[qid] = gen_chunks
                if ranked_chunks:
                    ranked_chunks_map[qid] = ranked_chunks

    return rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats