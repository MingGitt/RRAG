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
    all_chunks: [(idx, text, doc_id, score, meta), ...]
    """
    best = {}

    for idx, text, doc_id, score, meta in all_chunks:
        key = (str(doc_id), text)
        if key not in best or score > best[key][3]:
            best[key] = (idx, text, doc_id, score, meta)

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
    candidate_meta_map = {}

    for idx, text, doc_id, _, meta in retrieved:
        texts.append(text)
        candidate_indices.append(idx)
        candidate_doc_map[idx] = doc_id
        candidate_meta_map[idx] = meta

    metas = [candidate_meta_map[idx] for idx in candidate_indices]

    with reranker_lock:
        ranked_chunks_raw = reranker.rerank_texts(
            query,
            texts,
            candidate_indices,
            metas=metas,
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

    ranked_chunks = []
    for candidate_idx, text, score, meta in ranked_chunks_raw:
        ranked_chunks.append({
            "chunk_idx": candidate_idx,
            "text": text,
            "doc_id": candidate_doc_map.get(candidate_idx),
            "score": float(score),
            "meta": meta,
        })

    doc_level_for_eval = [
        (item["doc_id"], item["text"], item["score"])
        for item in ranked_chunks
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

    gen_chunks_input = [
        {
            "doc_id": item["doc_id"],
            "text": item["text"],
            "score": item["score"],
            "meta": item["meta"],
        }
        for item in ranked_chunks
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