from collections import defaultdict
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


def main():
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

    retriever = HybridRetriever(
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        ALPHA
    )

    reranker = Reranker()

    rerank_results_with_scores = {}
    generation_chunks = {}
    query_type_stats = defaultdict(int)
    
    for qid, query in queries.items():
        qtype, expanded_queries = QueryOptimizer.expand(query)
        query_type_stats[qtype] += 1

        print(f"[QueryRouter] qid={qid}, type={qtype}, expanded={expanded_queries}")

        all_retrieved = []
        for sub_q in expanded_queries:
            retrieved_chunks = retriever.retrieve(
                sub_q,
                top_k_chunks=TOP_K_CHUNKS,
                dense_k=DENSE_CANDIDATE_K,
                bm25_k=BM25_CANDIDATE_K
            )
            all_retrieved.extend(retrieved_chunks)

        retrieved = merge_retrieved_chunks(
            all_retrieved,
            top_k=TOP_K_CHUNKS
        )

        if not retrieved:
            continue

        texts = []
        doc_ids = []

        for _, text, doc_id, _ in retrieved:
            texts.append(text)
            doc_ids.append(doc_id)

        ranked_chunks = reranker.rerank_texts(
            query,
            texts,
            doc_ids,
            top_k=TOP_K_CHUNKS
        )

        if not ranked_chunks:
            continue

        doc_scores = aggregate_doc_scores(
            ranked_chunks,
            mode=DOC_AGG_MODE
        )

        if not doc_scores:
            continue

        rerank_results_with_scores[qid] = doc_scores
        generation_chunks[qid] = select_generation_chunks(
            ranked_chunks,
            top_n=5,
            max_per_doc=2
        )

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

    gen_scores = Generator.run(queries, generation_chunks)

    print("\n========== Generation Quality ==========")
    if gen_scores:
        print("Avg score:", sum(gen_scores) / len(gen_scores))
        print({i: gen_scores.count(i) for i in range(1, 6)})
    else:
        print("Avg score: N/A")
        print("No valid generation scores.")


if __name__ == "__main__":
    main()