from collections import defaultdict
from beir.datasets.data_loader import GenericDataLoader
from config import *

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from evaluator import RetrievalEvaluator
from generator import Generator


def aggregate_doc_scores(ranked_chunks, mode="top2_mean"):
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


def main():
    corpus, queries, qrels = GenericDataLoader(DATASET_PATH).load(split="test")

    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids = VectorStoreBuilder.build_or_load(
        corpus, EMBED_MODEL_NAME, PERSIST_DIR,
        collection_name, CHUNK_SIZE, CHUNK_OVERLAP
    )

    retriever = HybridRetriever(vectordb, chunk_texts, chunk_doc_ids, chunk_ids, ALPHA)
    reranker = Reranker(RERANK_MODEL_NAME)

    rerank_results_with_scores = {}
    generation_chunks = {}

    for qid, query in queries.items():
        retrieved = retriever.retrieve(query, TOP_K_CHUNKS)

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

        doc_scores = aggregate_doc_scores(ranked_chunks, mode=DOC_AGG_MODE)
        if not doc_scores:
            continue

        rerank_results_with_scores[qid] = doc_scores
        generation_chunks[qid] = select_generation_chunks(ranked_chunks, top_n=5, max_per_doc=2)

    ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
        qrels, rerank_results_with_scores
    )

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    gen_scores = Generator.run(queries, generation_chunks, qrels)

    print("\n========== Generation Quality ==========")
    print("Avg score:", sum(gen_scores) / len(gen_scores))
    print({i: gen_scores.count(i) for i in range(1, 6)})


if __name__ == "__main__":
    main()