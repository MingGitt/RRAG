from beir.datasets.data_loader import GenericDataLoader
from config import *

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from evaluator import RetrievalEvaluator
from generator import Generator


def main():

    corpus, queries, qrels = GenericDataLoader(DATASET_PATH).load(split="test")
    
    corpus_texts = {
        doc_id: ((doc.get("title") or "") + "\n" + (doc.get("text") or "")).strip()
        for doc_id, doc in corpus.items()
    }

    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET, CHUNK_SIZE, CHUNK_OVERLAP, EMBED_MODEL_NAME
    )

    vectordb, chunk_texts, chunk_doc_ids = VectorStoreBuilder.build_or_load(
        corpus, EMBED_MODEL_NAME, PERSIST_DIR,
        collection_name, CHUNK_SIZE, CHUNK_OVERLAP
    )

    retriever = HybridRetriever(vectordb, chunk_texts, chunk_doc_ids, ALPHA)
    reranker = Reranker(RERANK_MODEL_NAME)

    rerank_results = {}
    rerank_results_with_scores = {}
    generation_chunks = {}

    for qid, query in queries.items():

       # ===== Step1: chunk retrieval =====
        retrieved = retriever.retrieve(query, TOP_K_CHUNKS)

        texts = []
        doc_ids = []

        for _, text, doc_id, _ in retrieved:
            texts.append(text)
            doc_ids.append(doc_id)

        # ===== Step2: chunk rerank =====
        ranked_chunks = reranker.rerank_texts(
            query,
            texts,
            doc_ids,
            top_k=TOP_K_CHUNKS
        )

        # ranked_chunks: [(doc_id, text, score), ...]

        # ===== Step3: doc 聚合用于 BEIR 评测 =====
        doc_scores = {}

        for doc_id, text, score in ranked_chunks:
            doc_id = str(doc_id)
            score = float(score)

            if doc_id not in doc_scores:
                doc_scores[doc_id] = score
            else:
                doc_scores[doc_id] = max(score, doc_scores[doc_id])

        # 避免空查询（pytrec_eval 会崩）
        if not doc_scores:
            continue

        rerank_results_with_scores[qid] = doc_scores
        rerank_results[qid] = list(doc_scores.keys())
        # ===== Step4: top chunk 用于生成 =====
        generation_chunks[qid] = [text for _, text, _ in ranked_chunks[:5]]

    # ===== Retrieval evaluation =====
    ndcg, _map, recall, precision = RetrievalEvaluator.evaluate(
        qrels, rerank_results_with_scores
    )

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)

    # ===== Generation evaluation =====
    gen_scores = Generator.run(queries, generation_chunks, qrels)

    print("\n========== Generation Quality ==========")
    print("Avg score:", sum(gen_scores)/len(gen_scores))
    print({i: gen_scores.count(i) for i in range(1,6)})


if __name__ == "__main__":
    main()