# beir_eval_test.py
import asyncio
import threading
import time
from typing import Dict, Any

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET,
    DATASET_PATH,
    EMBED_MODEL_NAME,
    PERSIST_DIR,
    QUERY_WORKERS,
    SUBQUERY_WORKERS,
)
from generator import Generator
from hybrid_retriever import HybridRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from reranker import Reranker
from risk_controller import RiskController
from vector_store import VectorStoreBuilder


TEST_QUERY_LIMIT = 30


def load_beir_data():
    """
    加载 BEIR 数据，并构建/加载当前数据集的向量库
    """
    corpus, queries, qrels = GenericDataLoader(DATASET_PATH).load(split="test")

    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME,
    )

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load(
        corpus=corpus,
        persist_dir=PERSIST_DIR,
        collection_name=collection_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    return corpus, queries, qrels, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas


def select_first_n_queries(queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], n: int = 30):
    """
    只保留前 n 个 query，并同步裁剪 qrels
    """
    selected_items = list(queries.items())[:n]
    selected_queries = dict(selected_items)
    selected_qrels = {qid: qrels[qid] for qid in selected_queries if qid in qrels}
    return selected_queries, selected_qrels


def print_retrieval_metrics(qrels, rerank_results_with_scores):
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        rerank_results_with_scores,
        k_values=[1, 3, 5, 10],
    )

    print("\n========== Retrieval Quality ==========")
    print("NDCG:", ndcg)
    print("MAP:", _map)
    print("Recall:", recall)
    print("Precision:", precision)


def print_generation_outputs_for_queries(
    queries: Dict[str, str],
    ranked_chunks_map: Dict[str, Any],
    gen_outputs: Dict[str, Any],
):
    """
    逐条打印前30个 query 的生成结果
    """
    print("\n========== Generation Outputs (First 30 Queries) ==========")

    for idx, (qid, query_text) in enumerate(queries.items(), start=1):
        gen_output = gen_outputs.get(qid, {})
        ranked_chunks = ranked_chunks_map.get(qid, [])

        retrieval_risk = RiskController.assess_retrieval_risk(ranked_chunks)
        generation_risk = gen_output.get("generation_risk", {})

        print(f"\n\n==================== Query #{idx} | qid={qid} ====================")
        print("问题：")
        print(query_text)

        print("\n最终答案：")
        print(gen_output.get("answer", "未生成答案。"))

        print("\n---------- Retrieval Risk ----------")
        print(retrieval_risk)

        print("\n---------- Generation Risk ----------")
        print(generation_risk)

        print("\n---------- Self-RAG Reflections ----------")
        print(gen_output.get("reflections", {}))

        print("\n---------- Candidates ----------")
        candidates = gen_output.get("candidates", [])
        if not candidates:
            print("无候选答案")
        else:
            for i, cand in enumerate(candidates, start=1):
                print(f"\n[Candidate {i}]")
                print("answer:", cand.get("answer", ""))
                print("evidence_score:", cand.get("evidence_score", 0.0))
                print("final_score:", cand.get("final_score", 0.0))
                print("retrieve:", cand.get("retrieve", "Unknown"))
                print("isrel:", cand.get("isrel", "Unknown"))
                print("issup:", cand.get("issup", "Unknown"))
                print("isuse:", cand.get("isuse", 1))

        print("\n---------- Citations ----------")
        citations = gen_output.get("citations", [])
        if not citations:
            print("无证据来源")
        else:
            for c in citations:
                print(f"[{c['index']}] {c['source']}")
                print(c["excerpt"])
                print("-" * 80)

        print("\n---------- Raw Self-RAG Output ----------")
        print(gen_output.get("raw_text", ""))


def run_beir_evaluation():
    corpus, queries, qrels, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = load_beir_data()

    # 只取前30个 query
    queries, qrels = select_first_n_queries(queries, qrels, TEST_QUERY_LIMIT)

    print(f"\n========== Test Setting ==========")
    print(f"Dataset: {DATASET}")
    print(f"Using first {len(queries)} queries for retrieval + generation test")

    retriever = HybridRetriever(
        vectordb=vectordb,
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_ids=chunk_ids,
        chunk_metas=chunk_metas,
        alpha=ALPHA,
    )

    reranker = Reranker()
    reranker_lock = threading.Lock()

    start_time = time.time()

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
        subquery_workers=SUBQUERY_WORKERS,
    )
    print("\n========== Generation Chunk Stats ==========")
    for qid in list(queries.keys())[:30]:
        ranked_cnt = len(ranked_chunks_map.get(qid, []))
        gen_cnt = len(generation_chunks.get(qid, []))

        print(f"qid={qid} ranked_chunks={ranked_cnt}, generation_chunks={gen_cnt}")

        # 如果你还想顺便看一下前几个 generation chunk 的内容，可以保留下面这段
        if gen_cnt > 0:
            preview_chunks = generation_chunks.get(qid, [])[:2]
            for i, item in enumerate(preview_chunks, start=1):
                text = (item.get("text", "") or "").replace("\n", " ")
                print(f"  [gen_chunk_{i}] score={item.get('score', 0.0):.4f} text={text[:120]}")

    retrieval_end_time = time.time()

    print(f"\n检索和重排序总用时: {retrieval_end_time - start_time:.2f}s")
    print("\n========== Query Routing Stats ==========")
    print(query_type_stats)

    print_retrieval_metrics(qrels, rerank_results_with_scores)

    # 对前30个 query 全部做生成
    gen_outputs = asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=len(queries),
            max_concurrency=1,   # 如果你想并发生成，可以调大，但先稳妥起见设为1
        )
    )

    generation_end_time = time.time()
    print(f"\n生成总用时: {generation_end_time - retrieval_end_time:.2f}s")
    print(f"总流程用时: {generation_end_time - start_time:.2f}s")

    print_generation_outputs_for_queries(
        queries=queries,
        ranked_chunks_map=ranked_chunks_map,
        gen_outputs=gen_outputs,
    )


if __name__ == "__main__":
    run_beir_evaluation()