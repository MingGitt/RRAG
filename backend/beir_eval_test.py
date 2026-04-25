# beir_eval_test.py
import argparse
import asyncio
import json
import os
import threading
import time
from typing import Any, Dict, Tuple

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET,
    DATASET_PATH,
    EMBED_MODEL_NAME,
    MAX_CONCURRENCY,
    MAX_SAMPLES,
    PERSIST_DIR,
    QUERY_WORKERS,
    RERANK_TOP_DOCS,
    SUBQUERY_WORKERS,
)
from generator import Generator
from hybrid_retriever import HybridRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from reranker import Reranker
from risk_controller import RiskController
from vector_store import VectorStoreBuilder


def load_beir_data(dataset_name: str, dataset_path: str):
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    collection_name = VectorStoreBuilder.make_collection_name(
        dataset_name,
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


def select_queries(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    max_queries: int = 0,
):
    if max_queries and max_queries > 0:
        selected_items = list(queries.items())[:max_queries]
        selected_queries = dict(selected_items)
    else:
        selected_queries = queries

    selected_qrels = {
        qid: qrels[qid]
        for qid in selected_queries
        if qid in qrels
    }
    return selected_queries, selected_qrels


def evaluate_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]],
    results_with_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    evaluator = EvaluateRetrieval()
    ndcg, _map, recall, precision = evaluator.evaluate(
        qrels,
        results_with_scores,
        k_values=[1, 3, 5, 10, 20, 50, 100],
    )
    return {
        "NDCG": ndcg,
        "MAP": _map,
        "Recall": recall,
        "Precision": precision,
    }


def print_retrieval_metrics(metrics: Dict[str, Any]):
    print("NDCG:", json.dumps(metrics["NDCG"], ensure_ascii=False, indent=2))
    print("MAP:", json.dumps(metrics["MAP"], ensure_ascii=False, indent=2))
    print("Recall:", json.dumps(metrics["Recall"], ensure_ascii=False, indent=2))
    print("Precision:", json.dumps(metrics["Precision"], ensure_ascii=False, indent=2))


def print_generation_chunk_stats(queries, ranked_chunks_map, generation_chunks, max_show=10):
    print("\n========== Generation Chunk Stats ==========")
    for qid in list(queries.keys())[:max_show]:
        ranked_cnt = len(ranked_chunks_map.get(qid, []))
        gen_cnt = len(generation_chunks.get(qid, []))

        print(f"qid={qid} ranked_chunks={ranked_cnt}, generation_chunks={gen_cnt}")

        if gen_cnt > 0:
            preview_chunks = generation_chunks.get(qid, [])[:2]
            for i, item in enumerate(preview_chunks, start=1):
                text = (item.get("text", "") or "").replace("\n", " ")
                critique_rel = item.get("critique_relevance", "Unknown")
                critique_use = item.get("critique_usefulness", "Unknown")
                critique_score = float(item.get("critique_score", 0.0))
                rerank_score = float(item.get("score", 0.0))
                print(
                    f"  [gen_chunk_{i}] rerank_score={rerank_score:.4f} "
                    f"critique_relevance={critique_rel} "
                    f"critique_usefulness={critique_use} "
                    f"critique_score={critique_score:.4f} "
                    f"text={text[:120]}"
                )


def print_generation_outputs_for_queries(
    queries: Dict[str, str],
    ranked_chunks_map: Dict[str, Any],
    generation_chunks: Dict[str, Any],
    gen_outputs: Dict[str, Any],
    max_show: int = 5,
):
    print("\n========== Generation Outputs ==========")

    for idx, (qid, query_text) in enumerate(list(queries.items())[:max_show], start=1):
        gen_output = gen_outputs.get(qid, {})
        ranked_chunks = ranked_chunks_map.get(qid, [])
        gen_chunks = generation_chunks.get(qid, [])

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

        print("\n---------- Reflections ----------")
        print(gen_output.get("reflections", {}))

        print("\n---------- Selected Generation Chunks ----------")
        if not gen_chunks:
            print("无 generation chunks")
        else:
            for i, item in enumerate(gen_chunks, start=1):
                print(f"\n[Generation Chunk {i}]")
                print("doc_id:", item.get("doc_id", ""))
                print("rerank_score:", item.get("score", 0.0))
                print("critique_relevance:", item.get("critique_relevance", "Unknown"))
                print("critique_usefulness:", item.get("critique_usefulness", "Unknown"))
                print("critique_score:", item.get("critique_score", 0.0))
                print("critique_reason:", item.get("critique_reason", ""))
                print("text:", (item.get("text", "") or "")[:500])

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
                print("support_score:", cand.get("support_score", 0.0))
                print("completeness_score:", cand.get("completeness_score", 0.0))
                print("retrieve:", cand.get("retrieve", "Unknown"))
                print("isrel:", cand.get("isrel", "Unknown"))
                print("issup:", cand.get("issup", "Unknown"))
                print("isuse:", cand.get("isuse", 1))
                print("support_reason:", cand.get("support_reason", ""))

        print("\n---------- Citations ----------")
        citations = gen_output.get("citations", [])
        if not citations:
            print("无证据来源")
        else:
            for c in citations:
                print(f"[{c['index']}] {c['source']}")
                print(c["excerpt"])
                print("-" * 80)

        print("\n---------- Raw Generated Answer ----------")
        print(gen_output.get("raw_text", ""))


def aggregate_generation_chunks_to_doc_scores(
    generation_chunks: Dict[str, Any],
    agg_mode: str = "max",
    score_key: str = "critique_score",
    fallback_score_key: str = "score",
    top_k_docs: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    将 generation_chunks 聚合为 doc-level 检索结果:
    {
        qid: {
            doc_id: score,
            ...
        }
    }

    参数：
    - agg_mode:
        max  -> 同一 doc 取最大 chunk 分
        sum  -> 同一 doc 分数求和
        mean -> 同一 doc 分数平均
    - score_key:
        优先使用的分数字段，默认 critique_score
    - fallback_score_key:
        score_key 缺失时回退到这个字段，默认 rerank score
    - top_k_docs:
        >0 时只保留前 top_k_docs 个文档
    """
    doc_results: Dict[str, Dict[str, float]] = {}

    for qid, chunks in generation_chunks.items():
        if not isinstance(chunks, list) or not chunks:
            doc_results[qid] = {}
            continue

        temp_scores: Dict[str, float] = {}
        temp_counts: Dict[str, int] = {}

        for item in chunks:
            if not isinstance(item, dict):
                continue

            doc_id = str(item.get("doc_id", "")).strip()
            if not doc_id:
                continue

            score = item.get(score_key, None)
            if score is None:
                score = item.get(fallback_score_key, None)
            if score is None:
                score = 0.0

            try:
                score = float(score)
            except Exception:
                score = 0.0

            if agg_mode == "max":
                temp_scores[doc_id] = max(score, temp_scores.get(doc_id, float("-inf")))
            elif agg_mode == "sum":
                temp_scores[doc_id] = temp_scores.get(doc_id, 0.0) + score
            elif agg_mode == "mean":
                temp_scores[doc_id] = temp_scores.get(doc_id, 0.0) + score
                temp_counts[doc_id] = temp_counts.get(doc_id, 0) + 1
            else:
                raise ValueError(f"不支持的聚合方式: {agg_mode}")

        if agg_mode == "mean":
            for doc_id in list(temp_scores.keys()):
                cnt = temp_counts.get(doc_id, 1)
                temp_scores[doc_id] = temp_scores[doc_id] / max(cnt, 1)

        sorted_scores = dict(sorted(temp_scores.items(), key=lambda x: x[1], reverse=True))
        if top_k_docs and top_k_docs > 0:
            sorted_scores = dict(list(sorted_scores.items())[:top_k_docs])

        doc_results[qid] = sorted_scores

    return doc_results


def build_retrieval_components(
    dataset_name: str,
    dataset_path: str,
):
    corpus, queries, qrels, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = load_beir_data(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )

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

    return corpus, queries, qrels, retriever, reranker, reranker_lock


def run_full_retrieval(
    dataset_name: str,
    dataset_path: str,
    max_queries: int = 0,
    generation_doc_agg_mode: str = "max",
    generation_score_key: str = "critique_score",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    corpus, queries, qrels, retriever, reranker, reranker_lock = build_retrieval_components(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )

    queries, qrels = select_queries(queries, qrels, max_queries=max_queries)

    print(f"\n========== Test Setting ==========")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Queries: {len(queries)}")
    print(f"Generation doc aggregation mode: {generation_doc_agg_mode}")
    print(f"Generation score key: {generation_score_key}")

    start_time = time.time()

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
        subquery_workers=SUBQUERY_WORKERS,
    )

    retrieval_end_time = time.time()
    retrieval_elapsed = retrieval_end_time - start_time

    rerank_metrics = evaluate_retrieval_metrics(qrels, rerank_results_with_scores)

    generation_doc_results = aggregate_generation_chunks_to_doc_scores(
        generation_chunks=generation_chunks,
        agg_mode=generation_doc_agg_mode,
        score_key=generation_score_key,
        fallback_score_key="score",
        top_k_docs=RERANK_TOP_DOCS,
    )
    generation_metrics = evaluate_retrieval_metrics(qrels, generation_doc_results)

    summary = {
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "queries": len(queries),
        "retrieved_queries": len(rerank_results_with_scores),
        "top_k_docs": RERANK_TOP_DOCS,
        "retrieval_elapsed_sec": round(retrieval_elapsed, 3),
        "query_type_stats": query_type_stats,
        "generation_doc_agg_mode": generation_doc_agg_mode,
        "generation_score_key": generation_score_key,
        "metrics_rerank_docs": rerank_metrics,
        "metrics_generation_chunks_aggregated_to_docs": generation_metrics,
    }

    details = {
        "queries": queries,
        "qrels": qrels,
        "rerank_doc_results": rerank_results_with_scores,
        "generation_doc_results": generation_doc_results,
        "generation_chunks": generation_chunks,
        "ranked_chunks_map": ranked_chunks_map,
        "query_type_stats": query_type_stats,
    }

    print(f"\n检索和重排序总用时: {retrieval_elapsed:.2f}s")
    print("\n========== Query Routing Stats ==========")
    print(query_type_stats)

    print("\n========== Retrieval Quality (Rerank Docs) ==========")
    print_retrieval_metrics(rerank_metrics)

    print("\n========== Retrieval Quality (Generation Chunks -> Docs) ==========")
    print_retrieval_metrics(generation_metrics)

    print_generation_chunk_stats(queries, ranked_chunks_map, generation_chunks)

    return summary, details


def run_generation(
    queries: Dict[str, str],
    generation_chunks: Dict[str, Any],
):
    return asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=min(len(queries), MAX_SAMPLES),
            max_concurrency=MAX_CONCURRENCY,
        )
    )


def save_json(path: str, obj: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DATASET, help="如 scifact / nfcorpus")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH, help="本地 BEIR 数据集目录")
    parser.add_argument("--max_queries", type=int, default=0, help="只跑前多少条 query，0 表示全量")
    parser.add_argument("--with_generation", action="store_true", help="是否在检索后继续跑生成")
    parser.add_argument("--save_prefix", type=str, default="", help="输出文件名前缀")
    parser.add_argument("--show_generation", type=int, default=5, help="打印前多少条生成结果")
    parser.add_argument(
        "--generation_doc_agg_mode",
        type=str,
        default="max",
        choices=["max", "sum", "mean"],
        help="generation_chunks 聚合成 doc 的方式",
    )
    parser.add_argument(
        "--generation_score_key",
        type=str,
        default="critique_score",
        help="generation_chunks 聚合时优先使用的分数字段，如 critique_score / score",
    )
    args = parser.parse_args()

    dataset_name = args.dataset.lower().strip()
    dataset_path = os.path.abspath(args.dataset_path)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"dataset_path 不存在或不是目录: {dataset_path}")

    summary, details = run_full_retrieval(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        max_queries=args.max_queries,
        generation_doc_agg_mode=args.generation_doc_agg_mode,
        generation_score_key=args.generation_score_key,
    )

    prefix = args.save_prefix or dataset_name
    retrieval_summary_path = f"{prefix}_full_retrieval_summary.json"
    retrieval_details_path = f"{prefix}_full_retrieval_details.jsonl"

    save_json(retrieval_summary_path, summary)

    detail_rows = []
    queries = details["queries"]
    qrels = details["qrels"]
    rerank_doc_results = details["rerank_doc_results"]
    generation_doc_results = details["generation_doc_results"]
    generation_chunks = details["generation_chunks"]
    ranked_chunks_map = details["ranked_chunks_map"]

    for qid, query in queries.items():
        detail_rows.append({
            "qid": qid,
            "query": query,
            "qrels": qrels.get(qid, {}),
            "rerank_doc_scores": rerank_doc_results.get(qid, {}),
            "generation_doc_scores": generation_doc_results.get(qid, {}),
            "generation_chunks": generation_chunks.get(qid, []),
            "ranked_chunks": ranked_chunks_map.get(qid, []),
        })

    save_jsonl(retrieval_details_path, detail_rows)

    print("\n========== Full Retrieval Pipeline Evaluation ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {retrieval_summary_path}")
    print(f"Saved: {retrieval_details_path}")

    if args.with_generation:
        print("\nStart generation...")
        queries = details["queries"]
        generation_chunks = details["generation_chunks"]
        ranked_chunks_map = details["ranked_chunks_map"]

        generation_start = time.time()
        gen_outputs = run_generation(queries, generation_chunks)
        generation_elapsed = time.time() - generation_start

        gen_summary = {
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "queries": len(queries),
            "generated_queries": len(gen_outputs),
            "generation_elapsed_sec": round(generation_elapsed, 3),
        }

        gen_summary_path = f"{prefix}_generation_summary.json"
        gen_details_path = f"{prefix}_generation_details.jsonl"

        save_json(gen_summary_path, gen_summary)

        gen_rows = []
        for qid, query in queries.items():
            gen_rows.append({
                "qid": qid,
                "query": query,
                "generation_chunks": generation_chunks.get(qid, []),
                "ranked_chunks": ranked_chunks_map.get(qid, []),
                "gen_output": gen_outputs.get(qid, {}),
            })

        save_jsonl(gen_details_path, gen_rows)

        print(f"\n生成总用时: {generation_elapsed:.2f}s")
        print(f"Saved: {gen_summary_path}")
        print(f"Saved: {gen_details_path}")

        print_generation_outputs_for_queries(
            queries=queries,
            ranked_chunks_map=ranked_chunks_map,
            generation_chunks=generation_chunks,
            gen_outputs=gen_outputs,
            max_show=args.show_generation,
        )


if __name__ == "__main__":
    main()