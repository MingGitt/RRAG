# scifact_ablation_no_query_opt_eval.py
# 放置位置：
# D:\code\rag\FSR\eval\scifact_ablation_no_query_opt_eval.py
#
# 实验目的：
# 以完整基线流程为基础，只取消查询优化模块。
#
# 原完整流程：
# query -> QueryOptimizer.expand -> 多子查询检索 -> merge -> rerank
#       -> ChunkCritiqueJudge -> generation_chunks -> Generator
#
# 本消融流程：
# query -> 直接检索 -> rerank
#       -> ChunkCritiqueJudge -> generation_chunks -> Generator
#
# 运行示例：
# cd D:\code\rag\FSR\eval
# python scifact_ablation_no_query_opt_eval.py --max_queries 20
# python scifact_ablation_no_query_opt_eval.py --max_queries 0 --output_name scifact_ablation_no_query_opt_full
# python scifact_ablation_no_query_opt_eval.py --max_queries 50 --with_generation

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


# ============================================================
# 1. 路径设置
# ============================================================

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

DEFAULT_DATASET_NAME = "scifact"
DEFAULT_DATASET_PATH = r"D:\code\rag\FSR\data\scifact"
DEFAULT_PERSIST_DIR = str(PROJECT_ROOT / "chroma_store")

# 必须在 import backend/config.py 前设置
os.environ.setdefault("DATASET", DEFAULT_DATASET_NAME)
os.environ.setdefault("DATASET_PATH", DEFAULT_DATASET_PATH)
os.environ.setdefault("PERSIST_DIR", DEFAULT_PERSIST_DIR)

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# 2. 导入你后端已有接口
# ============================================================

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

from config import (
    ALPHA,
    BM25_CANDIDATE_K,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET,
    DATASET_PATH,
    DENSE_CANDIDATE_K,
    DOC_AGG_MODE,
    EMBED_MODEL_NAME,
    GENERATION_MAX_PER_DOC,
    GENERATION_TOP_N,
    MAX_CONCURRENCY,
    MAX_SAMPLES,
    PERSIST_DIR,
    QUERY_WORKERS,
    RERANK_TOP_DOCS,
    TOP_K_CHUNKS,
)

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from evaluator import RetrievalEvaluator
from chunk_critique_judge import ChunkCritiqueJudge

try:
    from generator import Generator
except Exception:
    Generator = None


# ============================================================
# 3. 通用工具函数
# ============================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_metrics(metrics: Dict[str, Any]):
    print("NDCG:", json.dumps(metrics["NDCG"], ensure_ascii=False, indent=2))
    print("MAP:", json.dumps(metrics["MAP"], ensure_ascii=False, indent=2))
    print("Recall:", json.dumps(metrics["Recall"], ensure_ascii=False, indent=2))
    print("Precision:", json.dumps(metrics["Precision"], ensure_ascii=False, indent=2))


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


def select_queries(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    max_queries: int = 0,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    if max_queries and max_queries > 0:
        selected_queries = dict(list(queries.items())[:max_queries])
    else:
        selected_queries = queries

    selected_qrels = {
        qid: qrels[qid]
        for qid in selected_queries
        if qid in qrels
    }

    return selected_queries, selected_qrels


def aggregate_generation_chunks_to_doc_scores(
    generation_chunks: Dict[str, Any],
    agg_mode: str = "max",
    score_key: str = "critique_score",
    fallback_score_key: str = "score",
    top_k_docs: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    将 generation_chunks 聚合为 BEIR 需要的 doc-level 结果：
    {
        qid: {
            doc_id: score
        }
    }
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
                temp_scores[doc_id] = max(
                    score,
                    temp_scores.get(doc_id, float("-inf")),
                )
            elif agg_mode == "sum":
                temp_scores[doc_id] = temp_scores.get(doc_id, 0.0) + score
            elif agg_mode == "mean":
                temp_scores[doc_id] = temp_scores.get(doc_id, 0.0) + score
                temp_counts[doc_id] = temp_counts.get(doc_id, 0) + 1
            else:
                raise ValueError(f"不支持的聚合方式: {agg_mode}")

        if agg_mode == "mean":
            for doc_id in list(temp_scores.keys()):
                cnt = max(temp_counts.get(doc_id, 1), 1)
                temp_scores[doc_id] = temp_scores[doc_id] / cnt

        sorted_scores = dict(
            sorted(temp_scores.items(), key=lambda x: x[1], reverse=True)
        )

        if top_k_docs and top_k_docs > 0:
            sorted_scores = dict(list(sorted_scores.items())[:top_k_docs])

        doc_results[qid] = sorted_scores

    return doc_results


# ============================================================
# 4. 数据加载与组件构建
# ============================================================

def load_beir_data_and_vector_store(
    dataset_name: str,
    dataset_path: str,
):
    print("\n========== Load BEIR Dataset ==========")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset path: {dataset_path}")

    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    print(f"Corpus size: {len(corpus)}")
    print(f"Query size: {len(queries)}")
    print(f"Qrels size: {len(qrels)}")

    collection_name = VectorStoreBuilder.make_collection_name(
        dataset_name,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME,
    )

    print("\n========== Build / Load Vector Store ==========")
    print(f"Persist dir: {PERSIST_DIR}")
    print(f"Collection name: {collection_name}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Embedding model: {EMBED_MODEL_NAME}")

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = (
        VectorStoreBuilder.build_or_load(
            corpus=corpus,
            persist_dir=PERSIST_DIR,
            collection_name=collection_name,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
    )

    print(f"Loaded chunks: {len(chunk_texts)}")

    return corpus, queries, qrels, vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas


def build_retrieval_components(
    dataset_name: str,
    dataset_path: str,
):
    (
        corpus,
        queries,
        qrels,
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        chunk_metas,
    ) = load_beir_data_and_vector_store(
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


# ============================================================
# 5. 消融核心：取消查询优化后的单 query 流程
# ============================================================

def process_single_query_no_query_opt(
    qid: str,
    query: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    reranker_lock: threading.Lock,
) -> Dict[str, Any]:
    """
    这是本消融实验的核心函数。

    相比原始 process_single_query：
    取消：
        qtype, expanded_queries = QueryOptimizer.expand(query)
        retrieve_expanded_queries_parallel(...)

    保留：
        原始 query 直接检索
        HybridRetriever.retrieve
        Reranker.rerank_texts
        RetrievalEvaluator.aggregate_doc_scores
        ChunkCritiqueJudge.judge_chunks
        ChunkCritiqueJudge.select_generation_chunks
        Reranker.select_generation_chunks fallback
    """

    qtype = "no_query_optimization"

    # 1. 原始 query 直接进入检索
    retrieved = retriever.retrieve(
        query,
        top_k_chunks=TOP_K_CHUNKS,
        dense_k=DENSE_CANDIDATE_K,
        bm25_k=BM25_CANDIDATE_K,
    )

    if not retrieved:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
            "ranked_chunks": None,
            "critiqued_chunks": None,
        }

    # 2. 准备 rerank 输入
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

    # 3. 保留原 reranker
    with reranker_lock:
        ranked_chunks_raw = reranker.rerank_texts(
            query,
            texts,
            candidate_indices,
            metas=metas,
            top_k=TOP_K_CHUNKS,
        )

    if not ranked_chunks_raw:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
            "ranked_chunks": None,
            "critiqued_chunks": None,
        }

    ranked_chunks = []
    for candidate_idx, text, score, meta in ranked_chunks_raw:
        ranked_chunks.append(
            {
                "chunk_idx": candidate_idx,
                "text": text,
                "doc_id": candidate_doc_map.get(candidate_idx),
                "score": float(score),
                "meta": meta,
            }
        )

    # 4. 保留原来的 doc-level 聚合评测逻辑
    doc_level_for_eval = [
        (item["doc_id"], item["text"], item["score"])
        for item in ranked_chunks
    ]

    doc_scores = RetrievalEvaluator.aggregate_doc_scores(
        doc_level_for_eval,
        mode=DOC_AGG_MODE,
    )

    if not doc_scores:
        return {
            "qid": qid,
            "qtype": qtype,
            "doc_scores": None,
            "generation_chunks": None,
            "ranked_chunks": ranked_chunks,
            "critiqued_chunks": None,
        }

    # 5. 保留 generation chunk 筛选
    critique_input = ranked_chunks[:TOP_K_CHUNKS]

    critiqued_chunks = ChunkCritiqueJudge.judge_chunks(
        query,
        critique_input,
    )

    gen_chunks = ChunkCritiqueJudge.select_generation_chunks(
        critiqued_chunks,
        top_n=GENERATION_TOP_N,
        max_per_doc=GENERATION_MAX_PER_DOC,
        min_score=0.35,
    )

    # 6. 保留 fallback：如果 critique 一个都没选中，就退回 rerank 结果
    if not gen_chunks:
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
            top_n=GENERATION_TOP_N,
            max_per_doc=GENERATION_MAX_PER_DOC,
        )

    return {
        "qid": qid,
        "qtype": qtype,
        "doc_scores": doc_scores,
        "generation_chunks": gen_chunks,
        "ranked_chunks": ranked_chunks,
        "critiqued_chunks": critiqued_chunks,
    }


def run_parallel_retrieval_pipeline_no_query_opt(
    queries: Dict[str, str],
    retriever: HybridRetriever,
    reranker: Reranker,
    reranker_lock: threading.Lock,
    query_workers: int = 4,
):
    """
    消融版并行 pipeline。

    不再创建 subquery_executor。
    因为取消查询优化后，每个 query 只有一个原始检索请求。
    """
    rerank_results_with_scores = {}
    generation_chunks = {}
    ranked_chunks_map = {}
    critiqued_chunks_map = {}
    query_type_stats = {}

    with ThreadPoolExecutor(max_workers=query_workers) as query_executor:
        future_to_qid = {
            query_executor.submit(
                process_single_query_no_query_opt,
                qid,
                query,
                retriever,
                reranker,
                reranker_lock,
            ): qid
            for qid, query in queries.items()
        }

        for future in as_completed(future_to_qid):
            try:
                result = future.result()
            except Exception as e:
                qid = future_to_qid[future]
                print(f"[ERROR] qid={qid} failed: {repr(e)}")
                continue

            qid = result["qid"]
            qtype = result["qtype"]
            doc_scores = result["doc_scores"]
            gen_chunks = result["generation_chunks"]
            ranked_chunks = result["ranked_chunks"]
            critiqued_chunks = result.get("critiqued_chunks")

            query_type_stats[qtype] = query_type_stats.get(qtype, 0) + 1

            if doc_scores:
                rerank_results_with_scores[qid] = doc_scores

            if gen_chunks:
                generation_chunks[qid] = gen_chunks

            if ranked_chunks:
                ranked_chunks_map[qid] = ranked_chunks

            if critiqued_chunks:
                critiqued_chunks_map[qid] = critiqued_chunks

    return (
        rerank_results_with_scores,
        generation_chunks,
        ranked_chunks_map,
        critiqued_chunks_map,
        query_type_stats,
    )


# ============================================================
# 6. 检索评测主逻辑
# ============================================================

def run_ablation_no_query_opt_retrieval(
    dataset_name: str,
    dataset_path: str,
    max_queries: int,
    generation_doc_agg_mode: str,
    generation_score_key: str,
):
    (
        corpus,
        queries,
        qrels,
        retriever,
        reranker,
        reranker_lock,
    ) = build_retrieval_components(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )

    selected_queries, selected_qrels = select_queries(
        queries=queries,
        qrels=qrels,
        max_queries=max_queries,
    )

    print("\n========== Ablation Setting ==========")
    print("Experiment: no_query_optimization")
    print("Removed module: QueryOptimizer.expand")
    print("Kept modules: hybrid retrieval, rerank, chunk critique, generation chunk selection")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Selected queries: {len(selected_queries)}")
    print(f"Query workers: {QUERY_WORKERS}")
    print(f"Top K chunks: {TOP_K_CHUNKS}")
    print(f"Dense candidate k: {DENSE_CANDIDATE_K}")
    print(f"BM25 candidate k: {BM25_CANDIDATE_K}")
    print(f"Rerank top docs: {RERANK_TOP_DOCS}")
    print(f"Generation top n: {GENERATION_TOP_N}")
    print(f"Generation max per doc: {GENERATION_MAX_PER_DOC}")
    print(f"Generation doc agg mode: {generation_doc_agg_mode}")
    print(f"Generation score key: {generation_score_key}")

    start_time = time.time()

    (
        rerank_results_with_scores,
        generation_chunks,
        ranked_chunks_map,
        critiqued_chunks_map,
        query_type_stats,
    ) = run_parallel_retrieval_pipeline_no_query_opt(
        queries=selected_queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
    )

    retrieval_elapsed = time.time() - start_time

    # 1. rerank doc-level 指标
    rerank_metrics = evaluate_retrieval_metrics(
        selected_qrels,
        rerank_results_with_scores,
    )

    # 2. generation chunks 聚合后指标
    generation_doc_results = aggregate_generation_chunks_to_doc_scores(
        generation_chunks=generation_chunks,
        agg_mode=generation_doc_agg_mode,
        score_key=generation_score_key,
        fallback_score_key="score",
        top_k_docs=RERANK_TOP_DOCS,
    )

    generation_chunk_metrics = evaluate_retrieval_metrics(
        selected_qrels,
        generation_doc_results,
    )

    summary = {
        "task": "scifact_ablation_no_query_optimization",
        "description": "取消查询优化模块，每个 query 直接进入检索；其余混合检索、重排序、generation chunk 筛选、生成流程保持不变。",
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "corpus_size": len(corpus),
        "query_size_total": len(queries),
        "query_size_eval": len(selected_queries),
        "retrieved_queries": len(rerank_results_with_scores),
        "generation_chunk_queries": len(generation_chunks),
        "persist_dir": PERSIST_DIR,
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "alpha": ALPHA,
        "top_k_chunks": TOP_K_CHUNKS,
        "dense_candidate_k": DENSE_CANDIDATE_K,
        "bm25_candidate_k": BM25_CANDIDATE_K,
        "query_workers": QUERY_WORKERS,
        "rerank_top_docs": RERANK_TOP_DOCS,
        "generation_top_n": GENERATION_TOP_N,
        "generation_max_per_doc": GENERATION_MAX_PER_DOC,
        "retrieval_elapsed_sec": round(retrieval_elapsed, 3),
        "avg_retrieval_elapsed_per_query_sec": round(
            retrieval_elapsed / max(len(selected_queries), 1),
            6,
        ),
        "removed_modules": [
            "QueryOptimizer.expand",
            "query_type_classification",
            "query_rewrite_or_expansion",
            "subquery_decomposition",
        ],
        "kept_modules": [
            "HybridRetriever.retrieve",
            "Reranker.rerank_texts",
            "RetrievalEvaluator.aggregate_doc_scores",
            "ChunkCritiqueJudge.judge_chunks",
            "ChunkCritiqueJudge.select_generation_chunks",
            "Reranker.select_generation_chunks_fallback",
        ],
        "query_type_stats": query_type_stats,
        "generation_doc_agg_mode": generation_doc_agg_mode,
        "generation_score_key": generation_score_key,
        "metrics_rerank_docs": rerank_metrics,
        "metrics_generation_chunks_aggregated_to_docs": generation_chunk_metrics,
    }

    details = {
        "queries": selected_queries,
        "qrels": selected_qrels,
        "rerank_doc_results": rerank_results_with_scores,
        "generation_doc_results": generation_doc_results,
        "generation_chunks": generation_chunks,
        "ranked_chunks_map": ranked_chunks_map,
        "critiqued_chunks_map": critiqued_chunks_map,
        "query_type_stats": query_type_stats,
    }

    print("\n========== Retrieval Time ==========")
    print(f"检索和重排序总用时: {retrieval_elapsed:.2f}s")
    print(
        f"平均每条 query 用时: "
        f"{retrieval_elapsed / max(len(selected_queries), 1):.4f}s"
    )

    print("\n========== Query Routing Stats ==========")
    print(json.dumps(query_type_stats, ensure_ascii=False, indent=2))

    print("\n========== Retrieval Quality: Rerank Docs ==========")
    print_metrics(rerank_metrics)

    print("\n========== Retrieval Quality: Generation Chunks -> Docs ==========")
    print_metrics(generation_chunk_metrics)

    return summary, details


# ============================================================
# 7. 可选：生成阶段
# ============================================================

def run_generation(
    queries: Dict[str, str],
    generation_chunks: Dict[str, Any],
):
    if Generator is None:
        raise RuntimeError(
            "无法导入 backend/generator.py 中的 Generator，不能运行生成阶段。"
        )

    return asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=min(len(queries), MAX_SAMPLES),
            max_concurrency=MAX_CONCURRENCY,
        )
    )


# ============================================================
# 8. 保存输出
# ============================================================

def save_retrieval_outputs(
    output_dir: Path,
    summary: Dict[str, Any],
    details: Dict[str, Any],
):
    summary_path = output_dir / "ablation_no_query_opt_retrieval_summary.json"
    details_path = output_dir / "ablation_no_query_opt_retrieval_details.jsonl"
    config_path = output_dir / "ablation_no_query_opt_runtime_config.json"

    save_json(summary_path, summary)

    queries = details["queries"]
    qrels = details["qrels"]
    rerank_doc_results = details["rerank_doc_results"]
    generation_doc_results = details["generation_doc_results"]
    generation_chunks = details["generation_chunks"]
    ranked_chunks_map = details["ranked_chunks_map"]
    critiqued_chunks_map = details["critiqued_chunks_map"]

    rows = []

    for qid, query in queries.items():
        rows.append(
            {
                "qid": qid,
                "query": query,
                "qrels": qrels.get(qid, {}),
                "rerank_doc_scores": rerank_doc_results.get(qid, {}),
                "generation_doc_scores": generation_doc_results.get(qid, {}),
                "generation_chunks": generation_chunks.get(qid, []),
                "ranked_chunks": ranked_chunks_map.get(qid, []),
                "critiqued_chunks": critiqued_chunks_map.get(qid, []),
            }
        )

    save_jsonl(details_path, rows)

    runtime_config = {
        "project_root": str(PROJECT_ROOT),
        "backend_dir": str(BACKEND_DIR),
        "eval_dir": str(EVAL_DIR),
        "persist_dir": PERSIST_DIR,
        "dataset": DATASET,
        "dataset_path_from_config": DATASET_PATH,
        "embed_model_name": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "alpha": ALPHA,
        "top_k_chunks": TOP_K_CHUNKS,
        "dense_candidate_k": DENSE_CANDIDATE_K,
        "bm25_candidate_k": BM25_CANDIDATE_K,
        "doc_agg_mode": DOC_AGG_MODE,
        "query_workers": QUERY_WORKERS,
        "rerank_top_docs": RERANK_TOP_DOCS,
        "generation_top_n": GENERATION_TOP_N,
        "generation_max_per_doc": GENERATION_MAX_PER_DOC,
        "max_samples": MAX_SAMPLES,
        "max_concurrency": MAX_CONCURRENCY,
        "ablation": {
            "name": "no_query_optimization",
            "removed": [
                "QueryOptimizer.expand",
                "query_type_classification",
                "query_rewrite_or_expansion",
                "subquery_decomposition",
            ],
            "kept": [
                "HybridRetriever",
                "Reranker",
                "ChunkCritiqueJudge",
                "Generator",
            ],
        },
    }

    save_json(config_path, runtime_config)

    print("\n========== Saved Retrieval Outputs ==========")
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {details_path}")
    print(f"Saved config: {config_path}")


def save_generation_outputs(
    output_dir: Path,
    queries: Dict[str, str],
    generation_chunks: Dict[str, Any],
    ranked_chunks_map: Dict[str, Any],
    gen_outputs: Dict[str, Any],
    generation_elapsed: float,
):
    gen_summary = {
        "task": "scifact_ablation_no_query_optimization_generation",
        "queries": len(queries),
        "generated_queries": len(gen_outputs),
        "generation_elapsed_sec": round(generation_elapsed, 3),
        "avg_generation_elapsed_per_query_sec": round(
            generation_elapsed / max(len(queries), 1),
            6,
        ),
    }

    gen_summary_path = output_dir / "ablation_no_query_opt_generation_summary.json"
    gen_details_path = output_dir / "ablation_no_query_opt_generation_details.jsonl"

    save_json(gen_summary_path, gen_summary)

    rows = []

    for qid, query in queries.items():
        rows.append(
            {
                "qid": qid,
                "query": query,
                "generation_chunks": generation_chunks.get(qid, []),
                "ranked_chunks": ranked_chunks_map.get(qid, []),
                "gen_output": gen_outputs.get(qid, {}),
            }
        )

    save_jsonl(gen_details_path, rows)

    print("\n========== Saved Generation Outputs ==========")
    print(f"Saved generation summary: {gen_summary_path}")
    print(f"Saved generation details: {gen_details_path}")


# ============================================================
# 9. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="数据集名称，默认 scifact",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=DEFAULT_DATASET_PATH,
        help=r"本地 SciFact 数据集目录，默认 D:\code\rag\FSR\data\scifact",
    )

    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="测试前多少条 query；0 表示全量",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="",
        help="输出文件夹名称；为空则自动生成时间戳目录",
    )

    parser.add_argument(
        "--with_generation",
        action="store_true",
        help="是否在检索后继续调用 Generator 生成答案",
    )

    parser.add_argument(
        "--generation_doc_agg_mode",
        type=str,
        default="max",
        choices=["max", "sum", "mean"],
        help="generation_chunks 聚合到 doc-level 的方式",
    )

    parser.add_argument(
        "--generation_score_key",
        type=str,
        default="critique_score",
        help="generation_chunks 聚合时优先使用的分数字段，例如 critique_score 或 score",
    )

    args = parser.parse_args()

    dataset_name = args.dataset.lower().strip()
    dataset_path = os.path.abspath(args.dataset_path)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"dataset_path 不存在或不是目录: {dataset_path}\n"
            f"请确认目录下包含 corpus.jsonl、queries.jsonl、qrels/test.tsv"
        )

    if args.output_name:
        output_dir = EVAL_DIR / args.output_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = EVAL_DIR / f"scifact_ablation_no_query_opt_results_{timestamp}"

    ensure_dir(output_dir)

    print("\n========== Output Directory ==========")
    print(output_dir)

    summary, details = run_ablation_no_query_opt_retrieval(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        max_queries=args.max_queries,
        generation_doc_agg_mode=args.generation_doc_agg_mode,
        generation_score_key=args.generation_score_key,
    )

    save_retrieval_outputs(
        output_dir=output_dir,
        summary=summary,
        details=details,
    )

    if args.with_generation:
        print("\n========== Start Generation ==========")
        gen_start = time.time()

        gen_outputs = run_generation(
            queries=details["queries"],
            generation_chunks=details["generation_chunks"],
        )

        generation_elapsed = time.time() - gen_start

        print(f"\n生成总用时: {generation_elapsed:.2f}s")

        save_generation_outputs(
            output_dir=output_dir,
            queries=details["queries"],
            generation_chunks=details["generation_chunks"],
            ranked_chunks_map=details["ranked_chunks_map"],
            gen_outputs=gen_outputs,
            generation_elapsed=generation_elapsed,
        )

    print("\n========== Done ==========")
    print(f"所有结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()