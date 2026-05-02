# scifact_ablation_vector_only_eval.py
# 放置位置：
# D:\code\rag\FSR\eval\scifact_ablation_vector_only_eval.py
#
# 实验目的：
# 基于完整基线，取消：
#   1. 查询优化 QueryOptimizer
#   2. 混合检索 HybridRetriever / BM25
#   3. 重排序 Reranker
#   4. 证据筛选 ChunkCritiqueJudge
#
# 只保留：
#   原始 query -> 向量检索 -> 检索结果直接给 Generator
#
# 运行示例：
# cd D:\code\rag\FSR\eval
# python scifact_ablation_vector_only_eval.py --max_queries 20
# python scifact_ablation_vector_only_eval.py --max_queries 0 --output_name scifact_ablation_vector_only_full
# python scifact_ablation_vector_only_eval.py --max_queries 50 --with_generation

import argparse
import asyncio
import json
import os
import sys
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

# 必须在 import backend/config.py 之前设置
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
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATASET,
    DATASET_PATH,
    DOC_AGG_MODE,
    EMBED_MODEL_NAME,
    GENERATION_TOP_N,
    MAX_CONCURRENCY,
    MAX_SAMPLES,
    PERSIST_DIR,
    QUERY_WORKERS,
    RERANK_TOP_DOCS,
    TOP_K_CHUNKS,
)

from vector_store import VectorStoreBuilder

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


def chroma_distance_to_score(distance: float) -> float:
    """
    Chroma similarity_search_with_score 返回的 score 通常是距离，越小越相似。
    BEIR EvaluateRetrieval 需要分数越大越相关。
    这里用 -distance 转成排序分数。
    """
    try:
        return -float(distance)
    except Exception:
        return 0.0


def aggregate_chunk_scores_to_doc_scores(
    chunks: List[Dict[str, Any]],
    mode: str = "max",
) -> Dict[str, float]:
    """
    将向量检索得到的 chunk 分数聚合为 doc-level 分数。
    用于和 BEIR qrels 计算 NDCG / MAP / Recall / Precision。
    """
    doc_scores: Dict[str, float] = {}
    doc_counts: Dict[str, int] = {}

    for item in chunks:
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue

        score = float(item.get("score", 0.0))

        if mode == "max":
            doc_scores[doc_id] = max(score, doc_scores.get(doc_id, float("-inf")))
        elif mode == "sum":
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
        elif mode == "mean":
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        else:
            raise ValueError(f"不支持的 DOC_AGG_MODE: {mode}")

    if mode == "mean":
        for doc_id in list(doc_scores.keys()):
            doc_scores[doc_id] = doc_scores[doc_id] / max(doc_counts.get(doc_id, 1), 1)

    return dict(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# 4. 加载 SciFact 和向量库
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


# ============================================================
# 5. 消融核心：只进行向量检索
# ============================================================

def process_single_query_vector_only(
    qid: str,
    query: str,
    vectordb,
    vector_top_k: int,
    doc_agg_mode: str,
) -> Dict[str, Any]:
    """
    本消融实验的核心流程：

    取消：
        QueryOptimizer.expand
        HybridRetriever.retrieve
        BM25
        RRF fusion
        Reranker.rerank_texts
        ChunkCritiqueJudge.judge_chunks
        ChunkCritiqueJudge.select_generation_chunks

    保留：
        原始 query
        Chroma 向量检索
        检索结果直接作为 generation_chunks 给 Generator
    """

    try:
        hits = vectordb.similarity_search_with_score(query, k=vector_top_k)
    except Exception as e:
        print(f"[ERROR] qid={qid} vector search failed: {repr(e)}")
        hits = []

    retrieved_chunks: List[Dict[str, Any]] = []

    for rank, pair in enumerate(hits, start=1):
        try:
            doc, distance = pair
        except Exception:
            continue

        meta = dict(doc.metadata or {})
        doc_id = str(meta.get("doc_id", "")).strip()
        chunk_id = str(meta.get("chunk_id", "")).strip()
        chunk_idx = meta.get("chunk_idx", meta.get("chunk_index", rank - 1))

        score = chroma_distance_to_score(distance)

        item = {
            "rank": rank,
            "chunk_idx": chunk_idx,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": doc.page_content or "",
            "score": float(score),
            "vector_distance": float(distance) if distance is not None else None,
            "meta": meta,

            # 兼容 Generator.build_citations 中可能读取的 critique 字段。
            # 注意：这里没有做证据筛选，这些字段只是占位，避免生成阶段读取时报错。
            "critique_relevance": "NotJudged",
            "critique_usefulness": "NotJudged",
            "critique_score": float(score),
            "critique_reason": "Ablation: evidence critique disabled; vector-retrieved chunk passed directly to generator.",
        }

        retrieved_chunks.append(item)

    doc_scores = aggregate_chunk_scores_to_doc_scores(
        chunks=retrieved_chunks,
        mode=doc_agg_mode,
    )

    return {
        "qid": qid,
        "query": query,
        "doc_scores": doc_scores,
        "generation_chunks": retrieved_chunks,
        "ranked_chunks": retrieved_chunks,
    }


def run_parallel_vector_only_pipeline(
    queries: Dict[str, str],
    vectordb,
    vector_top_k: int,
    doc_agg_mode: str,
    query_workers: int,
):
    retrieval_results_with_scores: Dict[str, Dict[str, float]] = {}
    generation_chunks: Dict[str, List[Dict[str, Any]]] = {}
    ranked_chunks_map: Dict[str, List[Dict[str, Any]]] = {}

    query_type_stats = {
        "vector_only_no_query_optimization": 0,
    }

    with ThreadPoolExecutor(max_workers=query_workers) as executor:
        future_to_qid = {
            executor.submit(
                process_single_query_vector_only,
                qid,
                query,
                vectordb,
                vector_top_k,
                doc_agg_mode,
            ): qid
            for qid, query in queries.items()
        }

        for future in as_completed(future_to_qid):
            qid = future_to_qid[future]

            try:
                result = future.result()
            except Exception as e:
                print(f"[ERROR] qid={qid} failed: {repr(e)}")
                continue

            doc_scores = result.get("doc_scores") or {}
            chunks = result.get("generation_chunks") or []

            if doc_scores:
                retrieval_results_with_scores[qid] = doc_scores

            if chunks:
                generation_chunks[qid] = chunks
                ranked_chunks_map[qid] = chunks

            query_type_stats["vector_only_no_query_optimization"] += 1

    return (
        retrieval_results_with_scores,
        generation_chunks,
        ranked_chunks_map,
        query_type_stats,
    )


# ============================================================
# 6. 主检索评测逻辑
# ============================================================

def run_ablation_vector_only_retrieval(
    dataset_name: str,
    dataset_path: str,
    max_queries: int,
    vector_top_k: int,
    doc_agg_mode: str,
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

    selected_queries, selected_qrels = select_queries(
        queries=queries,
        qrels=qrels,
        max_queries=max_queries,
    )

    print("\n========== Ablation Setting ==========")
    print("Experiment: vector_only_direct_to_generator")
    print("Removed modules:")
    print("  - QueryOptimizer / query rewrite / subquery decomposition")
    print("  - HybridRetriever / BM25 / RRF fusion")
    print("  - Reranker")
    print("  - ChunkCritiqueJudge / evidence filtering")
    print("Kept modules:")
    print("  - VectorStoreBuilder")
    print("  - Chroma vector similarity search")
    print("  - Generator, if --with_generation is enabled")
    print(f"Dataset: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Selected queries: {len(selected_queries)}")
    print(f"Vector top k: {vector_top_k}")
    print(f"Doc aggregation mode: {doc_agg_mode}")
    print(f"Query workers: {QUERY_WORKERS}")

    start_time = time.time()

    (
        retrieval_results_with_scores,
        generation_chunks,
        ranked_chunks_map,
        query_type_stats,
    ) = run_parallel_vector_only_pipeline(
        queries=selected_queries,
        vectordb=vectordb,
        vector_top_k=vector_top_k,
        doc_agg_mode=doc_agg_mode,
        query_workers=QUERY_WORKERS,
    )

    retrieval_elapsed = time.time() - start_time

    retrieval_metrics = evaluate_retrieval_metrics(
        selected_qrels,
        retrieval_results_with_scores,
    )

    # 因为本实验取消证据筛选，generation_chunks 就是向量检索结果本身。
    generation_doc_results = retrieval_results_with_scores
    generation_chunk_metrics = retrieval_metrics

    summary = {
        "task": "scifact_ablation_vector_only_direct_to_generator",
        "description": (
            "取消查询优化、混合检索、重排序、证据筛选；"
            "每个 query 直接做向量检索，并将向量检索 top-k chunks 直接给生成器。"
        ),
        "dataset": dataset_name,
        "dataset_path": dataset_path,
        "corpus_size": len(corpus),
        "query_size_total": len(queries),
        "query_size_eval": len(selected_queries),
        "retrieved_queries": len(retrieval_results_with_scores),
        "generation_chunk_queries": len(generation_chunks),
        "persist_dir": PERSIST_DIR,
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "vector_top_k": vector_top_k,
        "doc_agg_mode": doc_agg_mode,
        "query_workers": QUERY_WORKERS,
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
            "HybridRetriever.retrieve",
            "BM25",
            "RRF_fusion",
            "Reranker.rerank_texts",
            "ChunkCritiqueJudge.judge_chunks",
            "ChunkCritiqueJudge.select_generation_chunks",
        ],
        "kept_modules": [
            "VectorStoreBuilder.build_or_load",
            "Chroma.similarity_search_with_score",
            "Generator.run_optional",
        ],
        "query_type_stats": query_type_stats,
        "metrics_vector_docs": retrieval_metrics,
        "metrics_generation_chunks_aggregated_to_docs": generation_chunk_metrics,
    }

    details = {
        "queries": selected_queries,
        "qrels": selected_qrels,
        "vector_doc_results": retrieval_results_with_scores,
        "generation_doc_results": generation_doc_results,
        "generation_chunks": generation_chunks,
        "ranked_chunks_map": ranked_chunks_map,
        "query_type_stats": query_type_stats,
    }

    print("\n========== Retrieval Time ==========")
    print(f"向量检索总用时: {retrieval_elapsed:.2f}s")
    print(
        f"平均每条 query 用时: "
        f"{retrieval_elapsed / max(len(selected_queries), 1):.4f}s"
    )

    print("\n========== Query Routing Stats ==========")
    print(json.dumps(query_type_stats, ensure_ascii=False, indent=2))

    print("\n========== Retrieval Quality: Vector Search Docs ==========")
    print_metrics(retrieval_metrics)

    print("\n========== Generation Chunks -> Docs ==========")
    print("本实验取消证据筛选，因此 generation_chunks 就是向量检索结果本身。")
    print_metrics(generation_chunk_metrics)

    return summary, details


# ============================================================
# 7. 可选：生成阶段
# ============================================================

def run_generation(
    queries: Dict[str, str],
    generation_chunks: Dict[str, Any],
    max_samples: int,
    max_concurrency: int,
):
    if Generator is None:
        raise RuntimeError(
            "无法导入 backend/generator.py 中的 Generator，不能运行生成阶段。"
        )

    return asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=max_samples,
            max_concurrency=max_concurrency,
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
    summary_path = output_dir / "ablation_vector_only_retrieval_summary.json"
    details_path = output_dir / "ablation_vector_only_retrieval_details.jsonl"
    config_path = output_dir / "ablation_vector_only_runtime_config.json"

    save_json(summary_path, summary)

    queries = details["queries"]
    qrels = details["qrels"]
    vector_doc_results = details["vector_doc_results"]
    generation_doc_results = details["generation_doc_results"]
    generation_chunks = details["generation_chunks"]
    ranked_chunks_map = details["ranked_chunks_map"]

    rows = []

    for qid, query in queries.items():
        rows.append(
            {
                "qid": qid,
                "query": query,
                "qrels": qrels.get(qid, {}),
                "vector_doc_scores": vector_doc_results.get(qid, {}),
                "generation_doc_scores": generation_doc_results.get(qid, {}),
                "generation_chunks": generation_chunks.get(qid, []),
                "ranked_chunks": ranked_chunks_map.get(qid, []),
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
        "top_k_chunks_from_config": TOP_K_CHUNKS,
        "rerank_top_docs_from_config": RERANK_TOP_DOCS,
        "generation_top_n_from_config": GENERATION_TOP_N,
        "doc_agg_mode_from_config": DOC_AGG_MODE,
        "query_workers": QUERY_WORKERS,
        "max_samples": MAX_SAMPLES,
        "max_concurrency": MAX_CONCURRENCY,
        "ablation": {
            "name": "vector_only_direct_to_generator",
            "removed": [
                "QueryOptimizer",
                "HybridRetriever",
                "BM25",
                "Reranker",
                "ChunkCritiqueJudge",
            ],
            "kept": [
                "VectorStoreBuilder",
                "Chroma vector search",
                "Generator optional",
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
        "task": "scifact_ablation_vector_only_generation",
        "description": "向量检索结果直接给 Generator，不经过查询优化、混合检索、重排序、证据筛选。",
        "queries": len(queries),
        "generated_queries": len(gen_outputs),
        "generation_elapsed_sec": round(generation_elapsed, 3),
        "avg_generation_elapsed_per_query_sec": round(
            generation_elapsed / max(len(queries), 1),
            6,
        ),
    }

    gen_summary_path = output_dir / "ablation_vector_only_generation_summary.json"
    gen_details_path = output_dir / "ablation_vector_only_generation_details.jsonl"

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
        "--vector_top_k",
        type=int,
        default=TOP_K_CHUNKS,
        help="每个 query 直接向量检索的 chunk 数量，默认使用 config.TOP_K_CHUNKS",
    )

    parser.add_argument(
        "--doc_agg_mode",
        type=str,
        default=DOC_AGG_MODE,
        choices=["max", "sum", "mean"],
        help="chunk 分数聚合到 doc-level 的方式，默认使用 config.DOC_AGG_MODE",
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
        help="是否把向量检索结果直接给 Generator 生成答案",
    )

    parser.add_argument(
        "--generation_max_samples",
        type=int,
        default=0,
        help="生成阶段最多生成多少条；0 表示使用当前检索 query 数量和 config.MAX_SAMPLES 的较小值",
    )

    parser.add_argument(
        "--generation_max_concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="生成阶段并发数，默认使用 config.MAX_CONCURRENCY",
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
        output_dir = EVAL_DIR / f"scifact_ablation_vector_only_results_{timestamp}"

    ensure_dir(output_dir)

    print("\n========== Output Directory ==========")
    print(output_dir)

    summary, details = run_ablation_vector_only_retrieval(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        max_queries=args.max_queries,
        vector_top_k=args.vector_top_k,
        doc_agg_mode=args.doc_agg_mode,
    )

    save_retrieval_outputs(
        output_dir=output_dir,
        summary=summary,
        details=details,
    )

    if args.with_generation:
        print("\n========== Start Generation ==========")

        if args.generation_max_samples and args.generation_max_samples > 0:
            gen_max_samples = args.generation_max_samples
        else:
            gen_max_samples = min(len(details["queries"]), MAX_SAMPLES)

        gen_start = time.time()

        gen_outputs = run_generation(
            queries=details["queries"],
            generation_chunks=details["generation_chunks"],
            max_samples=gen_max_samples,
            max_concurrency=args.generation_max_concurrency,
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