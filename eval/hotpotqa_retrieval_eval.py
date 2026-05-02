# hotpotqa_retrieval_eval.py
# 放置位置：D:\code\rag\FSR\eval\hotpotqa_retrieval_eval.py
#
# 功能：
# 使用 BEIR HotpotQA 测试当前 RAG 系统的检索能力。
#
# 评测内容：
# 1. rerank_doc_results:
#    评测重排序后的 doc-level 检索结果
#
# 2. generation_chunks -> docs:
#    把最终选中的 generation_chunks 聚合回 doc_id 后再评测
#
# 输出：
# 1. hotpotqa_retrieval_summary.json
# 2. hotpotqa_retrieval_details.jsonl
#
# 推荐运行：
# conda activate all-in-rag
# cd D:\code\rag\FSR\eval
# $env:DASHSCOPE_API_KEY="你的 DashScope API Key"
# python hotpotqa_retrieval_eval.py --max_queries 20

import os
import sys
import json
import time
import argparse
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval


# =========================
# 1. 路径配置
# =========================

CURRENT_FILE = Path(__file__).resolve()

# D:\code\rag\FSR
BASE_DIR = CURRENT_FILE.parents[1]

# D:\code\rag\FSR\backend
BACKEND_DIR = BASE_DIR / "backend"

# D:\code\rag\FSR\eval
EVAL_DIR = BASE_DIR / "eval"

# D:\code\rag\FSR\beir_datasets\hotpotqa
DEFAULT_DATASET_PATH = BASE_DIR / "beir_datasets" / "hotpotqa"

# 把 backend 加入 Python 搜索路径，方便导入你的后端模块
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# =========================
# 2. 导入你项目中的后端模块
# =========================
#
# 这些模块来自你的 backend 目录：
# - config.py
# - vector_store.py
# - hybrid_retriever.py
# - reranker.py
# - pipeline_executor.py
#
# 你的原始 beir_eval_test.py 也是使用这些组件完成检索评测的。

from config import (  # noqa: E402
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL_NAME,
    PERSIST_DIR,
    QUERY_WORKERS,
    RERANK_TOP_DOCS,
    SUBQUERY_WORKERS,
)

from vector_store import VectorStoreBuilder  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402
from pipeline_executor import run_parallel_retrieval_pipeline  # noqa: E402


# =========================
# 3. 基础工具函数
# =========================

def save_json(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_metrics(metrics: Dict[str, Any], title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print("NDCG:")
    print(json.dumps(metrics.get("NDCG", {}), ensure_ascii=False, indent=2))
    print("MAP:")
    print(json.dumps(metrics.get("MAP", {}), ensure_ascii=False, indent=2))
    print("Recall:")
    print(json.dumps(metrics.get("Recall", {}), ensure_ascii=False, indent=2))
    print("Precision:")
    print(json.dumps(metrics.get("Precision", {}), ensure_ascii=False, indent=2))


# =========================
# 4. BEIR 数据加载
# =========================

def load_beir_hotpotqa(dataset_path: Path):
    """
    读取 BEIR 格式 HotpotQA。

    返回：
    corpus:
        {doc_id: {"title": ..., "text": ...}}

    queries:
        {qid: query_text}

    qrels:
        {qid: {doc_id: relevance_score}}
    """

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"数据集目录不存在：{dataset_path}\n"
            f"请先运行：python download_beir_hotpotqa.py"
        )

    print("=" * 80)
    print("加载 BEIR HotpotQA 数据集")
    print(f"dataset_path: {dataset_path}")
    print("=" * 80)

    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")

    print(f"corpus 文档数 : {len(corpus)}")
    print(f"queries 数量  : {len(queries)}")
    print(f"qrels 数量    : {len(qrels)}")

    return corpus, queries, qrels


def select_queries(
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    max_queries: int = 0,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    选择前 max_queries 条 query。
    max_queries=0 表示全量。
    """

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


# =========================
# 5. 构建检索组件
# =========================

def build_retrieval_components(
    dataset_name: str,
    corpus: Dict[str, Dict[str, str]],
):
    """
    构建或加载向量库，并初始化 HybridRetriever + Reranker。
    """

    collection_name = VectorStoreBuilder.make_collection_name(
        dataset_name,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME,
    )

    print("=" * 80)
    print("构建 / 加载向量库")
    print(f"dataset_name     : {dataset_name}")
    print(f"collection_name  : {collection_name}")
    print(f"persist_dir      : {PERSIST_DIR}")
    print(f"chunk_size       : {CHUNK_SIZE}")
    print(f"chunk_overlap    : {CHUNK_OVERLAP}")
    print(f"embed_model      : {EMBED_MODEL_NAME}")
    print("=" * 80)

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load(
        corpus=corpus,
        persist_dir=PERSIST_DIR,
        collection_name=collection_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
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

    print("=" * 80)
    print("检索组件初始化完成")
    print(f"chunk 数量       : {len(chunk_texts)}")
    print(f"alpha            : {ALPHA}")
    print(f"rerank_top_docs  : {RERANK_TOP_DOCS}")
    print(f"query_workers    : {QUERY_WORKERS}")
    print(f"subquery_workers : {SUBQUERY_WORKERS}")
    print("=" * 80)

    return retriever, reranker, reranker_lock


# =========================
# 6. 指标计算
# =========================

def evaluate_retrieval_metrics(
    qrels: Dict[str, Dict[str, int]],
    results_with_scores: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    使用 BEIR 的 EvaluateRetrieval 计算指标。

    输入格式：
    qrels:
        {
          qid: {
            doc_id: relevance_score
          }
        }

    results_with_scores:
        {
          qid: {
            doc_id: retrieval_score
          }
        }
    """

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


def aggregate_generation_chunks_to_doc_scores(
    generation_chunks: Dict[str, Any],
    agg_mode: str = "max",
    score_key: str = "critique_score",
    fallback_score_key: str = "score",
    top_k_docs: int = 0,
) -> Dict[str, Dict[str, float]]:
    """
    将 generation_chunks 聚合成 doc-level 检索结果。

    输入：
    generation_chunks:
        {
          qid: [
            {
              "doc_id": "...",
              "text": "...",
              "score": 0.8,
              "critique_score": 0.9
            }
          ]
        }

    输出：
        {
          qid: {
            doc_id: score
          }
        }

    agg_mode:
        max  : 同一 doc 多个 chunk 取最大分
        sum  : 同一 doc 多个 chunk 分数求和
        mean : 同一 doc 多个 chunk 分数平均

    score_key:
        默认优先使用 critique_score

    fallback_score_key:
        如果没有 critique_score，则使用 score
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
                raise ValueError(f"不支持的 agg_mode: {agg_mode}")

        if agg_mode == "mean":
            for doc_id in list(temp_scores.keys()):
                cnt = temp_counts.get(doc_id, 1)
                temp_scores[doc_id] = temp_scores[doc_id] / max(cnt, 1)

        sorted_scores = dict(
            sorted(
                temp_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        if top_k_docs and top_k_docs > 0:
            sorted_scores = dict(list(sorted_scores.items())[:top_k_docs])

        doc_results[qid] = sorted_scores

    return doc_results


# =========================
# 7. 主评测流程
# =========================

def run_hotpotqa_retrieval_eval(
    dataset_path: Path,
    max_queries: int,
    output_prefix: str,
    generation_doc_agg_mode: str,
    generation_score_key: str,
):
    dataset_name = "hotpotqa"

    # 1. 读取 BEIR HotpotQA
    corpus, queries, qrels = load_beir_hotpotqa(dataset_path)

    # 2. 选择部分 query
    queries, qrels = select_queries(
        queries=queries,
        qrels=qrels,
        max_queries=max_queries,
    )

    print("=" * 80)
    print("测试配置")
    print(f"dataset_name              : {dataset_name}")
    print(f"dataset_path              : {dataset_path}")
    print(f"selected_queries          : {len(queries)}")
    print(f"generation_doc_agg_mode   : {generation_doc_agg_mode}")
    print(f"generation_score_key      : {generation_score_key}")
    print("=" * 80)

    # 3. 构建检索器和重排序器
    retriever, reranker, reranker_lock = build_retrieval_components(
        dataset_name=dataset_name,
        corpus=corpus,
    )

    # 4. 运行你的并行检索 pipeline
    start_time = time.time()

    rerank_doc_results, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
        subquery_workers=SUBQUERY_WORKERS,
    )

    elapsed = time.time() - start_time

    # 5. 评测 rerank 后的 doc-level 检索结果
    rerank_metrics = evaluate_retrieval_metrics(
        qrels=qrels,
        results_with_scores=rerank_doc_results,
    )

    # 6. 评测 generation_chunks 聚合后的 doc-level 结果
    generation_doc_results = aggregate_generation_chunks_to_doc_scores(
        generation_chunks=generation_chunks,
        agg_mode=generation_doc_agg_mode,
        score_key=generation_score_key,
        fallback_score_key="score",
        top_k_docs=RERANK_TOP_DOCS,
    )

    generation_metrics = evaluate_retrieval_metrics(
        qrels=qrels,
        results_with_scores=generation_doc_results,
    )

    # 7. 汇总结果
    summary = {
        "dataset": dataset_name,
        "dataset_path": str(dataset_path),
        "queries": len(queries),
        "retrieved_queries": len(rerank_doc_results),
        "retrieval_elapsed_sec": round(elapsed, 3),
        "avg_time_per_query_sec": round(elapsed / max(len(queries), 1), 3),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL_NAME,
        "alpha": ALPHA,
        "rerank_top_docs": RERANK_TOP_DOCS,
        "query_workers": QUERY_WORKERS,
        "subquery_workers": SUBQUERY_WORKERS,
        "query_type_stats": query_type_stats,
        "generation_doc_agg_mode": generation_doc_agg_mode,
        "generation_score_key": generation_score_key,
        "metrics_rerank_docs": rerank_metrics,
        "metrics_generation_chunks_aggregated_to_docs": generation_metrics,
    }

    # 8. 保存明细
    detail_rows = []

    for qid, query_text in queries.items():
        detail_rows.append({
            "qid": qid,
            "query": query_text,
            "qrels": qrels.get(qid, {}),
            "rerank_doc_scores": rerank_doc_results.get(qid, {}),
            "generation_doc_scores": generation_doc_results.get(qid, {}),
            "generation_chunks": generation_chunks.get(qid, []),
            "ranked_chunks": ranked_chunks_map.get(qid, []),
        })

    output_prefix_path = Path(output_prefix)

    if not output_prefix_path.is_absolute():
        output_prefix_path = EVAL_DIR / output_prefix_path

    summary_path = output_prefix_path.with_name(output_prefix_path.name + "_summary.json")
    details_path = output_prefix_path.with_name(output_prefix_path.name + "_details.jsonl")

    save_json(summary_path, summary)
    save_jsonl(details_path, detail_rows)

    # 9. 打印结果
    print("\n" + "=" * 80)
    print("HotpotQA 检索评测完成")
    print("=" * 80)
    print(f"总耗时: {elapsed:.2f}s")
    print(f"平均每条 query 耗时: {elapsed / max(len(queries), 1):.3f}s")
    print("\nQuery Routing Stats:")
    print(query_type_stats)

    print_metrics(
        metrics=rerank_metrics,
        title="Retrieval Quality: Rerank Docs",
    )

    print_metrics(
        metrics=generation_metrics,
        title="Retrieval Quality: Generation Chunks Aggregated To Docs",
    )

    print("\n" + "=" * 80)
    print("结果文件")
    print(f"summary : {summary_path}")
    print(f"details : {details_path}")
    print("=" * 80)


# =========================
# 8. 命令行入口
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="使用 BEIR HotpotQA 测试当前 RAG 系统的检索能力"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help="BEIR HotpotQA 数据集路径，默认 D:\\code\\rag\\FSR\\beir_datasets\\hotpotqa",
    )

    parser.add_argument(
        "--max_queries",
        type=int,
        default=20,
        help="评测多少条 query。0 表示全量。建议先用 20 或 50 跑通。",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="hotpotqa_retrieval",
        help="输出文件名前缀。默认输出到 eval/hotpotqa_retrieval_summary.json 和 details.jsonl",
    )

    parser.add_argument(
        "--generation_doc_agg_mode",
        type=str,
        default="max",
        choices=["max", "sum", "mean"],
        help="generation_chunks 聚合到 doc_id 的方式",
    )

    parser.add_argument(
        "--generation_score_key",
        type=str,
        default="critique_score",
        help="generation_chunks 聚合时优先使用的分数字段，默认 critique_score，可改为 score",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()

    run_hotpotqa_retrieval_eval(
        dataset_path=dataset_path,
        max_queries=args.max_queries,
        output_prefix=args.output_prefix,
        generation_doc_agg_mode=args.generation_doc_agg_mode,
        generation_score_key=args.generation_score_key,
    )


if __name__ == "__main__":
    main()