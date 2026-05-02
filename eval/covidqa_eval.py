# covidqa_rrag_eval.py
# -*- coding: utf-8 -*-

"""
COVID-QA evaluation for RRAG.

目录约定：
- 数据集路径：D:/code/rag/FSR/data/covidqa
- 向量库路径：D:/code/rag/FSR/chroma_store
- 当前脚本路径：D:/code/rag/FSR/eval/covidqa_rrag_eval.py
- 结果保存路径：D:/code/rag/FSR/eval/results_xxx/

功能：
1. 自动读取 COVID-QA 数据集
2. 自动构造 corpus / queries / qrels / answers
3. 调用你现有 RRAG 后端流程：
   - VectorStoreBuilder
   - HybridRetriever
   - Reranker
   - run_parallel_retrieval_pipeline
   - 可选 Generator
4. 输出 json / jsonl 结果文件
"""

import os
import sys
import json
import time
import math
import argparse
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Any


# ============================================================
# 1. 固定你的项目路径
# ============================================================

FSR_ROOT = Path(r"D:\code\rag\FSR")
DATASET_PATH = FSR_ROOT / "data" / "covidqa"
CHROMA_STORE_PATH = FSR_ROOT / "chroma_store"
EVAL_DIR = FSR_ROOT / "eval"
BACKEND_DIR = FSR_ROOT / "backend"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# 2. 在导入 backend 模块前设置环境变量
#    注意：config.py 通常是在 import 时读取环境变量
# ============================================================

os.environ.setdefault("DATASET", "covidqa")
os.environ.setdefault("DATASET_PATH", str(DATASET_PATH))
os.environ.setdefault("PERSIST_DIR", str(CHROMA_STORE_PATH))

# 英文医学数据集，建议用英文 embedding / reranker
os.environ.setdefault("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
os.environ.setdefault("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# 检索参数，可按实验需要调整
os.environ.setdefault("CHUNK_SIZE", "400")
os.environ.setdefault("CHUNK_OVERLAP", "80")
os.environ.setdefault("TOP_K_CHUNKS", "50")
os.environ.setdefault("DENSE_CANDIDATE_K", "200")
os.environ.setdefault("BM25_CANDIDATE_K", "100")
os.environ.setdefault("RERANK_TOP_DOCS", "50")
os.environ.setdefault("GENERATION_TOP_N", "5")
os.environ.setdefault("GENERATION_MAX_PER_DOC", "2")
os.environ.setdefault("DOC_AGG_MODE", "top2_mean")

# 生成阶段并发不要太高，避免 API 报错
os.environ.setdefault("MAX_CONCURRENCY", "2")


# ============================================================
# 3. 导入你的后端模块
# ============================================================

from config import (  # noqa: E402
    DATASET,
    PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL_NAME,
    QUERY_WORKERS,
    SUBQUERY_WORKERS,
    RERANK_MODEL_NAME,
)

from vector_store import VectorStoreBuilder  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402
from evaluator import RetrievalEvaluator  # noqa: E402
from pipeline_executor import run_parallel_retrieval_pipeline  # noqa: E402


# Generator 只在 --with_generation 时使用
try:
    from generator import Generator  # noqa: E402
except Exception:
    Generator = None


# ============================================================
# 4. 工具函数
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def now_result_dir() -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = EVAL_DIR / f"covidqa_eval_{ts}"
    ensure_dir(out_dir)
    return out_dir


def write_json(path: Path, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_doc_id(text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
    return f"doc_{h}"


def normalize_answer(answers_field) -> str:
    """
    兼容 HuggingFace covid_qa_deepset 常见 answer 格式：
    answers = {
        "text": [...],
        "answer_start": [...]
    }
    """
    if answers_field is None:
        return ""

    if isinstance(answers_field, str):
        return answers_field.strip()

    if isinstance(answers_field, list):
        if not answers_field:
            return ""
        first = answers_field[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            return normalize_answer(first)

    if isinstance(answers_field, dict):
        text = answers_field.get("text", "")
        if isinstance(text, list):
            return str(text[0]).strip() if text else ""
        if isinstance(text, str):
            return text.strip()

        # 有些数据可能是 {"answer": "..."}
        ans = answers_field.get("answer", "")
        if isinstance(ans, str):
            return ans.strip()

    return str(answers_field).strip()


# ============================================================
# 5. 读取 COVID-QA 数据集
# ============================================================

def load_beir_format(data_dir: Path):
    """
    如果你的 covidqa 已经是 BEIR 格式：
    data/covidqa/
      corpus.jsonl
      queries.jsonl
      qrels/test.tsv
      answers.json 可选
    则直接读取。
    """
    corpus_file = data_dir / "corpus.jsonl"
    queries_file = data_dir / "queries.jsonl"
    qrels_file = data_dir / "qrels" / "test.tsv"
    answers_file = data_dir / "answers.json"

    if not corpus_file.exists() or not queries_file.exists() or not qrels_file.exists():
        return None

    corpus = {}
    queries = {}
    qrels = {}

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            doc_id = str(obj.get("_id"))
            corpus[doc_id] = {
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
            }

    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj.get("_id"))
            queries[qid] = obj.get("text", "")

    with open(qrels_file, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], int(float(parts[2]))
            qrels.setdefault(str(qid), {})[str(doc_id)] = score

    answers = {}
    if answers_file.exists():
        with open(answers_file, "r", encoding="utf-8") as f:
            answers = json.load(f)

    return corpus, queries, qrels, answers


def iter_json_or_jsonl_files(data_dir: Path):
    for pattern in ["*.jsonl", "*.json"]:
        for path in data_dir.glob(pattern):
            # 排除 BEIR 文件
            if path.name in {"corpus.jsonl", "queries.jsonl", "answers.json"}:
                continue
            yield path


def load_raw_json_dataset(data_dir: Path):
    """
    兼容常见 JSON / JSONL COVID-QA 格式。
    期望字段大致包括：
    - question
    - context
    - answers / answer
    - document_id 可选
    """
    rows = []

    for path in iter_json_or_jsonl_files(data_dir):
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
        elif path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    rows.extend(obj)
                elif isinstance(obj, dict):
                    # HuggingFace 导出的 dict 可能有 data/train 等字段
                    if "data" in obj and isinstance(obj["data"], list):
                        rows.extend(obj["data"])
                    elif "train" in obj and isinstance(obj["train"], list):
                        rows.extend(obj["train"])
                    else:
                        # 单条样本
                        rows.append(obj)

    if not rows:
        return None

    return build_eval_data_from_rows(rows)


def load_hf_dataset_from_disk(data_dir: Path):
    """
    如果你是用 datasets.save_to_disk 保存的，则尝试用 load_from_disk 读取。
    """
    try:
        from datasets import load_from_disk
    except Exception:
        return None

    try:
        ds = load_from_disk(str(data_dir))
    except Exception:
        return None

    rows = []

    # DatasetDict
    if hasattr(ds, "keys"):
        split_name = "test" if "test" in ds else ("train" if "train" in ds else list(ds.keys())[0])
        for item in ds[split_name]:
            rows.append(dict(item))
    else:
        for item in ds:
            rows.append(dict(item))

    if not rows:
        return None

    return build_eval_data_from_rows(rows)


def load_parquet_dataset(data_dir: Path):
    """
    如果目录下是 parquet 文件，则读取。
    """
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        return None

    try:
        import pandas as pd
    except Exception:
        print("[WARN] 检测到 parquet 文件，但未安装 pandas，跳过 parquet 读取。")
        return None

    rows = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        rows.extend(df.to_dict(orient="records"))

    if not rows:
        return None

    return build_eval_data_from_rows(rows)


def build_eval_data_from_rows(rows: List[Dict]):
    """
    把 COVID-QA 原始样本转成：
    corpus: {doc_id: {"title": ..., "text": ...}}
    queries: {qid: question}
    qrels: {qid: {doc_id: 1}}
    answers: {qid: answer}
    """
    corpus = {}
    queries = {}
    qrels = {}
    answers = {}

    for i, item in enumerate(rows):
        question = str(item.get("question", "") or "").strip()

        # 常见字段：context
        context = str(
            item.get("context", "")
            or item.get("document", "")
            or item.get("passage", "")
            or item.get("text", "")
            or ""
        ).strip()

        answer = normalize_answer(
            item.get("answers", None)
            if "answers" in item
            else item.get("answer", None)
        )

        if not question or not context:
            continue

        doc_id = stable_doc_id(context)

        title = str(
            item.get("title", "")
            or item.get("document_id", "")
            or item.get("doc_id", "")
            or f"COVID-QA Document {doc_id}"
        )

        if doc_id not in corpus:
            corpus[doc_id] = {
                "title": title,
                "text": context,
            }

        qid = f"q_{i}"
        queries[qid] = question
        qrels[qid] = {doc_id: 1}
        answers[qid] = answer

    return corpus, queries, qrels, answers


def load_covidqa_dataset(data_dir: Path):
    """
    按优先级读取：
    1. BEIR 格式
    2. datasets.save_to_disk 格式
    3. parquet
    4. json/jsonl
    """
    print(f"[DATA] Loading COVID-QA from: {data_dir}")

    if not data_dir.exists():
        raise FileNotFoundError(f"数据集路径不存在：{data_dir}")

    loaders = [
        load_beir_format,
        load_hf_dataset_from_disk,
        load_parquet_dataset,
        load_raw_json_dataset,
    ]

    for loader in loaders:
        result = loader(data_dir)
        if result is not None:
            corpus, queries, qrels, answers = result
            print(f"[DATA] corpus={len(corpus)}, queries={len(queries)}, qrels={len(qrels)}")
            return corpus, queries, qrels, answers

    raise RuntimeError(
        f"无法识别 COVID-QA 数据格式：{data_dir}\n"
        f"建议目录中至少包含 BEIR 格式 corpus.jsonl / queries.jsonl / qrels/test.tsv，"
        f"或者包含原始 json/jsonl/parquet 文件。"
    )


# ============================================================
# 6. 检索指标计算
# ============================================================

def sorted_docs(result_scores: Dict[str, float]) -> List[str]:
    return [
        doc_id
        for doc_id, _ in sorted(
            result_scores.items(),
            key=lambda x: float(x[1]),
            reverse=True,
        )
    ]


def dcg_at_k(rels: List[int], k: int) -> float:
    rels = rels[:k]
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(pred_docs: List[str], gold_docs: Dict[str, int], k: int) -> float:
    rels = [int(gold_docs.get(doc_id, 0)) for doc_id in pred_docs[:k]]
    ideal_rels = sorted([int(v) for v in gold_docs.values()], reverse=True)
    ideal = dcg_at_k(ideal_rels, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(rels, k) / ideal


def precision_at_k(pred_docs: List[str], gold_docs: Dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    pred_k = pred_docs[:k]
    hit = sum(1 for doc_id in pred_k if gold_docs.get(doc_id, 0) > 0)
    return hit / k


def recall_at_k(pred_docs: List[str], gold_docs: Dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in gold_docs.values() if int(v) > 0)
    if total_rel == 0:
        return 0.0
    pred_k = pred_docs[:k]
    hit = sum(1 for doc_id in pred_k if gold_docs.get(doc_id, 0) > 0)
    return hit / total_rel


def map_at_k(pred_docs: List[str], gold_docs: Dict[str, int], k: int) -> float:
    total_rel = sum(1 for v in gold_docs.values() if int(v) > 0)
    if total_rel == 0:
        return 0.0

    hits = 0
    precisions = []

    for i, doc_id in enumerate(pred_docs[:k], start=1):
        if gold_docs.get(doc_id, 0) > 0:
            hits += 1
            precisions.append(hits / i)

    if not precisions:
        return 0.0

    return sum(precisions) / min(total_rel, k)


def evaluate_retrieval_results(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values=(1, 3, 5, 10, 20, 50, 100),
):
    metrics = {
        "NDCG": {},
        "MAP": {},
        "Recall": {},
        "Precision": {},
    }

    valid_qids = [qid for qid in qrels.keys() if qid in results]

    for k in k_values:
        ndcgs = []
        maps = []
        recalls = []
        precisions = []

        for qid in valid_qids:
            pred = sorted_docs(results.get(qid, {}))
            gold = qrels.get(qid, {})

            ndcgs.append(ndcg_at_k(pred, gold, k))
            maps.append(map_at_k(pred, gold, k))
            recalls.append(recall_at_k(pred, gold, k))
            precisions.append(precision_at_k(pred, gold, k))

        denom = max(1, len(valid_qids))
        metrics["NDCG"][f"NDCG@{k}"] = round(sum(ndcgs) / denom, 5)
        metrics["MAP"][f"MAP@{k}"] = round(sum(maps) / denom, 5)
        metrics["Recall"][f"Recall@{k}"] = round(sum(recalls) / denom, 5)
        metrics["Precision"][f"Precision@{k}"] = round(sum(precisions) / denom, 5)

    metrics["evaluated_queries"] = len(valid_qids)
    return metrics


def generation_chunks_to_doc_scores(generation_chunks_map: Dict[str, List[Dict]]):
    """
    把 generation_chunks 聚合成文档级结果，用于评估：
    generation chunk 筛选后是否还保留了正确文档。
    """
    results = {}

    for qid, chunks in generation_chunks_map.items():
        ranked = []
        for item in chunks:
            doc_id = str(item.get("doc_id", ""))
            text = item.get("text", "")
            score = float(
                item.get("critique_score", 0.0)
                or item.get("score", 0.0)
                or 0.0
            )
            if doc_id:
                ranked.append((doc_id, text, score))

        doc_scores = RetrievalEvaluator.aggregate_doc_scores(
            ranked,
            mode=os.environ.get("DOC_AGG_MODE", "top2_mean"),
        )
        if doc_scores:
            results[qid] = doc_scores

    return results


# ============================================================
# 7. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_queries", type=int, default=200, help="最多评测多少条；0 表示全量")
    parser.add_argument("--with_generation", action="store_true", help="是否执行生成阶段")
    parser.add_argument("--result_name", type=str, default="", help="结果文件夹名，可不填")
    args = parser.parse_args()

    if args.result_name:
        result_dir = EVAL_DIR / args.result_name
        ensure_dir(result_dir)
    else:
        result_dir = now_result_dir()

    print("=" * 80)
    print("COVID-QA RRAG Evaluation")
    print("=" * 80)
    print(f"[PATH] FSR_ROOT        = {FSR_ROOT}")
    print(f"[PATH] DATASET_PATH    = {DATASET_PATH}")
    print(f"[PATH] CHROMA_STORE    = {CHROMA_STORE_PATH}")
    print(f"[PATH] BACKEND_DIR     = {BACKEND_DIR}")
    print(f"[PATH] RESULT_DIR      = {result_dir}")
    print("=" * 80)

    # 1. 读取数据
    corpus, queries, qrels, answers = load_covidqa_dataset(DATASET_PATH)

    if args.max_queries and args.max_queries > 0:
        selected_items = list(queries.items())[:args.max_queries]
        selected_qids = {qid for qid, _ in selected_items}
        queries = dict(selected_items)
        qrels = {qid: qrels[qid] for qid in selected_qids if qid in qrels}
        answers = {qid: answers.get(qid, "") for qid in selected_qids}

    # 2. 保存本次实际评测集
    write_json(result_dir / "covidqa_eval_dataset_info.json", {
        "dataset": "covidqa",
        "dataset_path": str(DATASET_PATH),
        "corpus_size": len(corpus),
        "query_size": len(queries),
        "qrels_size": len(qrels),
        "max_queries": args.max_queries,
        "with_generation": args.with_generation,
    })

    write_json(result_dir / "covidqa_answers.json", answers)

    # 3. 建库或加载向量库
    collection_name = VectorStoreBuilder.make_collection_name(
        DATASET,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME,
    )

    print(f"[VectorStore] persist_dir     = {PERSIST_DIR}")
    print(f"[VectorStore] collection_name = {collection_name}")
    print(f"[VectorStore] chunk_size      = {CHUNK_SIZE}")
    print(f"[VectorStore] chunk_overlap   = {CHUNK_OVERLAP}")
    print(f"[VectorStore] embed_model     = {EMBED_MODEL_NAME}")

    t0 = time.time()

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load(
        corpus=corpus,
        persist_dir=str(CHROMA_STORE_PATH),
        collection_name=collection_name,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    build_or_load_time = time.time() - t0

    print(f"[VectorStore] chunks = {len(chunk_texts)}")
    print(f"[VectorStore] build/load time = {build_or_load_time:.2f}s")

    # 4. 初始化检索器和重排序器
    retriever = HybridRetriever(
        vectordb=vectordb,
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_ids=chunk_ids,
        chunk_metas=chunk_metas,
    )

    reranker = Reranker()
    reranker_lock = threading.Lock()

    # 5. 跑检索 + rerank + generation chunk 筛选
    print("[Eval] Running retrieval pipeline...")

    retrieval_start = time.time()

    (
        rerank_doc_results,
        generation_chunks_map,
        ranked_chunks_map,
        query_type_stats,
    ) = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=QUERY_WORKERS,
        subquery_workers=SUBQUERY_WORKERS,
    )

    retrieval_elapsed = time.time() - retrieval_start

    print(f"[Eval] retrieval elapsed = {retrieval_elapsed:.2f}s")
    print(f"[Eval] query_type_stats = {query_type_stats}")

    # 6. 检索指标：rerank docs
    rerank_metrics = evaluate_retrieval_results(
        results=rerank_doc_results,
        qrels=qrels,
    )

    # 7. 检索指标：generation chunks 聚合回 doc
    gen_doc_results = generation_chunks_to_doc_scores(generation_chunks_map)
    generation_chunk_metrics = evaluate_retrieval_results(
        results=gen_doc_results,
        qrels=qrels,
    )

    # 8. 保存检索 summary
    retrieval_summary = {
        "task": "covidqa",
        "dataset_path": str(DATASET_PATH),
        "chroma_store_path": str(CHROMA_STORE_PATH),
        "result_dir": str(result_dir),
        "collection_name": collection_name,
        "config": {
            "embed_model": EMBED_MODEL_NAME,
            "rerank_model": RERANK_MODEL_NAME,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "retrieval_workers": QUERY_WORKERS,
            "subquery_workers": SUBQUERY_WORKERS,
            "doc_agg_mode": os.environ.get("DOC_AGG_MODE", "top2_mean"),
        },
        "data_stats": {
            "corpus_size": len(corpus),
            "query_size": len(queries),
            "qrels_size": len(qrels),
            "chunk_size": len(chunk_texts),
        },
        "time": {
            "vectorstore_build_or_load_sec": round(build_or_load_time, 4),
            "retrieval_elapsed_sec": round(retrieval_elapsed, 4),
            "avg_retrieval_sec_per_query": round(
                retrieval_elapsed / max(1, len(queries)), 4
            ),
        },
        "query_type_stats": query_type_stats,
        "rerank_doc_metrics": rerank_metrics,
        "generation_chunk_doc_metrics": generation_chunk_metrics,
    }

    write_json(result_dir / "covidqa_retrieval_summary.json", retrieval_summary)

    # 9. 保存检索 details
    retrieval_details = []

    for qid, query in queries.items():
        gold_docs = qrels.get(qid, {})
        rerank_scores = rerank_doc_results.get(qid, {})
        gen_doc_scores = gen_doc_results.get(qid, {})

        rerank_top_docs = sorted(
            rerank_scores.items(),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:20]

        gen_top_docs = sorted(
            gen_doc_scores.items(),
            key=lambda x: float(x[1]),
            reverse=True,
        )[:20]

        retrieval_details.append({
            "qid": qid,
            "query": query,
            "gold_answer": answers.get(qid, ""),
            "gold_docs": gold_docs,
            "rerank_top_docs": [
                {"doc_id": doc_id, "score": float(score)}
                for doc_id, score in rerank_top_docs
            ],
            "generation_chunk_top_docs": [
                {"doc_id": doc_id, "score": float(score)}
                for doc_id, score in gen_top_docs
            ],
            "ranked_chunks": ranked_chunks_map.get(qid, []),
            "generation_chunks": generation_chunks_map.get(qid, []),
        })

    write_jsonl(result_dir / "covidqa_retrieval_details.jsonl", retrieval_details)

    # 10. 可选：执行生成阶段
    generation_summary = None

    if args.with_generation:
        if Generator is None:
            print("[Generation][WARN] 无法导入 Generator，跳过生成阶段。")
        else:
            print("[Generation] Running Generator...")

            gen_start = time.time()

            import asyncio

            max_samples = len(queries)

            gen_outputs = asyncio.run(
                Generator.run(
                    queries=queries,
                    generation_chunks=generation_chunks_map,
                    max_samples=max_samples,
                    max_concurrency=int(os.environ.get("MAX_CONCURRENCY", "2")),
                )
            )

            gen_elapsed = time.time() - gen_start

            generation_rows = []

            for qid, output in gen_outputs.items():
                generation_rows.append({
                    "qid": qid,
                    "query": queries.get(qid, ""),
                    "gold_answer": answers.get(qid, ""),
                    "gold_docs": qrels.get(qid, {}),
                    "answer": output.get("answer", ""),
                    "raw_text": output.get("raw_text", ""),
                    "reflections": output.get("reflections", {}),
                    "citations": output.get("citations", []),
                    "candidates": output.get("candidates", []),
                    "generation_risk": output.get("generation_risk", {}),
                    "generation_chunks": generation_chunks_map.get(qid, []),
                })

            generation_summary = {
                "task": "covidqa_generation",
                "query_size": len(queries),
                "generated_size": len(gen_outputs),
                "generation_elapsed_sec": round(gen_elapsed, 4),
                "avg_generation_sec_per_query": round(
                    gen_elapsed / max(1, len(gen_outputs)), 4
                ),
            }

            write_json(result_dir / "covidqa_generation_summary.json", generation_summary)
            write_jsonl(result_dir / "covidqa_generation_details.jsonl", generation_rows)

    # 11. 最终索引文件
    final_index = {
        "result_dir": str(result_dir),
        "files": {
            "dataset_info": "covidqa_eval_dataset_info.json",
            "answers": "covidqa_answers.json",
            "retrieval_summary": "covidqa_retrieval_summary.json",
            "retrieval_details": "covidqa_retrieval_details.jsonl",
        }
    }

    if generation_summary is not None:
        final_index["files"]["generation_summary"] = "covidqa_generation_summary.json"
        final_index["files"]["generation_details"] = "covidqa_generation_details.jsonl"

    write_json(result_dir / "index.json", final_index)

    print("=" * 80)
    print("[DONE] COVID-QA evaluation finished.")
    print(f"[DONE] Results saved to: {result_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()