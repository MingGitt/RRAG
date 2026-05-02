# hotpotqa_distractor_no_query_opt_eval.py
# 放置位置：D:\code\rag\FSR\eval\hotpotqa_distractor_no_query_opt_eval.py
#
# 消融实验：
# 去掉查询优化 / 查询改写 / 查询类型判断。
# 每个 HotpotQA 问题的原始 question 直接进入检索。
#
# 保留：
# 1. HotpotQA Distractor 全局去重 collection
# 2. HybridRetriever 混合检索
# 3. Reranker 重排序
# 4. Reranker.select_generation_chunks
# 5. Generator.run_single 多候选生成与答案选择
# 6. 检索指标、generation chunk 指标、Answer EM/F1
#
# 路径约定：
# 数据集：D:\code\rag\FSR\data\hotpotqa_distractor
# 向量库：D:\code\rag\FSR\chroma_store
# 测试代码：D:\code\rag\FSR\eval
# 测试结果：D:\code\rag\FSR\eval\hotpotqa_distractor_ablation_outputs\...

import os
import re
import sys
import json
import time
import shutil
import hashlib
import argparse
import asyncio
import string
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Tuple, Set


# =========================================================
# 路径设置
# =========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "hotpotqa_distractor")
DEFAULT_CHROMA_ROOT = os.path.join(PROJECT_ROOT, "chroma_store")
DEFAULT_PERSIST_DIR = os.path.join(
    DEFAULT_CHROMA_ROOT,
    "hotpotqa_distractor_global_no_query_opt",
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    CURRENT_DIR,
    "hotpotqa_distractor_ablation_outputs",
)


# =========================================================
# 导入后端已有接口
# =========================================================

from config import (  # noqa: E402
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ALPHA,
    DENSE_CANDIDATE_K,
    BM25_CANDIDATE_K,
    TOP_K_CHUNKS,
    RERANK_TOP_DOCS,
    GENERATION_TOP_N,
    GENERATION_MAX_PER_DOC,
)

from vector_store import VectorStoreBuilder  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402

try:
    from generator import Generator  # noqa: E402
except Exception as e:
    Generator = None
    GENERATOR_IMPORT_ERROR = repr(e)


# =========================================================
# 通用工具
# =========================================================

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def md5_text(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def save_json(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, obj: Any):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clear_file(path: str):
    if os.path.exists(path):
        os.remove(path)


def normalize_answer(s: str) -> str:
    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(s or ""))))


def answer_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def avg(rows: List[Dict[str, Any]], key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in rows if key in r]
    return sum(vals) / len(vals) if vals else 0.0


def group_avg(rows: List[Dict[str, Any]], key: str, group_key: str):
    groups: Dict[str, List[float]] = {}

    for row in rows:
        group = row.get(group_key, "") or "unknown"

        if key not in row:
            continue

        groups.setdefault(group, []).append(float(row[key]))

    return {
        group: sum(vals) / len(vals) if vals else 0.0
        for group, vals in groups.items()
    }


# =========================================================
# HotpotQA 本地数据读取
# =========================================================

def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []

    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            return obj

        if isinstance(obj, dict):
            for key in ["data", "examples", "rows", "validation", "train", "test"]:
                if key in obj and isinstance(obj[key], list):
                    return obj[key]

            if all(isinstance(v, dict) for v in obj.values()):
                return list(obj.values())

        raise RuntimeError(f"无法识别 JSON 文件结构：{path}")

    raise RuntimeError(f"不支持的文件格式：{path}")


def find_hotpotqa_files(data_dir: str, split: str) -> List[str]:
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    candidates = []

    for root, _, files in os.walk(data_dir):
        for name in files:
            lower = name.lower()
            if not lower.endswith((".json", ".jsonl")):
                continue

            full_path = os.path.join(root, name)

            if split.lower() in lower:
                candidates.append(full_path)
            elif split.lower() == "validation" and ("dev" in lower or "valid" in lower):
                candidates.append(full_path)
            elif split.lower() == "train" and "train" in lower:
                candidates.append(full_path)

    if candidates:
        return sorted(candidates)

    all_files = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            lower = name.lower()
            if lower.endswith((".json", ".jsonl")):
                all_files.append(os.path.join(root, name))

    return sorted(all_files)


def load_hotpotqa_local(data_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    兼容：
    1. HuggingFace datasets.save_to_disk 保存的 Dataset / DatasetDict
    2. 原始 HotpotQA json
    3. jsonl
    """
    try:
        from datasets import load_from_disk, Dataset, DatasetDict

        ds = load_from_disk(data_dir)

        if isinstance(ds, DatasetDict):
            if split in ds:
                return [dict(x) for x in ds[split]]

            if split == "validation":
                for key in ["validation", "dev", "test", "train"]:
                    if key in ds:
                        print(f"[Data] split={split} 不存在，自动使用 split={key}")
                        return [dict(x) for x in ds[key]]

            first_key = list(ds.keys())[0]
            print(f"[Data] split={split} 不存在，自动使用 split={first_key}")
            return [dict(x) for x in ds[first_key]]

        if isinstance(ds, Dataset):
            return [dict(x) for x in ds]

    except Exception:
        pass

    files = find_hotpotqa_files(data_dir, split=split)

    if not files:
        raise RuntimeError(
            f"没有在 {data_dir} 中找到 HotpotQA 数据文件。"
            f"请确认目录里有 .json/.jsonl，或 HuggingFace save_to_disk 格式数据。"
        )

    print("[Data] 使用以下本地文件：")
    for f in files:
        print(f"  - {f}")

    rows = []
    for f in files:
        rows.extend(load_json_or_jsonl(f))

    return rows


def extract_context_pairs(example: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    HF 格式：
    context = {
        "title": [...],
        "sentences": [[...], [...]]
    }

    原始格式：
    context = [
        [title, [sent1, sent2, ...]],
        [title, [sent1, sent2, ...]]
    ]
    """
    context = example.get("context", None)
    pairs = []

    if context is None:
        return pairs

    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        for title, sentences in zip(titles, sentences_list):
            title = str(title or "").strip()

            if isinstance(sentences, list):
                para_text = " ".join([str(s) for s in sentences if s is not None])
            else:
                para_text = str(sentences or "")

            para_text = normalize_space(para_text)

            if title and para_text:
                pairs.append((title, para_text))

        return pairs

    if isinstance(context, list):
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            title = str(item[0] or "").strip()
            sentences = item[1]

            if isinstance(sentences, list):
                para_text = " ".join([str(s) for s in sentences if s is not None])
            else:
                para_text = str(sentences or "")

            para_text = normalize_space(para_text)

            if title and para_text:
                pairs.append((title, para_text))

        return pairs

    return pairs


def extract_supporting_titles(example: Dict[str, Any]) -> Set[str]:
    """
    HF 格式：
    supporting_facts = {
        "title": [...],
        "sent_id": [...]
    }

    原始格式：
    supporting_facts = [
        [title, sent_id],
        [title, sent_id]
    ]
    """
    sf = example.get("supporting_facts", None)
    titles = set()

    if sf is None:
        return titles

    if isinstance(sf, dict):
        for t in sf.get("title", []):
            if t is not None:
                t = str(t).strip()
                if t:
                    titles.add(t)
        return titles

    if isinstance(sf, list):
        for item in sf:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                t = str(item[0] or "").strip()
                if t:
                    titles.add(t)
            elif isinstance(item, dict):
                t = str(item.get("title", "") or "").strip()
                if t:
                    titles.add(t)

    return titles


def get_example_id(example: Dict[str, Any], idx: int) -> str:
    for key in ["_id", "id", "qid", "question_id"]:
        if key in example and example[key] is not None:
            return str(example[key])
    return f"hotpotqa_q_{idx}"


def get_question(example: Dict[str, Any]) -> str:
    for key in ["question", "query"]:
        if key in example and example[key] is not None:
            return str(example[key]).strip()
    return ""


def get_answer(example: Dict[str, Any]) -> str:
    for key in ["answer", "answers", "reference", "gold_answer"]:
        if key not in example:
            continue

        value = example[key]

        if isinstance(value, str):
            return value.strip()

        if isinstance(value, list):
            if not value:
                return ""
            first = value[0]
            if isinstance(first, str):
                return first.strip()
            if isinstance(first, dict):
                for k in ["text", "answer"]:
                    if k in first:
                        return str(first[k]).strip()

        if isinstance(value, dict):
            if "text" in value:
                text = value["text"]
                if isinstance(text, list):
                    return str(text[0]).strip() if text else ""
                return str(text).strip()

    return ""


def make_doc_id(title: str, para_text: str) -> str:
    """
    全局去重粒度：title + paragraph text。
    不把 qid 放进 doc_id。
    """
    key = normalize_space(title) + "\n" + normalize_space(para_text)
    return "hotpot_doc_" + md5_text(key)[:16]


def build_hotpotqa_global_chunks(
    data_dir: str,
    split: str,
    max_samples: int,
    chunk_size: int,
    chunk_overlap: int,
):
    rows = load_hotpotqa_local(data_dir=data_dir, split=split)

    if max_samples and max_samples > 0:
        rows = rows[:max_samples]

    doc_map: Dict[str, Dict[str, Any]] = {}
    qrels: Dict[str, List[str]] = {}
    queries: Dict[str, str] = {}
    answers: Dict[str, str] = {}
    sample_meta: Dict[str, Dict[str, Any]] = {}

    skipped = 0
    no_gold = 0

    for idx, ex in enumerate(rows):
        qid = get_example_id(ex, idx)
        question = get_question(ex)
        answer = get_answer(ex)

        if not question:
            skipped += 1
            continue

        context_pairs = extract_context_pairs(ex)
        supporting_titles = extract_supporting_titles(ex)

        if not context_pairs:
            skipped += 1
            continue

        queries[qid] = question
        answers[qid] = answer

        sample_meta[qid] = {
            "type": ex.get("type", ""),
            "level": ex.get("level", ""),
            "supporting_titles": sorted(list(supporting_titles)),
        }

        gold_doc_ids = set()

        for title, para_text in context_pairs:
            doc_id = make_doc_id(title, para_text)

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "text": para_text,
                }

            if title in supporting_titles:
                gold_doc_ids.add(doc_id)

        if not gold_doc_ids:
            no_gold += 1

        qrels[qid] = sorted(list(gold_doc_ids))

    chunks: List[Dict[str, Any]] = []

    for doc_id, doc in doc_map.items():
        title = doc["title"]
        text = doc["text"]
        full_text = f"{title}\n{text}".strip()

        parts = VectorStoreBuilder.split_by_structure_then_length(
            full_text,
            max_len=chunk_size,
            overlap=chunk_overlap,
        )

        for chunk_idx, part in enumerate(parts):
            chunks.append({
                "chunk_id": f"{doc_id}::chunk_{chunk_idx}",
                "doc_id": doc_id,
                "text": part,
                "chunk_index": chunk_idx,
                "source_path": data_dir,
                "file_type": "hotpotqa_distractor",
                "title": title,
                "page_no": 0,
            })

    print("\n========== HotpotQA Distractor Data ==========")
    print(f"data_dir          = {data_dir}")
    print(f"split             = {split}")
    print(f"samples           = {len(queries)}")
    print(f"unique_paragraphs = {len(doc_map)}")
    print(f"chunks            = {len(chunks)}")
    print(f"skipped           = {skipped}")
    print(f"no_gold_qrels     = {no_gold}")

    return chunks, queries, answers, qrels, doc_map, sample_meta


# =========================================================
# 指标
# =========================================================

def compute_retrieval_metrics(
    retrieved_doc_ids: List[str],
    gold_doc_ids: Set[str],
    ks=(1, 2, 3, 5, 10, 20, 50),
):
    metrics = {}

    for k in ks:
        topk = retrieved_doc_ids[:k]
        hit_count = len(set(topk) & gold_doc_ids)

        metrics[f"hit@{k}"] = 1.0 if hit_count > 0 else 0.0
        metrics[f"recall@{k}"] = (
            hit_count / len(gold_doc_ids) if gold_doc_ids else 0.0
        )

    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            mrr = 1.0 / rank
            break

    metrics["mrr"] = mrr
    return metrics


# =========================================================
# 生成阶段
# =========================================================

async def run_generator_once(qid: str, query: str, generation_chunks: List[Dict[str, Any]]):
    if Generator is None:
        raise RuntimeError(
            f"generator.py 导入失败，无法执行生成。错误：{GENERATOR_IMPORT_ERROR}"
        )

    result = await Generator.run_single(
        qid=qid,
        query=query,
        chunks=generation_chunks,
    )

    if isinstance(result, tuple) and len(result) == 2:
        return result[1]

    if isinstance(result, dict):
        return result

    return {"answer": str(result)}


# =========================================================
# 主流程：no query optimization
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="validation")

    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="用于构建全局 collection 的样本数。0 表示使用全部数据。",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=0,
        help="实际评测的问题数。0 表示评测全部问题。",
    )

    parser.add_argument("--chroma_root", type=str, default=DEFAULT_CHROMA_ROOT)
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR)
    parser.add_argument(
        "--collection_name",
        type=str,
        default="hotpotqa_distractor_no_query_opt",
    )

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)

    parser.add_argument("--top_k_chunks", type=int, default=TOP_K_CHUNKS)
    parser.add_argument("--dense_k", type=int, default=DENSE_CANDIDATE_K)
    parser.add_argument("--bm25_k", type=int, default=BM25_CANDIDATE_K)
    parser.add_argument("--alpha", type=float, default=ALPHA)

    parser.add_argument("--with_rerank", action="store_true")
    parser.add_argument("--rerank_top_k", type=int, default=RERANK_TOP_DOCS)

    parser.add_argument("--with_generation", action="store_true")
    parser.add_argument("--generation_top_n", type=int, default=GENERATION_TOP_N)
    parser.add_argument("--generation_max_per_doc", type=int, default=GENERATION_MAX_PER_DOC)

    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--result_name",
        type=str,
        default="hotpotqa_no_query_opt",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除当前 persist_dir 后重新构建向量库。",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_root, f"{args.result_name}_{timestamp}")
    ensure_dir(output_dir)

    details_path = os.path.join(output_dir, "details.jsonl")
    summary_path = os.path.join(output_dir, "summary.json")
    doc_map_path = os.path.join(output_dir, "doc_map.json")
    qrels_path = os.path.join(output_dir, "qrels.json")
    sample_meta_path = os.path.join(output_dir, "sample_meta.json")

    clear_file(details_path)

    print("\n========== Ablation Setting ==========")
    print("Ablation name: no_query_optimization")
    print("Query optimizer: DISABLED")
    print("Query rewrite   : DISABLED")
    print("Query routing   : DISABLED")
    print("Retrieval input : original question")

    print("\n========== Paths ==========")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"BACKEND_DIR  = {BACKEND_DIR}")
    print(f"DATA_DIR     = {args.data_dir}")
    print(f"CHROMA_ROOT  = {args.chroma_root}")
    print(f"PERSIST_DIR  = {args.persist_dir}")
    print(f"OUTPUT_DIR   = {output_dir}")

    ensure_dir(args.chroma_root)

    if args.rebuild and os.path.exists(args.persist_dir):
        print(f"[Rebuild] 删除旧向量库目录：{args.persist_dir}")
        shutil.rmtree(args.persist_dir, ignore_errors=True)

    ensure_dir(args.persist_dir)

    # 1. 数据 -> 全局 chunks
    chunks, queries, answers, qrels, doc_map, sample_meta = build_hotpotqa_global_chunks(
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    save_json(doc_map_path, doc_map)
    save_json(qrels_path, qrels)
    save_json(sample_meta_path, sample_meta)

    # 2. 构建 / 加载向量库
    print("\n========== VectorStore ==========")
    t0 = time.time()

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = (
        VectorStoreBuilder.build_or_load_from_chunks(
            chunks=chunks,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
        )
    )

    vector_time = time.time() - t0
    print(f"[VectorStore] build/load time = {vector_time:.2f}s")
    print(f"[VectorStore] loaded chunks   = {len(chunk_texts)}")

    # 3. 检索器
    retriever = HybridRetriever(
        vectordb=vectordb,
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_ids=chunk_ids,
        chunk_metas=chunk_metas,
        alpha=args.alpha,
    )

    # 4. reranker
    reranker = None
    if args.with_rerank:
        print("\n========== Reranker ==========")
        reranker = Reranker()
        print("[Reranker] loaded")

    if args.with_generation and Generator is None:
        raise RuntimeError(
            f"你开启了 --with_generation，但 generator.py 导入失败：{GENERATOR_IMPORT_ERROR}"
        )

    # 5. 选择问题
    qids = list(queries.keys())
    if args.max_queries and args.max_queries > 0:
        qids = qids[:args.max_queries]

    print("\n========== Evaluation ==========")
    print(f"evaluated queries = {len(qids)}")
    print(f"with_rerank       = {args.with_rerank}")
    print(f"with_generation   = {args.with_generation}")

    metric_rows: List[Dict[str, Any]] = []

    total_retrieval_time = 0.0
    total_rerank_time = 0.0
    total_generation_time = 0.0

    for idx, qid in enumerate(qids, start=1):
        original_query = queries[qid]
        gold_answer = answers.get(qid, "")
        gold_doc_ids = set(qrels.get(qid, []))
        meta_for_sample = sample_meta.get(qid, {})

        # =================================================
        # 核心消融点：
        # 不调用 query_optimizer / query_rewrite / query_router。
        # 直接使用 original_query 进入检索。
        # =================================================
        retrieval_query = original_query

        # 5.1 直接检索
        rt0 = time.time()

        raw_hits = retriever.retrieve(
            query=retrieval_query,
            top_k_chunks=args.top_k_chunks,
            dense_k=args.dense_k,
            bm25_k=args.bm25_k,
        )

        retrieval_time = time.time() - rt0
        total_retrieval_time += retrieval_time

        retrieved_items = []
        for chunk_key, text, doc_id, score, meta in raw_hits:
            retrieved_items.append({
                "chunk_id": str(chunk_key),
                "doc_id": str(doc_id),
                "text": text,
                "score": float(score),
                "meta": meta or {},
            })

        # 5.2 rerank 保持不变
        ranked_items = retrieved_items
        rerank_time = 0.0

        if reranker is not None and retrieved_items:
            rrt0 = time.time()

            texts = [x["text"] for x in retrieved_items]
            ids = list(range(len(retrieved_items)))
            metas = [
                {
                    "chunk_id": x["chunk_id"],
                    "doc_id": x["doc_id"],
                    **(x.get("meta") or {}),
                }
                for x in retrieved_items
            ]

            rerank_results = reranker.rerank_texts(
                query=original_query,
                texts=texts,
                ids=ids,
                metas=metas,
                top_k=args.rerank_top_k,
            )

            ranked_items = []
            for local_idx, text, score, meta in rerank_results:
                original = retrieved_items[local_idx]
                ranked_items.append({
                    "chunk_id": original["chunk_id"],
                    "doc_id": original["doc_id"],
                    "text": text,
                    "score": float(score),
                    "meta": meta or {},
                })

            rerank_time = time.time() - rrt0
            total_rerank_time += rerank_time

        # 5.3 doc-level 聚合
        ranked_doc_ids = []
        seen_doc_ids = set()

        for item in ranked_items:
            doc_id = item["doc_id"]
            if doc_id not in seen_doc_ids:
                ranked_doc_ids.append(doc_id)
                seen_doc_ids.add(doc_id)

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_doc_ids=ranked_doc_ids,
            gold_doc_ids=gold_doc_ids,
            ks=(1, 2, 3, 5, 10, 20, 50),
        )

        # 5.4 generation chunks 保持不变
        if reranker is not None:
            selected_items = Reranker.select_generation_chunks(
                ranked_chunks=ranked_items,
                top_n=args.generation_top_n,
                max_per_doc=args.generation_max_per_doc,
            )
        else:
            selected_items = ranked_items[:args.generation_top_n]

        generation_chunks = []
        for item in selected_items:
            generation_chunks.append({
                "chunk_idx": item["chunk_id"],
                "chunk_id": item["chunk_id"],
                "doc_id": item["doc_id"],
                "text": item["text"],
                "score": item["score"],
                "meta": item.get("meta", {}),
            })

        generation_doc_ids = []
        seen_gen_docs = set()
        for item in generation_chunks:
            doc_id = item["doc_id"]
            if doc_id not in seen_gen_docs:
                generation_doc_ids.append(doc_id)
                seen_gen_docs.add(doc_id)

        generation_chunk_metrics = compute_retrieval_metrics(
            retrieved_doc_ids=generation_doc_ids,
            gold_doc_ids=gold_doc_ids,
            ks=(1, 2, 3, 5, 10, 20, 50),
        )

        # 5.5 生成保持不变
        gen_output = None
        answer_metrics = {}
        generation_time = 0.0

        if args.with_generation:
            gt0 = time.time()

            gen_output = asyncio.run(
                run_generator_once(
                    qid=qid,
                    query=original_query,
                    generation_chunks=generation_chunks,
                )
            )

            generation_time = time.time() - gt0
            total_generation_time += generation_time

            prediction = (gen_output.get("answer") or "").strip()

            answer_metrics = {
                "answer_em": answer_em(prediction, gold_answer),
                "answer_f1": answer_f1(prediction, gold_answer),
                "prediction": prediction,
                "gold_answer": gold_answer,
            }

        row_metrics = {
            **retrieval_metrics,
            **{
                f"genchunk_{k}": v
                for k, v in generation_chunk_metrics.items()
            },
            **{
                k: v
                for k, v in answer_metrics.items()
                if isinstance(v, (int, float))
            },
            "type": meta_for_sample.get("type", ""),
            "level": meta_for_sample.get("level", ""),
        }

        metric_rows.append(row_metrics)

        detail_row = {
            "qid": qid,
            "query": original_query,
            "retrieval_query": retrieval_query,
            "query_optimization_enabled": False,
            "query_rewrite_enabled": False,
            "query_routing_enabled": False,
            "type": meta_for_sample.get("type", ""),
            "level": meta_for_sample.get("level", ""),
            "supporting_titles": meta_for_sample.get("supporting_titles", []),
            "gold_answer": gold_answer,
            "gold_doc_ids": sorted(list(gold_doc_ids)),
            "retrieved_doc_ids": ranked_doc_ids,
            "generation_doc_ids": generation_doc_ids,
            "ranked_chunks": ranked_items,
            "generation_chunks": generation_chunks,
            "retrieval_metrics": retrieval_metrics,
            "generation_chunk_metrics": generation_chunk_metrics,
            "answer_metrics": answer_metrics,
            "timing": {
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": generation_time,
            },
        }

        if gen_output is not None:
            detail_row["gen_output"] = gen_output

        append_jsonl(details_path, detail_row)

        if idx % 10 == 0 or idx == len(qids):
            print(
                f"[{idx}/{len(qids)}] "
                f"R@2={retrieval_metrics['recall@2']:.3f} "
                f"R@5={retrieval_metrics['recall@5']:.3f} "
                f"R@10={retrieval_metrics['recall@10']:.3f} "
                f"GenR@5={generation_chunk_metrics['recall@5']:.3f} "
                f"MRR={retrieval_metrics['mrr']:.3f} "
                f"time={retrieval_time + rerank_time + generation_time:.2f}s"
            )

    # 6. 汇总
    retrieval_summary = {
        "hit@1": avg(metric_rows, "hit@1"),
        "hit@2": avg(metric_rows, "hit@2"),
        "hit@3": avg(metric_rows, "hit@3"),
        "hit@5": avg(metric_rows, "hit@5"),
        "hit@10": avg(metric_rows, "hit@10"),
        "hit@20": avg(metric_rows, "hit@20"),
        "hit@50": avg(metric_rows, "hit@50"),
        "recall@1": avg(metric_rows, "recall@1"),
        "recall@2": avg(metric_rows, "recall@2"),
        "recall@3": avg(metric_rows, "recall@3"),
        "recall@5": avg(metric_rows, "recall@5"),
        "recall@10": avg(metric_rows, "recall@10"),
        "recall@20": avg(metric_rows, "recall@20"),
        "recall@50": avg(metric_rows, "recall@50"),
        "mrr": avg(metric_rows, "mrr"),
    }

    generation_chunk_summary = {
        "hit@1": avg(metric_rows, "genchunk_hit@1"),
        "hit@2": avg(metric_rows, "genchunk_hit@2"),
        "hit@3": avg(metric_rows, "genchunk_hit@3"),
        "hit@5": avg(metric_rows, "genchunk_hit@5"),
        "hit@10": avg(metric_rows, "genchunk_hit@10"),
        "hit@20": avg(metric_rows, "genchunk_hit@20"),
        "hit@50": avg(metric_rows, "genchunk_hit@50"),
        "recall@1": avg(metric_rows, "genchunk_recall@1"),
        "recall@2": avg(metric_rows, "genchunk_recall@2"),
        "recall@3": avg(metric_rows, "genchunk_recall@3"),
        "recall@5": avg(metric_rows, "genchunk_recall@5"),
        "recall@10": avg(metric_rows, "genchunk_recall@10"),
        "recall@20": avg(metric_rows, "genchunk_recall@20"),
        "recall@50": avg(metric_rows, "genchunk_recall@50"),
        "mrr": avg(metric_rows, "genchunk_mrr"),
    }

    answer_summary = {}
    if args.with_generation:
        answer_summary = {
            "answer_em": avg(metric_rows, "answer_em"),
            "answer_f1": avg(metric_rows, "answer_f1"),
        }

    by_type = {
        "recall@2": group_avg(metric_rows, "recall@2", "type"),
        "recall@5": group_avg(metric_rows, "recall@5", "type"),
        "recall@10": group_avg(metric_rows, "recall@10", "type"),
    }

    by_level = {
        "recall@2": group_avg(metric_rows, "recall@2", "level"),
        "recall@5": group_avg(metric_rows, "recall@5", "level"),
        "recall@10": group_avg(metric_rows, "recall@10", "level"),
    }

    summary = {
        "dataset": "hotpotqa_distractor",
        "setting": "ablation_no_query_optimization",
        "ablation": {
            "query_optimization": False,
            "query_rewrite": False,
            "query_routing": False,
            "retrieval_query": "original_question",
            "kept_modules": [
                "global_deduplicated_collection",
                "hybrid_retrieval",
                "rerank",
                "generation_chunk_selection",
                "generator",
            ],
        },
        "data_dir": args.data_dir,
        "chroma_root": args.chroma_root,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection_name,
        "output_dir": output_dir,
        "split": args.split,
        "max_samples_for_collection": args.max_samples,
        "max_queries": args.max_queries,
        "evaluated_queries": len(qids),
        "unique_paragraphs": len(doc_map),
        "chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "retrieval_params": {
            "top_k_chunks": args.top_k_chunks,
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "alpha": args.alpha,
            "with_rerank": args.with_rerank,
            "rerank_top_k": args.rerank_top_k,
        },
        "generation_params": {
            "with_generation": args.with_generation,
            "generation_top_n": args.generation_top_n,
            "generation_max_per_doc": args.generation_max_per_doc,
        },
        "retrieval": retrieval_summary,
        "generation_chunk_selection": generation_chunk_summary,
        "answer": answer_summary,
        "by_type": by_type,
        "by_level": by_level,
        "timing": {
            "vector_build_or_load_time": vector_time,
            "total_retrieval_time": total_retrieval_time,
            "total_rerank_time": total_rerank_time,
            "total_generation_time": total_generation_time,
            "avg_retrieval_time": total_retrieval_time / len(qids) if qids else 0.0,
            "avg_rerank_time": total_rerank_time / len(qids) if qids else 0.0,
            "avg_generation_time": total_generation_time / len(qids) if qids else 0.0,
        },
        "files": {
            "summary": summary_path,
            "details": details_path,
            "doc_map": doc_map_path,
            "qrels": qrels_path,
            "sample_meta": sample_meta_path,
        },
    }

    save_json(summary_path, summary)

    print("\n========== No Query Optimization Ablation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"summary     : {summary_path}")
    print(f"details     : {details_path}")
    print(f"doc_map     : {doc_map_path}")
    print(f"qrels       : {qrels_path}")
    print(f"sample_meta : {sample_meta_path}")


if __name__ == "__main__":
    main()