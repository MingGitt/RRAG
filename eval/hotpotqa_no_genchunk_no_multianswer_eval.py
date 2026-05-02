# hotpotqa_no_genchunk_no_multianswer_eval.py
# 放置位置：D:\code\rag\FSR\eval\hotpotqa_no_genchunk_no_multianswer_eval.py
#
# 消融实验：
# 基于完整基线，去掉：
# 1. generation chunk 筛选 / critique
# 2. 多候选答案生成
# 3. 多候选答案打分 / 筛选
#
# 保留：
# 1. 查询优化 / 查询改写 / 复杂查询拆分
# 2. HybridRetriever 混合检索
# 3. Reranker 重排序
# 4. 子查询分别检索、分别生成
# 5. 多子答案最终 LLM 融合
#
# 固定实验设置：
# 数据：D:\code\rag\FSR\data\hotpotqa_distractor
# 向量库：D:\code\rag\FSR\chroma_store\hotpotqa_distractor_global_1000
# collection：hotpotqa_distractor_global
# max_samples=1000
# max_queries=200

import os
import re
import sys
import json
import time
import shutil
import hashlib
import argparse
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
    "hotpotqa_distractor_global_1000",
)
DEFAULT_OUTPUT_ROOT = os.path.join(
    CURRENT_DIR,
    "hotpotqa_distractor_ablation_outputs",
)


# =========================================================
# 后端已有接口
# =========================================================

from config import (  # noqa: E402
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ALPHA,
    DENSE_CANDIDATE_K,
    BM25_CANDIDATE_K,
    TOP_K_CHUNKS,
    RERANK_TOP_DOCS,
    QWEN_ANSWER_MODEL,
)

from vector_store import VectorStoreBuilder  # noqa: E402
from hybrid_retriever import HybridRetriever  # noqa: E402
from reranker import Reranker  # noqa: E402
from query_optimizer import QueryOptimizer  # noqa: E402
from qwen_client import QwenClient  # noqa: E402


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
# HotpotQA 本地读取
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
# 检索、rerank、直接生成
# =========================================================

def hybrid_retrieve_and_rerank(
    query: str,
    retriever: HybridRetriever,
    reranker: Reranker,
    top_k_chunks: int,
    dense_k: int,
    bm25_k: int,
    rerank_top_k: int,
):
    raw_hits = retriever.retrieve(
        query=query,
        top_k_chunks=top_k_chunks,
        dense_k=dense_k,
        bm25_k=bm25_k,
    )

    retrieved_items = []
    for chunk_key, text, doc_id, score, meta in raw_hits:
        retrieved_items.append({
            "chunk_idx": str(chunk_key),
            "chunk_id": str(chunk_key),
            "doc_id": str(doc_id),
            "text": text,
            "score": float(score),
            "meta": meta or {},
        })

    if not retrieved_items:
        return []

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
        query=query,
        texts=texts,
        ids=ids,
        metas=metas,
        top_k=rerank_top_k,
    )

    ranked_items = []
    for local_idx, text, score, meta in rerank_results:
        original = retrieved_items[local_idx]
        ranked_items.append({
            "chunk_idx": original["chunk_id"],
            "chunk_id": original["chunk_id"],
            "doc_id": original["doc_id"],
            "text": text,
            "score": float(score),
            "meta": meta or {},
        })

    return ranked_items


def doc_ids_from_chunks(chunks: List[Dict[str, Any]]) -> List[str]:
    doc_ids = []
    seen = set()

    for item in chunks:
        doc_id = item.get("doc_id", "")
        if doc_id and doc_id not in seen:
            doc_ids.append(doc_id)
            seen.add(doc_id)

    return doc_ids


def build_context_from_rerank_topk(chunks: List[Dict[str, Any]], max_chunks: int, max_chars_per_chunk: int):
    blocks = []

    for i, item in enumerate(chunks[:max_chunks], start=1):
        meta = item.get("meta", {}) or {}
        title = meta.get("title", "") or item.get("doc_id", "")
        text = normalize_space(item.get("text", ""))

        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."

        blocks.append(f"[{i}] Source: {title}\n{text}")

    return "\n\n".join(blocks) if blocks else "No evidence."


def direct_generate_once(
    query: str,
    rerank_chunks: List[Dict[str, Any]],
    evidence_top_k: int,
    max_chars_per_chunk: int,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    单次生成：
    - 不调用 CandidateGenerator
    - 不生成多候选
    - 不调用 AnswerSupportJudge
    - 不调用 RiskController
    - 直接将 rerank top-k 拼入 prompt
    """
    evidence_text = build_context_from_rerank_topk(
        chunks=rerank_chunks,
        max_chunks=evidence_top_k,
        max_chars_per_chunk=max_chars_per_chunk,
    )

    prompt = f"""
你是一个基于证据的问答助手。

请只根据给定证据回答问题。
如果证据不足以回答，请回答：根据当前证据无法确定。
请输出简洁答案，不要输出分析过程。

问题：
{query}

证据：
{evidence_text}

最终答案：
""".strip()

    answer = QwenClient.call(
        messages=[{"role": "user", "content": prompt}],
        model=QWEN_ANSWER_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    answer = (answer or "").strip()
    if not answer:
        answer = "根据当前证据无法确定。"

    return {
        "answer": answer,
        "raw_text": answer,
        "query": query,
        "generation_method": "single_qwen_call_no_multi_candidate",
        "chunks": rerank_chunks[:evidence_top_k],
        "citations": [
            {
                "index": i,
                "source": (item.get("meta", {}) or {}).get("title", item.get("doc_id", "")),
                "doc_id": item.get("doc_id", ""),
                "chunk_id": item.get("chunk_id", ""),
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
            }
            for i, item in enumerate(rerank_chunks[:evidence_top_k], start=1)
        ],
        "candidates": [
            {
                "answer": answer,
                "raw_text": answer,
                "note": "single_direct_generation_no_candidate_filtering",
            }
        ],
        "generation_risk": {
            "risk_score": None,
            "risk_level": "not_evaluated",
            "confidence": None,
            "reasons": [
                "本消融实验去掉多答案筛选、支持度判断和风险评估。"
            ],
        },
    }


def fuse_subanswers_once(
    original_query: str,
    qtype: str,
    sub_results: List[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    """
    对 fuzzy/complex 的多个子查询答案进行一次 LLM 融合。
    simple 只有一个子答案时也可以不融合，直接返回子答案。
    """
    if len(sub_results) == 1:
        only = sub_results[0]
        return {
            "answer": only.get("answer", ""),
            "raw_text": only.get("answer", ""),
            "fusion_method": "single_subanswer_no_fusion",
            "subanswers": sub_results,
        }

    sub_texts = []
    for i, item in enumerate(sub_results, start=1):
        sub_texts.append(
            f"[子答案{i}]\n"
            f"子查询：{item.get('subquery', '')}\n"
            f"答案：{item.get('answer', '')}"
        )

    prompt = f"""
你是一个答案融合助手。

原始问题：
{original_query}

查询类型：
{qtype}

下面是针对该问题的多个子查询答案。请综合这些子答案，生成一个能直接回答原始问题的最终答案。

要求：
1. 只输出最终答案。
2. 不要输出分析过程。
3. 如果多个子答案存在冲突，请优先保留证据更明确的结论。
4. 如果子答案都无法支持结论，请回答：根据当前证据无法确定。

子查询答案：
{chr(10).join(sub_texts)}

最终答案：
""".strip()

    answer = QwenClient.call(
        messages=[{"role": "user", "content": prompt}],
        model=QWEN_ANSWER_MODEL,
        temperature=0.0,
        max_tokens=max_tokens,
    )

    answer = (answer or "").strip()
    if not answer:
        answer = "根据当前证据无法确定。"

    return {
        "answer": answer,
        "raw_text": answer,
        "fusion_method": "llm_fuse_subanswers_once",
        "subanswers": sub_results,
    }


# =========================================================
# 主流程
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="validation")

    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--max_queries", type=int, default=200)

    parser.add_argument("--chroma_root", type=str, default=DEFAULT_CHROMA_ROOT)
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection_name", type=str, default="hotpotqa_distractor_global")

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)

    parser.add_argument("--top_k_chunks", type=int, default=TOP_K_CHUNKS)
    parser.add_argument("--dense_k", type=int, default=DENSE_CANDIDATE_K)
    parser.add_argument("--bm25_k", type=int, default=BM25_CANDIDATE_K)
    parser.add_argument("--alpha", type=float, default=ALPHA)

    parser.add_argument("--rerank_top_k", type=int, default=RERANK_TOP_DOCS)

    parser.add_argument(
        "--evidence_top_k",
        type=int,
        default=5,
        help="不做 generation chunk 筛选，直接取 rerank top-k 进入单次生成。",
    )
    parser.add_argument("--max_chars_per_chunk", type=int, default=1200)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--fusion_max_tokens", type=int, default=256)

    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--result_name",
        type=str,
        default="no_genchunk_no_multianswer_1000_200",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="复用之前向量库时不要加。只有需要重建库才加。",
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
    print("Ablation name                 : no_generation_chunk_filter_no_multi_answer_filter")
    print("Query optimizer               : ENABLED")
    print("Hybrid retrieval              : ENABLED")
    print("Reranker                      : ENABLED")
    print("Generation chunk filtering    : DISABLED")
    print("Multi-candidate generation    : DISABLED")
    print("Candidate scoring/filtering   : DISABLED")
    print("Subquery generation           : each subquery single Qwen generation")
    print("Final answer for subqueries   : one LLM fusion call")
    print("Evidence to generator         : rerank top-k directly")

    print("\n========== Paths ==========")
    print(f"PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"DATA_DIR     = {args.data_dir}")
    print(f"PERSIST_DIR  = {args.persist_dir}")
    print(f"OUTPUT_DIR   = {output_dir}")

    ensure_dir(args.chroma_root)

    if args.rebuild and os.path.exists(args.persist_dir):
        print(f"[Rebuild] 删除旧向量库目录：{args.persist_dir}")
        shutil.rmtree(args.persist_dir, ignore_errors=True)

    ensure_dir(args.persist_dir)

    # 1. 数据与 qrels
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

    # 2. 向量库
    print("\n========== VectorStore ==========")
    t0 = time.time()

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = (
        VectorStoreBuilder.build_or_load_from_chunks(
            chunks=chunks,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
        )
    )

    vector_build_time = time.time() - t0
    print(f"[VectorStore] build/load time = {vector_build_time:.2f}s")
    print(f"[VectorStore] loaded chunks   = {len(chunk_texts)}")

    # 3. 检索器和 reranker
    retriever = HybridRetriever(
        vectordb=vectordb,
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_ids=chunk_ids,
        chunk_metas=chunk_metas,
        alpha=args.alpha,
    )

    print("\n========== Reranker ==========")
    reranker = Reranker()
    print("[Reranker] loaded")

    # 4. 评测问题
    qids = list(queries.keys())
    if args.max_queries and args.max_queries > 0:
        qids = qids[:args.max_queries]

    print("\n========== Evaluation ==========")
    print(f"evaluated queries = {len(qids)}")
    print(f"max_samples       = {args.max_samples}")
    print(f"evidence_top_k    = {args.evidence_top_k}")

    metric_rows: List[Dict[str, Any]] = []

    total_query_opt_time = 0.0
    total_retrieval_time = 0.0
    total_rerank_time = 0.0
    total_generation_time = 0.0
    total_fusion_time = 0.0

    query_type_counts = {
        "simple": 0,
        "fuzzy": 0,
        "complex": 0,
        "NO": 0,
        "unknown": 0,
    }

    for idx, qid in enumerate(qids, start=1):
        original_query = queries[qid]
        gold_answer = answers.get(qid, "")
        gold_doc_ids = set(qrels.get(qid, []))
        meta_for_sample = sample_meta.get(qid, {})

        # 4.1 查询优化保留
        qo_t0 = time.time()
        try:
            qtype, expanded_queries = QueryOptimizer.expand(original_query)
        except Exception as e:
            print(f"[QueryOptimizer][ERROR] qid={qid}, error={repr(e)}")
            qtype, expanded_queries = "simple", [original_query]

        total_query_opt_time += time.time() - qo_t0

        if qtype not in query_type_counts:
            query_type_counts["unknown"] += 1
        else:
            query_type_counts[qtype] += 1

        if not expanded_queries:
            expanded_queries = [original_query]

        # 去重，避免重复子查询
        dedup_subqueries = []
        seen_subq = set()
        for sq in expanded_queries:
            sq = str(sq or "").strip()
            if sq and sq not in seen_subq:
                seen_subq.add(sq)
                dedup_subqueries.append(sq)

        expanded_queries = dedup_subqueries or [original_query]

        # 4.2 每个子查询：hybrid retrieval + rerank + rerank top-k 单次生成
        sub_results = []
        all_ranked_chunks = []
        all_generation_chunks = []

        for sub_idx, subquery in enumerate(expanded_queries, start=1):
            rt0 = time.time()

            raw_hits = retriever.retrieve(
                query=subquery,
                top_k_chunks=args.top_k_chunks,
                dense_k=args.dense_k,
                bm25_k=args.bm25_k,
            )

            retrieval_time = time.time() - rt0
            total_retrieval_time += retrieval_time

            retrieved_items = []
            for chunk_key, text, doc_id, score, meta in raw_hits:
                retrieved_items.append({
                    "chunk_idx": str(chunk_key),
                    "chunk_id": str(chunk_key),
                    "doc_id": str(doc_id),
                    "text": text,
                    "score": float(score),
                    "meta": meta or {},
                })

            rrt0 = time.time()

            ranked_items = []
            if retrieved_items:
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
                    query=subquery,
                    texts=texts,
                    ids=ids,
                    metas=metas,
                    top_k=args.rerank_top_k,
                )

                for local_idx, text, score, meta in rerank_results:
                    original = retrieved_items[local_idx]
                    ranked_items.append({
                        "chunk_idx": original["chunk_id"],
                        "chunk_id": original["chunk_id"],
                        "doc_id": original["doc_id"],
                        "text": text,
                        "score": float(score),
                        "meta": meta or {},
                    })

            rerank_time = time.time() - rrt0
            total_rerank_time += rerank_time

            # 核心消融点：
            # 不调用 critique / generation chunk 筛选
            # 直接取 rerank top-k 进入生成器
            generation_chunks = ranked_items[:args.evidence_top_k]

            gt0 = time.time()

            gen_output = direct_generate_once(
                query=subquery,
                rerank_chunks=ranked_items,
                evidence_top_k=args.evidence_top_k,
                max_chars_per_chunk=args.max_chars_per_chunk,
                max_tokens=args.max_tokens,
            )

            generation_time = time.time() - gt0
            total_generation_time += generation_time

            sub_result = {
                "subquery_index": sub_idx,
                "subquery": subquery,
                "answer": gen_output.get("answer", ""),
                "gen_output": gen_output,
                "retrieval_time": retrieval_time,
                "rerank_time": rerank_time,
                "generation_time": generation_time,
                "ranked_chunks": ranked_items,
                "generation_chunks": generation_chunks,
            }

            sub_results.append(sub_result)
            all_ranked_chunks.extend(ranked_items)
            all_generation_chunks.extend(generation_chunks)

        # 4.3 子答案融合
        ft0 = time.time()

        fused_output = fuse_subanswers_once(
            original_query=original_query,
            qtype=qtype,
            sub_results=[
                {
                    "subquery_index": x["subquery_index"],
                    "subquery": x["subquery"],
                    "answer": x["answer"],
                }
                for x in sub_results
            ],
            max_tokens=args.fusion_max_tokens,
        )

        fusion_time = time.time() - ft0
        total_fusion_time += fusion_time

        prediction = (fused_output.get("answer") or "").strip()

        answer_metrics = {
            "answer_em": answer_em(prediction, gold_answer),
            "answer_f1": answer_f1(prediction, gold_answer),
            "prediction": prediction,
            "gold_answer": gold_answer,
        }

        # 4.4 doc-level 聚合指标
        retrieved_doc_ids = doc_ids_from_chunks(all_ranked_chunks)
        generation_doc_ids = doc_ids_from_chunks(all_generation_chunks)

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_doc_ids=retrieved_doc_ids,
            gold_doc_ids=gold_doc_ids,
            ks=(1, 2, 3, 5, 10, 20, 50),
        )

        generation_chunk_metrics = compute_retrieval_metrics(
            retrieved_doc_ids=generation_doc_ids,
            gold_doc_ids=gold_doc_ids,
            ks=(1, 2, 3, 5, 10, 20, 50),
        )

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
            "qtype": qtype,
        }

        metric_rows.append(row_metrics)

        detail_row = {
            "qid": qid,
            "query": original_query,
            "qtype": qtype,
            "expanded_queries": expanded_queries,
            "query_optimization_enabled": True,
            "hybrid_retrieval_enabled": True,
            "rerank_enabled": True,
            "generation_chunk_filter_enabled": False,
            "multi_candidate_generation_enabled": False,
            "candidate_scoring_enabled": False,
            "subquery_single_generation_enabled": True,
            "final_fusion_enabled": len(sub_results) > 1,
            "type": meta_for_sample.get("type", ""),
            "level": meta_for_sample.get("level", ""),
            "supporting_titles": meta_for_sample.get("supporting_titles", []),
            "gold_answer": gold_answer,
            "gold_doc_ids": sorted(list(gold_doc_ids)),
            "retrieved_doc_ids": retrieved_doc_ids,
            "generation_doc_ids": generation_doc_ids,
            "ranked_chunks": all_ranked_chunks,
            "generation_chunks": all_generation_chunks,
            "sub_results": sub_results,
            "gen_output": fused_output,
            "answer_metrics": answer_metrics,
            "retrieval_metrics": retrieval_metrics,
            "generation_chunk_metrics": generation_chunk_metrics,
            "timing": {
                "query_optimization_time": None,
                "fusion_time": fusion_time,
            },
        }

        append_jsonl(details_path, detail_row)

        if idx % 10 == 0 or idx == len(qids):
            print(
                f"[{idx}/{len(qids)}] "
                f"qtype={qtype} "
                f"subq={len(expanded_queries)} "
                f"R@5={retrieval_metrics['recall@5']:.3f} "
                f"GenR@5={generation_chunk_metrics['recall@5']:.3f} "
                f"AnswerF1={answer_metrics['answer_f1']:.3f}"
            )

    # 5. 汇总
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

    answer_summary = {
        "answer_em": avg(metric_rows, "answer_em"),
        "answer_f1": avg(metric_rows, "answer_f1"),
    }

    summary = {
        "dataset": "hotpotqa_distractor",
        "setting": "ablation_no_generation_chunk_filter_no_multi_answer_filter",
        "ablation": {
            "query_optimization": True,
            "query_rewrite_or_decomposition": True,
            "hybrid_retrieval": True,
            "rerank": True,
            "generation_chunk_filter": False,
            "multi_candidate_generation": False,
            "candidate_scoring": False,
            "answer_support_judge": False,
            "risk_controller": False,
            "subquery_generation": "each_subquery_single_generation",
            "final_answer": "llm_fusion_for_multiple_subanswers",
            "generation_input": "rerank_topk_directly",
        },
        "data_dir": args.data_dir,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection_name,
        "output_dir": output_dir,
        "split": args.split,
        "max_samples_for_collection": args.max_samples,
        "max_queries": args.max_queries,
        "evaluated_queries": len(qids),
        "unique_paragraphs": len(doc_map),
        "chunks": len(chunks),
        "query_type_counts": query_type_counts,
        "retrieval_params": {
            "top_k_chunks": args.top_k_chunks,
            "dense_k": args.dense_k,
            "bm25_k": args.bm25_k,
            "alpha": args.alpha,
            "rerank_top_k": args.rerank_top_k,
        },
        "generation_params": {
            "evidence_top_k": args.evidence_top_k,
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "max_tokens": args.max_tokens,
            "fusion_max_tokens": args.fusion_max_tokens,
            "model": QWEN_ANSWER_MODEL,
        },
        "retrieval": retrieval_summary,
        "generation_chunk_selection": generation_chunk_summary,
        "answer": answer_summary,
        "by_type": {
            "recall@5": group_avg(metric_rows, "recall@5", "type"),
            "answer_f1": group_avg(metric_rows, "answer_f1", "type"),
        },
        "by_level": {
            "recall@5": group_avg(metric_rows, "recall@5", "level"),
            "answer_f1": group_avg(metric_rows, "answer_f1", "level"),
        },
        "by_qtype": {
            "recall@5": group_avg(metric_rows, "recall@5", "qtype"),
            "answer_f1": group_avg(metric_rows, "answer_f1", "qtype"),
        },
        "timing": {
            "vector_build_or_load_time": vector_build_time,
            "total_query_opt_time": total_query_opt_time,
            "total_retrieval_time": total_retrieval_time,
            "total_rerank_time": total_rerank_time,
            "total_generation_time": total_generation_time,
            "total_fusion_time": total_fusion_time,
            "avg_query_opt_time": total_query_opt_time / len(qids) if qids else 0.0,
            "avg_retrieval_time": total_retrieval_time / len(qids) if qids else 0.0,
            "avg_rerank_time": total_rerank_time / len(qids) if qids else 0.0,
            "avg_generation_time": total_generation_time / len(qids) if qids else 0.0,
            "avg_fusion_time": total_fusion_time / len(qids) if qids else 0.0,
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

    print("\n========== Ablation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"summary     : {summary_path}")
    print(f"details     : {details_path}")
    print(f"doc_map     : {doc_map_path}")
    print(f"qrels       : {qrels_path}")
    print(f"sample_meta : {sample_meta_path}")


if __name__ == "__main__":
    main()