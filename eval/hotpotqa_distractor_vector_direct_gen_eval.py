# hotpotqa_distractor_vector_direct_gen_eval.py
# 放置位置：D:\code\rag\FSR\eval\hotpotqa_distractor_vector_direct_gen_eval.py
#
# 消融实验：
# 去掉：
# 1. 查询优化模块
# 2. 查询改写模块
# 3. 查询路由模块
# 4. 混合检索模块
# 5. BM25 检索模块
# 6. 重排序模块
# 7. generation chunk 多证据筛选模块
# 8. 多候选生成与候选打分模块
#
# 保留：
# 1. HotpotQA Distractor 全局去重 collection
# 2. Chroma 向量库
# 3. 原始 question
# 4. 向量检索 top-k
# 5. 向量检索前 5 个证据块进入单次 Qwen 生成
# 6. Recall@k / Hit@k / MRR / Answer EM / Answer F1
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
# 导入后端已有接口
# =========================================================

from config import (  # noqa: E402
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_CHUNKS,
    QWEN_ANSWER_MODEL,
)

try:
    from config import QWEN_REQUEST_TIMEOUT  # noqa: E402
except Exception:
    QWEN_REQUEST_TIMEOUT = 300

from vector_store import VectorStoreBuilder  # noqa: E402
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
# 向量检索
# =========================================================

def vector_search(vectordb, query: str, top_k: int) -> List[Dict[str, Any]]:
    """
    直接调用 Chroma 向量检索：
    不经过 QueryOptimizer
    不经过 HybridRetriever
    不经过 BM25
    不经过 Reranker
    """
    results = vectordb.similarity_search_with_score(query, k=top_k)

    items = []
    for doc, distance in results:
        meta = dict(doc.metadata or {})
        chunk_id = str(meta.get("chunk_id", ""))
        doc_id = str(meta.get("doc_id", ""))

        score = 1.0 / (1.0 + float(distance))

        items.append({
            "chunk_idx": chunk_id,
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": doc.page_content,
            "score": score,
            "distance": float(distance),
            "meta": meta,
        })

    return items


def select_vector_top5_generation_chunks(
    vector_items: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    向量检索 + 直接生成消融实验专用：

    只取向量检索排序前 top_n 个证据块进入直接生成。
    默认 top_n=5。
    """
    return vector_items[:top_n]


# =========================================================
# 单次生成：不调用 Generator.run_single
# =========================================================

def build_context_from_vector_chunks(
    chunks: List[Dict[str, Any]],
    max_chars_per_chunk: int = 1200,
) -> str:
    blocks = []

    for i, item in enumerate(chunks, start=1):
        meta = item.get("meta", {}) or {}
        title = meta.get("title", "") or item.get("doc_id", "")
        text = item.get("text", "") or ""
        text = normalize_space(text)

        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + "..."

        blocks.append(f"[{i}] Source: {title}\n{text}")

    return "\n\n".join(blocks) if blocks else "No evidence."


def direct_generate_once(
    query: str,
    vector_chunks: List[Dict[str, Any]],
    max_tokens: int = 256,
    max_chars_per_chunk: int = 1200,
) -> Dict[str, Any]:
    """
    只调用一次 Qwen 生成答案。
    不生成多个候选。
    不调用 AnswerSupportJudge。
    不调用 RiskController。
    不做多证据筛选。
    """
    evidence_text = build_context_from_vector_chunks(
        chunks=vector_chunks,
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
        messages=[
            {"role": "user", "content": prompt}
        ],
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
        "reflections": {
            "retrieve": "Yes",
            "isrel": "Unknown",
            "issup": "NotJudged",
            "isuse": 0,
            "followup_query": "None",
        },
        "chunks": vector_chunks,
        "citations": [
            {
                "index": i,
                "source": (item.get("meta", {}) or {}).get("title", item.get("doc_id", "")),
                "doc_id": item.get("doc_id", ""),
                "chunk_id": item.get("chunk_id", ""),
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0)),
            }
            for i, item in enumerate(vector_chunks, start=1)
        ],
        "candidates": [
            {
                "answer": answer,
                "raw_text": answer,
                "final_score": None,
                "support_score": None,
                "note": "single_direct_generation_no_candidate_scoring",
            }
        ],
        "generation_risk": {
            "risk_score": None,
            "risk_level": "not_evaluated",
            "confidence": None,
            "reasons": [
                "该消融实验不调用多候选生成、支持度评估和风险评估模块。"
            ],
        },
    }


# =========================================================
# 主流程：vector direct generation ablation
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="validation")

    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="用于构建或复用全局 collection 的样本数。你的固定实验设置为 1000。",
    )
    parser.add_argument(
        "--max_queries",
        type=int,
        default=200,
        help="实际评测的问题数。你的固定实验设置为 200。",
    )

    parser.add_argument("--chroma_root", type=str, default=DEFAULT_CHROMA_ROOT)
    parser.add_argument("--persist_dir", type=str, default=DEFAULT_PERSIST_DIR)
    parser.add_argument(
        "--collection_name",
        type=str,
        default="hotpotqa_distractor_global",
    )

    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)

    parser.add_argument(
        "--vector_top_k",
        type=int,
        default=TOP_K_CHUNKS,
        help="向量检索返回的 top-k chunks，用于检索指标统计。",
    )

    parser.add_argument(
        "--generation_top_n",
        type=int,
        default=5,
        help="进入直接生成器的证据块数量。本消融实验固定为向量检索前 5 个 chunk。",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="单次答案生成最大 token 数。",
    )

    parser.add_argument(
        "--max_chars_per_chunk",
        type=int,
        default=1200,
        help="每个证据块拼入 prompt 的最大字符数。",
    )

    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--result_name",
        type=str,
        default="hotpotqa_vector_direct_gen_top5_1000_200",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="删除当前 persist_dir 后重新构建向量库。复用旧库时不要加。",
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
    print("Ablation name             : vector_direct_single_generation_top5")
    print("Query optimizer           : DISABLED")
    print("Query rewrite             : DISABLED")
    print("Query routing             : DISABLED")
    print("Hybrid retrieval          : DISABLED")
    print("BM25 retrieval            : DISABLED")
    print("Reranker                  : DISABLED")
    print("Generation chunk critique : DISABLED")
    print("Multi-candidate generation: DISABLED")
    print("Candidate scoring         : DISABLED")
    print("Risk assessment           : DISABLED")
    print("Retrieval method          : Chroma vector similarity_search_with_score")
    print("Retrieval metrics input   : vector top-k")
    print("Generation input          : vector top-5 chunks")
    print("Generation method         : one QwenClient.call")

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

    # 1. 读取数据，构造 chunks/qrels。
    # 即使复用向量库，也需要 qrels/doc_map 用于评测。
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

    vector_build_time = time.time() - t0
    print(f"[VectorStore] build/load time = {vector_build_time:.2f}s")
    print(f"[VectorStore] loaded chunks   = {len(chunk_texts)}")

    # 3. 选择问题
    qids = list(queries.keys())
    if args.max_queries and args.max_queries > 0:
        qids = qids[:args.max_queries]

    print("\n========== Evaluation ==========")
    print(f"evaluated queries = {len(qids)}")
    print(f"max_samples       = {args.max_samples}")
    print(f"vector_top_k      = {args.vector_top_k}")
    print(f"generation_top_n  = {args.generation_top_n}")
    print(f"single generation = True")

    metric_rows: List[Dict[str, Any]] = []

    total_vector_search_time = 0.0
    total_generation_time = 0.0

    for idx, qid in enumerate(qids, start=1):
        original_query = queries[qid]
        retrieval_query = original_query
        gold_answer = answers.get(qid, "")
        gold_doc_ids = set(qrels.get(qid, []))
        meta_for_sample = sample_meta.get(qid, {})

        # =================================================
        # 核心消融点：
        # 不调用 query_optimizer / HybridRetriever / Reranker。
        # 原始 query 直接进入 Chroma 向量检索。
        # =================================================
        vt0 = time.time()

        vector_items = vector_search(
            vectordb=vectordb,
            query=retrieval_query,
            top_k=args.vector_top_k,
        )

        vector_search_time = time.time() - vt0
        total_vector_search_time += vector_search_time

        # 4.1 doc-level 聚合，用于 retrieval 指标
        retrieved_doc_ids = []
        seen_doc_ids = set()

        for item in vector_items:
            doc_id = item["doc_id"]
            if doc_id not in seen_doc_ids:
                retrieved_doc_ids.append(doc_id)
                seen_doc_ids.add(doc_id)

        retrieval_metrics = compute_retrieval_metrics(
            retrieved_doc_ids=retrieved_doc_ids,
            gold_doc_ids=gold_doc_ids,
            ks=(1, 2, 3, 5, 10, 20, 50),
        )

        # 4.2 不做 generation chunk 筛选，但只取向量检索前 5 个 chunk 进入直接生成
        generation_chunks = select_vector_top5_generation_chunks(
            vector_items=vector_items,
            top_n=args.generation_top_n,
        )

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

        # 4.3 单次直接生成答案
        gt0 = time.time()

        gen_output = direct_generate_once(
            query=original_query,
            vector_chunks=generation_chunks,
            max_tokens=args.max_tokens,
            max_chars_per_chunk=args.max_chars_per_chunk,
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
            "hybrid_retrieval_enabled": False,
            "bm25_enabled": False,
            "rerank_enabled": False,
            "generation_chunk_filter_enabled": False,
            "multi_candidate_generation_enabled": False,
            "candidate_scoring_enabled": False,
            "generation_method": "single_qwen_call",
            "retrieval_method": "vector_only_chroma_similarity_search_with_score",
            "generation_chunk_source": "vector_top5",
            "type": meta_for_sample.get("type", ""),
            "level": meta_for_sample.get("level", ""),
            "supporting_titles": meta_for_sample.get("supporting_titles", []),
            "gold_answer": gold_answer,
            "gold_doc_ids": sorted(list(gold_doc_ids)),
            "retrieved_doc_ids": retrieved_doc_ids,
            "generation_doc_ids": generation_doc_ids,
            "ranked_chunks": vector_items,
            "generation_chunks": generation_chunks,
            "retrieval_metrics": retrieval_metrics,
            "generation_chunk_metrics": generation_chunk_metrics,
            "answer_metrics": answer_metrics,
            "gen_output": gen_output,
            "timing": {
                "vector_search_time": vector_search_time,
                "retrieval_time": vector_search_time,
                "rerank_time": 0.0,
                "generation_time": generation_time,
            },
        }

        append_jsonl(details_path, detail_row)

        if idx % 10 == 0 or idx == len(qids):
            print(
                f"[{idx}/{len(qids)}] "
                f"R@2={retrieval_metrics['recall@2']:.3f} "
                f"R@5={retrieval_metrics['recall@5']:.3f} "
                f"R@10={retrieval_metrics['recall@10']:.3f} "
                f"GenR@5={generation_chunk_metrics['recall@5']:.3f} "
                f"AnswerF1={answer_metrics['answer_f1']:.3f} "
                f"time={vector_search_time + generation_time:.2f}s"
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

    by_type = {
        "recall@2": group_avg(metric_rows, "recall@2", "type"),
        "recall@5": group_avg(metric_rows, "recall@5", "type"),
        "recall@10": group_avg(metric_rows, "recall@10", "type"),
        "answer_f1": group_avg(metric_rows, "answer_f1", "type"),
    }

    by_level = {
        "recall@2": group_avg(metric_rows, "recall@2", "level"),
        "recall@5": group_avg(metric_rows, "recall@5", "level"),
        "recall@10": group_avg(metric_rows, "recall@10", "level"),
        "answer_f1": group_avg(metric_rows, "answer_f1", "level"),
    }

    summary = {
        "dataset": "hotpotqa_distractor",
        "setting": "ablation_vector_direct_single_generation_top5",
        "ablation": {
            "query_optimization": False,
            "query_rewrite": False,
            "query_routing": False,
            "hybrid_retrieval": False,
            "bm25_retrieval": False,
            "rerank": False,
            "generation_chunk_filter": False,
            "multi_candidate_generation": False,
            "candidate_scoring": False,
            "answer_support_judge": False,
            "risk_controller": False,
            "retrieval_query": "original_question",
            "retrieval_method": "chroma_vector_similarity_search_with_score",
            "retrieval_metrics_input": "vector_topk",
            "generation_chunks": "vector_top5_directly_to_llm",
            "generation_method": "single_qwen_call",
            "kept_modules": [
                "global_deduplicated_collection",
                "vector_store",
                "qwen_client",
                "answer_em_f1_eval",
                "retrieval_recall_eval",
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
            "vector_top_k": args.vector_top_k,
            "with_hybrid": False,
            "with_bm25": False,
            "with_rerank": False,
        },
        "generation_params": {
            "single_generation": True,
            "generation_top_n": args.generation_top_n,
            "generation_chunk_source": "vector_top5",
            "max_tokens": args.max_tokens,
            "max_chars_per_chunk": args.max_chars_per_chunk,
            "model": QWEN_ANSWER_MODEL,
        },
        "retrieval": retrieval_summary,
        "generation_chunk_selection": generation_chunk_summary,
        "answer": answer_summary,
        "by_type": by_type,
        "by_level": by_level,
        "timing": {
            "vector_build_or_load_time": vector_build_time,
            "total_vector_search_time": total_vector_search_time,
            "total_retrieval_time": total_vector_search_time,
            "total_rerank_time": 0.0,
            "total_generation_time": total_generation_time,
            "avg_vector_search_time": total_vector_search_time / len(qids) if qids else 0.0,
            "avg_retrieval_time": total_vector_search_time / len(qids) if qids else 0.0,
            "avg_rerank_time": 0.0,
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

    print("\n========== Vector Direct Single Generation Top5 Ablation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"summary     : {summary_path}")
    print(f"details     : {details_path}")
    print(f"doc_map     : {doc_map_path}")
    print(f"qrels       : {qrels_path}")
    print(f"sample_meta : {sample_meta_path}")


if __name__ == "__main__":
    main()