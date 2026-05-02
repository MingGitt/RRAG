import os
import sys
import re
import json
import time
import argparse
import asyncio
import threading
import string
from collections import Counter, defaultdict
from typing import Dict, Any, List, Set

from tqdm import tqdm


# ============================================================
# 1. 导入 backend 模块
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from vector_store import VectorStoreBuilder
from hybrid_retriever import HybridRetriever
from reranker import Reranker
from qa_service import answer_one_query
from config import ALPHA


# ============================================================
# 2. 答案评测指标：EM / F1
# ============================================================

def normalize_answer(s: str) -> str:
    """
    对英文开放域问答答案进行归一化。
    主要用于计算 EM / F1。
    """
    if s is None:
        return ""

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """
    Exact Match。
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1。
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def metric_max_over_gold(metric_fn, pred: str, gold_answers: List[str]) -> float:
    """
    KILT / NQ 可能有多个标准答案，取最高分。
    """
    if not gold_answers:
        return 0.0
    return max(metric_fn(pred, gold) for gold in gold_answers)


# ============================================================
# 3. 数据读取
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def normalize_title(s: str) -> str:
    """
    归一化 Wikipedia title / 文件名。
    用于判断检索结果是否命中 gold provenance title。
    """
    if not s:
        return ""

    s = os.path.splitext(os.path.basename(str(s)))[0]
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def get_gold_answers(question_record: Dict[str, Any]) -> List[str]:
    """
    从 kilt_nq_200_questions.jsonl 中读取标准答案。
    """
    answers = question_record.get("answers", []) or []
    return [str(a) for a in answers if str(a).strip()]


def get_gold_titles(question_record: Dict[str, Any]) -> Set[str]:
    """
    从 kilt_nq_200_questions.jsonl 中读取 gold provenance title。
    """
    titles = set()

    for prov in question_record.get("provenance", []) or []:
        title = prov.get("title")
        if title:
            titles.add(normalize_title(title))

    return titles


# ============================================================
# 4. 从系统输出中提取答案、证据、contexts
# ============================================================

def extract_answer_from_result(result: Dict[str, Any], qid: str) -> str:
    """
    从 qa_service.answer_one_query 的返回结果中提取最终答案。

    预期结构：
        result["gen_outputs"][qid]["answer"]
    """
    gen_outputs = result.get("gen_outputs", {}) or {}
    output = gen_outputs.get(qid, {}) or {}
    return output.get("answer", "") or ""


def extract_citations_from_result(result: Dict[str, Any], qid: str) -> List[Dict[str, Any]]:
    """
    提取 citations。
    simple 查询一般在 output["citations"]。
    fuzzy / complex 查询可能在 output["sub_answers"][].citations。
    """
    gen_outputs = result.get("gen_outputs", {}) or {}
    output = gen_outputs.get(qid, {}) or {}

    citations = []

    for c in output.get("citations", []) or []:
        citations.append(c)

    for sub in output.get("sub_answers", []) or []:
        for c in sub.get("citations", []) or []:
            citations.append(c)

    return citations


def extract_ranked_titles_from_result(result: Dict[str, Any], qid: str) -> List[str]:
    """
    从 ranked_chunks_map 中提取检索到的文档 title。
    用于计算 provenance Recall@K。

    对于复杂问题，ranked_chunks_map 里可能存在：
        qid
        qid__sub_0
        qid__sub_1
        ...
    所以这里会把主问题和子问题的 ranked chunks 都统计进来。
    """
    titles = []

    ranked_map = result.get("ranked_chunks_map", {}) or {}

    candidate_keys = [qid]
    candidate_keys.extend([
        k for k in ranked_map.keys()
        if str(k).startswith(str(qid) + "__sub_")
    ])

    for key in candidate_keys:
        ranked_chunks = ranked_map.get(key, []) or []

        for item in ranked_chunks:
            meta = item.get("meta", {}) or {}
            title = (
                meta.get("title")
                or meta.get("kilt_title")
                or meta.get("source")
                or item.get("doc_id")
                or ""
            )

            if title:
                titles.append(normalize_title(title))

    return titles


def extract_contexts_from_result(
    result: Dict[str, Any],
    qid: str,
    top_k: int = 5,
    max_context_chars: int = 4000,
) -> List[str]:
    """
    提取 RAGAS 所需 contexts / retrieved_contexts。

    RAGAS 需要的是检索到的文本块列表，而不是标题。
    这里优先从 ranked_chunks_map 里取 text/content/page_content。

    参数：
        top_k:
            最多保留多少个上下文块。
        max_context_chars:
            单个上下文最大字符数，避免写入文件过大。
    """
    contexts = []

    ranked_map = result.get("ranked_chunks_map", {}) or {}

    candidate_keys = [qid]
    candidate_keys.extend([
        k for k in ranked_map.keys()
        if str(k).startswith(str(qid) + "__sub_")
    ])

    for key in candidate_keys:
        ranked_chunks = ranked_map.get(key, []) or []

        for item in ranked_chunks:
            text = (
                item.get("text")
                or item.get("content")
                or item.get("page_content")
                or item.get("chunk_text")
                or ""
            )

            # 有些代码会把文本放进 meta
            if not text:
                meta = item.get("meta", {}) or {}
                text = (
                    meta.get("text")
                    or meta.get("content")
                    or meta.get("page_content")
                    or ""
                )

            text = str(text).strip()

            if not text:
                continue

            if max_context_chars and len(text) > max_context_chars:
                text = text[:max_context_chars]

            if text not in contexts:
                contexts.append(text)

            if len(contexts) >= top_k:
                return contexts

    return contexts


def provenance_hit_at_k(
    pred_titles: List[str],
    gold_titles: Set[str],
    k: int,
) -> int:
    """
    判断 top-k 检索结果是否命中 gold Wikipedia title。
    """
    if not gold_titles:
        return 0

    top_titles = pred_titles[:k]

    for gold in gold_titles:
        gold_norm = normalize_title(gold)

        for pred in top_titles:
            pred_norm = normalize_title(pred)

            if gold_norm and (
                gold_norm == pred_norm
                or gold_norm in pred_norm
                or pred_norm in gold_norm
            ):
                return 1

    return 0


def extract_query_type_stats(result: Dict[str, Any]) -> Dict[str, int]:
    """
    提取 simple / fuzzy / complex 查询路由统计。
    """
    stats = result.get("query_type_stats", {}) or {}
    clean = {}

    for k, v in stats.items():
        try:
            clean[str(k)] = int(v)
        except Exception:
            clean[str(k)] = 1

    return clean


# ============================================================
# 5. 加载统一索引
# ============================================================

def load_single_index(
    persist_dir: str,
    collection_name: str,
):
    """
    加载 build_kilt_nq_single_index.py 构建好的统一 Chroma collection。

    注意：
        这里传入 chunks=[]。
        如果 collection 已经存在并且有数据，
        VectorStoreBuilder.build_or_load_from_chunks 会加载已有 collection。
    """
    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = (
        VectorStoreBuilder.build_or_load_from_chunks(
            chunks=[],
            persist_dir=persist_dir,
            collection_name=collection_name,
        )
    )

    try:
        count = vectordb._collection.count()
    except Exception:
        count = len(chunk_texts)

    if count <= 0:
        raise RuntimeError(
            f"统一索引为空，请先运行 build_kilt_nq_single_index.py。persist_dir={persist_dir}"
        )

    print(f"Loaded Chroma collection: {collection_name}")
    print(f"persist_dir: {persist_dir}")
    print(f"chunk count: {count}")

    retriever = HybridRetriever(
        vectordb=vectordb,
        chunk_texts=chunk_texts,
        chunk_doc_ids=chunk_doc_ids,
        chunk_ids=chunk_ids,
        chunk_metas=chunk_metas,
        alpha=ALPHA,
    )

    return retriever


# ============================================================
# 6. 主评测逻辑
# ============================================================

async def run_eval_async(
    persist_dir: str,
    collection_name: str,
    questions_path: str,
    output_dir: str,
    sample_size: int,
    ragas_top_k: int,
    max_context_chars: int,
):
    os.makedirs(output_dir, exist_ok=True)

    records = read_jsonl(questions_path)

    if sample_size > 0:
        records = records[:sample_size]

    retriever = load_single_index(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    print("Loading reranker...")
    reranker = Reranker()
    reranker_lock = threading.Lock()

    details_path = os.path.join(output_dir, "kilt_nq_single_index_eval_details.jsonl")
    summary_path = os.path.join(output_dir, "kilt_nq_single_index_eval_summary.json")

    # RAGAS 两种格式
    ragas_legacy_path = os.path.join(output_dir, "kilt_nq_single_index_ragas_legacy.jsonl")
    ragas_new_path = os.path.join(output_dir, "kilt_nq_single_index_ragas_new.jsonl")

    total = 0
    errors = 0

    em_sum = 0.0
    f1_sum = 0.0
    latency_sum = 0.0

    recall_at = {
        1: 0,
        3: 0,
        5: 0,
        10: 0,
    }

    query_type_stats = defaultdict(int)

    with open(details_path, "w", encoding="utf-8") as fout, \
         open(ragas_legacy_path, "w", encoding="utf-8") as ragas_legacy_f, \
         open(ragas_new_path, "w", encoding="utf-8") as ragas_new_f:

        for idx, item in enumerate(tqdm(records, desc="Evaluating KILT NQ single index")):
            total += 1

            raw_qid = item.get("id", f"q_{idx}")
            qid = f"kilt_nq_{idx}_{raw_qid}"
            question = item.get("question", "")

            gold_answers = get_gold_answers(item)
            gold_titles = get_gold_titles(item)

            start = time.time()

            try:
                result = await answer_one_query(
                    query_id=qid,
                    query_text=question,
                    retriever=retriever,
                    reranker=reranker,
                    reranker_lock=reranker_lock,
                )

                latency = time.time() - start
                latency_sum += latency

                pred_answer = extract_answer_from_result(result, qid)
                pred_titles = extract_ranked_titles_from_result(result, qid)
                contexts = extract_contexts_from_result(
                    result=result,
                    qid=qid,
                    top_k=ragas_top_k,
                    max_context_chars=max_context_chars,
                )

                em = metric_max_over_gold(
                    exact_match_score,
                    pred_answer,
                    gold_answers,
                )

                f1 = metric_max_over_gold(
                    f1_score,
                    pred_answer,
                    gold_answers,
                )

                em_sum += em
                f1_sum += f1

                recall_hit_record = {}

                for k in recall_at.keys():
                    hit = provenance_hit_at_k(
                        pred_titles=pred_titles,
                        gold_titles=gold_titles,
                        k=k,
                    )
                    recall_at[k] += hit
                    recall_hit_record[f"R@{k}"] = hit

                one_stats = extract_query_type_stats(result)
                for k, v in one_stats.items():
                    query_type_stats[k] += v

                citations = extract_citations_from_result(result, qid)

                # 详细评测记录
                result_record = {
                    "id": raw_qid,
                    "eval_qid": qid,
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "pred_answer": pred_answer,
                    "em": em,
                    "f1": f1,
                    "recall_hit_at": recall_hit_record,
                    "latency_sec": latency,
                    "pred_titles_top10": pred_titles[:10],
                    "query_type_stats": one_stats,
                    "citations": citations,
                    "num_ragas_contexts": len(contexts),
                }

                fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")

                # RAGAS legacy 格式
                ragas_legacy_record = {
                    "question": question,
                    "answer": pred_answer,
                    "contexts": contexts,
                    "ground_truth": gold_answers[0] if gold_answers else "",
                }

                ragas_legacy_f.write(
                    json.dumps(ragas_legacy_record, ensure_ascii=False) + "\n"
                )

                # RAGAS new 格式
                ragas_new_record = {
                    "user_input": question,
                    "response": pred_answer,
                    "retrieved_contexts": contexts,
                    "reference": gold_answers[0] if gold_answers else "",
                }

                ragas_new_f.write(
                    json.dumps(ragas_new_record, ensure_ascii=False) + "\n"
                )

            except Exception as e:
                errors += 1
                latency = time.time() - start

                error_record = {
                    "id": raw_qid,
                    "eval_qid": qid,
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "error": repr(e),
                    "latency_sec": latency,
                }

                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")

                # 出错样本不写入 RAGAS 文件，避免 RAGAS 评测时报空 answer / contexts 错误

    valid_total = max(total - errors, 1)

    summary = {
        "task": "kilt_nq_200_single_index",
        "persist_dir": persist_dir,
        "collection_name": collection_name,
        "questions_path": questions_path,
        "total": total,
        "errors": errors,
        "valid_total": valid_total,
        "answer_em": em_sum / valid_total,
        "answer_f1": f1_sum / valid_total,
        "provenance_recall": {
            f"R@{k}": v / valid_total
            for k, v in recall_at.items()
        },
        "avg_latency_sec": latency_sum / valid_total,
        "query_type_stats": dict(query_type_stats),
        "ragas_top_k": ragas_top_k,
        "max_context_chars": max_context_chars,
        "details_path": details_path,
        "summary_path": summary_path,
        "ragas_legacy_path": ragas_legacy_path,
        "ragas_new_path": ragas_new_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== KILT NQ Single Index Evaluation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved files:")
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")
    print(f"ragas legacy: {ragas_legacy_path}")
    print(f"ragas new: {ragas_new_path}")


# ============================================================
# 7. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--persist_dir",
        type=str,
        default=r"D:\code\rag\FSR\chroma_kilt_nq_200_single",
        help="统一 Chroma 向量库目录。",
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default="kilt_nq_200_single",
        help="统一 Chroma collection 名称。",
    )

    parser.add_argument(
        "--questions_path",
        type=str,
        default=r"D:\code\rag\FSR\data\kilt_nq_200_fulltext\kilt_nq_200_questions.jsonl",
        help="fetch_kilt_nq_200_fulltext.py 生成的问题文件。",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\code\rag\FSR\eval\kilt_nq_single_index_eval_outputs",
        help="评测结果输出目录。",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="评测样本数量。先用 5 或 20 测试，确认速度后再改成 200。",
    )

    parser.add_argument(
        "--ragas_top_k",
        type=int,
        default=5,
        help="写入 RAGAS 文件的 context 数量。",
    )

    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=4000,
        help="单个 context 最大字符数，防止 RAGAS 文件过大。",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.persist_dir):
        raise FileNotFoundError(
            f"向量库目录不存在: {args.persist_dir}\n"
            f"请先运行 build_kilt_nq_single_index.py 构建统一索引。"
        )

    if not os.path.exists(args.questions_path):
        raise FileNotFoundError(
            f"问题文件不存在: {args.questions_path}"
        )

    asyncio.run(
        run_eval_async(
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            questions_path=args.questions_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            ragas_top_k=args.ragas_top_k,
            max_context_chars=args.max_context_chars,
        )
    )


if __name__ == "__main__":
    main()