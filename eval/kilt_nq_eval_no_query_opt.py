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
from typing import Dict, Any, List, Set, Tuple

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
from generator import Generator
from evaluator import RetrievalEvaluator
from chunk_critique_judge import ChunkCritiqueJudge

from config import (
    ALPHA,
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    DOC_AGG_MODE,
    TOP_K_CHUNKS,
    GENERATION_TOP_N,
    GENERATION_MAX_PER_DOC,
)


# ============================================================
# 2. 答案评测指标：EM / F1
# ============================================================

def normalize_answer(s: str) -> str:
    """
    对英文开放域问答答案进行归一化。
    用于计算 EM / F1。
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
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
    answers = question_record.get("answers", []) or []
    return [str(a) for a in answers if str(a).strip()]


def get_gold_titles(question_record: Dict[str, Any]) -> Set[str]:
    titles = set()

    for prov in question_record.get("provenance", []) or []:
        title = prov.get("title")
        if title:
            titles.add(normalize_title(title))

    return titles


# ============================================================
# 4. 加载统一索引
# ============================================================

def load_single_index(
    persist_dir: str,
    collection_name: str,
):
    """
    加载 build_kilt_nq_single_index.py 构建好的统一 Chroma collection。
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
# 5. 取消查询优化的 RAG 链路
# ============================================================

def run_raw_query_retrieval_chain(
    qid: str,
    query: str,
    retriever,
    reranker,
    reranker_lock,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    消融实验核心函数：取消查询优化。

    不调用：
        QueryOptimizer.classify()
        QueryOptimizer.expand()
        QueryOptimizer.rewrite_query()
        QueryOptimizer.decompose_query()

    只做：
        原始 query 检索
        rerank
        chunk critique
        选择 generation chunks
    """

    # 1. 原始 query 直接检索
    retrieved = retriever.retrieve(
        query,
        top_k_chunks=TOP_K_CHUNKS,
        dense_k=DENSE_CANDIDATE_K,
        bm25_k=BM25_CANDIDATE_K,
    )

    if not retrieved:
        return {}, [], [], []

    # retrieved item:
    # (candidate_key, text, doc_id, score, meta)
    texts = []
    candidate_indices = []
    candidate_doc_map = {}
    candidate_meta_map = {}

    for idx, text, doc_id, score, meta in retrieved:
        texts.append(text)
        candidate_indices.append(idx)
        candidate_doc_map[idx] = doc_id
        candidate_meta_map[idx] = meta

    metas = [candidate_meta_map[idx] for idx in candidate_indices]

    # 2. rerank
    with reranker_lock:
        ranked_chunks_raw = reranker.rerank_texts(
            query,
            texts,
            candidate_indices,
            metas=metas,
            top_k=TOP_K_CHUNKS,
        )

    if not ranked_chunks_raw:
        return {}, [], [], []

    ranked_chunks = []

    for candidate_idx, text, score, meta in ranked_chunks_raw:
        ranked_chunks.append({
            "chunk_idx": candidate_idx,
            "text": text,
            "doc_id": candidate_doc_map.get(candidate_idx),
            "score": float(score),
            "meta": meta or {},
        })

    # 3. 文档级聚合分数
    doc_level_for_eval = [
        (item["doc_id"], item["text"], item["score"])
        for item in ranked_chunks
    ]

    doc_scores = RetrievalEvaluator.aggregate_doc_scores(
        doc_level_for_eval,
        mode=DOC_AGG_MODE,
    )

    # 4. chunk critique：保留原系统的 evidence critique 功能
    critique_input = ranked_chunks[:TOP_K_CHUNKS]

    critiqued_chunks = ChunkCritiqueJudge.judge_chunks(
        query,
        critique_input,
    )

    generation_chunks = ChunkCritiqueJudge.select_generation_chunks(
        critiqued_chunks,
        top_n=GENERATION_TOP_N,
        max_per_doc=GENERATION_MAX_PER_DOC,
        min_score=0.35,
    )

    # 5. 如果 critique 一个都没留下，则回退到 rerank top chunks
    if not generation_chunks:
        gen_chunks_input = [
            {
                "doc_id": item["doc_id"],
                "text": item["text"],
                "score": item["score"],
                "meta": item["meta"],
            }
            for item in ranked_chunks
        ]

        generation_chunks = Reranker.select_generation_chunks(
            gen_chunks_input,
            top_n=GENERATION_TOP_N,
            max_per_doc=GENERATION_MAX_PER_DOC,
        )

    return doc_scores, generation_chunks, ranked_chunks, critiqued_chunks


async def answer_one_query_no_query_opt(
    query_id: str,
    query_text: str,
    retriever,
    reranker,
    reranker_lock,
) -> Dict[str, Any]:
    """
    取消查询优化后的 answer_one_query 替代函数。

    输入问题后，不做查询分类、不改写、不分解。
    直接使用原始 query 走检索和生成。
    """

    doc_scores, generation_chunks, ranked_chunks, critiqued_chunks = (
        run_raw_query_retrieval_chain(
            qid=query_id,
            query=query_text,
            retriever=retriever,
            reranker=reranker,
            reranker_lock=reranker_lock,
        )
    )

    generation_chunks_map = {}
    ranked_chunks_map = {}
    critiqued_chunks_map = {}

    if generation_chunks:
        generation_chunks_map[query_id] = generation_chunks

    if ranked_chunks:
        ranked_chunks_map[query_id] = ranked_chunks

    if critiqued_chunks:
        critiqued_chunks_map[query_id] = critiqued_chunks

    # 没有检索到 chunk 时，返回空答案，避免生成阶段报错
    if not generation_chunks:
        return {
            "query_type_stats": {"raw_no_query_opt": 1},
            "rerank_results_with_scores": {
                query_id: doc_scores or {}
            },
            "generation_chunks": generation_chunks_map,
            "ranked_chunks_map": ranked_chunks_map,
            "critiqued_chunks_map": critiqued_chunks_map,
            "gen_outputs": {
                query_id: {
                    "answer": "根据当前检索到的文档，暂时无法确定答案。",
                    "raw_text": "",
                    "reflections": {
                        "retrieve": "Yes",
                        "isrel": "Irrelevant",
                        "issup": "No Support",
                        "isuse": 2,
                        "followup_query": "None",
                    },
                    "chunks": [],
                    "citations": [],
                    "candidates": [],
                    "generation_risk": {
                        "risk_score": 0.8,
                        "risk_level": "high",
                        "confidence": 0.3,
                        "reasons": ["未检索到可用于生成的证据块"],
                    },
                }
            },
        }

    queries = {
        query_id: query_text
    }

    gen_outputs = await Generator.run(
        queries,
        generation_chunks_map,
        max_samples=1,
        max_concurrency=1,
    )

    return {
        "query_type_stats": {"raw_no_query_opt": 1},
        "rerank_results_with_scores": {
            query_id: doc_scores or {}
        },
        "generation_chunks": generation_chunks_map,
        "ranked_chunks_map": ranked_chunks_map,
        "critiqued_chunks_map": critiqued_chunks_map,
        "gen_outputs": gen_outputs,
    }


# ============================================================
# 6. 从系统输出中提取答案、证据、contexts
# ============================================================

def extract_answer_from_result(result: Dict[str, Any], qid: str) -> str:
    gen_outputs = result.get("gen_outputs", {}) or {}
    output = gen_outputs.get(qid, {}) or {}
    return output.get("answer", "") or ""


def extract_citations_from_result(result: Dict[str, Any], qid: str) -> List[Dict[str, Any]]:
    gen_outputs = result.get("gen_outputs", {}) or {}
    output = gen_outputs.get(qid, {}) or {}

    citations = []

    for c in output.get("citations", []) or []:
        citations.append(c)

    return citations


def extract_ranked_titles_from_result(result: Dict[str, Any], qid: str) -> List[str]:
    titles = []

    ranked_map = result.get("ranked_chunks_map", {}) or {}
    ranked_chunks = ranked_map.get(qid, []) or []

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
    contexts = []

    ranked_map = result.get("ranked_chunks_map", {}) or {}
    ranked_chunks = ranked_map.get(qid, []) or []

    for item in ranked_chunks:
        text = (
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or item.get("chunk_text")
            or ""
        )

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
    stats = result.get("query_type_stats", {}) or {}
    clean = {}

    for k, v in stats.items():
        try:
            clean[str(k)] = int(v)
        except Exception:
            clean[str(k)] = 1

    return clean


# ============================================================
# 7. 主评测逻辑
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

    details_path = os.path.join(
        output_dir,
        "kilt_nq_no_query_opt_eval_details.jsonl",
    )

    summary_path = os.path.join(
        output_dir,
        "kilt_nq_no_query_opt_eval_summary.json",
    )

    ragas_legacy_path = os.path.join(
        output_dir,
        "kilt_nq_no_query_opt_ragas_legacy.jsonl",
    )

    ragas_new_path = os.path.join(
        output_dir,
        "kilt_nq_no_query_opt_ragas_new.jsonl",
    )

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

        for idx, item in enumerate(tqdm(records, desc="Evaluating KILT NQ no query optimization")):
            total += 1

            raw_qid = item.get("id", f"q_{idx}")
            qid = f"kilt_nq_noopt_{idx}_{raw_qid}"
            question = item.get("question", "")

            gold_answers = get_gold_answers(item)
            gold_titles = get_gold_titles(item)

            start = time.time()

            try:
                result = await answer_one_query_no_query_opt(
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

                result_record = {
                    "id": raw_qid,
                    "eval_qid": qid,
                    "ablation": "no_query_optimization",
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

                ragas_legacy_record = {
                    "question": question,
                    "answer": pred_answer,
                    "contexts": contexts,
                    "ground_truth": gold_answers[0] if gold_answers else "",
                }

                ragas_legacy_f.write(
                    json.dumps(ragas_legacy_record, ensure_ascii=False) + "\n"
                )

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
                    "ablation": "no_query_optimization",
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "error": repr(e),
                    "latency_sec": latency,
                }

                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    valid_total = max(total - errors, 1)

    summary = {
        "task": "kilt_nq_200_single_index_ablation_no_query_optimization",
        "ablation": "no_query_optimization",
        "description": "原始 query 直接进入检索，不进行 QueryOptimizer.classify / expand / rewrite / decompose。",
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

    print("\n========== KILT NQ Ablation: No Query Optimization Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved files:")
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")
    print(f"ragas legacy: {ragas_legacy_path}")
    print(f"ragas new: {ragas_new_path}")


# ============================================================
# 8. 命令行入口
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
        help="KILT NQ 200 问题文件。",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\code\rag\FSR\eval\kilt_nq_no_query_opt_outputs",
        help="消融实验结果输出目录。",
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
        help="单个 context 最大字符数。",
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