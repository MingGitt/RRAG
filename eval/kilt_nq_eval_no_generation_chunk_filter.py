import os
import sys
import re
import json
import time
import argparse
import asyncio
import threading
import string
from collections import Counter
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
from chunk_critique_judge import ChunkCritiqueJudge

from config import (
    ALPHA,
    GENERATION_TOP_N,
    GENERATION_MAX_PER_DOC,
)


# ============================================================
# 2. 只去掉 generation chunk 筛选：Monkey Patch
# ============================================================

def bypass_chunk_critique(
    query: str,
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    替代 ChunkCritiqueJudge.judge_chunks。

    作用：
    - 不调用 LLM 判断 chunk 是否相关 / 有用
    - 不过滤 chunk
    - 直接保留 rerank 后的 chunks
    - 补充伪 critique 字段，保证后续接口结构兼容

    注意：
    这里只绕过 generation chunk 筛选，不影响其他模块。
    """
    if not chunks:
        return []

    patched_chunks = []

    for item in chunks:
        new_item = dict(item)

        new_item["critique_relevance"] = "Bypassed"
        new_item["critique_usefulness"] = "Bypassed"
        new_item["critique_score"] = float(item.get("score", 0.0) or 0.0)
        new_item["critique_reason"] = (
            "Ablation: ChunkCritiqueJudge.judge_chunks is bypassed; "
            "reranked chunks are passed through directly."
        )
        new_item["from_rerank_direct"] = True
        new_item["ablation_note"] = "only_no_generation_chunk_filter"

        patched_chunks.append(new_item)

    patched_chunks.sort(
        key=lambda x: float(x.get("score", 0.0) or 0.0),
        reverse=True,
    )

    return patched_chunks


def select_rerank_chunks_directly(
    chunks: List[Dict[str, Any]],
    top_n: int = GENERATION_TOP_N,
    max_per_doc: int = GENERATION_MAX_PER_DOC,
    min_score: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    替代 ChunkCritiqueJudge.select_generation_chunks。

    作用：
    - 忽略 min_score
    - 忽略 critique_score
    - 忽略 critique_relevance / critique_usefulness
    - 直接按照 rerank score 取 top-N chunks 作为 generation_chunks

    为了保持和原系统输入规模一致，默认仍使用：
    - GENERATION_TOP_N
    - GENERATION_MAX_PER_DOC

    如果你想完全不限制每篇文档的 chunk 数，可以在 config.py 里调大
    GENERATION_MAX_PER_DOC，或者这里直接改成 999。
    """
    if not chunks:
        return []

    selected = []
    per_doc_count: Dict[str, int] = {}

    sorted_chunks = sorted(
        chunks,
        key=lambda x: float(x.get("score", 0.0) or 0.0),
        reverse=True,
    )

    for item in sorted_chunks:
        doc_id = str(item.get("doc_id", ""))

        if per_doc_count.get(doc_id, 0) >= max_per_doc:
            continue

        new_item = dict(item)
        new_item["from_rerank_direct"] = True
        new_item["ablation_note"] = "only_no_generation_chunk_filter"

        selected.append(new_item)
        per_doc_count[doc_id] = per_doc_count.get(doc_id, 0) + 1

        if len(selected) >= top_n:
            break

    return selected


def apply_only_no_generation_chunk_filter_patch():
    """
    关键补丁。

    只替换 ChunkCritiqueJudge 的两个接口：
    - judge_chunks
    - select_generation_chunks

    其他完整基线流程全部保留：
    - qa_service.answer_one_query
    - QueryOptimizer
    - HybridRetriever
    - Reranker
    - Generator
    - 多子查询生成与融合
    """
    ChunkCritiqueJudge.judge_chunks = staticmethod(bypass_chunk_critique)
    ChunkCritiqueJudge.select_generation_chunks = staticmethod(select_rerank_chunks_directly)

    print("\n[Ablation Patch Applied]")
    print("Only removed generation chunk filtering.")
    print("Patched: ChunkCritiqueJudge.judge_chunks -> bypass_chunk_critique")
    print("Patched: ChunkCritiqueJudge.select_generation_chunks -> select_rerank_chunks_directly")
    print("Unchanged: answer_one_query / QueryOptimizer / HybridRetriever / Reranker / Generator\n")


# ============================================================
# 3. 答案指标：EM / F1
# ============================================================

def normalize_answer(s: str) -> str:
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
# 4. 数据读取
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
# 5. 加载统一索引
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
# 6. 结果提取函数
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

    for sub in output.get("sub_answers", []) or []:
        for c in sub.get("citations", []) or []:
            citations.append(c)

    return citations


def extract_ranked_titles_from_result(result: Dict[str, Any], qid: str) -> List[str]:
    """
    从 ranked_chunks_map 中提取 top 检索标题，用于 provenance Recall@K。
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
    提取 RAGAS 所需 contexts。
    优先从 ranked_chunks_map 取文本。
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


def count_generation_chunks(result: Dict[str, Any]) -> int:
    generation_chunks = result.get("generation_chunks", {}) or {}
    return sum(len(v or []) for v in generation_chunks.values())


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

    # 关键：先应用补丁，再调用 answer_one_query
    apply_only_no_generation_chunk_filter_patch()

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
        "kilt_nq_only_no_generation_chunk_filter_eval_details.jsonl",
    )

    summary_path = os.path.join(
        output_dir,
        "kilt_nq_only_no_generation_chunk_filter_eval_summary.json",
    )

    ragas_legacy_path = os.path.join(
        output_dir,
        "kilt_nq_only_no_generation_chunk_filter_ragas_legacy.jsonl",
    )

    ragas_new_path = os.path.join(
        output_dir,
        "kilt_nq_only_no_generation_chunk_filter_ragas_new.jsonl",
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

    query_type_stats: Dict[str, int] = {}
    generation_chunk_count_sum = 0

    with open(details_path, "w", encoding="utf-8") as fout, \
         open(ragas_legacy_path, "w", encoding="utf-8") as ragas_legacy_f, \
         open(ragas_new_path, "w", encoding="utf-8") as ragas_new_f:

        for idx, item in enumerate(
            tqdm(records, desc="Evaluating strict ablation: only no generation chunk filter")
        ):
            total += 1

            raw_qid = item.get("id", f"q_{idx}")
            qid = f"kilt_nq_only_no_gen_filter_{idx}_{raw_qid}"
            question = item.get("question", "")

            gold_answers = get_gold_answers(item)
            gold_titles = get_gold_titles(item)

            start = time.time()

            try:
                # ====================================================
                # 核心：仍然调用你的原始代码接口 answer_one_query
                # ====================================================
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

                one_query_type_stats = result.get("query_type_stats", {}) or {}
                for k, v in one_query_type_stats.items():
                    try:
                        query_type_stats[k] = query_type_stats.get(k, 0) + int(v)
                    except Exception:
                        query_type_stats[k] = query_type_stats.get(k, 0) + 1

                generation_chunk_count = count_generation_chunks(result)
                generation_chunk_count_sum += generation_chunk_count

                citations = extract_citations_from_result(result, qid)

                result_record = {
                    "id": raw_qid,
                    "eval_qid": qid,
                    "ablation": "only_no_generation_chunk_filter",
                    "description": (
                        "Strict baseline ablation: only bypass ChunkCritiqueJudge "
                        "generation chunk filtering; all other baseline functions are unchanged."
                    ),
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "pred_answer": pred_answer,
                    "em": em,
                    "f1": f1,
                    "recall_hit_at": recall_hit_record,
                    "latency_sec": latency,
                    "query_type_stats": one_query_type_stats,
                    "pred_titles_top10": pred_titles[:10],
                    "citations": citations,
                    "num_generation_chunks": generation_chunk_count,
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
                    "ablation": "only_no_generation_chunk_filter",
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "error": repr(e),
                    "latency_sec": latency,
                }

                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    valid_total = max(total - errors, 1)

    summary = {
        "task": "kilt_nq_200_single_index_ablation_only_no_generation_chunk_filter",
        "ablation": "only_no_generation_chunk_filter",
        "description": (
            "严格基线消融：只去掉 generation chunk 筛选，即绕过 "
            "ChunkCritiqueJudge.judge_chunks 和 ChunkCritiqueJudge.select_generation_chunks；"
            "保留 NO/simple/fuzzy/complex 判断、查询改写/分解、混合检索、rerank、生成和融合。"
        ),
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
        "query_type_stats": query_type_stats,
        "avg_generation_chunk_count": generation_chunk_count_sum / valid_total,
        "ragas_top_k": ragas_top_k,
        "max_context_chars": max_context_chars,
        "details_path": details_path,
        "summary_path": summary_path,
        "ragas_legacy_path": ragas_legacy_path,
        "ragas_new_path": ragas_new_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Strict Ablation: Only No Generation Chunk Filter ==========")
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
        default=r"D:\code\rag\FSR\eval\kilt_nq_only_no_generation_chunk_filter_outputs",
        help="消融实验输出目录。",
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