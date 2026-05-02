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
from query_optimizer import QueryOptimizer
from qwen_client import QwenClient

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
# 5. 只保留查询类型判断：simple / fuzzy / complex
# ============================================================

def normalize_query_type(label: str) -> str:
    label = str(label or "").strip().lower()

    if "complex" in label:
        return "complex"
    if "fuzzy" in label:
        return "fuzzy"
    if "simple" in label:
        return "simple"

    # 兜底：无法解析时按 simple 处理，保证进入检索
    return "simple"


def classify_query_type_only(query: str) -> Tuple[str, str]:
    """
    消融实验专用分类器：

    只允许输出：
        simple / fuzzy / complex

    不允许输出：
        YES / NO
        NO
        no_retrieval

    也就是说，所有问题都默认需要检索，只判断采用哪种查询处理方式。
    """

    prompt = f"""
你是一个 RAG 系统的查询类型分类器。

现在不需要判断是否检索，因为所有问题都会进入检索流程。

你的任务：判断用户问题属于以下哪一种查询类型。

只能输出一个标签：
simple
fuzzy
complex

分类标准：
1. simple：问题单一明确，可以直接用原始问题检索。
2. fuzzy：问题表述模糊、口语化、同义表达较多，适合改写成多个检索查询。
3. complex：问题包含多个子问题、多个约束、比较关系或组合需求，适合拆分成多个子查询。

注意：
- 不要输出 YES 或 NO。
- 不要输出解释。
- 不要输出 JSON。
- 只能输出 simple / fuzzy / complex 三者之一。

用户问题：
{query}
""".strip()

    raw = QwenClient.call(
        [{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=16,
    )

    raw_text = (raw or "").strip()
    return normalize_query_type(raw_text), raw_text


# ============================================================
# 6. 原始 query 检索链路：不再调用 QueryOptimizer.expand()
# ============================================================

def run_raw_query_retrieval_chain(
    qid: str,
    query: str,
    retriever,
    reranker,
    reranker_lock,
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    单个查询的检索链路。

    注意：
    这里不调用 run_parallel_retrieval_pipeline，
    因为 pipeline_executor.process_single_query() 内部会调用 QueryOptimizer.expand()。
    本消融实验要自己控制 simple/fuzzy/complex 逻辑。
    """

    retrieved = retriever.retrieve(
        query,
        top_k_chunks=TOP_K_CHUNKS,
        dense_k=DENSE_CANDIDATE_K,
        bm25_k=BM25_CANDIDATE_K,
    )

    if not retrieved:
        return {}, [], [], []

    texts = []
    candidate_indices = []
    candidate_doc_map = {}
    candidate_meta_map = {}

    for idx, text, doc_id, score, meta in retrieved:
        texts.append(text)
        candidate_indices.append(idx)
        candidate_doc_map[idx] = doc_id
        candidate_meta_map[idx] = meta or {}

    metas = [candidate_meta_map[idx] for idx in candidate_indices]

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

    doc_level_for_eval = [
        (item["doc_id"], item["text"], item["score"])
        for item in ranked_chunks
    ]

    doc_scores = RetrievalEvaluator.aggregate_doc_scores(
        doc_level_for_eval,
        mode=DOC_AGG_MODE,
    )

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


# ============================================================
# 7. 多子查询融合工具函数
# ============================================================

def merge_dict_max_score(dst: Dict[str, float], src: Dict[str, float]) -> Dict[str, float]:
    if not src:
        return dst

    for k, v in src.items():
        try:
            score = float(v)
        except Exception:
            score = 0.0

        if k not in dst or score > float(dst[k]):
            dst[k] = score

    return dst


def merge_ranked_chunks(all_ranked_chunks: List[List[Dict[str, Any]]], top_k: int = 30) -> List[Dict[str, Any]]:
    merged = []
    seen = set()

    for ranked_chunks in all_ranked_chunks:
        for item in ranked_chunks or []:
            key = (
                str(item.get("doc_id", "")),
                str(item.get("chunk_idx", "")),
                (item.get("text", "") or "")[:120],
            )

            if key in seen:
                continue

            seen.add(key)
            merged.append(item)

    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]


def merge_citations(sub_outputs: List[Dict[str, Any]], max_total: int = 8) -> List[Dict[str, Any]]:
    merged = []
    seen = set()

    for out in sub_outputs:
        for c in out.get("citations", []) or []:
            key = (c.get("source", ""), c.get("excerpt", ""))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)

    return merged[:max_total]


def merge_candidates(sub_outputs: List[Dict[str, Any]], max_total: int = 12) -> List[Dict[str, Any]]:
    merged = []

    for out in sub_outputs:
        for cand in out.get("candidates", []) or []:
            merged.append(cand)

    merged.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return merged[:max_total]


def merge_generation_risk(sub_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not sub_outputs:
        return {
            "risk_score": 0.8,
            "risk_level": "high",
            "confidence": 0.3,
            "reasons": ["没有有效子查询答案"],
        }

    best = None

    for out in sub_outputs:
        risk = out.get("generation_risk", {}) or {}
        score = float(risk.get("risk_score", 0.0) or 0.0)

        if best is None or score > float(best.get("risk_score", 0.0)):
            best = risk

    if not best:
        return {
            "risk_score": 0.5,
            "risk_level": "medium",
            "confidence": 0.5,
            "reasons": ["未获得有效风险评估，使用默认值"],
        }

    return best


def build_fusion_prompt(original_query: str, sub_results: List[Dict[str, Any]]) -> str:
    blocks = []

    for i, item in enumerate(sub_results, start=1):
        sub_q = item.get("subquery", "")
        sub_a = item.get("answer", "")
        blocks.append(
            f"子查询{i}：{sub_q}\n"
            f"子答案{i}：{sub_a}"
        )

    joined = "\n\n".join(blocks)

    prompt = f"""
你是一个文档问答融合助手。

用户原始问题：
{original_query}

下面是围绕该问题得到的多个子查询及其对应答案：
{joined}

请输出最终融合答案，要求：
1. 直接回答用户原始问题。
2. 对多个子答案去重、整合、补充，不要重复表述。
3. 如果多个子答案互补，就合并成一段连贯回答。
4. 如果存在不确定之处，要明确说明“根据当前证据”或“从现有信息看”。
5. 不要输出“子查询”“子答案”“融合结果”等过程性词语。
6. 只输出最终给用户看的答案正文。
""".strip()

    return prompt


def fuse_sub_answers(original_query: str, sub_results: List[Dict[str, Any]]) -> str:
    if not sub_results:
        return "根据当前证据，暂时无法生成最终答案。"

    if len(sub_results) == 1:
        return sub_results[0].get("answer", "") or "根据当前证据，暂时无法生成最终答案。"

    prompt = build_fusion_prompt(original_query, sub_results)

    fused = QwenClient.call(
        [{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )

    fused = (fused or "").strip()

    if fused:
        return fused

    sorted_subs = sorted(
        sub_results,
        key=lambda x: float(x.get("best_score", 0.0)),
        reverse=True,
    )

    return sorted_subs[0].get("answer", "") if sorted_subs else "根据当前证据，暂时无法生成最终答案。"


# ============================================================
# 8. 按 simple / fuzzy / complex 类型执行
# ============================================================

async def run_one_subquery_chain(
    parent_query_id: str,
    subquery_text: str,
    sub_idx: int,
    retriever,
    reranker,
    reranker_lock,
) -> Dict[str, Any]:
    sub_qid = f"{parent_query_id}__sub_{sub_idx}"

    doc_scores, generation_chunks, ranked_chunks, critiqued_chunks = (
        run_raw_query_retrieval_chain(
            qid=sub_qid,
            query=subquery_text,
            retriever=retriever,
            reranker=reranker,
            reranker_lock=reranker_lock,
        )
    )

    generation_chunks_map = {}

    if generation_chunks:
        generation_chunks_map[sub_qid] = generation_chunks

    if generation_chunks:
        gen_outputs = await Generator.run(
            {sub_qid: subquery_text},
            generation_chunks_map,
            max_samples=1,
            max_concurrency=1,
        )
    else:
        gen_outputs = {}

    sub_output = gen_outputs.get(sub_qid, {}) or {}

    return {
        "sub_qid": sub_qid,
        "subquery": subquery_text,
        "doc_scores": doc_scores or {},
        "generation_chunks": generation_chunks or [],
        "ranked_chunks": ranked_chunks or [],
        "critiqued_chunks": critiqued_chunks or [],
        "answer": sub_output.get("answer", ""),
        "raw_text": sub_output.get("raw_text", ""),
        "reflections": sub_output.get("reflections", {}) or {},
        "citations": sub_output.get("citations", []) or [],
        "candidates": sub_output.get("candidates", []) or [],
        "generation_risk": sub_output.get("generation_risk", {}) or {},
        "best_score": max(
            [float(c.get("final_score", 0.0)) for c in (sub_output.get("candidates", []) or [])]
            or [0.0]
        ),
    }


async def answer_one_query_type_only(
    query_id: str,
    query_text: str,
    retriever,
    reranker,
    reranker_lock,
) -> Dict[str, Any]:
    """
    核心消融逻辑：

    1. 只做 simple / fuzzy / complex 三分类
    2. 不做 YES / NO 是否检索判断
    3. 所有问题都进入检索
    4. simple：原始 query 检索生成
    5. fuzzy：QueryOptimizer.rewrite_query 后多个查询分别检索生成，再融合
    6. complex：QueryOptimizer.decompose_query 后多个子查询分别检索生成，再融合
    """

    query_type, query_type_raw = classify_query_type_only(query_text)

    # ========== simple：原始 query 直接检索 ==========
    if query_type == "simple":
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

        if generation_chunks:
            gen_outputs = await Generator.run(
                {query_id: query_text},
                generation_chunks_map,
                max_samples=1,
                max_concurrency=1,
            )
        else:
            gen_outputs = {
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
            }

        return {
            "ablation": "query_type_only",
            "query_type": query_type,
            "query_type_raw": query_type_raw,
            "expanded_queries": [query_text],
            "query_type_stats": {"simple": 1},
            "rerank_results_with_scores": {
                query_id: doc_scores or {}
            },
            "generation_chunks": generation_chunks_map,
            "ranked_chunks_map": ranked_chunks_map,
            "critiqued_chunks_map": critiqued_chunks_map,
            "gen_outputs": gen_outputs,
        }

    # ========== fuzzy / complex：保留查询类型对应的改写或分解 ==========
    if query_type == "fuzzy":
        expanded_queries = QueryOptimizer.rewrite_query(query_text)
    elif query_type == "complex":
        expanded_queries = QueryOptimizer.decompose_query(query_text)
    else:
        query_type = "simple"
        expanded_queries = [query_text]

    if not expanded_queries:
        expanded_queries = [query_text]

    sub_results = []

    for idx, sub_q in enumerate(expanded_queries):
        sub_result = await run_one_subquery_chain(
            parent_query_id=query_id,
            subquery_text=sub_q,
            sub_idx=idx,
            retriever=retriever,
            reranker=reranker,
            reranker_lock=reranker_lock,
        )
        sub_results.append(sub_result)

    valid_sub_results = [
        x for x in sub_results
        if str(x.get("answer", "")).strip()
    ]

    final_answer = fuse_sub_answers(query_text, valid_sub_results)

    merged_citations = merge_citations(valid_sub_results, max_total=8)
    merged_candidates = merge_candidates(valid_sub_results, max_total=12)
    merged_generation_risk = merge_generation_risk(valid_sub_results)

    merged_doc_scores = {}
    merged_generation_chunks = {}
    merged_ranked_chunks_map = {}
    merged_critiqued_chunks_map = {}

    all_ranked_chunks = []

    for idx, item in enumerate(sub_results):
        merge_dict_max_score(merged_doc_scores, item.get("doc_scores", {}))

        sub_key = f"{query_id}__sub_{idx}"
        merged_generation_chunks[sub_key] = item.get("generation_chunks", [])
        merged_ranked_chunks_map[sub_key] = item.get("ranked_chunks", [])
        merged_critiqued_chunks_map[sub_key] = item.get("critiqued_chunks", [])

        all_ranked_chunks.append(item.get("ranked_chunks", []))

    merged_ranked_chunks_map[query_id] = merge_ranked_chunks(
        all_ranked_chunks,
        top_k=30,
    )

    final_reflection = {
        "retrieve": "Yes",
        "isrel": "Relevant" if valid_sub_results else "Irrelevant",
        "issup": "Fully" if valid_sub_results else "No Support",
        "isuse": 4 if valid_sub_results else 2,
        "followup_query": "None",
    }

    final_output = {
        "answer": final_answer,
        "raw_text": final_answer,
        "reflections": final_reflection,
        "citations": merged_citations,
        "candidates": merged_candidates,
        "generation_risk": merged_generation_risk,
        "sub_answers": valid_sub_results,
    }

    return {
        "ablation": "query_type_only",
        "query_type": query_type,
        "query_type_raw": query_type_raw,
        "expanded_queries": expanded_queries,
        "query_type_stats": {
            query_type: 1,
            "subquery_count": len(expanded_queries),
        },
        "rerank_results_with_scores": {
            query_id: merged_doc_scores or {}
        },
        "generation_chunks": merged_generation_chunks,
        "ranked_chunks_map": merged_ranked_chunks_map,
        "critiqued_chunks_map": merged_critiqued_chunks_map,
        "gen_outputs": {
            query_id: final_output
        },
    }


# ============================================================
# 9. 结果提取函数
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


# ============================================================
# 10. 主评测逻辑
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
        "kilt_nq_query_type_only_eval_details.jsonl",
    )

    summary_path = os.path.join(
        output_dir,
        "kilt_nq_query_type_only_eval_summary.json",
    )

    ragas_legacy_path = os.path.join(
        output_dir,
        "kilt_nq_query_type_only_ragas_legacy.jsonl",
    )

    ragas_new_path = os.path.join(
        output_dir,
        "kilt_nq_query_type_only_ragas_new.jsonl",
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
    expanded_query_count_sum = 0

    with open(details_path, "w", encoding="utf-8") as fout, \
         open(ragas_legacy_path, "w", encoding="utf-8") as ragas_legacy_f, \
         open(ragas_new_path, "w", encoding="utf-8") as ragas_new_f:

        for idx, item in enumerate(tqdm(records, desc="Evaluating KILT NQ query type only")):
            total += 1

            raw_qid = item.get("id", f"q_{idx}")
            qid = f"kilt_nq_query_type_only_{idx}_{raw_qid}"
            question = item.get("question", "")

            gold_answers = get_gold_answers(item)
            gold_titles = get_gold_titles(item)

            start = time.time()

            try:
                result = await answer_one_query_type_only(
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

                query_type = result.get("query_type", "unknown")
                query_type_stats[query_type] += 1

                expanded_queries = result.get("expanded_queries", []) or []
                expanded_query_count_sum += len(expanded_queries)

                citations = extract_citations_from_result(result, qid)

                result_record = {
                    "id": raw_qid,
                    "eval_qid": qid,
                    "ablation": "query_type_only_no_retrieve_decision",
                    "question": question,
                    "query_type": query_type,
                    "query_type_raw": result.get("query_type_raw", ""),
                    "expanded_queries": expanded_queries,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "pred_answer": pred_answer,
                    "em": em,
                    "f1": f1,
                    "recall_hit_at": recall_hit_record,
                    "latency_sec": latency,
                    "pred_titles_top10": pred_titles[:10],
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
                    "ablation": "query_type_only_no_retrieve_decision",
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "error": repr(e),
                    "latency_sec": latency,
                }

                fout.write(json.dumps(error_record, ensure_ascii=False) + "\n")

    valid_total = max(total - errors, 1)

    summary = {
        "task": "kilt_nq_200_single_index_ablation_query_type_only",
        "ablation": "query_type_only_no_retrieve_decision",
        "description": "取消是否检索判断，不再输出 YES/NO，所有问题都进入检索；只保留 simple/fuzzy/complex 查询类型判断，并统计平均端到端响应时间。",
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
        "avg_expanded_query_count": expanded_query_count_sum / valid_total,
        "ragas_top_k": ragas_top_k,
        "max_context_chars": max_context_chars,
        "details_path": details_path,
        "summary_path": summary_path,
        "ragas_legacy_path": ragas_legacy_path,
        "ragas_new_path": ragas_new_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== KILT NQ Ablation: Query Type Only Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved files:")
    print(f"details: {details_path}")
    print(f"summary: {summary_path}")
    print(f"ragas legacy: {ragas_legacy_path}")
    print(f"ragas new: {ragas_new_path}")


# ============================================================
# 11. 命令行入口
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
        default=r"D:\code\rag\FSR\eval\kilt_nq_query_type_only_outputs",
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