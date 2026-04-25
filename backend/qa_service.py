# qa_service.py

from typing import Dict, List

from generator import Generator
from pipeline_executor import run_parallel_retrieval_pipeline
from query_optimizer import QueryOptimizer
from qwen_client import QwenClient


def _merge_dict_max_score(dst: Dict, src: Dict):
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


def _merge_ranked_chunks(all_ranked_chunks: List[List[Dict]], top_k: int = 30):
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


def _merge_citations(sub_outputs: List[Dict], max_total: int = 8):
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


def _merge_candidates(sub_outputs: List[Dict], max_total: int = 12):
    merged = []
    for out in sub_outputs:
        for cand in out.get("candidates", []) or []:
            merged.append(cand)

    merged.sort(key=lambda x: float(x.get("final_score", 0.0)), reverse=True)
    return merged[:max_total]


def _merge_generation_risk(sub_outputs: List[Dict]):
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


def _build_fusion_prompt(original_query: str, sub_results: List[Dict]) -> str:
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

下面是围绕该问题拆分得到的多个子查询及其对应答案：
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


def _fuse_sub_answers(original_query: str, sub_results: List[Dict]) -> str:
    if not sub_results:
        return "根据当前证据，暂时无法生成最终答案。"

    if len(sub_results) == 1:
        return sub_results[0].get("answer", "") or "根据当前证据，暂时无法生成最终答案。"

    prompt = _build_fusion_prompt(original_query, sub_results)
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


async def _run_one_subquery_full_chain(
    parent_query_id: str,
    subquery_text: str,
    sub_idx: int,
    retriever,
    reranker,
    reranker_lock,
):
    """
    每个子查询完整走一遍原本单查询链路：
    检索 -> rerank -> generation chunk -> generate
    """
    sub_qid = f"{parent_query_id}__sub_{sub_idx}"
    sub_queries = {sub_qid: subquery_text}

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=sub_queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=1,
        subquery_workers=1,
    )

    gen_outputs = await Generator.run(
        sub_queries,
        generation_chunks,
        max_samples=1,
        max_concurrency=1,
    )

    sub_output = gen_outputs.get(sub_qid, {}) or {}

    return {
        "subquery": subquery_text,
        "qtype": query_type_stats,
        "doc_scores": rerank_results_with_scores.get(sub_qid, {}) or {},
        "generation_chunks": generation_chunks.get(sub_qid, []) or [],
        "ranked_chunks": ranked_chunks_map.get(sub_qid, []) or [],
        "answer": sub_output.get("answer", ""),
        "raw_text": sub_output.get("raw_text", ""),
        "reflections": sub_output.get("reflections", {}) or {},
        "citations": sub_output.get("citations", []) or [],
        "candidates": sub_output.get("candidates", []) or [],
        "generation_risk": sub_output.get("generation_risk", {}) or {},
        "best_score": max(
            [float(c.get("final_score", 0.0)) for c in (sub_output.get("candidates", []) or [])] or [0.0]
        ),
    }


async def answer_one_query(
    query_id,
    query_text,
    retriever,
    reranker,
    reranker_lock,
):
    """
    新版逻辑：

    第一次 LLM：
      QueryOptimizer.classify(query)
      输出：
        - NO
        - simple
        - fuzzy
        - complex

    第二次 LLM（只在需要时）：
      - fuzzy -> rewrite_query
      - complex -> decompose_query

    然后：
      - NO -> 直接回答
      - simple -> 原查询直接走单查询链路
      - fuzzy / complex -> 多子查询分别走单查询链路 + 最后融合
    """
    label = QueryOptimizer.classify(query_text)

    # ========== 不检索 ==========
    if label == "NO":
        direct_output = Generator.generate_direct_answer(query_text)
        return {
            "query_type_stats": {"no_retrieval": 1},
            "rerank_results_with_scores": {},
            "generation_chunks": {},
            "ranked_chunks_map": {},
            "gen_outputs": {
                query_id: direct_output
            },
        }

    # ========== simple：不再额外调第二次 LLM ==========
    if label == "simple":
        queries = {query_id: query_text}
        rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
            queries=queries,
            retriever=retriever,
            reranker=reranker,
            reranker_lock=reranker_lock,
            query_workers=1,
            subquery_workers=1,
        )

        gen_outputs = await Generator.run(
            queries,
            generation_chunks,
            max_samples=1,
            max_concurrency=1,
        )

        return {
            "query_type_stats": {"simple": 1},
            "rerank_results_with_scores": rerank_results_with_scores,
            "generation_chunks": generation_chunks,
            "ranked_chunks_map": ranked_chunks_map,
            "gen_outputs": gen_outputs,
        }

    # ========== fuzzy / complex：第二次 LLM 做查询扩展 ==========
    if label == "fuzzy":
        expanded_queries = QueryOptimizer.rewrite_query(query_text)
        parent_qtype = "fuzzy"
    elif label == "complex":
        expanded_queries = QueryOptimizer.decompose_query(query_text)
        parent_qtype = "complex"
    else:
        expanded_queries = [query_text]
        parent_qtype = "simple"

    if not expanded_queries:
        expanded_queries = [query_text]

    sub_results = []
    for idx, sub_q in enumerate(expanded_queries):
        sub_result = await _run_one_subquery_full_chain(
            parent_query_id=query_id,
            subquery_text=sub_q,
            sub_idx=idx,
            retriever=retriever,
            reranker=reranker,
            reranker_lock=reranker_lock,
        )
        sub_results.append(sub_result)

    valid_sub_results = [x for x in sub_results if x.get("answer", "").strip()]
    final_answer = _fuse_sub_answers(query_text, valid_sub_results)

    merged_citations = _merge_citations(valid_sub_results, max_total=8)
    merged_candidates = _merge_candidates(valid_sub_results, max_total=12)
    merged_generation_risk = _merge_generation_risk(valid_sub_results)

    merged_doc_scores = {}
    merged_generation_chunks = {}
    merged_ranked_chunks_map = {}

    all_ranked_chunks = []

    for idx, item in enumerate(sub_results):
        _merge_dict_max_score(merged_doc_scores, item.get("doc_scores", {}))
        merged_generation_chunks[f"{query_id}__sub_{idx}"] = item.get("generation_chunks", [])
        merged_ranked_chunks_map[f"{query_id}__sub_{idx}"] = item.get("ranked_chunks", [])
        all_ranked_chunks.append(item.get("ranked_chunks", []))

    merged_ranked_chunks_map[query_id] = _merge_ranked_chunks(all_ranked_chunks, top_k=30)

    final_reflection = {
        "retrieve": "Yes",
        "isrel": "Relevant" if valid_sub_results else "Irrelevant",
        "issup": "Fully" if valid_sub_results else "No Support",
        "isuse": 4 if valid_sub_results else 2,
        "followup_query": "None",
    }

    if valid_sub_results:
        supports = []
        for item in valid_sub_results:
            refl = item.get("reflections", {}) or {}
            supports.append(refl.get("issup", "Unknown"))

        if all(s == "No Support" for s in supports):
            final_reflection["issup"] = "No Support"
            final_reflection["isuse"] = 2
        elif any(s == "Partially" for s in supports) or any(s == "No Support" for s in supports):
            final_reflection["issup"] = "Partially"
            final_reflection["isuse"] = 3

    final_output = {
        "answer": final_answer,
        "raw_text": final_answer,
        "reflections": final_reflection,
        "citations": merged_citations,
        "candidates": merged_candidates,
        "generation_risk": merged_generation_risk,
        "sub_answers": valid_sub_results,
    }

    query_type_stats = {
        parent_qtype: 1,
        "subquery_count": len(expanded_queries),
    }

    return {
        "query_type_stats": query_type_stats,
        "rerank_results_with_scores": {
            query_id: merged_doc_scores
        },
        "generation_chunks": merged_generation_chunks,
        "ranked_chunks_map": merged_ranked_chunks_map,
        "gen_outputs": {
            query_id: final_output
        },
    }