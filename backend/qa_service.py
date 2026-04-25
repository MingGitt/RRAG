# qa_service.py
import asyncio

from config import SHOW_TOP_K, SUBQUERY_WORKERS
from generator import Generator
from pipeline_executor import run_parallel_retrieval_pipeline
from retrieval_judge import RetrievalJudge


def format_source(meta: dict, doc_id: str):
    title = meta.get("title", "") if meta else ""
    page_no = int(meta.get("page_no", 0)) if meta else 0
    source = title if title else doc_id
    if page_no > 0:
        source = f"{source} | page {page_no}"
    return source


def print_citations(citations):
    if not citations:
        print("无")
        return

    print("\n参考来源：")
    for item in citations:
        print(f'[{item["index"]}] {item["source"]}')

    print("\n引用片段：")
    for item in citations:
        print(f'\n[{item["index"]}] {item["source"]}')
        print(f'“{item["excerpt"]}”')


def answer_one_query(
    query_id,
    query_text,
    retriever,
    reranker,
    reranker_lock,
):
    """
    新版完整问答流程：
    Step 1: qwen-plus 检索决策
      - No: 直接 qwen-plus 回答
      - Yes: 进入原检索链路，再由新版 Generator 完成 Step 3~6
    """
    queries = {query_id: query_text}

    # Step 1: 检索决策
    retrieve_decision = RetrievalJudge.decide_retrieval(
        user_query=query_text,
        history_text="",
    )

    if retrieve_decision == "No":
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

    # Step 2: 执行检索
    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
        reranker=reranker,
        reranker_lock=reranker_lock,
        query_workers=1,
        subquery_workers=SUBQUERY_WORKERS,
    )

    # 如果 generation_chunks 为空，兜底用 ranked_chunks_map 前几条
    if query_id not in generation_chunks or not generation_chunks.get(query_id):
        ranked = ranked_chunks_map.get(query_id, [])
        if ranked:
            generation_chunks[query_id] = ranked[:5]

    # Step 3~6: relevance -> answer -> support -> hard constraint
    gen_outputs = asyncio.run(
        Generator.run(
            queries,
            generation_chunks,
            max_samples=1,
            max_concurrency=1
        )
    )

    return {
        "query_type_stats": query_type_stats,
        "rerank_results_with_scores": rerank_results_with_scores,
        "generation_chunks": generation_chunks,
        "ranked_chunks_map": ranked_chunks_map,
        "gen_outputs": gen_outputs,
    }


def print_single_answer_result(query_id, query_text, result):
    ranked_chunks = result["ranked_chunks_map"].get(query_id, [])
    gen_output = result["gen_outputs"].get(query_id)

    print("\n========== Query ==========")
    print(query_text)

    print("\n========== Query Routing Stats ==========")
    print(result.get("query_type_stats", {}))

    print(f"\n========== Top-{SHOW_TOP_K} Retrieved Chunks ==========")
    if not ranked_chunks:
        print("没有检索到相关内容。")
    else:
        for rank, item in enumerate(ranked_chunks[:SHOW_TOP_K], start=1):
            chunk_idx = item["chunk_idx"]
            text = item["text"]
            doc_id = item["doc_id"]
            score = item["score"]
            meta = item.get("meta", {})
            source = format_source(meta, doc_id)

            print(f"\n[Rank {rank}] chunk_idx={chunk_idx} doc_id={doc_id} score={score:.6f}")
            print(f"Source: {source}")
            print(text[:1000])

    print("\n========== Answer ==========")
    if not gen_output:
        print("未生成答案。")
        return

    print(gen_output["answer"])

    citations = gen_output.get("citations", [])
    print_citations(citations)

    print("\n========== Reflections ==========")
    print(gen_output.get("reflections", {}))

    print("\n========== Raw Output ==========")
    print(gen_output.get("raw_text", ""))


def interactive_local_qa(retriever, reranker, reranker_lock):
    print("\n===== 本地文档问答模式 =====")
    print("输入你的问题后回车即可提问；输入 exit / quit / q 退出。")

    qid_counter = 1

    while True:
        query = input("\n请输入问题：").strip()

        if not query:
            print("问题不能为空，请重新输入。")
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("已退出问答模式。")
            break

        qid = f"user_query_{qid_counter}"
        qid_counter += 1

        try:
            result = answer_one_query(
                qid,
                query,
                retriever,
                reranker,
                reranker_lock,
            )
            print_single_answer_result(qid, query, result)
        except Exception as e:
            print(f"\n[ERROR] 本次问答失败：{e}")