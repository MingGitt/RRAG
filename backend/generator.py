import asyncio
import re

import aiohttp

from config import QW_HEADERS, MAX_CONCURRENCY, QWEN_MODEL, QWEN_URL


class Generator:
    @staticmethod
    def _format_source(meta, doc_id):
        title = meta.get("title", "") if meta else ""
        page_no = int(meta.get("page_no", 0)) if meta else 0
        source = title if title else doc_id
        if page_no > 0:
            source = f"{source} | page {page_no}"
        return source

    @staticmethod
    def _clean_excerpt(text, max_len=220):
        text = (text or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) <= max_len:
            return text
        return text[:max_len].rstrip() + "..."

    @staticmethod
    def build_citations(chunks, max_citations=3, excerpt_len=220):
        citations = []
        for i, item in enumerate(chunks[:max_citations], start=1):
            meta = item.get("meta", {})
            source = Generator._format_source(meta, item.get("doc_id", "unknown_doc"))
            full_text = item.get("text", "") or ""
            excerpt = Generator._clean_excerpt(full_text, max_len=excerpt_len)

            citations.append({
                "index": i,
                "source": source,
                "excerpt": excerpt,
                "doc_id": item.get("doc_id", ""),
                "page_no": int(meta.get("page_no", 0) or 0),
                "text": full_text,
            })
        return citations

    @staticmethod
    def build_context_from_citations(citations):
        blocks = []
        for item in citations:
            idx = item["index"]
            source = item["source"]
            text = item["text"]
            blocks.append(f"[{idx}] 来源: {source}\n{text}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_prompt(query, citations):
        context = Generator.build_context_from_citations(citations)
        return f"""你是一个文档问答助手。请严格依据给定证据回答问题，不要编造。

问题：
{query}

证据：
{context}

回答要求：
1. 只能根据给定证据回答；
2. 如果证据不足，请明确写“证据不足”；
3. 回答尽量准确、简洁、有条理；
4. 每一个关键结论后面都要标注证据编号，例如 [1] 或 [2]；
5. 如果一句话同时由多个证据支持，可以写成 [1][2]；
6. 不要使用未提供的编号；
7. 不要在回答末尾单独列“参考文献”，只在正文句子后标注编号；
8. 不要编造证据中不存在的事实。

请直接输出最终答案正文。
"""

    @staticmethod
    def build_judge_prompt(query, answer, citations):
        context = Generator.build_context_from_citations(citations)
        return f"""请你作为评审，判断下面回答是否忠实于证据，是否回答了问题。

问题：
{query}

证据：
{context}

回答：
{answer}

请输出 JSON，格式如下：
{{
  "faithfulness": 0-10,
  "relevance": 0-10,
  "citation_correctness": 0-10,
  "overall": 0-10,
  "comment": "简短评价"
}}
"""

    @staticmethod
    def _extract_answer(data):
        try:
            return data["output"]["text"].strip()
        except Exception:
            pass

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return ""

    @staticmethod
    async def call_llm(session, prompt):
        payload = {
            "model": QWEN_MODEL,
            "input": {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            "parameters": {
                "temperature": 0.2,
                "max_tokens": 1024,
            }
        }

        async with session.post(QWEN_URL, headers=QW_HEADERS, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return Generator._extract_answer(data)

    @staticmethod
    def postprocess_answer(answer: str, available_indices):
        if not answer:
            return answer

        valid_set = set(int(x) for x in available_indices)

        def replace_ref(match):
            num = int(match.group(1))
            return f"[{num}]" if num in valid_set else ""

        answer = re.sub(r"\[(\d+)\]", replace_ref, answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        return answer

    @staticmethod
    async def run_single(session, qid, query, chunks):
        citations = Generator.build_citations(chunks, max_citations=3, excerpt_len=220)

        answer_prompt = Generator.build_prompt(query, citations)
        answer = await Generator.call_llm(session, answer_prompt)
        answer = Generator.postprocess_answer(answer, [c["index"] for c in citations])

        judge_prompt = Generator.build_judge_prompt(query, answer, citations)
        judge = await Generator.call_llm(session, judge_prompt)

        return qid, {
            "answer": answer,
            "judge": judge,
            "chunks": chunks,
            "citations": citations,
        }

    @staticmethod
    async def run(queries, generation_chunks, max_samples=1, max_concurrency=None):
        if max_concurrency is None:
            max_concurrency = MAX_CONCURRENCY

        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)

        async def limited_run(session, qid, query, chunks):
            async with semaphore:
                return await Generator.run_single(session, qid, query, chunks)

        selected_items = list(queries.items())[:max_samples]

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = []
            for qid, query in selected_items:
                chunks = generation_chunks.get(qid, [])
                if not chunks:
                    continue
                tasks.append(limited_run(session, qid, query, chunks))

            outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in outputs:
            if isinstance(output, Exception):
                continue
            qid, result = output
            results[qid] = result

        return results