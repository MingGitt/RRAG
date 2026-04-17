import asyncio
import json

import aiohttp

from config import HEADERS, MAX_CONCURRENCY, QWEN_MODEL, QWEN_URL


class Generator:
    @staticmethod
    def build_context(chunks):
        """
        chunks: List[{
            "doc_id": ...,
            "text": ...,
            "score": ...,
            "meta": {...}
        }]
        """
        blocks = []
        for i, item in enumerate(chunks, start=1):
            text = item["text"]
            meta = item.get("meta", {})
            title = meta.get("title", "")
            page_no = meta.get("page_no", 0)

            source = title if title else item.get("doc_id", "unknown_doc")
            if page_no and int(page_no) > 0:
                source = f"{source} | page {page_no}"

            blocks.append(f"[证据{i}] 来源: {source}\n{text}")

        return "\n\n".join(blocks)

    @staticmethod
    def build_prompt(query, chunks):
        context = Generator.build_context(chunks)
        return f"""你是一个文档问答助手。请严格依据给定证据回答问题，不要编造。

问题：
{query}

证据：
{context}

要求：
1. 只根据证据回答；
2. 如果证据不足，请明确说“证据不足”；
3. 回答尽量准确、简洁、有条理。
"""

    @staticmethod
    def build_judge_prompt(query, answer, chunks):
        context = Generator.build_context(chunks)
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
  "overall": 0-10,
  "comment": "简短评价"
}}
"""

    @staticmethod
    async def call_llm(session, prompt):
        payload = {
            "model": QWEN_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
        }

        async with session.post(QWEN_URL, headers=HEADERS, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]

    @staticmethod
    async def run_single(session, qid, query, chunks):
        answer_prompt = Generator.build_prompt(query, chunks)
        answer = await Generator.call_llm(session, answer_prompt)

        judge_prompt = Generator.build_judge_prompt(query, answer, chunks)
        judge = await Generator.call_llm(session, judge_prompt)

        return qid, {
            "answer": answer,
            "judge": judge,
            "chunks": chunks,
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