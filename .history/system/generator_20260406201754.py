import asyncio
import json
import re

import aiohttp

from config import HEADERS, MAX_CONCURRENCY, QWEN_MODEL, QWEN_URL


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
            source = Generator._format_source(meta, item.get("doc_id", "unknown_doc"))
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
3. 回答尽量准确、简洁、有条理；
4. 不要输出“根据证据1/证据2”之类字样；
5. 不要自行添加证据中不存在的事实。
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
    def build_citations(chunks, max_citations=3, excerpt_len=220):
        """
        给最终答案展示用：
        [
          {
            "index": 1,
            "source": "xxx.pdf | page 3",
            "excerpt": "...",
            "doc_id": "...",
            "page_no": 3
          }
        ]
        """
        citations = []
        for i, item in enumerate(chunks[:max_citations], start=1):
            meta = item.get("meta", {})
            source = Generator._format_source(meta, item.get("doc_id", "unknown_doc"))
            excerpt = Generator._clean_excerpt(item.get("text", ""), max_len=excerpt_len)
            citations.append({
                "index": i,
                "source": source,
                "excerpt": excerpt,
                "doc_id": item.get("doc_id", ""),
                "page_no": int(meta.get("page_no", 0) or 0),
            })
        return citations

    @staticmethod
    async def run_single(session, qid, query, chunks):
        answer_prompt = Generator.build_prompt(query, chunks)
        answer = await Generator.call_llm(session, answer_prompt)

        judge_prompt = Generator.build_judge_prompt(query, answer, chunks)
        judge = await Generator.call_llm(session, judge_prompt)

        citations = Generator.build_citations(chunks, max_citations=3, excerpt_len=220)

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