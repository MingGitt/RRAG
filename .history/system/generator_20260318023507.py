import asyncio
from qwen_client import QwenClient


class Generator:
    """
    生成器：
    - 输入 query + generation_chunks
    - 基于证据文本块生成对查询的回答
    - 不再做相关性判断/打分
    """

    @staticmethod
    def build_answer_prompt(query, context_chunks):
        context = "\n\n".join(
            f"[证据块 {i + 1}]\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ).strip()

        prompt = f"""
你是一名严谨的智能问答助手。请基于给定的证据文本块，直接回答用户查询。

要求：
1. 只依据提供的证据作答，不要引入证据之外的事实。
2. 回答要综合多个证据块的信息，尽量完整、准确、连贯。
3. 若证据足以回答问题，请给出明确答案，并用简洁自然的语言组织。
4. 若证据存在不一致之处，请指出不一致点并给出谨慎结论。
5. 若证据不足以完整回答，请明确说明“根据当前证据无法完全确定”，并说明已知信息。
6. 不要评价查询与证据是否相关，直接回答查询本身。

用户查询：
{query}

证据文本块：
{context}

请直接输出最终回答：
""".strip()

        return prompt

    @staticmethod
    async def process_single_query(qid, query, generation_chunks, semaphore):
        context_chunks = generation_chunks.get(qid, [])
        if not context_chunks:
            return {
                "qid": qid,
                "answer": "未检索到可用于生成回答的证据文本块，因此暂时无法回答该查询。",
                "context": []
            }

        prompt = Generator.build_answer_prompt(query, context_chunks)

        async with semaphore:
            answer = await QwenClient.async_call(
                [{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=512
            )

        if not answer:
            answer = "生成失败：模型未返回有效内容。"

        return {
            "qid": qid,
            "answer": answer,
            "context": context_chunks
        }

    @staticmethod
    async def run(queries, generation_chunks, max_samples=100, max_concurrency=4):
        """
        返回：
        {
            qid: {
                "qid": qid,
                "answer": "...",
                "context": [...]
            }
        }
        """
        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)

        items = list(queries.items())[:max_samples]
        if not items:
            return results

        tasks = [
            Generator.process_single_query(qid, query, generation_chunks, semaphore)
            for qid, query in items
        ]

        outputs = await asyncio.gather(*tasks, return_exceptions=True)

        for output in outputs:
            if isinstance(output, Exception):
                print(f"[WARN] generation task failed: {output}")
                continue
            if output is not None:
                results[output["qid"]] = output

        return results