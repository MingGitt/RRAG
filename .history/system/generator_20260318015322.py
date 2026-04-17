import json
import re
import asyncio
from qwen_client import QwenClient
from deepseek_client import DeepSeekClient


class Generator:

    @staticmethod
    def parse_judge(text):
        if not text:
            return None

        candidates = [text]

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.append(match.group(0))

        for item in candidates:
            try:
                data = json.loads(item)
                score = int(data.get("score"))
                if 1 <= score <= 5:
                    return {
                        "label": data.get("label", ""),
                        "score": score,
                        "reason": data.get("reason", "")
                    }
            except Exception:
                continue

        return None

    @staticmethod
    async def process_single_query(qid, query, generation_chunks, semaphore):
        context_chunks = generation_chunks.get(qid, [])
        context = "\n\n".join(context_chunks)
        if not context.strip():
            return None

        answer_prompt = f"""
You are a scientific fact-checking assistant.

Claim:
{query}

Evidence:
{context}

Task:
Based only on the evidence, output one label:
- supported
- refuted
- not_enough_info

Then give one short explanation.
Return plain text.
""".strip()

        async with semaphore:
            answer = await QwenClient.async_call(
                [{"role": "user", "content": answer_prompt}],
                max_tokens=1024
            )

        if not answer:
            return None

        judge_prompt = f"""
You are grading whether an answer is grounded in the provided evidence.

Claim:
{query}

Evidence:
{context}

Answer:
{answer}

Scoring rubric:
5 = correct, evidence-grounded, complete
4 = mostly correct, minor omission
3 = partially correct or somewhat weak grounding
2 = mostly incorrect or weakly grounded
1 = incorrect or unsupported

Return JSON only:
{{"label":"supported|refuted|not_enough_info","score":1,"reason":"..."}}
""".strip()

        async with semaphore:
            judge = await DeepSeekClient.async_call(
                [{"role": "user", "content": judge_prompt}],
                max_tokens=1024
            )

        parsed = Generator.parse_judge(judge)

        return {
            "qid": qid,
            "answer": answer,
            "judge": parsed,
            "context": context_chunks
        }

    @staticmethod
    async def run(queries, generation_chunks, max_samples=100, max_concurrency=4):
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