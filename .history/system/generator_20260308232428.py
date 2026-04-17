import json
from qwen_client import QwenClient


class Generator:

    @staticmethod
    def parse_judge(text):
        if not text:
            return None

        try:
            data = json.loads(text)
            score = int(data.get("score"))
            if 1 <= score <= 5:
                return {
                    "label": data.get("label", ""),
                    "score": score,
                    "reason": data.get("reason", "")
                }
        except Exception:
            return None

        return None

    @staticmethod
    def run(queries, generation_chunks, qrels):
        scores = []

        for qid in list(queries.keys())[:100]:
            context = "\n\n".join(generation_chunks.get(qid, []))
            if not context.strip():
                continue

            answer_prompt = f"""
You are a scientific fact-checking assistant.

Claim:
{queries[qid]}

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

            answer = QwenClient.call(
                [{"role": "user", "content": answer_prompt}],
                max_tokens=256
            )

            judge_prompt = f"""
You are grading whether an answer is grounded in the provided evidence.

Claim:
{queries[qid]}

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

            judge = QwenClient.call(
                [{"role": "user", "content": judge_prompt}],
                max_tokens=128
            )

            parsed = Generator.parse_judge(judge)
            if parsed is not None:
                scores.append(parsed["score"])

        return scores