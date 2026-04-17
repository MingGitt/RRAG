import re
from qwen_client import QwenClient

class Generator:

    @staticmethod
    def parse_score(text):
        if not text:
            return None
        m = re.search(r"[1-5]", text)
        return int(m.group()) if m else None

    @staticmethod
    def run(queries, generation_chunks, qrels):

        scores = []

        for qid in list(queries.keys())[:100]:

            context = "\n\n".join(generation_chunks[qid])

            prompt = f"""
Claim:
{queries[qid]}

Evidence:
{context}

Based on the evidence, answer whether the claim is supported.
"""

            answer = QwenClient.call([{"role": "user", "content": prompt}])

            judge_prompt = f"""
Question:
{queries[qid]}

Answer:
{answer}

Give a score from 1 to 5.
"""

            score = QwenClient.call(
                [{"role": "user", "content": judge_prompt}],
                max_tokens=32
            )

            parsed = Generator.parse_score(score)
            if parsed:
                scores.append(parsed)

        return scores