# hotpotqa_ragas_eval.py
# 放置位置：D:\code\rag\FSR\eval\hotpotqa_ragas_eval.py
#
# 功能：
# 读取各个 HotpotQA 实验输出的 details.jsonl，
# 使用 Ragas + DashScope/Qwen 评估：
# faithfulness / context_precision / context_recall
#
# 注意：
# 这里不跑 answer_relevancy，因为它通常依赖 embeddings，
# DashScope embedding 在部分 Ragas/langchain 版本中容易报：
# Value error, contents is neither str nor list of str.: input.contents

import os
import json
import argparse
from datetime import datetime

from datasets import Dataset

from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def extract_answer(row):
    """
    兼容不同实验输出：
    1. gen_output.answer
    2. answer_metrics.prediction
    """
    gen_output = row.get("gen_output", {})
    if isinstance(gen_output, dict):
        ans = gen_output.get("answer", "")
        if ans:
            return clean_text(ans)

    answer_metrics = row.get("answer_metrics", {})
    if isinstance(answer_metrics, dict):
        ans = answer_metrics.get("prediction", "")
        if ans:
            return clean_text(ans)

    return ""


def build_ragas_dataset(details_path: str, max_samples: int = 0):
    rows = load_jsonl(details_path)

    data = []
    skipped = 0

    for row in rows:
        question = clean_text(row.get("query", ""))
        reference = clean_text(row.get("gold_answer", ""))
        response = extract_answer(row)

        generation_chunks = row.get("generation_chunks", []) or []
        contexts = []

        for ch in generation_chunks:
            if not isinstance(ch, dict):
                continue

            text = clean_text(ch.get("text", ""))
            if text:
                contexts.append(text)

        if not question or not response or not reference or not contexts:
            skipped += 1
            continue

        data.append({
            "user_input": question,
            "response": response,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

        if max_samples and max_samples > 0 and len(data) >= max_samples:
            break

    print(f"[RagasData] valid samples = {len(data)}")
    print(f"[RagasData] skipped       = {skipped}")

    return Dataset.from_list(data)


def build_dashscope_llm(args):
    api_key = args.dashscope_api_key or os.getenv("DASHSCOPE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到 DASHSCOPE_API_KEY。\n"
            "请先在 PowerShell 中设置：\n"
            '$env:DASHSCOPE_API_KEY=\"你的DashScope_API_Key\"'
        )

    llm = ChatOpenAI(
        model=args.llm_model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=0,
        max_retries=2,
        timeout=120,
    )

    return llm


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--details_path",
        type=str,
        required=True,
        help="实验输出目录中的 details.jsonl 路径",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Ragas 结果保存目录。默认保存到 details.jsonl 同级目录下的 ragas_eval_xxx 文件夹。",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=50,
        help="最多评估多少条样本。0 表示全部。",
    )

    parser.add_argument(
        "--dashscope_api_key",
        type=str,
        default=None,
        help="可选：直接通过参数传入 DashScope API Key；更推荐用环境变量。",
    )

    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    parser.add_argument(
        "--llm_model",
        type=str,
        default="qwen-plus",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        parent = os.path.dirname(os.path.abspath(args.details_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(parent, f"ragas_eval_{timestamp}")

    ensure_dir(args.output_dir)

    print("Loading details:")
    print(args.details_path)

    dataset = build_ragas_dataset(
        details_path=args.details_path,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "没有可用于 Ragas 的样本。请确认 details.jsonl 中存在 "
            "query、gold_answer、generation_chunks、gen_output.answer。"
        )

    llm = build_dashscope_llm(args)

    print("Using evaluator LLM:")
    print(f"  llm_model = {args.llm_model}")
    print(f"  base_url  = {args.base_url}")

    metrics = [
        faithfulness,
        context_precision,
        context_recall,
    ]

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    details_out = os.path.join(args.output_dir, "ragas_details.jsonl")
    summary_out = os.path.join(args.output_dir, "ragas_summary.json")

    df.to_json(
        details_out,
        orient="records",
        lines=True,
        force_ascii=False,
    )

    summary = {}
    for col in [
        "faithfulness",
        "context_precision",
        "context_recall",
    ]:
        if col in df.columns:
            try:
                summary[col] = float(df[col].mean())
            except Exception:
                summary[col] = None

    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== Ragas Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"details: {details_out}")
    print(f"summary: {summary_out}")


if __name__ == "__main__":
    main()