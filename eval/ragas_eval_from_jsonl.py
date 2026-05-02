# ragas_eval_from_jsonl.py
# 放置位置：D:\code\rag\FSR\eval\ragas_eval_from_jsonl.py
#
# 功能：
# 1. 读取你的 RAG 系统输出的 JSONL 评测结果
# 2. 抽取 question、answer、gold_answer、generation_chunks
# 3. 转换成 RAGAS 需要的数据格式
# 4. 使用 Qwen / DashScope OpenAI 兼容接口作为 RAGAS 评判模型
# 5. 输出 RAGAS 明细 CSV 和平均分 JSON
#
# 推荐运行：
# conda activate ragas-eval
# cd D:\code\rag\FSR\eval
# $env:OPENAI_API_KEY="你的DashScope API Key"
# $env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# python ragas_eval_from_jsonl.py

import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)


# =========================
# 1. 路径配置
# =========================

# 当前文件路径：
# D:\code\rag\FSR\eval\ragas_eval_from_jsonl.py
CURRENT_FILE = Path(__file__).resolve()

# 项目根目录：
# D:\code\rag\FSR
BASE_DIR = CURRENT_FILE.parents[1]

# 后端代码目录：
# D:\code\rag\FSR\backend
BACKEND_DIR = BASE_DIR / "backend"

# 评测代码目录：
# D:\code\rag\FSR\eval
EVAL_DIR = BASE_DIR / "eval"

# 默认读取 eval 目录下的结果文件
INPUT_FILE = EVAL_DIR / "hotpotqa_eval_details.jsonl"

# 如果你的 hotpotqa_eval_details.jsonl 在 backend 目录，请改成这一行：
# INPUT_FILE = BACKEND_DIR / "hotpotqa_eval_details.jsonl"

OUTPUT_CSV = EVAL_DIR / "hotpotqa_ragas_result.csv"
OUTPUT_SUMMARY_JSON = EVAL_DIR / "hotpotqa_ragas_summary.json"


# =========================
# 2. RAGAS 评判模型配置
# =========================

# DashScope OpenAI 兼容接口
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 你可以换成：
# qwen-plus
# qwen-turbo
# qwen-max
# qwen3-next-80b-a3b-instruct
JUDGE_MODEL_NAME = os.getenv("RAGAS_JUDGE_MODEL", "qwen-plus")

# 为了先跑通，建议先限制样本数量
# None 表示跑全部
# 例如改成 10，表示只评测前 10 条
MAX_SAMPLES = None

# 如果你的数据很多，建议先用：
# MAX_SAMPLES = 10


# =========================
# 3. 工具函数
# =========================

def safe_str(value: Any) -> str:
    """
    安全地把任意值转成字符串。
    """
    if value is None:
        return ""
    return str(value).strip()


def extract_answer(item: Dict[str, Any]) -> str:
    """
    兼容不同字段名，抽取系统生成答案。
    """
    possible_keys = [
        "answer",
        "generated_answer",
        "prediction",
        "pred",
        "response",
        "final_answer",
    ]

    for key in possible_keys:
        value = safe_str(item.get(key))
        if value:
            return value

    return ""


def extract_question(item: Dict[str, Any]) -> str:
    """
    兼容不同字段名，抽取问题。
    """
    possible_keys = [
        "question",
        "query",
        "user_input",
        "input",
    ]

    for key in possible_keys:
        value = safe_str(item.get(key))
        if value:
            return value

    return ""


def extract_ground_truth(item: Dict[str, Any]) -> str:
    """
    兼容不同字段名，抽取标准答案。
    """
    possible_keys = [
        "gold_answer",
        "ground_truth",
        "reference",
        "label",
        "target",
        "expected_answer",
    ]

    for key in possible_keys:
        value = item.get(key)

        if isinstance(value, list):
            # HotpotQA / PopQA 有时标准答案可能是列表
            joined = " | ".join([safe_str(v) for v in value if safe_str(v)])
            if joined:
                return joined

        value_str = safe_str(value)
        if value_str:
            return value_str

    return ""


def extract_contexts(item: Dict[str, Any]) -> List[str]:
    """
    抽取 RAGAS 的 retrieved_contexts。

    优先使用 generation_chunks，因为它代表最终送入生成模型的文本块。
    如果没有 generation_chunks，则尝试使用 reranked_chunks / retrieved_chunks / contexts。

    支持以下格式：

    1. generation_chunks: ["文本1", "文本2"]

    2. generation_chunks: [
           {"text": "文本1"},
           {"text": "文本2"}
       ]

    3. generation_chunks: [
           {"content": "文本1"},
           {"page_content": "文本2"},
           {"chunk_text": "文本3"}
       ]
    """

    possible_context_keys = [
        "generation_chunks",
        "final_chunks",
        "selected_chunks",
        "reranked_chunks",
        "retrieved_chunks",
        "contexts",
        "context",
    ]

    raw_chunks = None

    for key in possible_context_keys:
        if key in item and item.get(key):
            raw_chunks = item.get(key)
            break

    if raw_chunks is None:
        return []

    # 如果 context 是单个字符串
    if isinstance(raw_chunks, str):
        raw_chunks = [raw_chunks]

    contexts: List[str] = []

    if isinstance(raw_chunks, list):
        for chunk in raw_chunks:
            text = ""

            if isinstance(chunk, str):
                text = chunk

            elif isinstance(chunk, dict):
                # 兼容常见字段
                text = (
                    chunk.get("text")
                    or chunk.get("content")
                    or chunk.get("page_content")
                    or chunk.get("chunk_text")
                    or chunk.get("document")
                    or ""
                )

                # 有些代码可能把文本放在 metadata 里面
                if not text and isinstance(chunk.get("metadata"), dict):
                    metadata = chunk.get("metadata", {})
                    text = (
                        metadata.get("text")
                        or metadata.get("content")
                        or metadata.get("page_content")
                        or ""
                    )

            text = safe_str(text)

            if text:
                contexts.append(text)

    return contexts


def load_jsonl_results(jsonl_path: Path, max_samples=None) -> List[Dict[str, Any]]:
    """
    读取 JSONL 文件，并转换成 RAGAS v0.4 需要的字段：

    user_input:
        用户问题

    response:
        RAG 系统生成的答案

    retrieved_contexts:
        检索/筛选后的上下文文本块列表

    reference:
        标准答案
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{jsonl_path}")

    rows: List[Dict[str, Any]] = []

    skipped = 0
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            total += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"[跳过] 第 {line_no} 行不是合法 JSON")
                skipped += 1
                continue

            question = extract_question(item)
            answer = extract_answer(item)
            ground_truth = extract_ground_truth(item)
            contexts = extract_contexts(item)

            if not question:
                print(f"[跳过] 第 {line_no} 行缺少 question/query/user_input")
                skipped += 1
                continue

            if not answer:
                print(f"[跳过] 第 {line_no} 行缺少 answer/response")
                skipped += 1
                continue

            if not contexts:
                print(f"[跳过] 第 {line_no} 行缺少 generation_chunks/contexts")
                skipped += 1
                continue

            # RAGAS v0.4 字段
            rows.append({
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": ground_truth,
            })

            if max_samples is not None and len(rows) >= max_samples:
                break

    print("=" * 80)
    print("JSONL 读取完成")
    print(f"输入文件: {jsonl_path}")
    print(f"原始行数: {total}")
    print(f"有效样本: {len(rows)}")
    print(f"跳过样本: {skipped}")
    print("=" * 80)

    return rows


def build_judge_llm():
    """
    构建 RAGAS 使用的评判模型。

    默认使用 DashScope OpenAI 兼容接口：

    OPENAI_API_KEY:
        你的 DashScope API Key

    OPENAI_BASE_URL:
        https://dashscope.aliyuncs.com/compatible-mode/v1
    """

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL)

    if not api_key:
        raise ValueError(
            "未检测到 OPENAI_API_KEY。\n"
            "请先在 PowerShell 中执行：\n"
            '$env:OPENAI_API_KEY="你的DashScope API Key"\n'
            '$env:OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"\n'
        )

    print("=" * 80)
    print("RAGAS 评判模型配置")
    print(f"model    : {JUDGE_MODEL_NAME}")
    print(f"base_url : {base_url}")
    print("=" * 80)

    llm = ChatOpenAI(
        model=JUDGE_MODEL_NAME,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
        max_retries=3,
    )

    return LangchainLLMWrapper(llm)


def save_summary(df: pd.DataFrame, output_path: Path):
    """
    保存平均分汇总。
    """

    metric_cols = [
        col for col in df.columns
        if col not in [
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
        ]
    ]

    summary = {}

    for col in metric_cols:
        try:
            summary[col] = float(pd.to_numeric(df[col], errors="coerce").mean())
        except Exception:
            pass

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved summary: {output_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# =========================
# 4. 主函数
# =========================

def main():
    try:
        print("=" * 80)
        print("RAGAS 离线评测开始")
        print(f"BASE_DIR     : {BASE_DIR}")
        print(f"BACKEND_DIR  : {BACKEND_DIR}")
        print(f"EVAL_DIR     : {EVAL_DIR}")
        print(f"INPUT_FILE   : {INPUT_FILE}")
        print(f"OUTPUT_CSV   : {OUTPUT_CSV}")
        print("=" * 80)

        # 1. 读取 JSONL 结果
        rows = load_jsonl_results(INPUT_FILE, max_samples=MAX_SAMPLES)

        if not rows:
            print("没有可评测样本，请检查 JSONL 字段是否包含 question、answer、generation_chunks。")
            return

        # 2. 转换成 HuggingFace Dataset
        dataset = Dataset.from_list(rows)

        # 3. 构建评判模型
        evaluator_llm = build_judge_llm()

        # 4. 配置 RAGAS 指标
        metrics = [
            Faithfulness(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
        ]

        print("=" * 80)
        print("开始执行 RAGAS evaluate()")
        print("指标:")
        print("- Faithfulness")
        print("- ResponseRelevancy")
        print("- LLMContextPrecisionWithoutReference")
        print("- LLMContextRecall")
        print("=" * 80)

        # 5. 执行评测
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )

        print("=" * 80)
        print("RAGAS 评测完成")
        print(result)
        print("=" * 80)

        # 6. 保存明细结果
        df = result.to_pandas()
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved details: {OUTPUT_CSV}")

        # 7. 保存平均分汇总
        save_summary(df, OUTPUT_SUMMARY_JSON)

    except Exception as e:
        print("=" * 80)
        print("RAGAS 评测失败")
        print(str(e))
        print("=" * 80)
        traceback.print_exc()


if __name__ == "__main__":
    main()