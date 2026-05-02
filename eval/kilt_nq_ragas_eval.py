# kilt_nq_ragas_eval.py
# 放置位置：
# D:\code\rag\FSR\eval\kilt_nq_ragas_eval.py
#
# 功能：
# 使用 RAGAS 评测 KILT-NQ 轻量版 JSONL 结果。
#
# 默认输入：
# D:\code\rag\FSR\eval\kilt_nq_single_index_eval_outputs\kilt_nq_single_index_ragas_legacy.jsonl
#
# 默认输出：
# D:\code\rag\FSR\eval\kilt_nq_single_index_eval_outputs\kilt_nq_ragas_result.csv
# D:\code\rag\FSR\eval\kilt_nq_single_index_eval_outputs\kilt_nq_ragas_summary.json
#
# 运行示例：
# conda activate ragas-eval
# cd D:\code\rag\FSR\eval
#
# $env:DASHSCOPE_API_KEY="你的DashScope API Key"
# $env:RAGAS_JUDGE_MODEL="qwen-plus"
# $env:RAGAS_EMBEDDING_MODEL="text-embedding-v3"
#
# python kilt_nq_ragas_eval.py --max_samples 20
#
# 全量 200 条：
# python kilt_nq_ragas_eval.py --max_samples 0
#
# 如果仍然想禁用 ResponseRelevancy：
# python kilt_nq_ragas_eval.py --max_samples 20 --disable_response_relevancy

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from datasets import Dataset

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

try:
    from ragas.embeddings import LangchainEmbeddingsWrapper
except Exception:
    LangchainEmbeddingsWrapper = None


# ============================================================
# 1. 路径配置
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# D:\code\rag\FSR
BASE_DIR = CURRENT_FILE.parents[1]

# D:\code\rag\FSR\eval
EVAL_DIR = BASE_DIR / "eval"

# D:\code\rag\FSR\backend
BACKEND_DIR = BASE_DIR / "backend"

# 把 backend 加入 sys.path，方便导入 config.py
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# 2. 从 backend/config.py 读取 DashScope / Qwen 配置
# ============================================================

try:
    from config import DASHSCOPE_API_KEY, QWEN_URL
except Exception as e:
    print("[警告] 无法从 backend/config.py 导入 DASHSCOPE_API_KEY 和 QWEN_URL")
    print(f"[警告] 原因：{e}")

    DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
    QWEN_URL = os.environ.get(
        "QWEN_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )


# ============================================================
# 3. 默认输入输出路径
# ============================================================

DEFAULT_OUTPUT_DIR = EVAL_DIR / "scifact_baseline_output"

DEFAULT_INPUT_FILE = (
    DEFAULT_OUTPUT_DIR / "baseline_generation_details.jsonl"
)

DEFAULT_OUTPUT_CSV = (
    DEFAULT_OUTPUT_DIR / "baseline_ragas_result.csv"
)

DEFAULT_OUTPUT_SUMMARY = (
    DEFAULT_OUTPUT_DIR / "baseline_ragas_summary.json"
)


# ============================================================
# 4. RAGAS 模型配置
# ============================================================

DEFAULT_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "qwen-plus")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-v3")


def qwen_chat_url_to_base_url(url: str) -> str:
    """
    将 config.py 里的 QWEN_URL 转成 ChatOpenAI / OpenAIEmbeddings 需要的 base_url。

    config.py 中通常是：
        https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions

    LangChain OpenAI 兼容接口需要的是：
        https://dashscope.aliyuncs.com/compatible-mode/v1
    """
    url = (url or "").strip().rstrip("/")

    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]

    return url


# ============================================================
# 5. 字段抽取工具函数
# ============================================================

def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_non_empty(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in item:
            value = item.get(key)
            if value is not None and value != "":
                return value
    return ""


def extract_question(item: Dict[str, Any]) -> str:
    """
    抽取问题字段。
    兼容：
    - question
    - query
    - input
    - user_input
    - prompt
    """
    value = first_non_empty(
        item,
        [
            "question",
            "query",
            "input",
            "user_input",
            "prompt",
        ],
    )
    return safe_str(value)


def extract_answer(item: Dict[str, Any]) -> str:
    """
    抽取系统生成答案。
    兼容：
    - answer
    - response
    - generated_answer
    - prediction
    - pred
    - final_answer
    - output
    """
    value = first_non_empty(
        item,
        [
            "answer",
            "response",
            "generated_answer",
            "prediction",
            "pred",
            "final_answer",
        ],
    )

    if safe_str(value):
        return safe_str(value)

    output = item.get("output")

    if isinstance(output, str):
        return safe_str(output)

    if isinstance(output, dict):
        return safe_str(
            output.get("answer")
            or output.get("text")
            or output.get("response")
            or ""
        )

    return ""


def extract_reference(item: Dict[str, Any]) -> str:
    """
    抽取标准答案。
    兼容：
    - reference
    - ground_truth
    - gold_answer
    - target
    - label
    - expected_answer
    - answers
    - output
    """

    value = first_non_empty(
        item,
        [
            "reference",
            "ground_truth",
            "gold_answer",
            "target",
            "label",
            "expected_answer",
        ],
    )

    if isinstance(value, list):
        refs = [safe_str(v) for v in value if safe_str(v)]
        if refs:
            return " | ".join(refs)

    if safe_str(value):
        return safe_str(value)

    answers = item.get("answers")
    if isinstance(answers, list):
        refs = [safe_str(v) for v in answers if safe_str(v)]
        if refs:
            return " | ".join(refs)

    # 兼容 KILT 原始 output 字段
    output = item.get("output")

    if isinstance(output, list):
        refs = []

        for obj in output:
            if isinstance(obj, dict):
                ans = (
                    obj.get("answer")
                    or obj.get("text")
                    or obj.get("label")
                    or ""
                )

                if isinstance(ans, list):
                    refs.extend([safe_str(x) for x in ans if safe_str(x)])
                elif safe_str(ans):
                    refs.append(safe_str(ans))

            elif safe_str(obj):
                refs.append(safe_str(obj))

        if refs:
            return " | ".join(refs)

    elif isinstance(output, dict):
        ans = (
            output.get("answer")
            or output.get("text")
            or output.get("label")
            or ""
        )
        if safe_str(ans):
            return safe_str(ans)

    return ""


def normalize_context_item(ctx: Any) -> str:
    """
    把单个 context/chunk 转成字符串。

    兼容：
    1. "文本"
    2. {"text": "..."}
    3. {"content": "..."}
    4. {"page_content": "..."}
    5. {"title": "...", "text": "..."}
    6. {"document": "..."}
    7. {"metadata": {"text": "..."}}
    """

    if isinstance(ctx, str):
        return safe_str(ctx)

    if isinstance(ctx, dict):
        title = safe_str(ctx.get("title"))

        text = (
            ctx.get("text")
            or ctx.get("content")
            or ctx.get("page_content")
            or ctx.get("chunk_text")
            or ctx.get("document")
            or ctx.get("passage")
            or ctx.get("context")
            or ""
        )

        text = safe_str(text)

        if not text and isinstance(ctx.get("metadata"), dict):
            meta = ctx.get("metadata", {})
            text = safe_str(
                meta.get("text")
                or meta.get("content")
                or meta.get("page_content")
                or meta.get("chunk_text")
                or meta.get("document")
                or ""
            )

        if title and text:
            return f"{title}\n{text}"

        if text:
            return text

        if title:
            return title

    return ""


def extract_contexts(item: Dict[str, Any]) -> List[str]:
    """
    抽取 RAGAS 的 retrieved_contexts。

    优先级：
    1. retrieved_contexts
    2. contexts
    3. context
    4. generation_chunks
    5. selected_chunks
    6. final_chunks
    7. reranked_chunks
    8. retrieved_chunks
    9. docs / documents / passages
    """

    possible_keys = [
        "retrieved_contexts",
        "contexts",
        "context",
        "generation_chunks",
        "selected_chunks",
        "final_chunks",
        "reranked_chunks",
        "retrieved_chunks",
        "docs",
        "documents",
        "passages",
    ]

    raw_contexts = None

    for key in possible_keys:
        if key in item and item.get(key):
            raw_contexts = item.get(key)
            break

    if raw_contexts is None:
        return []

    if isinstance(raw_contexts, str):
        raw_contexts = [raw_contexts]

    contexts: List[str] = []

    if isinstance(raw_contexts, list):
        for ctx in raw_contexts:
            text = normalize_context_item(ctx)
            if text:
                contexts.append(text)

    elif isinstance(raw_contexts, dict):
        text = normalize_context_item(raw_contexts)
        if text:
            contexts.append(text)

    # 去重，避免重复 chunk 影响评测
    deduped: List[str] = []
    seen = set()

    for text in contexts:
        text = text.strip()
        if not text:
            continue

        key = text[:300]
        if key in seen:
            continue

        seen.add(key)
        deduped.append(text)

    return deduped


# ============================================================
# 6. JSONL 读取与转换
# ============================================================

def load_jsonl_for_ragas(
    input_file: Path,
    max_samples: Optional[int] = None,
    require_reference: bool = False,
) -> List[Dict[str, Any]]:
    """
    将 legacy JSONL 转成 RAGAS v0.4 推荐字段：

    user_input:
        问题

    response:
        系统生成答案

    retrieved_contexts:
        检索到的上下文列表

    reference:
        标准答案
    """

    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_file}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    total = 0

    with open(input_file, "r", encoding="utf-8") as f:
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
            contexts = extract_contexts(item)
            reference = extract_reference(item)

            if not question:
                print(f"[跳过] 第 {line_no} 行缺少 question/query/user_input")
                skipped += 1
                continue

            if not answer:
                print(f"[跳过] 第 {line_no} 行缺少 answer/response/generated_answer")
                skipped += 1
                continue

            if not contexts:
                print(f"[跳过] 第 {line_no} 行缺少 contexts/retrieved_contexts/generation_chunks")
                skipped += 1
                continue

            if require_reference and not reference:
                print(f"[跳过] 第 {line_no} 行缺少 reference/ground_truth/gold_answer")
                skipped += 1
                continue

            rows.append(
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                    "reference": reference,
                }
            )

            if max_samples is not None and len(rows) >= max_samples:
                break

    print("=" * 80)
    print("JSONL 读取完成")
    print(f"输入文件 : {input_file}")
    print(f"原始行数 : {total}")
    print(f"有效样本 : {len(rows)}")
    print(f"跳过样本 : {skipped}")
    print("=" * 80)

    if rows:
        print("首条样本预览：")
        preview = json.dumps(rows[0], ensure_ascii=False, indent=2)
        print(preview[:2000])
        print("=" * 80)

    return rows


# ============================================================
# 7. 构建 RAGAS 评判 LLM 和 Embedding
# ============================================================

def get_dashscope_config() -> Tuple[str, str]:
    """
    统一获取 DashScope API Key 和 base_url。

    优先级：
    1. backend/config.py 的 DASHSCOPE_API_KEY / QWEN_URL
    2. 环境变量 DASHSCOPE_API_KEY / QWEN_URL
    3. 环境变量 OPENAI_API_KEY / OPENAI_BASE_URL
    """

    api_key = (
        DASHSCOPE_API_KEY
        or os.getenv("DASHSCOPE_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
    )

    raw_qwen_url = (
        QWEN_URL
        or os.getenv(
            "QWEN_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
    )

    base_url = qwen_chat_url_to_base_url(raw_qwen_url)

    # 如果用户显式设置了 OPENAI_BASE_URL，则优先使用它
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if openai_base_url:
        base_url = qwen_chat_url_to_base_url(openai_base_url)

    if not api_key:
        raise ValueError(
            "未检测到 API Key。\n"
            "请在 PowerShell 中设置：\n"
            '$env:DASHSCOPE_API_KEY="你的DashScope API Key"\n'
            "或者设置：\n"
            '$env:OPENAI_API_KEY="你的DashScope API Key"\n'
        )

    # 关键修复：
    # RAGAS 内部部分组件会读取 OPENAI_API_KEY / OPENAI_BASE_URL。
    # 这里统一写入，避免自动创建 OpenAIEmbeddings 时找不到 key。
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_BASE_URL"] = base_url

    return api_key, base_url


def build_ragas_llm(model_name: str) -> LangchainLLMWrapper:
    """
    构建 RAGAS 使用的评判 LLM。
    """

    api_key, base_url = get_dashscope_config()

    print("=" * 80)
    print("RAGAS 评判模型配置")
    print(f"llm_model   : {model_name}")
    print(f"base_url    : {base_url}")
    print(f"api_key     : {'已设置' if api_key else '未设置'}")
    print("=" * 80)

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
        max_retries=3,
    )

    return LangchainLLMWrapper(llm)


class SafeOpenAIEmbeddings(OpenAIEmbeddings):
    """
    防止 RAGAS 传给 DashScope embedding 的内容不是 str/list[str]。

    DashScope embedding 报错：
        contents is neither str nor list of str.: input.contents

    这个类会把所有输入强制转成字符串。
    """

    def _to_text(self, value: Any) -> str:
        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            texts = [self._to_text(v) for v in value]
            return " ".join([t for t in texts if t])

        if isinstance(value, dict):
            for key in [
                "text",
                "content",
                "page_content",
                "response",
                "question",
                "user_input",
                "reference",
                "context",
            ]:
                if key in value and value[key]:
                    return self._to_text(value[key])

            return json.dumps(value, ensure_ascii=False)

        return str(value)

    def embed_documents(self, texts: List[Any]) -> List[List[float]]:
        texts = [self._to_text(t) for t in texts]
        texts = [t if t else " " for t in texts]
        return super().embed_documents(texts)

    async def aembed_documents(self, texts: List[Any]) -> List[List[float]]:
        texts = [self._to_text(t) for t in texts]
        texts = [t if t else " " for t in texts]
        return await super().aembed_documents(texts)

    def embed_query(self, text: Any) -> List[float]:
        text = self._to_text(text)
        if not text:
            text = " "
        return super().embed_query(text)

    async def aembed_query(self, text: Any) -> List[float]:
        text = self._to_text(text)
        if not text:
            text = " "
        return await super().aembed_query(text)


def build_ragas_embeddings():
    """
    构建 RAGAS 使用的 Embedding。

    重要：
    RAGAS 默认会使用 OpenAI 的 text-embedding-ada-002。
    DashScope 不支持这个模型，所以必须显式指定：
    - text-embedding-v3
    - text-embedding-v4

    这里使用 SafeOpenAIEmbeddings，避免 DashScope 报：
        contents is neither str nor list of str
    """

    api_key, base_url = get_dashscope_config()

    embedding_model = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL,
    )

    print("=" * 80)
    print("RAGAS Embedding 模型配置")
    print(f"embedding_model : {embedding_model}")
    print(f"base_url        : {base_url}")
    print(f"api_key         : {'已设置' if api_key else '未设置'}")
    print("=" * 80)

    embeddings = SafeOpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        timeout=120,
        max_retries=3,
    )

    # RAGAS v0.4 一般推荐包装成 LangchainEmbeddingsWrapper
    if LangchainEmbeddingsWrapper is not None:
        return LangchainEmbeddingsWrapper(embeddings)

    return embeddings


# ============================================================
# 8. 保存结果
# ============================================================

def save_summary(df: pd.DataFrame, output_summary: Path):
    """
    保存各项 RAGAS 指标均值。
    """

    ignore_cols = {
        "user_input",
        "response",
        "retrieved_contexts",
        "reference",
    }

    summary: Dict[str, float] = {}

    for col in df.columns:
        if col in ignore_cols:
            continue

        numeric = pd.to_numeric(df[col], errors="coerce")

        if numeric.notna().sum() > 0:
            summary[col] = float(numeric.mean())

    output_summary.parent.mkdir(parents=True, exist_ok=True)

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("RAGAS 平均分")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)
    print(f"Saved summary: {output_summary}")


# ============================================================
# 9. 主评测流程
# ============================================================

def run_ragas_eval(
    input_file: Path,
    output_csv: Path,
    output_summary: Path,
    model_name: str,
    max_samples: Optional[int],
    require_reference: bool,
    disable_response_relevancy: bool,
):
    print("=" * 80)
    print("KILT-NQ RAGAS 离线评测开始")
    print(f"BASE_DIR                  : {BASE_DIR}")
    print(f"BACKEND_DIR               : {BACKEND_DIR}")
    print(f"EVAL_DIR                  : {EVAL_DIR}")
    print(f"INPUT_FILE                : {input_file}")
    print(f"OUTPUT_CSV                : {output_csv}")
    print(f"OUTPUT_SUMMARY            : {output_summary}")
    print(f"MAX_SAMPLES               : {max_samples}")
    print(f"REQUIRE_REFERENCE         : {require_reference}")
    print(f"DISABLE_RESPONSE_RELEVANCY: {disable_response_relevancy}")
    print("=" * 80)

    rows = load_jsonl_for_ragas(
        input_file=input_file,
        max_samples=max_samples,
        require_reference=require_reference,
    )

    if not rows:
        print("没有可评测样本，请检查 JSONL 字段。")
        return

    dataset = Dataset.from_list(rows)

    evaluator_llm = build_ragas_llm(model_name)
    evaluator_embeddings = build_ragas_embeddings()

    # 指标说明：
    #
    # Faithfulness:
    #   回答是否被 retrieved_contexts 支撑。
    #
    # ResponseRelevancy:
    #   回答是否切题。该指标通常需要 Embedding。
    #
    # LLMContextPrecisionWithoutReference:
    #   不依赖 reference，判断检索上下文是否和问题相关。
    #
    # LLMContextRecall:
    #   依赖 reference，判断上下文是否覆盖标准答案需要的信息。
    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
    ]

    if not disable_response_relevancy:
        try:
            metrics.append(
                ResponseRelevancy(
                    llm=evaluator_llm,
                    embeddings=evaluator_embeddings,
                )
            )
        except TypeError:
            # 兼容部分 RAGAS 版本：构造函数不接受 embeddings 参数时，
            # 只添加 llm，embedding 通过 evaluate(..., embeddings=...) 传入。
            metrics.append(ResponseRelevancy(llm=evaluator_llm))

    has_reference = any(safe_str(row.get("reference")) for row in rows)

    if has_reference:
        metrics.append(LLMContextRecall(llm=evaluator_llm))
    else:
        print("警告：未检测到 reference/ground_truth/gold_answer，跳过 LLMContextRecall。")

    print("=" * 80)
    print("开始执行 RAGAS evaluate()")
    print("评测指标：")
    for metric in metrics:
        print(f"- {metric.__class__.__name__}")
    print("=" * 80)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    print("=" * 80)
    print("RAGAS evaluate() 完成")
    print(result)
    print("=" * 80)

    df = result.to_pandas()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved details: {output_csv}")

    save_summary(df, output_summary)


# ============================================================
# 10. 命令行参数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 RAGAS 评测 KILT-NQ 轻量版 JSONL 结果"
    )

    parser.add_argument(
        "--input_file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help="输入 JSONL 文件路径",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(DEFAULT_OUTPUT_CSV),
        help="输出 RAGAS 明细 CSV 路径",
    )

    parser.add_argument(
        "--output_summary",
        type=str,
        default=str(DEFAULT_OUTPUT_SUMMARY),
        help="输出 RAGAS 平均分 JSON 路径",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="RAGAS 评判模型名称，例如 qwen-plus / qwen-turbo / qwen3-next-80b-a3b-instruct",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="最多评测多少条。默认 20。设为 0 表示全部。",
    )

    parser.add_argument(
        "--require_reference",
        action="store_true",
        help="开启后，如果样本没有 reference/gold_answer，则跳过该样本。",
    )

    parser.add_argument(
        "--disable_response_relevancy",
        action="store_true",
        help="禁用 ResponseRelevancy。若 embedding 接口仍报错，可先加这个参数跑通。",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    max_samples = None if args.max_samples == 0 else args.max_samples

    try:
        run_ragas_eval(
            input_file=Path(args.input_file).resolve(),
            output_csv=Path(args.output_csv).resolve(),
            output_summary=Path(args.output_summary).resolve(),
            model_name=args.model,
            max_samples=max_samples,
            require_reference=args.require_reference,
            disable_response_relevancy=args.disable_response_relevancy,
        )

    except Exception as e:
        print("=" * 80)
        print("RAGAS 评测失败")
        print(str(e))
        print("=" * 80)
        traceback.print_exc()


if __name__ == "__main__":
    main()