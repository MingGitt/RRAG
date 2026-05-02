import os
import re
import json
import time
import argparse
import string
from collections import Counter, defaultdict
from typing import Dict, List, Any, Set, Tuple

import requests
from tqdm import tqdm


# ============================================================
# 1. 答案评测指标：EM / F1
# ============================================================

def normalize_answer(s: str) -> str:
    """
    对英文开放域问答答案做归一化：
    - 小写
    - 去标点
    - 去冠词 a/an/the
    - 合并空格
    """
    if s is None:
        return ""

    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def metric_max_over_gold(metric_fn, pred: str, gold_answers: List[str]) -> float:
    if not gold_answers:
        return 0.0
    return max(metric_fn(pred, gold) for gold in gold_answers)


# ============================================================
# 2. 文件和 KILT 数据读取
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def safe_title_from_filename(filename: str) -> str:
    """
    你的 txt 文件名大概率由 Wikipedia title 转换而来。
    例如：
        Albert_Einstein.txt -> Albert Einstein
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace("_", " ")
    return name.strip().lower()


def normalize_title(s: str) -> str:
    if not s:
        return ""
    s = os.path.splitext(os.path.basename(str(s)))[0]
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def get_gold_titles(question_record: Dict[str, Any]) -> Set[str]:
    """
    读取 fetch_kilt_nq_200_fulltext.py 生成的问题文件：
    kilt_nq_200_questions.jsonl
    """
    titles = set()

    for prov in question_record.get("provenance", []) or []:
        title = prov.get("title")
        if title:
            titles.add(normalize_title(title))

    return titles


def get_gold_answers(question_record: Dict[str, Any]) -> List[str]:
    answers = question_record.get("answers", []) or []
    return [str(a) for a in answers if str(a).strip()]


# ============================================================
# 3. 调用你的后端 API
# ============================================================

def check_backend(api_base: str) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/health"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def list_backend_files(api_base: str) -> List[Dict[str, Any]]:
    url = api_base.rstrip("/") + "/files"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json().get("files", [])


def upload_txt_files(
    api_base: str,
    txt_dir: str,
    batch_size: int = 5,
    skip_existing: bool = True,
):
    """
    批量调用你的 /upload 接口。
    注意：你的后端 /upload 会把文件保存到 LOCAL_DOCS_DIR/uploads，
    并为每个文件单独建立向量库。
    """
    upload_url = api_base.rstrip("/") + "/upload"

    all_txt_files = []
    for root, _, files in os.walk(txt_dir):
        for name in files:
            if name.lower().endswith(".txt"):
                all_txt_files.append(os.path.join(root, name))

    all_txt_files = sorted(all_txt_files)

    if not all_txt_files:
        raise FileNotFoundError(f"没有找到 txt 文件: {txt_dir}")

    existing_names = set()
    if skip_existing:
        try:
            existing_files = list_backend_files(api_base)
            existing_names = {x.get("name") for x in existing_files if x.get("status") == "indexed"}
        except Exception:
            existing_names = set()

    pending_files = [
        p for p in all_txt_files
        if os.path.basename(p) not in existing_names
    ]

    print(f"总 txt 文件数: {len(all_txt_files)}")
    print(f"已存在 indexed 文件数: {len(existing_names)}")
    print(f"本次需要上传文件数: {len(pending_files)}")

    if not pending_files:
        print("没有需要上传的新文件。")
        return

    success_batches = 0
    failed_batches = 0

    for start in tqdm(range(0, len(pending_files), batch_size), desc="Uploading txt files"):
        batch_paths = pending_files[start:start + batch_size]

        files_payload = []
        opened_files = []

        try:
            for path in batch_paths:
                f = open(path, "rb")
                opened_files.append(f)
                files_payload.append(
                    ("files", (os.path.basename(path), f, "text/plain"))
                )

            resp = requests.post(upload_url, files=files_payload, timeout=600)

            if resp.status_code == 200:
                success_batches += 1
            else:
                failed_batches += 1
                print("\n[UPLOAD ERROR]")
                print("status:", resp.status_code)
                print("text:", resp.text[:1000])

        except Exception as e:
            failed_batches += 1
            print("\n[UPLOAD EXCEPTION]", repr(e))

        finally:
            for f in opened_files:
                try:
                    f.close()
                except Exception:
                    pass

        # 给后端一点时间完成索引
        time.sleep(0.2)

    print("\n========== Upload Done ==========")
    print(f"success_batches: {success_batches}")
    print(f"failed_batches: {failed_batches}")


def ask_backend(
    api_base: str,
    query: str,
    session_id: str = "kilt_nq_eval",
    timeout: int = 300,
) -> Dict[str, Any]:
    """
    调用你的 /ask 接口。
    根据 api_server.py，AskRequest 字段是：
        query: str
        session_id: Optional[str]
    """
    url = api_base.rstrip("/") + "/ask"

    payload = {
        "query": query,
        "session_id": session_id,
    }

    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()

    return resp.json()


# ============================================================
# 4. 从后端返回结果中提取答案和证据
# ============================================================

def extract_answer(api_result: Dict[str, Any]) -> str:
    return api_result.get("answer", "") or ""


def collect_citation_sources(api_result: Dict[str, Any]) -> List[str]:
    """
    你的后端 AskResponse 中有：
    - citations
    - sub_answers[].citations

    fuzzy / complex 问题时，你的前端主要展示 sub_answers；
    因此这里两类 citation 都收集。
    """
    sources = []

    for c in api_result.get("citations", []) or []:
        src = c.get("source", "")
        if src:
            sources.append(src)

    for sub in api_result.get("sub_answers", []) or []:
        for c in sub.get("citations", []) or []:
            src = c.get("source", "")
            if src:
                sources.append(src)

    return sources


def provenance_hit_by_source(
    api_result: Dict[str, Any],
    gold_titles: Set[str],
) -> int:
    """
    轻量版 provenance hit：
    判断后端返回的 citations/source 是否包含 gold Wikipedia title。

    注意：
    你的后端当前 citation source 多半是文件名或 title。
    所以这里用 title / filename 的模糊匹配。
    """
    if not gold_titles:
        return 0

    pred_sources = collect_citation_sources(api_result)

    pred_norms = set()
    for src in pred_sources:
        pred_norms.add(normalize_title(src))

    for gold in gold_titles:
        gold_norm = normalize_title(gold)

        for pred in pred_norms:
            if gold_norm and (gold_norm in pred or pred in gold_norm):
                return 1

    return 0


def extract_query_mode(api_result: Dict[str, Any]) -> str:
    return api_result.get("query_mode", "unknown") or "unknown"


# ============================================================
# 5. 主评测逻辑
# ============================================================

def run_eval(
    api_base: str,
    questions_path: str,
    output_dir: str,
    sample_size: int = 200,
    timeout: int = 300,
):
    os.makedirs(output_dir, exist_ok=True)

    records = read_jsonl(questions_path)

    if sample_size > 0:
        records = records[:sample_size]

    details_path = os.path.join(output_dir, "kilt_nq_200_eval_details.jsonl")
    summary_path = os.path.join(output_dir, "kilt_nq_200_eval_summary.json")

    total = 0
    error_count = 0

    em_sum = 0.0
    f1_sum = 0.0
    latency_sum = 0.0

    provenance_hit_sum = 0
    query_mode_stats = defaultdict(int)

    with open(details_path, "w", encoding="utf-8") as fout:
        for item in tqdm(records, desc="Evaluating KILT NQ"):
            total += 1

            qid = item.get("id", "")
            question = item.get("question", "")
            gold_answers = get_gold_answers(item)
            gold_titles = get_gold_titles(item)

            start = time.time()

            try:
                # 每个问题使用独立 session，避免多轮历史影响评测
                session_id = f"kilt_nq_eval_{qid}"

                api_result = ask_backend(
                    api_base=api_base,
                    query=question,
                    session_id=session_id,
                    timeout=timeout,
                )

                latency = time.time() - start
                latency_sum += latency

                pred_answer = extract_answer(api_result)

                em = metric_max_over_gold(exact_match_score, pred_answer, gold_answers)
                f1 = metric_max_over_gold(f1_score, pred_answer, gold_answers)
                hit = provenance_hit_by_source(api_result, gold_titles)

                em_sum += em
                f1_sum += f1
                provenance_hit_sum += hit

                query_mode = extract_query_mode(api_result)
                query_mode_stats[query_mode] += 1

                result_record = {
                    "id": qid,
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "pred_answer": pred_answer,
                    "em": em,
                    "f1": f1,
                    "provenance_hit": hit,
                    "latency_sec": latency,
                    "query_mode": query_mode,
                    "citation_sources": collect_citation_sources(api_result),
                    "raw_api_result": api_result,
                }

            except Exception as e:
                error_count += 1
                latency = time.time() - start

                result_record = {
                    "id": qid,
                    "question": question,
                    "gold_answers": gold_answers,
                    "gold_titles": sorted(list(gold_titles)),
                    "error": repr(e),
                    "latency_sec": latency,
                }

            fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")

    valid_total = max(total - error_count, 1)

    summary = {
        "task": "kilt_nq_200_lightweight",
        "questions_path": questions_path,
        "total": total,
        "errors": error_count,
        "answer_em": em_sum / valid_total,
        "answer_f1": f1_sum / valid_total,
        "provenance_hit_rate_by_citation_source": provenance_hit_sum / valid_total,
        "avg_latency_sec": latency_sum / valid_total,
        "query_mode_stats": dict(query_mode_stats),
        "details_path": details_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n========== KILT NQ 200 Evaluation Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved details: {details_path}")
    print(f"Saved summary: {summary_path}")


# ============================================================
# 6. 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:5173",
        help="你的 FastAPI 后端地址。",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\code\rag\FSR\data\kilt_nq_200_fulltext",
        help="kilt_nq_200_fulltext 文件夹路径。",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"D:\code\rag\FSR\eval\kilt_nq_200_eval_outputs",
        help="评测结果输出目录。",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["upload", "eval", "all"],
        default="all",
        help="upload=只上传建库；eval=只评测；all=先上传再评测。",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="评测问题数量。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="上传 txt 文件时每批上传多少个。",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="每个问题调用 /ask 的超时时间。",
    )

    args = parser.parse_args()

    txt_dir = os.path.join(args.data_dir, "txt")
    questions_path = os.path.join(args.data_dir, "kilt_nq_200_questions.jsonl")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"txt 目录不存在: {txt_dir}")

    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"问题文件不存在: {questions_path}")

    print("Checking backend...")
    health = check_backend(args.api_base)
    print("Backend health:", health)

    if args.mode in ["upload", "all"]:
        upload_txt_files(
            api_base=args.api_base,
            txt_dir=txt_dir,
            batch_size=args.batch_size,
            skip_existing=True,
        )

    if args.mode in ["eval", "all"]:
        run_eval(
            api_base=args.api_base,
            questions_path=questions_path,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    main()