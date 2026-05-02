import os
import re
import json
import time
import argparse
from typing import Dict, Any, Set, List, Optional

import requests
from tqdm import tqdm
from datasets import load_dataset


def safe_filename(name: str, max_len: int = 150) -> str:
    """
    将 Wikipedia title 转成 Windows 安全文件名。
    """
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:max_len] if name else "untitled"


def get_gold_provenance(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 KILT 样本中提取 gold provenance。
    每条 provenance 通常包含：
    - wikipedia_id
    - title
    - start_paragraph_id
    - end_paragraph_id
    """
    results = []

    for out in example.get("output", []) or []:
        for prov in out.get("provenance", []) or []:
            title = prov.get("title")
            wikipedia_id = prov.get("wikipedia_id")

            if title or wikipedia_id:
                results.append({
                    "title": title,
                    "wikipedia_id": wikipedia_id,
                    "start_paragraph_id": prov.get("start_paragraph_id"),
                    "end_paragraph_id": prov.get("end_paragraph_id"),
                })

    return results


def fetch_wikipedia_fulltext_by_title(
    title: str,
    lang: str = "en",
    timeout: int = 30,
    max_retries: int = 3,
    sleep: float = 0.2,
) -> Optional[Dict[str, Any]]:
    """
    使用 Wikipedia API 获取页面纯文本正文。

    API 参数说明：
    - action=query
    - prop=extracts
    - explaintext=1：返回纯文本
    - redirects=1：自动处理重定向
    """
    api_url = f"https://{lang}.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts|info",
        "titles": title,
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
    }

    headers = {
        "User-Agent": "RRAG-KILT-Eval/1.0 (local research script)"
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(
                api_url,
                params=params,
                headers=headers,
                timeout=timeout,
            )

            if resp.status_code != 200:
                time.sleep(sleep * attempt)
                continue

            data = resp.json()
            pages = data.get("query", {}).get("pages", {})

            if not pages:
                return None

            # pages 是 dict，key 是 pageid
            page = next(iter(pages.values()))

            if "missing" in page:
                return None

            extract = page.get("extract", "")
            if not extract.strip():
                return None

            return {
                "pageid": page.get("pageid"),
                "title": page.get("title"),
                "fullurl": page.get("fullurl"),
                "extract": extract.strip(),
            }

        except Exception:
            time.sleep(sleep * attempt)

    return None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="nq",
        help="KILT 子数据集名称，这里默认 nq。",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="本地评测建议使用 validation。",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="读取前多少条 KILT NQ 样本。",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./kilt_nq_200_fulltext",
        help="正文输出目录。",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Wikipedia 语言版本，KILT 默认对应英文 Wikipedia。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="每次请求后的暂停时间，避免请求过快。",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    txt_dir = os.path.join(args.output_dir, "txt")
    os.makedirs(txt_dir, exist_ok=True)

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    questions_path = os.path.join(args.output_dir, "kilt_nq_200_questions.jsonl")
    failed_path = os.path.join(args.output_dir, "failed_titles.jsonl")

    print(f"Loading KILT dataset: facebook/kilt_tasks, name={args.dataset}, split={args.split}")

    dataset = load_dataset(
        "facebook/kilt_tasks",
        name=args.dataset,
        split=args.split,
    )

    dataset = dataset.select(range(min(args.sample_size, len(dataset))))

    # 1. 保存问题集，并收集 gold provenance title
    title_to_info: Dict[str, Dict[str, Any]] = {}

    with open(questions_path, "w", encoding="utf-8") as qf:
        for ex in dataset:
            qid = ex.get("id")
            question = ex.get("input", "")

            answers = []
            for out in ex.get("output", []) or []:
                ans = out.get("answer")
                if ans and ans not in answers:
                    answers.append(ans)

            provenances = get_gold_provenance(ex)

            q_record = {
                "id": qid,
                "question": question,
                "answers": answers,
                "provenance": provenances,
            }
            qf.write(json.dumps(q_record, ensure_ascii=False) + "\n")

            for prov in provenances:
                title = prov.get("title")
                if not title:
                    continue

                if title not in title_to_info:
                    title_to_info[title] = {
                        "title": title,
                        "wikipedia_ids": set(),
                        "used_by_question_ids": set(),
                    }

                if prov.get("wikipedia_id") is not None:
                    title_to_info[title]["wikipedia_ids"].add(str(prov.get("wikipedia_id")))

                if qid is not None:
                    title_to_info[title]["used_by_question_ids"].add(str(qid))

    titles = sorted(title_to_info.keys())

    print(f"Loaded questions: {len(dataset)}")
    print(f"Unique gold Wikipedia titles: {len(titles)}")
    print(f"Questions saved to: {questions_path}")

    # 2. 获取每个 title 的 Wikipedia 正文
    success = 0
    failed = 0

    with open(metadata_path, "w", encoding="utf-8") as meta_f, \
         open(failed_path, "w", encoding="utf-8") as failed_f:

        for title in tqdm(titles, desc="Fetching Wikipedia fulltext"):
            page = fetch_wikipedia_fulltext_by_title(
                title=title,
                lang=args.lang,
                sleep=args.sleep,
            )

            if page is None:
                failed += 1
                failed_f.write(json.dumps({
                    "title": title,
                    "reason": "missing_or_empty_extract",
                }, ensure_ascii=False) + "\n")
                continue

            file_stem = safe_filename(title)
            filename = f"{file_stem}.txt"
            file_path = os.path.join(txt_dir, filename)

            # 避免同名覆盖
            if os.path.exists(file_path):
                filename = f"{file_stem}_{page.get('pageid')}.txt"
                file_path = os.path.join(txt_dir, filename)

            full_text = page["extract"]

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Title: {page.get('title')}\n")
                f.write(f"PageID: {page.get('pageid')}\n")
                f.write(f"URL: {page.get('fullurl')}\n")
                f.write("\n")
                f.write(full_text)

            info = title_to_info[title]

            meta_record = {
                "source": "kilt_nq_200_wikipedia_api_fulltext",
                "kilt_dataset": args.dataset,
                "kilt_split": args.split,
                "original_kilt_title": title,
                "wikipedia_api_title": page.get("title"),
                "wikipedia_api_pageid": page.get("pageid"),
                "kilt_wikipedia_ids": sorted(list(info["wikipedia_ids"])),
                "used_by_question_ids": sorted(list(info["used_by_question_ids"])),
                "filename": filename,
                "file_path": file_path,
                "url": page.get("fullurl"),
                "text_length": len(full_text),
            }

            meta_f.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

            success += 1
            time.sleep(args.sleep)

    print("\n========== Done ==========")
    print(f"Output dir: {args.output_dir}")
    print(f"TXT dir: {txt_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Questions: {questions_path}")
    print(f"Failed titles: {failed_path}")
    print(f"Success pages: {success}")
    print(f"Failed pages: {failed}")


if __name__ == "__main__":
    main()