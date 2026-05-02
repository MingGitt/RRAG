# download_covidqa_hf.py

from datasets import load_dataset
import json
from pathlib import Path

OUT_DIR = Path("data/covidqa")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    ds = load_dataset("deepset/covid_qa_deepset")

    print(ds)

    # 一般只有 train split
    split_name = list(ds.keys())[0]
    data = ds[split_name]

    out_jsonl = OUT_DIR / "covidqa.jsonl"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for i, item in enumerate(data):
            # 先打印第一条，方便你确认字段结构
            if i == 0:
                print("Example item:")
                print(item)

            record = {
                "id": str(i),
                "question": item.get("question", ""),
                "answer": item.get("answers", item.get("answer", "")),
                "context": item.get("context", ""),
                "title": item.get("title", ""),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {out_jsonl}")

if __name__ == "__main__":
    main()