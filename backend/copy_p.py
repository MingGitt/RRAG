import os
import requests

URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

SAVE_DIR = r"D:\code\rag\FSR\data\hotpotqa_distractor"
SAVE_PATH = os.path.join(SAVE_DIR, "hotpot_dev_distractor_v1.json")

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Downloading HotpotQA dev distractor...")
    print("URL:", URL)

    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(SAVE_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"\r{downloaded / total * 100:.2f}%", end="")

    print("\nSaved to:", SAVE_PATH)

if __name__ == "__main__":
    main()