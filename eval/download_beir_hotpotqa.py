# download_beir_hotpotqa.py
# 放在：D:\code\rag\FSR\eval\download_beir_hotpotqa.py

import os
from beir import util

BASE_DIR = r"D:\code\rag\FSR"
DATASET_NAME = "hotpotqa"
OUT_DIR = os.path.join(BASE_DIR, "beir_datasets")

url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{DATASET_NAME}.zip"

print("Downloading:", url)
data_path = util.download_and_unzip(url, OUT_DIR)

print("HotpotQA BEIR dataset saved to:")
print(data_path)