import os

# ========= API =========
QWEN_API_KEY = os.environ.get("QWEN_API_KEY")
assert QWEN_API_KEY, "请先设置 QWEN_API_KEY"

QWEN_URL = "https://api.edgefn.net/v1/chat/completions"
QWEN_MODEL = "Qwen3-Next-80B-A3B-Instruct"

HEADERS = {
    "Authorization": f"Bearer {QWEN_API_KEY}", 
    "Content-Type": "application/json"
}

# ========= 数据 =========
DATASET = "scifact"
DATASET_PATH = "../beir_datasets/scifact"

# ========= 模型 =========
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ========= 参数 =========
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_CHUNKS = 50
RERANK_TOP_DOCS = 5
ALPHA = 0.6

PERSIST_DIR = "./chroma_scifact_chunks"