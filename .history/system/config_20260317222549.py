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
#EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
#EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#手动下载后的本地目录，没有就设为 None
#EMBED_MODEL_LOCAL_DIR = r"D:/code/model/models--sentence-transformers--all-MiniLM-L6-v2"
EMBED_MODEL_LOCAL_DIR = None
RERANK_MODEL_LOCAL_DIR = r"D:/code/model/models--cross-encoder--ms-marco-MiniLM-L-6-v2"

# ========= 参数 =========
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K_CHUNKS = 50
RERANK_TOP_DOCS = 10
ALPHA = 0.7
DENSE_CANDIDATE_K = 200
BM25_CANDIDATE_K = 100

DOC_AGG_MODE = "max"

PERSIST_DIR = "./chroma_scifact_chunks"

# 统一缓存目录
HF_CACHE_DIR = r"D:/code/hf_cache"

# 网络较慢时有用
HF_HUB_ETAG_TIMEOUT = "30"
HF_HUB_DOWNLOAD_TIMEOUT = "120"

#并发
QUERY_WORKERS = 4
SUBQUERY_WORKERS = 8
MAX_SAMPLES=100
MAX_CONCURRENCY=4