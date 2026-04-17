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

# ========= 数据源模式 =========
# 可选: "beir" / "local_docs"
DATA_SOURCE = os.environ.get("DATA_SOURCE", "beir")

# ========= BEIR 数据 =========
DATASET = os.environ.get("DATASET", "scifact")
DATASET_PATH = os.environ.get("DATASET_PATH", "../beir_datasets/scifact")
TARGET_QID = os.environ.get("TARGET_QID", "35")

# ========= 本地文档模式 =========
LOCAL_DOCS_DIR = os.environ.get("LOCAL_DOCS_DIR", "./docs")
LOCAL_TEST_QUERY = os.environ.get("LOCAL_TEST_QUERY", "请总结这些文档的核心内容，并给出关键结论。")
LOCAL_QUERY_ID = os.environ.get("LOCAL_QUERY_ID", "local_query_1")

# ========= 模型 =========
# EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
# EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# 手动下载后的本地目录，没有就设为 None
# EMBED_MODEL_LOCAL_DIR = r"D:/code/model/models--sentence-transformers--all-MiniLM-L6-v2"
EMBED_MODEL_LOCAL_DIR = os.environ.get("EMBED_MODEL_LOCAL_DIR") or None
RERANK_MODEL_LOCAL_DIR = os.environ.get("RERANK_MODEL_LOCAL_DIR") or None

# ========= 参数 =========
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", 50))
RERANK_TOP_DOCS = int(os.environ.get("RERANK_TOP_DOCS", 10))
ALPHA = float(os.environ.get("ALPHA", 0.7))
DENSE_CANDIDATE_K = int(os.environ.get("DENSE_CANDIDATE_K", 200))
BM25_CANDIDATE_K = int(os.environ.get("BM25_CANDIDATE_K", 100))

DOC_AGG_MODE = os.environ.get("DOC_AGG_MODE", "max")

# Chroma 持久化目录
PERSIST_DIR = os.environ.get("PERSIST_DIR", "./chroma_store")

# 统一缓存目录
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")

# 网络较慢时有用
HF_HUB_ETAG_TIMEOUT = os.environ.get("HF_HUB_ETAG_TIMEOUT", "30")
HF_HUB_DOWNLOAD_TIMEOUT = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# 并发
QUERY_WORKERS = int(os.environ.get("QUERY_WORKERS", 4))
SUBQUERY_WORKERS = int(os.environ.get("SUBQUERY_WORKERS", 8))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 100))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 4))

SHOW_TOP_K = int(os.environ.get("SHOW_TOP_K", 5))