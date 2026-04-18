import os

# ========= Qwen API =========
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY")
assert DASHSCOPE_API_KEY, "请先设置 DASHSCOPE_API_KEY"

QWEN_URL = os.environ.get(
    "QWEN_URL",
    "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

QW_HEADERS = {
    "Authorization": f"Bearer {DASHSCOPE_API_KEY}",
    "Content-Type": "application/json"
}

# ========= 本地文档模式 =========
LOCAL_DOCS_DIR = os.environ.get("LOCAL_DOCS_DIR", "../docs")
LOCAL_QUERY_ID = os.environ.get("LOCAL_QUERY_ID", "local_query_1")

# ========= BEIR 评测（仅供独立测试脚本使用） =========
DATASET = os.environ.get("DATASET", "scifact")
DATASET_PATH = os.environ.get("DATASET_PATH", "../beir_datasets/scifact")
TARGET_QID = os.environ.get("TARGET_QID", "35")

# ========= 模型 =========
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

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

# ========= 向量库 =========
PERSIST_DIR = os.environ.get("PERSIST_DIR", "../chroma_store")

# ========= HuggingFace 缓存 =========
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./hf_cache")
HF_HUB_ETAG_TIMEOUT = os.environ.get("HF_HUB_ETAG_TIMEOUT", "30")
HF_HUB_DOWNLOAD_TIMEOUT = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "120")

# ========= 并发 =========
QUERY_WORKERS = int(os.environ.get("QUERY_WORKERS", 4))
SUBQUERY_WORKERS = int(os.environ.get("SUBQUERY_WORKERS", 8))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", 100))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", 4))

# ========= 展示 =========
SHOW_TOP_K = int(os.environ.get("SHOW_TOP_K", 5))

# ========= Self-RAG: evidence critique =========
ENABLE_SELF_RAG_CRITIQUE = os.environ.get("ENABLE_SELF_RAG_CRITIQUE", "true").lower() == "true"
SELF_RAG_CRITIQUE_TOP_K = int(os.environ.get("SELF_RAG_CRITIQUE_TOP_K", 3))
SELF_RAG_GENERATION_TOP_N = int(os.environ.get("SELF_RAG_GENERATION_TOP_N", 3))
SELF_RAG_MAX_PER_DOC = int(os.environ.get("SELF_RAG_MAX_PER_DOC", 3))
SELF_RAG_MIN_SCORE = float(os.environ.get("SELF_RAG_MIN_SCORE", 0.55))
SELF_RAG_W_REL = float(os.environ.get("SELF_RAG_W_REL", 0.4))
SELF_RAG_W_SUP = float(os.environ.get("SELF_RAG_W_SUP", 0.4))
SELF_RAG_W_USE = float(os.environ.get("SELF_RAG_W_USE", 0.2))