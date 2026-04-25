# config.py
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ========= DashScope / Qwen =========
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

QWEN_URL = os.environ.get(
    "QWEN_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
)

# 通用默认模型
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")

# 分步骤模型
QWEN_DECISION_MODEL = os.environ.get("QWEN_DECISION_MODEL", "qwen-plus")      # Step1 检索决策
QWEN_RELEVANCE_MODEL = os.environ.get("QWEN_RELEVANCE_MODEL", "qwen-turbo")   # Step3 文档相关性判断
QWEN_ANSWER_MODEL = os.environ.get("QWEN_ANSWER_MODEL", "qwen-plus")          # Step4 答案生成
QWEN_SUPPORT_MODEL = os.environ.get("QWEN_SUPPORT_MODEL", QWEN_MODEL)         # Step5 支持度评估（你可改成 qwen3）

QW_HEADERS = {
    "Authorization": f"Bearer {DASHSCOPE_API_KEY}" if DASHSCOPE_API_KEY else "",
    "Content-Type": "application/json"
}

QWEN_REQUEST_TIMEOUT = int(os.environ.get("QWEN_REQUEST_TIMEOUT", 300))


# ========= 本地文档模式 / BEIR =========
LOCAL_DOCS_DIR = os.environ.get("LOCAL_DOCS_DIR", "../docs")
LOCAL_QUERY_ID = os.environ.get("LOCAL_QUERY_ID", "local_query_1")

DATASET = os.environ.get("DATASET", "scifact")
DATASET_PATH = os.environ.get("DATASET_PATH", "../beir_datasets/scifact")
TARGET_QID = os.environ.get("TARGET_QID", "35")


# ========= 模型 =========
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "BAAI/bge-base-en-v1.5")
RERANK_MODEL_NAME = os.environ.get("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

EMBED_MODEL_LOCAL_DIR = os.environ.get("EMBED_MODEL_LOCAL_DIR") or None
RERANK_MODEL_LOCAL_DIR = os.environ.get("RERANK_MODEL_LOCAL_DIR") or None


# ========= 检索参数 =========
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 400))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 80))
TOP_K_CHUNKS = int(os.environ.get("TOP_K_CHUNKS", 50))
RERANK_TOP_DOCS = int(os.environ.get("RERANK_TOP_DOCS", 50))
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

# relevance judge 并发
RELEVANCE_MAX_WORKERS = int(os.environ.get("RELEVANCE_MAX_WORKERS", 8))


# ========= 展示 =========
SHOW_TOP_K = int(os.environ.get("SHOW_TOP_K", 5))


# ========= 生成阶段参数 =========
GENERATION_TOP_N = int(os.environ.get("GENERATION_TOP_N", 50))
GENERATION_MAX_PER_DOC = int(os.environ.get("GENERATION_MAX_PER_DOC", 3))
MIN_RELEVANT_EVIDENCE = int(os.environ.get("MIN_RELEVANT_EVIDENCE", 1))

GEN_CANDIDATE_COUNT = int(os.environ.get("GEN_CANDIDATE_COUNT", 3))
QWEN_CANDIDATE_MODEL = os.environ.get("QWEN_CANDIDATE_MODEL", QWEN_ANSWER_MODEL)