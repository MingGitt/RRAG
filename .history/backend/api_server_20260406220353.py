import asyncio
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL_NAME,
    LOCAL_DOCS_DIR,
    PERSIST_DIR,
)
from data_loader import MultiFormatLoader
from generator import Generator
from hybrid_retriever import HybridRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from reranker import Reranker
from vector_store import VectorStoreBuilder


# =========================
# 基础目录
# =========================
BASE_DIR = Path(LOCAL_DOCS_DIR)
BASE_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# FastAPI
# =========================
app = FastAPI(title="Local Document RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段先放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 全局状态
# =========================
class FileRecord(BaseModel):
    name: str
    path: str
    status: str  # uploaded / indexing / indexed / upload_failed / index_failed
    error: str = ""


class AskRequest(BaseModel):
    query: str


class CitationItem(BaseModel):
    index: int
    source: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    judge: str


app_state = {
    "files": {},                # {filename: FileRecord.dict()}
    "vectordb": None,
    "chunk_texts": [],
    "chunk_doc_ids": [],
    "chunk_ids": [],
    "chunk_metas": [],
    "retriever": None,
    "reranker": None,
    "reranker_lock": threading.Lock(),
    "last_index_time": None,
}


# =========================
# 工具函数
# =========================
def build_local_index():
    """
    从 LOCAL_DOCS_DIR 下的所有文档重建索引
    """
    loader = MultiFormatLoader(max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    docs = loader.load_directory(str(BASE_DIR))

    all_chunks = []
    indexed_filenames = set()

    for doc in docs.values():
        all_chunks.extend(loader.chunk_document(doc))
        title = doc.get("title", "")
        if title:
            indexed_filenames.add(title)

    collection_name = VectorStoreBuilder.make_collection_name(
        "localdocs",
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        EMBED_MODEL_NAME
    )

    # 为了避免旧 collection 的 metadata 结构污染，重建前清空持久化目录
    persist_path = Path(PERSIST_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path, ignore_errors=True)
    persist_path.mkdir(parents=True, exist_ok=True)

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder.build_or_load_from_chunks(
        all_chunks,
        str(persist_path),
        collection_name,
    )

    retriever = HybridRetriever(
        vectordb,
        chunk_texts,
        chunk_doc_ids,
        chunk_ids,
        chunk_metas,
        ALPHA
    )

    if app_state["reranker"] is None:
        app_state["reranker"] = Reranker()

    app_state["vectordb"] = vectordb
    app_state["chunk_texts"] = chunk_texts
    app_state["chunk_doc_ids"] = chunk_doc_ids
    app_state["chunk_ids"] = chunk_ids
    app_state["chunk_metas"] = chunk_metas
    app_state["retriever"] = retriever
    app_state["last_index_time"] = time.time()

    # 更新文件状态
    for filename, record in app_state["files"].items():
        if filename in indexed_filenames:
            record["status"] = "indexed"
            record["error"] = ""
        else:
            # 文件存在但没有进入 docs，视为索引失败
            if record["status"] != "upload_failed":
                record["status"] = "index_failed"
                record["error"] = "文件未进入索引结果，请检查文件内容或格式。"


async def answer_one_query_api(query_text: str):
    if app_state["retriever"] is None or app_state["reranker"] is None:
        raise HTTPException(status_code=400, detail="当前没有可用索引，请先上传并索引文档。")

    queries = {"web_query_1": query_text}

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=app_state["retriever"],
        reranker=app_state["reranker"],
        reranker_lock=app_state["reranker_lock"],
        query_workers=1,
        subquery_workers=4,
    )

    gen_outputs = await Generator.run(
        queries,
        generation_chunks,
        max_samples=1,
        max_concurrency=1,
    )

    result = gen_outputs.get("web_query_1")
    if not result:
        raise HTTPException(status_code=500, detail="未生成答案。")

    citations = []
    for item in result.get("citations", []):
        citations.append(
            CitationItem(
                index=item["index"],
                source=item["source"],
                excerpt=item["excerpt"],
            )
        )

    return AskResponse(
        answer=result.get("answer", ""),
        citations=citations,
        judge=result.get("judge", ""),
    )


def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".pdf", ".docx", ".txt"}


# =========================
# 启动初始化
# =========================
@app.on_event("startup")
def startup_event():
    # 初始化 reranker，避免首次提问时再加载
    if app_state["reranker"] is None:
        app_state["reranker"] = Reranker()

    # 扫描已有文件
    for path in BASE_DIR.rglob("*"):
        if path.is_file() and allowed_file(path.name):
            app_state["files"][path.name] = FileRecord(
                name=path.name,
                path=str(path),
                status="uploaded",
                error=""
            ).dict()

    # 如果目录里已有文件，启动时尝试建库
    if app_state["files"]:
        try:
            for record in app_state["files"].values():
                record["status"] = "indexing"
            build_local_index()
        except Exception as e:
            for record in app_state["files"].values():
                record["status"] = "index_failed"
                record["error"] = str(e)


# =========================
# 接口
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "file_count": len(app_state["files"]),
        "indexed_chunk_count": len(app_state["chunk_texts"]),
        "last_index_time": app_state["last_index_time"],
    }


@app.get("/files")
def list_files():
    return {"files": list(app_state["files"].values())}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="没有接收到文件。")

    uploaded_names = []

    for f in files:
        if not allowed_file(f.filename):
            app_state["files"][f.filename] = FileRecord(
                name=f.filename,
                path="",
                status="upload_failed",
                error="仅支持 PDF、DOCX、TXT 文件。"
            ).dict()
            continue

        save_path = UPLOAD_DIR / f.filename

        try:
            with open(save_path, "wb") as out:
                content = await f.read()
                out.write(content)

            app_state["files"][f.filename] = FileRecord(
                name=f.filename,
                path=str(save_path),
                status="uploaded",
                error=""
            ).dict()
            uploaded_names.append(f.filename)
        except Exception as e:
            app_state["files"][f.filename] = FileRecord(
                name=f.filename,
                path=str(save_path),
                status="upload_failed",
                error=str(e)
            ).dict()

    # 上传成功的文件进入索引
    if uploaded_names:
        try:
            for name in uploaded_names:
                app_state["files"][name]["status"] = "indexing"
                app_state["files"][name]["error"] = ""

            build_local_index()
        except Exception as e:
            for name in uploaded_names:
                if name in app_state["files"]:
                    app_state["files"][name]["status"] = "index_failed"
                    app_state["files"][name]["error"] = str(e)

    return {"files": list(app_state["files"].values())}


@app.post("/files/{filename}/retry-index")
def retry_index(filename: str):
    if filename not in app_state["files"]:
        raise HTTPException(status_code=404, detail="文件不存在。")

    record = app_state["files"][filename]
    if record["status"] != "index_failed":
        return {"message": "当前文件不是索引失败状态。", "file": record}

    try:
        record["status"] = "indexing"
        record["error"] = ""
        build_local_index()
    except Exception as e:
        record["status"] = "index_failed"
        record["error"] = str(e)

    return {"file": app_state["files"][filename]}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    return await answer_one_query_api(query)