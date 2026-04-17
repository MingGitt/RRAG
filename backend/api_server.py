import asyncio
import os
import threading
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LOCAL_DOCS_DIR,
    PERSIST_DIR,
)
from data_loader import MultiFormatLoader
from generator import Generator
from hybrid_retriever import MultiVectorRetriever
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

PERSIST_ROOT = Path(PERSIST_DIR)
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# FastAPI
# =========================
app = FastAPI(title="Local Document RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段先全部放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 数据结构
# =========================
class FileRecord(BaseModel):
    name: str
    path: str
    status: str   # uploaded / indexing / indexed / upload_failed / index_failed
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


# =========================
# 全局状态
# =========================
app_state = {
    "files": {},          # filename -> FileRecord dict
    "file_indexes": {},   # file_id -> index object
    "reranker": None,
    "reranker_lock": threading.Lock(),
}


# =========================
# 工具函数
# =========================
def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".pdf", ".docx", ".txt"}


def get_loader():
    return MultiFormatLoader(max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


def build_single_uploaded_file_index(file_path: str, filename: str):
    """
    只为一个文件建库，并刷新 app_state["file_indexes"]
    """
    loader = get_loader()

    doc = loader.load_document(file_path)
    if not doc:
        raise ValueError("文档解析后为空。")

    chunks = loader.chunk_document(doc)
    if not chunks:
        raise ValueError("文档分块结果为空。")

    file_id = doc["doc_id"]

    print(f"[INDEX] building single-file index for: {filename}")
    print(f"[INDEX] file_id = {file_id}")
    print(f"[INDEX] chunk count = {len(chunks)}")

    index_obj = VectorStoreBuilder.build_single_file_index(
        file_id=file_id,
        chunks=chunks,
        persist_root=str(PERSIST_ROOT),
    )
    index_obj["filename"] = filename

    # 覆盖或新增该文件索引
    app_state["file_indexes"][file_id] = index_obj

    return file_id


def rebuild_existing_files_on_startup():
    """
    启动时扫描 uploads 下的已有文件，为每个文件恢复单独索引
    """
    loader = get_loader()

    for path in UPLOAD_DIR.iterdir():
        if not path.is_file() or not allowed_file(path.name):
            continue

        try:
            doc = loader.load_document(str(path))
            if not doc:
                app_state["files"][path.name] = FileRecord(
                    name=path.name,
                    path=str(path),
                    status="index_failed",
                    error="文档解析为空。"
                ).dict()
                continue

            chunks = loader.chunk_document(doc)
            if not chunks:
                app_state["files"][path.name] = FileRecord(
                    name=path.name,
                    path=str(path),
                    status="index_failed",
                    error="文档分块结果为空。"
                ).dict()
                continue

            file_id = doc["doc_id"]

            index_obj = VectorStoreBuilder.build_single_file_index(
                file_id=file_id,
                chunks=chunks,
                persist_root=str(PERSIST_ROOT),
            )
            index_obj["filename"] = path.name

            app_state["file_indexes"][file_id] = index_obj
            app_state["files"][path.name] = FileRecord(
                name=path.name,
                path=str(path),
                status="indexed",
                error=""
            ).dict()

            print(f"[STARTUP] indexed {path.name}, file_id={file_id}, chunks={len(chunks)}")

        except Exception as e:
            print(f"[STARTUP][ERROR] {path.name}: {e}")
            app_state["files"][path.name] = FileRecord(
                name=path.name,
                path=str(path),
                status="index_failed",
                error=str(e)
            ).dict()


async def answer_one_query_api(query_text: str):
    if not app_state["file_indexes"]:
        raise HTTPException(status_code=400, detail="当前没有已索引文档。")

    if app_state["reranker"] is None:
        app_state["reranker"] = Reranker()

    retriever = MultiVectorRetriever(
        file_indexes=app_state["file_indexes"],
        alpha=ALPHA
    )

    queries = {"web_query_1": query_text}

    rerank_results_with_scores, generation_chunks, ranked_chunks_map, query_type_stats = run_parallel_retrieval_pipeline(
        queries=queries,
        retriever=retriever,
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


def find_file_id_by_filename(filename: str):
    for file_id, index_obj in app_state["file_indexes"].items():
        if index_obj.get("filename") == filename:
            return file_id
    return None


# =========================
# 启动初始化
# =========================
@app.on_event("startup")
def startup_event():
    print("[STARTUP] api server starting...")

    if app_state["reranker"] is None:
        app_state["reranker"] = Reranker()

    print(f"[STARTUP] UPLOAD_DIR = {UPLOAD_DIR.resolve()}")
    print(f"[STARTUP] PERSIST_ROOT = {PERSIST_ROOT.resolve()}")

    rebuild_existing_files_on_startup()

    print(f"[STARTUP] recovered files = {list(app_state['files'].keys())}")
    print(f"[STARTUP] recovered indexes = {list(app_state['file_indexes'].keys())}")


# =========================
# 接口
# =========================
@app.get("/health")
def health():
    return {
        "ok": True,
        "file_count": len(app_state["files"]),
        "index_count": len(app_state["file_indexes"]),
    }


@app.get("/files")
def list_files():
    return {"files": list(app_state["files"].values())}


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="没有接收到文件。")

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
            print(f"[UPLOAD] receiving file: {f.filename}")
            print(f"[UPLOAD] save path: {save_path}")

            with open(save_path, "wb") as out:
                content = await f.read()
                out.write(content)

            app_state["files"][f.filename] = FileRecord(
                name=f.filename,
                path=str(save_path),
                status="indexing",
                error=""
            ).dict()

            file_id = build_single_uploaded_file_index(str(save_path), f.filename)

            app_state["files"][f.filename]["status"] = "indexed"
            app_state["files"][f.filename]["error"] = ""

            print(f"[UPLOAD] indexed ok: filename={f.filename}, file_id={file_id}")

        except Exception as e:
            print(f"[UPLOAD][ERROR] {f.filename}: {e}")
            app_state["files"][f.filename] = FileRecord(
                name=f.filename,
                path=str(save_path),
                status="index_failed",
                error=str(e)
            ).dict()

    return {"files": list(app_state["files"].values())}


@app.post("/files/{filename}/retry-index")
def retry_index(filename: str):
    if filename not in app_state["files"]:
        raise HTTPException(status_code=404, detail="文件不存在。")

    record = app_state["files"][filename]
    path = record.get("path", "")

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="文件路径不存在。")

    try:
        app_state["files"][filename]["status"] = "indexing"
        app_state["files"][filename]["error"] = ""

        old_file_id = find_file_id_by_filename(filename)
        if old_file_id:
            VectorStoreBuilder.delete_single_file_index(old_file_id, str(PERSIST_ROOT))
            del app_state["file_indexes"][old_file_id]

        new_file_id = build_single_uploaded_file_index(path, filename)

        app_state["files"][filename]["status"] = "indexed"
        app_state["files"][filename]["error"] = ""

        print(f"[RETRY] indexed ok: filename={filename}, file_id={new_file_id}")

    except Exception as e:
        print(f"[RETRY][ERROR] {filename}: {e}")
        app_state["files"][filename]["status"] = "index_failed"
        app_state["files"][filename]["error"] = str(e)

    return {"file": app_state["files"][filename]}


@app.delete("/files/{filename}")
def delete_file(filename: str):
    if filename not in app_state["files"]:
        raise HTTPException(status_code=404, detail="文件不存在。")

    record = app_state["files"][filename]
    path = record.get("path", "")

    try:
        # 删除磁盘文件
        if path and os.path.exists(path):
            os.remove(path)
            print(f"[DELETE] removed file: {path}")

        # 删除对应索引
        target_file_id = find_file_id_by_filename(filename)
        if target_file_id:
            VectorStoreBuilder.delete_single_file_index(target_file_id, str(PERSIST_ROOT))
            del app_state["file_indexes"][target_file_id]
            print(f"[DELETE] removed index: file_id={target_file_id}")

        del app_state["files"][filename]

        return {"ok": True, "files": list(app_state["files"].values())}

    except Exception as e:
        print(f"[DELETE][ERROR] {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    return await answer_one_query_api(query)