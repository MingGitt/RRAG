# api_server.py
import os
import threading
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    ALPHA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LOCAL_DOCS_DIR,
    PERSIST_DIR,
    SELF_RAG_EAS_URL,
    SELF_RAG_EAS_TOKEN,
)
from data_loader import MultiFormatLoader
from hybrid_retriever import MultiVectorRetriever
from pipeline_executor import run_parallel_retrieval_pipeline
from qa_service import answer_one_query
from reranker import Reranker
from risk_controller import RiskController
from vector_store import VectorStoreBuilder


# =========================
# 路径初始化
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

app = FastAPI(title="Local Document QA API", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 生产环境建议改成前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 全局状态
# =========================

app_state = {
    "files": {},           # filename -> record
    "file_indexes": {},    # file_id -> index_obj
    "reranker": None,
    "reranker_lock": threading.Lock(),
    "sessions": {},        # session_id -> history list
}


# =========================
# Pydantic Models
# =========================

class FileRecord(BaseModel):
    name: str
    path: str
    status: str
    error: str = ""


class AskRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"


class CitationItem(BaseModel):
    index: int
    source: str
    excerpt: str
    doc_id: str = ""
    page_no: int = 0


class ReflectionItem(BaseModel):
    round: int
    retrieve: str
    isrel: str
    issup: str
    isuse: int
    followup_query: str = "None"
    raw_text: str = ""


class TraceItem(BaseModel):
    stage: str
    query: Optional[str] = None
    round: Optional[int] = None
    retrieve: Optional[str] = None
    retrieved_count: Optional[int] = None
    top_sources: Optional[List[str]] = None
    detail: Optional[str] = None


class CandidateItem(BaseModel):
    answer: str
    evidence_score: float
    final_score: float
    retrieve: str
    isrel: str
    issup: str
    isuse: int


class RiskItem(BaseModel):
    risk_score: float
    risk_level: str
    reasons: List[str]
    confidence: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationItem]
    reflections: List[ReflectionItem]
    trace: List[TraceItem]
    candidates: List[CandidateItem] = []
    retrieval_risk: Optional[RiskItem] = None
    generation_risk: Optional[RiskItem] = None


# =========================
# 工具函数
# =========================

def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".pdf", ".docx", ".txt"}


def get_loader():
    return MultiFormatLoader(max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


def format_source(meta: dict, doc_id: str) -> str:
    title = meta.get("title", "") if meta else ""
    page_no = int(meta.get("page_no", 0)) if meta else 0
    source = title if title else doc_id
    if page_no > 0:
        source = f"{source} | page {page_no}"
    return source


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    if session_id not in app_state["sessions"]:
        app_state["sessions"][session_id] = []
    return app_state["sessions"][session_id]


def find_file_id_by_filename(filename: str) -> Optional[str]:
    for file_id, index_obj in app_state["file_indexes"].items():
        if index_obj.get("filename") == filename:
            return file_id
    return None


def build_retriever() -> MultiVectorRetriever:
    if not app_state["file_indexes"]:
        raise HTTPException(status_code=400, detail="当前没有已索引文档。")

    return MultiVectorRetriever(
        file_indexes=app_state["file_indexes"],
        alpha=ALPHA
    )


def to_risk_item(data: Optional[Dict[str, Any]]) -> Optional[RiskItem]:
    if not data:
        return None
    return RiskItem(
        risk_score=float(data.get("risk_score", 1.0)),
        risk_level=data.get("risk_level", "high"),
        reasons=data.get("reasons", []),
        confidence=(
            float(data.get("confidence", 0.0))
            if data.get("confidence") is not None
            else None
        ),
    )


def build_citations(gen_output: Dict[str, Any]) -> List[CitationItem]:
    citations = []
    for item in gen_output.get("citations", []):
        citations.append(
            CitationItem(
                index=int(item.get("index", 0)),
                source=item.get("source", ""),
                excerpt=item.get("excerpt", ""),
                doc_id=item.get("doc_id", ""),
                page_no=int(item.get("page_no", 0) or 0),
            )
        )
    return citations


def build_reflections(gen_output: Dict[str, Any]) -> List[ReflectionItem]:
    reflections = []
    if gen_output:
        ref = gen_output.get("reflections", {})
        reflections.append(
            ReflectionItem(
                round=1,
                retrieve=ref.get("retrieve", "Unknown"),
                isrel=ref.get("isrel", "Unknown"),
                issup=ref.get("issup", "Unknown"),
                isuse=int(ref.get("isuse", 1)),
                followup_query=ref.get("followup_query", "None"),
                raw_text=gen_output.get("raw_text", ""),
            )
        )
    return reflections


def build_candidates(gen_output: Dict[str, Any]) -> List[CandidateItem]:
    items = []
    for item in gen_output.get("candidates", []):
        items.append(
            CandidateItem(
                answer=item.get("answer", ""),
                evidence_score=float(item.get("evidence_score", 0.0)),
                final_score=float(item.get("final_score", 0.0)),
                retrieve=item.get("retrieve", "Unknown"),
                isrel=item.get("isrel", "Unknown"),
                issup=item.get("issup", "Unknown"),
                isuse=int(item.get("isuse", 1)),
            )
        )
    return items


def build_trace(
    query_text: str,
    ranked_chunks: List[Dict[str, Any]],
    query_type_stats: Dict[str, Any],
    retrieval_risk: Dict[str, Any],
    generation_risk: Optional[Dict[str, Any]],
) -> List[TraceItem]:
    top_sources = []
    for item in ranked_chunks[:5]:
        top_sources.append(
            format_source(item.get("meta", {}), item.get("doc_id", "unknown_doc"))
        )

    trace = [
        TraceItem(
            stage="retrieval_pipeline",
            query=query_text,
            round=1,
            retrieve="Yes" if ranked_chunks else "No",
            retrieved_count=len(ranked_chunks),
            top_sources=top_sources,
        ),
        TraceItem(
            stage="query_routing",
            detail=str(query_type_stats),
        ),
        TraceItem(
            stage="retrieval_risk",
            detail=(
                f"risk_level={retrieval_risk.get('risk_level', 'unknown')}, "
                f"risk_score={retrieval_risk.get('risk_score', 1.0)}"
            ),
        ),
    ]

    if generation_risk:
        trace.append(
            TraceItem(
                stage="generation_risk",
                detail=(
                    f"risk_level={generation_risk.get('risk_level', 'unknown')}, "
                    f"risk_score={generation_risk.get('risk_score', 1.0)}, "
                    f"confidence={generation_risk.get('confidence', 0.0)}"
                ),
            )
        )

    return trace


# =========================
# 建库 / 恢复索引
# =========================

def build_single_uploaded_file_index(file_path: str, filename: str):
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

    app_state["file_indexes"][file_id] = index_obj
    return file_id


def rebuild_existing_files_on_startup():
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


# =========================
# 生命周期
# =========================

@app.on_event("startup")
def startup_event():
    print("[STARTUP] api server starting...")

    if not SELF_RAG_EAS_URL:
        print("[STARTUP][WARN] SELF_RAG_EAS_URL 未配置，生成阶段会失败。")
    if not SELF_RAG_EAS_TOKEN:
        print("[STARTUP][WARN] SELF_RAG_EAS_TOKEN 未配置，生成阶段会失败。")

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

@app.get("/")
def root():
    return {"message": "Local Document QA API is running."}


@app.get("/health")
def health():
    return {
        "ok": True,
        "file_count": len(app_state["files"]),
        "index_count": len(app_state["file_indexes"]),
        "session_count": len(app_state["sessions"]),
        "selfrag_configured": bool(SELF_RAG_EAS_URL and SELF_RAG_EAS_TOKEN),
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
        if path and os.path.exists(path):
            os.remove(path)
            print(f"[DELETE] removed file: {path}")

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
    query_text = (req.query or "").strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="问题不能为空。")

    if not app_state["file_indexes"]:
        raise HTTPException(status_code=400, detail="当前没有已索引文档。")

    session_id = (req.session_id or "default").strip() or "default"

    try:
        if app_state["reranker"] is None:
            app_state["reranker"] = Reranker()

        retriever = build_retriever()

        query_id = f"api_query_{session_id}"

        result = answer_one_query(
            query_id=query_id,
            query_text=query_text,
            retriever=retriever,
            reranker=app_state["reranker"],
            reranker_lock=app_state["reranker_lock"],
        )

        ranked_chunks_map = result.get("ranked_chunks_map", {}) or {}
        gen_outputs = result.get("gen_outputs", {}) or {}
        query_type_stats = result.get("query_type_stats", {}) or {}

        ranked_chunks = ranked_chunks_map.get(query_id, []) or []
        gen_output = gen_outputs.get(query_id, {}) or {}

        # 会话记录
        history = get_session_history(session_id)
        history.append({"role": "user", "content": query_text})
        history.append({"role": "assistant", "content": gen_output.get("answer", "")})

        retrieval_risk_dict = RiskController.assess_retrieval_risk(ranked_chunks)
        generation_risk_dict = gen_output.get("generation_risk")

        citations = build_citations(gen_output)
        reflections = build_reflections(gen_output)
        candidates = build_candidates(gen_output)

        trace = build_trace(
            query_text=query_text,
            ranked_chunks=ranked_chunks,
            query_type_stats=query_type_stats,
            retrieval_risk=retrieval_risk_dict,
            generation_risk=generation_risk_dict,
        )

        return AskResponse(
            answer=gen_output.get("answer", ""),
            citations=citations,
            reflections=reflections,
            trace=trace,
            candidates=candidates,
            retrieval_risk=to_risk_item(retrieval_risk_dict),
            generation_risk=to_risk_item(generation_risk_dict),
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")