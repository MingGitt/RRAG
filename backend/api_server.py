import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

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
from hybrid_retriever import MultiVectorRetriever
from qa_service import answer_one_query
from reranker import Reranker
from vector_store import VectorStoreBuilder


BASE_DIR = Path(LOCAL_DOCS_DIR)
BASE_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PERSIST_ROOT = Path(PERSIST_DIR)
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Local Document QA API", version="8.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class ReflectionItem(BaseModel):
    round: int
    retrieve: str
    isrel: str
    issup: str
    isuse: int
    followup_query: str = "None"
    raw_text: str


class TraceItem(BaseModel):
    stage: str
    query: Optional[str] = None
    round: Optional[int] = None
    retrieve: Optional[str] = None
    retrieved_count: Optional[int] = None
    top_sources: Optional[List[str]] = None
    detail: Optional[str] = None


class CandidateItem(BaseModel):
    answer: str = ""
    retrieve: str = "Unknown"
    isrel: str = "Unknown"
    issup: str = "Unknown"
    isuse: int = 1
    evidence_score: float = 0.0
    final_score: float = 0.0
    subquery: str = ""


class RiskInfo(BaseModel):
    risk_level: str = "unknown"
    risk_score: float = 0.0
    detail: str = ""


class SubAnswerItem(BaseModel):
    subquery: str
    answer: str
    citations: List[CitationItem] = []
    best_score: float = 0.0


class AskResponse(BaseModel):
    answer: str
    query_mode: str = "simple"
    citations: List[CitationItem]
    reflections: List[ReflectionItem]
    trace: List[TraceItem]
    candidates: List[CandidateItem] = []
    sub_answers: List[SubAnswerItem] = []
    retrieval_risk: Optional[RiskInfo] = None
    generation_risk: Optional[RiskInfo] = None


app_state = {
    "files": {},
    "file_indexes": {},
    "reranker": None,
    "reranker_lock": threading.Lock(),
    "sessions": {},
}


def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".pdf", ".docx", ".txt"}


def get_loader():
    return MultiFormatLoader(max_chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)


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
            app_state["files"][path.name]["file_id"] = file_id

            print(f"[STARTUP] indexed {path.name}, file_id={file_id}, chunks={len(chunks)}")

        except Exception as e:
            print(f"[STARTUP][ERROR] {path.name}: {e}")
            app_state["files"][path.name] = FileRecord(
                name=path.name,
                path=str(path),
                status="index_failed",
                error=str(e)
            ).dict()


def find_file_id_by_filename(filename: str):
    for file_id, index_obj in app_state["file_indexes"].items():
        if index_obj.get("filename") == filename:
            return file_id
    return None


def build_retriever():
    if not app_state["file_indexes"]:
        raise HTTPException(status_code=400, detail="当前没有已索引文档。")

    return MultiVectorRetriever(
        file_indexes=app_state["file_indexes"],
        alpha=ALPHA
    )


def get_session_history(session_id: str) -> List[Dict]:
    if session_id not in app_state["sessions"]:
        app_state["sessions"][session_id] = []
    return app_state["sessions"][session_id]


def history_to_text(history: List[Dict]) -> str:
    if not history:
        return ""
    lines = []
    for item in history[-6:]:
        role = item.get("role", "user")
        content = (item.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def compact_assistant_history(text: str, max_len: int = 180) -> str:
    if not text:
        return ""

    text = " ".join(str(text).split())

    bad_patterns = [
        r"\[\d+\]\s*来源[:：].*?(?=\[\d+\]|\Z)",
        r"\[\d+\]\s*Source:.*?(?=\[\d+\]|\Z)",
        r"The evidence includes.*",
        r"From the current evidence.*",
        r"根据当前证据.*",
        r"从当前证据来看.*",
    ]
    for p in bad_patterns:
        text = re.sub(p, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def _safe_float(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val, default=1):
    try:
        return int(val)
    except Exception:
        return default


def _build_query_mode(query_type_stats: Dict) -> str:
    if not query_type_stats:
        return "simple"
    if query_type_stats.get("complex"):
        return "complex"
    if query_type_stats.get("fuzzy"):
        return "fuzzy"
    return "simple"


def _convert_sub_answers(raw_sub_answers: List[Dict]) -> List[SubAnswerItem]:
    sub_answers = []
    for item in raw_sub_answers or []:
        raw_citations = item.get("citations", []) or []
        citations = [
            CitationItem(
                index=c.get("index", idx + 1),
                source=c.get("source", ""),
                excerpt=c.get("excerpt", ""),
            )
            for idx, c in enumerate(raw_citations)
        ]
        sub_answers.append(
            SubAnswerItem(
                subquery=item.get("subquery", ""),
                answer=item.get("answer", ""),
                citations=citations,
                best_score=_safe_float(item.get("best_score", 0.0), 0.0),
            )
        )
    return sub_answers


async def answer_one_query_api(query_text: str, session_id: str = "default"):
    history = get_session_history(session_id)

    if app_state["reranker"] is None:
        app_state["reranker"] = Reranker()

    retriever = build_retriever()
    qid = f"api_query_{session_id}"

    result = await answer_one_query(
        query_id=qid,
        query_text=query_text,
        retriever=retriever,
        reranker=app_state["reranker"],
        reranker_lock=app_state["reranker_lock"],
    )

    query_type_stats = result.get("query_type_stats", {}) or {}
    gen_output = (result.get("gen_outputs", {}) or {}).get(qid, {}) or {}
    ranked_chunks_map = result.get("ranked_chunks_map", {}) or {}

    query_mode = _build_query_mode(query_type_stats)

    trace: List[TraceItem] = [
        TraceItem(
            stage="query_pipeline",
            query=query_text,
            round=1,
            retrieve=gen_output.get("reflections", {}).get("retrieve", "Unknown"),
            retrieved_count=len(ranked_chunks_map.get(qid, [])),
            top_sources=[
                f'{item["meta"].get("title", item["doc_id"])}'
                for item in ranked_chunks_map.get(qid, [])[:5]
            ] if ranked_chunks_map.get(qid) else [],
            detail=f"query_mode={query_mode}, query_type_stats={query_type_stats}",
        )
    ]

    citations = [
        CitationItem(
            index=item["index"],
            source=item["source"],
            excerpt=item["excerpt"],
        )
        for item in gen_output.get("citations", [])
    ]

    reflections = []
    ref = gen_output.get("reflections", {})
    if ref:
        reflections.append(
            ReflectionItem(
                round=1,
                retrieve=ref.get("retrieve", "Unknown"),
                isrel=ref.get("isrel", "Unknown"),
                issup=ref.get("issup", "Unknown"),
                isuse=_safe_int(ref.get("isuse", 1), 1),
                followup_query=ref.get("followup_query", "None"),
                raw_text=gen_output.get("raw_text", ""),
            )
        )

    candidates = []
    for item in gen_output.get("candidates", []) or []:
        candidates.append(
            CandidateItem(
                answer=item.get("answer", ""),
                retrieve=item.get("retrieve", "Unknown"),
                isrel=item.get("isrel", "Unknown"),
                issup=item.get("issup", "Unknown"),
                isuse=_safe_int(item.get("isuse", 1), 1),
                evidence_score=_safe_float(item.get("evidence_score", 0.0), 0.0),
                final_score=_safe_float(item.get("final_score", 0.0), 0.0),
                subquery=item.get("subquery", ""),
            )
        )

    sub_answers = _convert_sub_answers(gen_output.get("sub_answers", []) or [])

    retrieval_risk = None

    generation_risk = None
    raw_generation_risk = gen_output.get("generation_risk")
    if raw_generation_risk:
        generation_risk = RiskInfo(
            risk_level=raw_generation_risk.get("risk_level", "unknown"),
            risk_score=_safe_float(raw_generation_risk.get("risk_score", 0.0), 0.0),
            detail=str(raw_generation_risk.get("reasons", "")),
        )

    answer = gen_output.get("answer", "")

    history.append({"role": "user", "content": query_text})
    history.append({"role": "assistant", "content": compact_assistant_history(answer)})

    # fuzzy / complex 时，前端主要展示 sub_answers；总 citations / candidates 仍返回，但可由前端隐藏
    return AskResponse(
        answer=answer,
        query_mode=query_mode,
        citations=citations,
        reflections=reflections,
        trace=trace,
        candidates=candidates,
        sub_answers=sub_answers,
        retrieval_risk=retrieval_risk,
        generation_risk=generation_risk,
    )


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


@app.get("/health")
def health():
    return {
        "ok": True,
        "file_count": len(app_state["files"]),
        "index_count": len(app_state["file_indexes"]),
        "session_count": len(app_state["sessions"]),
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
            app_state["files"][f.filename]["file_id"] = file_id

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

        old_file_id = record.get("file_id")
        if not old_file_id:
            old_file_id = find_file_id_by_filename(filename)

        if old_file_id:
            VectorStoreBuilder.delete_single_file_index(old_file_id, str(PERSIST_ROOT))
            if old_file_id in app_state["file_indexes"]:
                del app_state["file_indexes"][old_file_id]

        new_file_id = build_single_uploaded_file_index(path, filename)

        app_state["files"][filename]["status"] = "indexed"
        app_state["files"][filename]["error"] = ""
        app_state["files"][filename]["file_id"] = new_file_id

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
    target_file_id = record.get("file_id")

    try:
        if path and os.path.exists(path):
            os.remove(path)
            print(f"[DELETE] removed file: {path}")

        if not target_file_id:
            target_file_id = find_file_id_by_filename(filename)

        if target_file_id:
            VectorStoreBuilder.delete_single_file_index(target_file_id, str(PERSIST_ROOT))
            if target_file_id in app_state["file_indexes"]:
                del app_state["file_indexes"][target_file_id]
            print(f"[DELETE] removed index: file_id={target_file_id}")
        else:
            print(f"[DELETE][WARN] no file_id found for filename={filename}, index may remain on disk")

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

    session_id = (req.session_id or "default").strip() or "default"
    return await answer_one_query_api(query, session_id=session_id)