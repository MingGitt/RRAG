import hashlib
import os
import re
import shutil
from typing import Dict, Iterable, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    EMBED_MODEL_LOCAL_DIR,
    EMBED_MODEL_NAME,
    HF_CACHE_DIR,
    HF_HUB_DOWNLOAD_TIMEOUT,
    HF_HUB_ETAG_TIMEOUT,
)
from model_utils import resolve_or_download_model


class VectorStoreBuilder:
    @staticmethod
    def make_collection_name(base, chunk_size, chunk_overlap, embed_model):
        key = f"{base}|cs={chunk_size}|co={chunk_overlap}|emb={embed_model}"
        suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        safe_base = re.sub(r"[^a-zA-Z0-9_\\-]", "_", str(base))
        return f"{safe_base}_chunks_{suffix}"

    @staticmethod
    def split_by_structure_then_length(text: str, max_len: int = 400, overlap: int = 50):
        structural_chunks = re.split(r"\n\s*\n|(?=\n[A-Z][^\n]{0,80}\n)", text)
        structural_chunks = [c.strip() for c in structural_chunks if c.strip()]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_len,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        final_chunks = []
        for chunk in structural_chunks:
            if len(chunk) <= max_len:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(splitter.split_text(chunk))

        return final_chunks

    @staticmethod
    def _build_embeddings():
        os.environ["HF_HOME"] = HF_CACHE_DIR
        os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
        os.environ["HF_HUB_ETAG_TIMEOUT"] = HF_HUB_ETAG_TIMEOUT
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = HF_HUB_DOWNLOAD_TIMEOUT

        model_path = resolve_or_download_model(
            repo_id=EMBED_MODEL_NAME,
            local_dir=EMBED_MODEL_LOCAL_DIR,
            cache_dir=HF_CACHE_DIR,
            offline=False,
        )

        print(f"[Embedding] loading model from: {model_path}")

        embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings

    @staticmethod
    def _create_vectordb(collection_name: str, persist_dir: str):
        embeddings = VectorStoreBuilder._build_embeddings()
        return Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    @staticmethod
    def _load_existing(vectordb):
        chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = [], [], [], []
        data = vectordb.get(include=["documents", "metadatas"])

        for text, meta in zip(data["documents"], data["metadatas"]):
            chunk_texts.append(text)
            chunk_doc_ids.append(meta["doc_id"])
            chunk_ids.append(meta.get("chunk_id"))
            chunk_metas.append(meta)

        return chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas

    @staticmethod
    def _sanitize_metadata(metadata: Dict) -> Dict:
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    @staticmethod
    def _insert_chunks(vectordb, chunks: Iterable[Dict]):
        texts: List[str] = []
        metadatas: List[Dict] = []
        ids: List[str] = []

        chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = [], [], [], []

        for ch in chunks:
            text = (ch.get("text") or "").strip()
            if not text:
                continue

            chunk_id = ch.get("chunk_id")
            if not chunk_id:
                raise ValueError("chunk 缺少 chunk_id")

            doc_id = ch.get("doc_id")
            if not doc_id:
                raise ValueError("chunk 缺少 doc_id")

            metadata = {
                "doc_id": str(doc_id),
                "chunk_id": str(chunk_id),
                "chunk_idx": int(ch.get("chunk_index", 0)),
                "source_path": ch.get("source_path", "") or "",
                "file_type": ch.get("file_type", "") or "",
                "title": ch.get("title", "") or "",
                "page_no": int(ch.get("page_no", 0) or 0),
            }
            metadata = VectorStoreBuilder._sanitize_metadata(metadata)

            texts.append(text)
            metadatas.append(metadata)
            ids.append(str(chunk_id))

            chunk_texts.append(text)
            chunk_doc_ids.append(str(doc_id))
            chunk_ids.append(str(chunk_id))
            chunk_metas.append(metadata)

        batch_size = 4000
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            vectordb.add_texts(
                texts=texts[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
            print(f"[Chroma] inserted {end}/{len(texts)}")

        return chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas

    @staticmethod
    def build_or_load(corpus: Dict[str, Dict], persist_dir, collection_name, chunk_size, chunk_overlap):
        vectordb = VectorStoreBuilder._create_vectordb(collection_name, persist_dir)

        try:
            existing = vectordb._collection.count()
        except Exception:
            existing = 0

        if existing > 0:
            return (vectordb, *VectorStoreBuilder._load_existing(vectordb))

        chunks = []
        for doc_id, doc in corpus.items():
            full_text = ((doc.get("title") or "") + "\n" + (doc.get("text") or "")).strip()
            if not full_text:
                continue

            parts = VectorStoreBuilder.split_by_structure_then_length(
                full_text, max_len=chunk_size, overlap=chunk_overlap
            )

            for i, ch in enumerate(parts):
                chunks.append({
                    "chunk_id": f"{doc_id}::chunk_{i}",
                    "doc_id": doc_id,
                    "text": ch,
                    "chunk_index": i,
                    "source_path": "",
                    "file_type": "beir",
                    "title": doc.get("title") or "",
                    "page_no": 0,
                })

        chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder._insert_chunks(vectordb, chunks)
        return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas

    @staticmethod
    def build_or_load_from_chunks(chunks: List[Dict], persist_dir, collection_name):
        vectordb = VectorStoreBuilder._create_vectordb(collection_name, persist_dir)

        try:
            existing = vectordb._collection.count()
        except Exception:
            existing = 0

        if existing > 0:
            return (vectordb, *VectorStoreBuilder._load_existing(vectordb))

        chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder._insert_chunks(vectordb, chunks)
        return vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas

    @staticmethod
    def build_single_file_index(file_id: str, chunks: List[Dict], persist_root: str):
        """
        为单个文件单独建库
        """
        persist_dir = os.path.join(persist_root, file_id)

        safe_file_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(file_id))
        safe_file_id = re.sub(r"_+", "_", safe_file_id).strip("_")

        if not safe_file_id:
            safe_file_id = "fileindex"

        collection_name = f"{safe_file_id}_collection"

        # Chroma collection name 最长 63
        if len(collection_name) > 63:
            collection_name = collection_name[:63]

        # 保证首尾是字母或数字
        collection_name = re.sub(r"^[^a-zA-Z0-9]+", "", collection_name)
        collection_name = re.sub(r"[^a-zA-Z0-9]+$", "", collection_name)

        if len(collection_name) < 3:
            collection_name = "file_collection"

        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
        os.makedirs(persist_dir, exist_ok=True)

        vectordb = VectorStoreBuilder._create_vectordb(collection_name, persist_dir)
        chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = VectorStoreBuilder._insert_chunks(vectordb, chunks)

        return {
            "file_id": file_id,
            "vectordb": vectordb,
            "chunk_texts": chunk_texts,
            "chunk_doc_ids": chunk_doc_ids,
            "chunk_ids": chunk_ids,
            "chunk_metas": chunk_metas,
            "persist_dir": persist_dir,
            "collection_name": collection_name,
        }

    @staticmethod
    def delete_single_file_index(file_id: str, persist_root: str):
        """
        删除单个文件对应的向量库目录
        """
        persist_dir = os.path.join(persist_root, file_id)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)