import hashlib
import re
from typing import Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # 新写法


class VectorStoreBuilder:

    @staticmethod
    def make_collection_name(base, chunk_size, chunk_overlap, embed_model):
        key = f"{base}|cs={chunk_size}|co={chunk_overlap}|emb={embed_model}"
        suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{base}_chunks_{suffix}"

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
    def build_or_load(corpus: Dict[str, Dict], embed_model, persist_dir,
                      collection_name, chunk_size, chunk_overlap):
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

        chunk_texts, chunk_doc_ids, chunk_ids = [], [], []

        try:
            existing = vectordb._collection.count()
        except Exception:
            existing = 0

        if existing == 0:
            texts, metadatas, ids = [], [], []

            for doc_id, doc in corpus.items():
                full_text = ((doc.get("title") or "") + "\n" + (doc.get("text") or "")).strip()
                if not full_text:
                    continue

                chunks = VectorStoreBuilder.split_by_structure_then_length(
                    full_text, max_len=chunk_size, overlap=chunk_overlap
                )

                for i, ch in enumerate(chunks):
                    chunk_id = f"{doc_id}::chunk_{i}"
                    texts.append(ch)
                    metadatas.append({"doc_id": doc_id, "chunk_id": chunk_id, "chunk_idx": i})
                    ids.append(chunk_id)

                    chunk_texts.append(ch)
                    chunk_doc_ids.append(doc_id)
                    chunk_ids.append(chunk_id)

            vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        else:
            data = vectordb.get(include=["documents", "metadatas"])
            for text, meta in zip(data["documents"], data["metadatas"]):
                chunk_texts.append(text)
                chunk_doc_ids.append(meta["doc_id"])
                chunk_ids.append(meta.get("chunk_id"))

        return vectordb, chunk_texts, chunk_doc_ids, chunk_ids