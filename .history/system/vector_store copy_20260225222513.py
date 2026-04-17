import hashlib
import re
from typing import Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreBuilder:

    @staticmethod
    def make_collection_name(base, chunk_size, chunk_overlap, embed_model):
        key = f"{base}|cs={chunk_size}|co={chunk_overlap}|emb={embed_model}"
        suffix = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{base}_chunks_{suffix}"

    @staticmethod
    def split_by_structure_then_length(text: str, max_len: int = 400, overlap: int = 50):
        # ===== Step1: 按结构切 =====
        # 识别标题 / 空行 / 段落
        structural_chunks = re.split(r"\n\s*\n|(?=\n[A-Z][^\n]{0,80}\n)", text)

        structural_chunks = [c.strip() for c in structural_chunks if c.strip()]

        """ # ===== Step2: 长度 fallback =====
        splitter = RecursiveCharacterTextSplitter(chunk_size=max_len, chunk_overlap=overlap)

        final_chunks = []

        for chunk in structural_chunks:
            # 小块直接保留
            if len(chunk) <= max_len:
                final_chunks.append(chunk)
            # 大块再递归切
            else:
                sub_chunks = splitter.split_text(chunk)
                final_chunks.extend(sub_chunks)

        return final_chunks """
        return structural_chunks

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

        chunk_texts, chunk_doc_ids = [], []

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

                # ===== 使用结构感知分块 =====
                chunks = VectorStoreBuilder.split_by_structure_then_length(full_text, max_len=chunk_size, overlap=chunk_overlap)

                for i, ch in enumerate(chunks):
                    texts.append(ch)
                    metadatas.append({"doc_id": doc_id})
                    ids.append(f"{doc_id}::chunk_{i}")
                    chunk_texts.append(ch)
                    chunk_doc_ids.append(doc_id)

            vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        else:
            data = vectordb.get(include=["documents", "metadatas"])
            for text, meta in zip(data["documents"], data["metadatas"]):
                chunk_texts.append(text)
                chunk_doc_ids.append(meta["doc_id"])

        return vectordb, chunk_texts, chunk_doc_ids