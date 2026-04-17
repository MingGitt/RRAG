import hashlib
import os
import re
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from docx import Document


class MultiFormatLoader:
    def __init__(self, max_chunk_size: int = 800, overlap: int = 100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    # ========= 兼容旧接口：返回 List[str] =========
    def load_file(self, filepath: str) -> List[str]:
        doc = self.load_document(filepath)
        if not doc:
            return []
        return [chunk["text"] for chunk in self.chunk_document(doc)]

    # ========= 新接口：目录 -> 文档字典 =========
    def load_directory(self, dir_path: str) -> Dict[str, Dict]:
        docs: Dict[str, Dict] = {}

        if not os.path.isdir(dir_path):
            raise FileNotFoundError(f"文档目录不存在: {dir_path}")

        for root, _, files in os.walk(dir_path):
            for filename in files:
                path = os.path.join(root, filename)
                ext = os.path.splitext(path)[1].lower()
                if ext not in {".pdf", ".txt", ".docx"}:
                    continue

                doc = self.load_document(path)
                if doc:
                    docs[doc["doc_id"]] = doc

        return docs

    # ========= 新接口：单文件 -> 文档对象 =========
    def load_document(self, filepath: str) -> Optional[Dict]:
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdf":
            raw_text, page_texts = self._load_pdf(filepath)
        elif ext == ".txt":
            raw_text = self._load_txt(filepath)
            page_texts = None
        elif ext == ".docx":
            raw_text = self._load_docx(filepath)
            page_texts = None
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        cleaned_text = self._clean_text(raw_text)
        if not cleaned_text.strip():
            return None

        cleaned_pages = None
        if page_texts is not None:
            cleaned_pages = []
            for i, page_text in enumerate(page_texts, start=1):
                page_cleaned = self._clean_text(page_text)
                if page_cleaned.strip():
                    cleaned_pages.append({
                        "page_no": i,
                        "text": page_cleaned
                    })

        return {
            "doc_id": self._make_doc_id(filepath),
            "title": os.path.basename(filepath),
            "text": cleaned_text,
            "source_path": os.path.abspath(filepath),
            "file_type": ext.lstrip("."),
            "page_texts": cleaned_pages,
        }

    # ========= 新接口：文档对象 -> chunk 对象列表 =========
    def chunk_document(self, doc: Dict) -> List[Dict]:
        # PDF：按页分块并保留 page_no
        if doc.get("file_type") == "pdf" and doc.get("page_texts"):
            return self._chunk_pdf_document(doc)

        # TXT / DOCX：普通分块
        structure_blocks = self._structure_split(doc["text"])
        final_chunks = self._secondary_split(structure_blocks)

        results: List[Dict] = []
        for i, chunk_text in enumerate(final_chunks):
            results.append({
                "chunk_id": f'{doc["doc_id"]}::chunk_{i}',
                "doc_id": doc["doc_id"],
                "text": chunk_text,
                "chunk_index": i,
                "source_path": doc.get("source_path", ""),
                "file_type": doc.get("file_type", ""),
                "title": doc.get("title", ""),
                "page_no": 0,
            })

        return results

    def _chunk_pdf_document(self, doc: Dict) -> List[Dict]:
        results: List[Dict] = []
        global_chunk_idx = 0

        for page in doc["page_texts"]:
            page_no = page["page_no"]
            page_text = page["text"]

            structure_blocks = self._structure_split(page_text)
            final_chunks = self._secondary_split(structure_blocks)

            for local_idx, chunk_text in enumerate(final_chunks):
                results.append({
                    "chunk_id": f'{doc["doc_id"]}::p{page_no}::chunk_{global_chunk_idx}',
                    "doc_id": doc["doc_id"],
                    "text": chunk_text,
                    "chunk_index": global_chunk_idx,
                    "source_path": doc.get("source_path", ""),
                    "file_type": doc.get("file_type", ""),
                    "title": doc.get("title", ""),
                    "page_no": page_no,
                    "page_chunk_index": local_idx,
                })
                global_chunk_idx += 1

        return results

    # ========= PDF =========
    def _load_pdf(self, path: str) -> Tuple[str, List[str]]:
        pdf = fitz.open(path)
        page_texts = []
        for page in pdf:
            page_texts.append(page.get_text("text"))
        full_text = "\n\n".join(page_texts)
        return full_text, page_texts

    # ========= TXT =========
    def _load_txt(self, path: str) -> str:
        encodings = ["utf-8", "utf-8-sig", "gb18030", "latin-1"]
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    # ========= DOCX =========
    def _load_docx(self, path: str) -> str:
        doc = Document(path)
        texts = []
        for para in doc.paragraphs:
            texts.append(para.text)
        return "\n".join(texts)

    # ========= 清洗：保留段落边界 =========
    def _clean_text(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\x00", "")

        cleaned_lines: List[str] = []
        blank_count = 0

        for raw_line in text.split("\n"):
            line = raw_line.replace("\u00a0", " ")
            line = re.sub(r"[ \t]+", " ", line).strip()

            # 去掉很常见的孤立页码行
            if re.fullmatch(r"\d{1,4}", line or ""):
                continue

            if not line:
                blank_count += 1
                if blank_count <= 1:
                    cleaned_lines.append("")
            else:
                blank_count = 0
                cleaned_lines.append(line)

        text = "\n".join(cleaned_lines)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ========= 结构分块 =========
    def _structure_split(self, text: str) -> List[str]:
        blocks = re.split(r"\n{2,}", text)

        cleaned = []
        for block in blocks:
            block = block.strip()
            if len(block) >= 10:
                cleaned.append(block)

        return cleaned

    # ========= 中英文二次切分 =========
    def _secondary_split(self, blocks: List[str]) -> List[str]:
        final_chunks: List[str] = []

        for block in blocks:
            if len(block) <= self.max_chunk_size:
                final_chunks.append(block)
                continue

            sentences = self._split_mixed_sentences(block)

            current = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                if len(current) + len(sent) <= self.max_chunk_size:
                    current += sent
                else:
                    if current.strip():
                        final_chunks.append(current.strip())

                    # 单句过长时，继续硬切
                    if len(sent) > self.max_chunk_size:
                        hard_splits = self._hard_split_long_text(sent)
                        final_chunks.extend(hard_splits[:-1])
                        current = hard_splits[-1] if hard_splits else ""
                    else:
                        current = sent

            if current.strip():
                final_chunks.append(current.strip())

        return final_chunks

    def _split_mixed_sentences(self, text: str) -> List[str]:
        """
        中英文混合切句：
        - 中文标点：。！？；
        - 英文标点：.!?;
        - 次级分隔：，、, 空格（仅在块太长时辅助切分）
        """
        text = text.strip()
        if not text:
            return []

        # 先按强分隔符切句
        parts = re.split(r'(?<=[。！？；.!?;])\s+', text)
        sentences: List[str] = []

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 如果句子仍过长，再按弱分隔符细切
            if len(part) > self.max_chunk_size:
                sub_parts = re.split(r'(?<=[，、,])', part)
                for sub in sub_parts:
                    sub = sub.strip()
                    if sub:
                        sentences.append(sub)
            else:
                sentences.append(part)

        return sentences

    def _hard_split_long_text(self, text: str) -> List[str]:
        step = max(1, self.max_chunk_size - self.overlap)
        results = []
        start = 0
        while start < len(text):
            end = start + self.max_chunk_size
            chunk = text[start:end].strip()
            if chunk:
                results.append(chunk)
            if end >= len(text):
                break
            start += step
        return results

    def _make_doc_id(self, filepath: str) -> str:
        normalized = os.path.abspath(filepath)
        digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:10]
        stem = os.path.splitext(os.path.basename(filepath))[0]
        safe_stem = re.sub(r"[^\w\-]+", "_", stem)
        return f"{safe_stem}_{digest}"