import os
import re
import fitz  # PyMuPDF
from docx import Document
from typing import List


class MultiFormatLoader:

    def __init__(self, max_chunk_size=800, overlap=100):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    # ========= 公共入口 =========
    def load_file(self, filepath: str) -> List[str]:
        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdf":
            text = self._load_pdf(filepath)
        elif ext == ".txt":
            text = self._load_txt(filepath)
        elif ext == ".docx":
            text = self._load_docx(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        text = self._clean_text(text)
        structure_blocks = self._structure_split(text)
        final_chunks = self._secondary_split(structure_blocks)

        return final_chunks

    # ========= PDF =========
    def _load_pdf(self, path):
        doc = fitz.open(path)
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)

    # ========= TXT =========
    def _load_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ========= DOCX =========
    def _load_docx(self, path):
        doc = Document(path)
        texts = []
        for para in doc.paragraphs:
            texts.append(para.text)
        return "\n".join(texts)

    # ========= 清洗 =========
    def _clean_text(self, text):

        # 去多余空格
        text = re.sub(r"\s+", " ", text)

        # 去页眉页脚样式数字
        text = re.sub(r"\n?\d+\n?", " ", text)

        # 去多重空行
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    # ========= 结构分块 =========
    def _structure_split(self, text):

        # 以标题、空行、段落为结构边界
        blocks = re.split(r"\n{2,}", text)

        cleaned = []
        for block in blocks:
            block = block.strip()
            if len(block) > 30:
                cleaned.append(block)

        return cleaned

    # ========= 过大块再固定长度切分 =========
    def _secondary_split(self, blocks):

        final_chunks = []

        for block in blocks:

            if len(block) <= self.max_chunk_size:
                final_chunks.append(block)
            else:
                # 滑动窗口切分
                start = 0
                while start < len(block):
                    end = start + self.max_chunk_size
                    chunk = block[start:end]
                    final_chunks.append(chunk)
                    start += self.max_chunk_size - self.overlap

        return final_chunks