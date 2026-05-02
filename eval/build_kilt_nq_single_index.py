import os
import sys
import re
import json
import shutil
import argparse
import hashlib
from typing import Dict, Any, List

from tqdm import tqdm


# ============================================================
# 1. 让 eval 脚本可以导入 backend 里的模块
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_ROOT, "backend")

if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from data_loader import MultiFormatLoader
from vector_store import VectorStoreBuilder
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ============================================================
# 2. 工具函数
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    if not os.path.exists(path):
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def safe_id(text: str, max_len: int = 80) -> str:
    """
    构造稳定、安全的 doc_id。
    """
    text = text or "untitled"
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    text = text[:max_len] if text else "untitled"
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{text}_{digest}"


def load_metadata_map(metadata_path: str) -> Dict[str, Dict[str, Any]]:
    """
    读取 fetch_kilt_nq_200_fulltext.py 生成的 metadata.jsonl。

    返回：
        filename -> metadata
    """
    records = read_jsonl(metadata_path)
    mapping = {}

    for r in records:
        filename = r.get("filename")
        if filename:
            mapping[filename] = r

    return mapping


def build_kilt_chunks(
    data_dir: str,
    max_chunk_size: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    读取 kilt_nq_200_fulltext/txt 目录下所有 txt，
    进行清洗、分块，并保留 KILT title / wikipedia_id 等元信息。
    """
    txt_dir = os.path.join(data_dir, "txt")
    metadata_path = os.path.join(data_dir, "metadata.jsonl")

    if not os.path.isdir(txt_dir):
        raise FileNotFoundError(f"txt 目录不存在: {txt_dir}")

    metadata_map = load_metadata_map(metadata_path)

    loader = MultiFormatLoader(
        max_chunk_size=max_chunk_size,
        overlap=overlap,
    )

    all_chunks: List[Dict[str, Any]] = []

    txt_files = []
    for root, _, files in os.walk(txt_dir):
        for name in files:
            if name.lower().endswith(".txt"):
                txt_files.append(os.path.join(root, name))

    txt_files = sorted(txt_files)

    print(f"发现 txt 文件数: {len(txt_files)}")

    for path in tqdm(txt_files, desc="Loading and chunking KILT txt"):
        filename = os.path.basename(path)
        meta = metadata_map.get(filename, {})

        original_title = (
            meta.get("original_kilt_title")
            or meta.get("wikipedia_api_title")
            or os.path.splitext(filename)[0].replace("_", " ")
        )

        wikipedia_api_pageid = meta.get("wikipedia_api_pageid")
        kilt_wikipedia_ids = meta.get("kilt_wikipedia_ids", [])

        doc = loader.load_document(path)
        if not doc:
            continue

        # 用 KILT title 作为文档 title，方便后续 provenance 命中判断
        doc_id = safe_id(original_title)

        doc["doc_id"] = doc_id
        doc["title"] = original_title
        doc["source_path"] = os.path.abspath(path)
        doc["file_type"] = "txt"

        chunks = loader.chunk_document(doc)

        for ch in chunks:
            ch["doc_id"] = doc_id
            ch["title"] = original_title
            ch["source_path"] = os.path.abspath(path)
            ch["file_type"] = "txt"

            # 重新生成稳定 chunk_id，避免路径变化导致 id 不一致
            chunk_index = int(ch.get("chunk_index", 0))
            ch["chunk_id"] = f"{doc_id}::chunk_{chunk_index}"

            # 额外保存 KILT / Wikipedia 元数据
            ch["wikipedia_api_pageid"] = wikipedia_api_pageid or ""
            ch["kilt_wikipedia_ids"] = ",".join([str(x) for x in kilt_wikipedia_ids])
            ch["kilt_title"] = original_title
            ch["source"] = "kilt_nq_200_fulltext"

            all_chunks.append(ch)

    print(f"总 chunk 数: {len(all_chunks)}")
    return all_chunks


# ============================================================
# 3. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"D:\code\rag\FSR\data\kilt_nq_200_fulltext",
        help="kilt_nq_200_fulltext 文件夹路径。",
    )

    parser.add_argument(
        "--persist_dir",
        type=str,
        default=r"D:\code\rag\FSR\chroma_kilt_nq_200_single",
        help="统一 Chroma 向量库保存目录。",
    )

    parser.add_argument(
        "--collection_name",
        type=str,
        default="kilt_nq_200_single",
        help="Chroma collection 名称。",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE,
    )

    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=CHUNK_OVERLAP,
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="如果指定，则删除旧向量库后重新构建。",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    if args.rebuild and os.path.exists(args.persist_dir):
        print(f"删除旧索引目录: {args.persist_dir}")
        shutil.rmtree(args.persist_dir, ignore_errors=True)

    os.makedirs(args.persist_dir, exist_ok=True)

    chunks = build_kilt_chunks(
        data_dir=args.data_dir,
        max_chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )

    if not chunks:
        raise RuntimeError("没有构建出任何 chunk，请检查 txt 文件是否为空。")

    print("\n开始构建统一 Chroma collection...")
    print(f"persist_dir: {args.persist_dir}")
    print(f"collection_name: {args.collection_name}")

    vectordb, chunk_texts, chunk_doc_ids, chunk_ids, chunk_metas = (
        VectorStoreBuilder.build_or_load_from_chunks(
            chunks=chunks,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
        )
    )

    try:
        count = vectordb._collection.count()
    except Exception:
        count = len(chunk_texts)

    index_info = {
        "data_dir": args.data_dir,
        "persist_dir": args.persist_dir,
        "collection_name": args.collection_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_chunks": len(chunk_texts),
        "chroma_count": count,
    }

    info_path = os.path.join(args.persist_dir, "index_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(index_info, f, ensure_ascii=False, indent=2)

    print("\n========== Build Done ==========")
    print(json.dumps(index_info, ensure_ascii=False, indent=2))
    print(f"索引信息已保存: {info_path}")


if __name__ == "__main__":
    main()