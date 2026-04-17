import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, scan_cache_dir


def is_valid_sentence_transformer_dir(model_dir: str) -> bool:
    """
    粗略判断一个本地目录是不是可用的 sentence-transformers 模型目录。
    """
    if not model_dir:
        return False

    p = Path(model_dir)
    if not p.exists() or not p.is_dir():
        return False

    # sentence-transformers 常见关键文件，命中部分即可认为大概率可用
    required_any = [
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
        "config_sentence_transformers.json",
    ]
    return any((p / name).exists() for name in required_any)


def find_model_in_hf_cache(repo_id: str) -> Optional[str]:
    """
    在 Hugging Face 本地缓存里查找 repo_id，对应最新 snapshot 路径。
    找到则返回本地路径，找不到返回 None。
    """
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return None

    matched = [repo for repo in cache_info.repos if repo.repo_id == repo_id]
    if not matched:
        return None

    repo = matched[0]

    # revisions 可能为空；取最后一个可用 revision 的 snapshot_path
    revisions = list(repo.revisions)
    if not revisions:
        return None

    for rev in reversed(revisions):
        snapshot_path = getattr(rev, "snapshot_path", None)
        if snapshot_path and Path(snapshot_path).exists():
            return str(snapshot_path)

    return None


def resolve_or_download_model(
    repo_id: str,
    local_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    offline: bool = False,
) -> str:
    """
    优先级：
    1. 显式本地目录
    2. HF 已有缓存
    3. 在线下载到 local_dir / cache_dir
    """
    # 1) 显式本地目录优先
    if local_dir and is_valid_sentence_transformer_dir(local_dir):
        print(f"[ModelResolver] use local dir: {local_dir}")
        return local_dir

    # 2) 查 HF cache
    cached = find_model_in_hf_cache(repo_id)
    if cached and is_valid_sentence_transformer_dir(cached):
        print(f"[ModelResolver] use HF cache: {cached}")
        return cached

    # 3) 离线模式则不再联网
    if offline:
        raise FileNotFoundError(
            f"未找到本地模型或缓存，且当前处于离线模式：{repo_id}"
        )

    # 4) 在线下载
    target_dir = local_dir if local_dir else None

    print(f"[ModelResolver] downloading from HF Hub: {repo_id}")
    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        cache_dir=cache_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    # snapshot_download 返回下载后的目录
    if is_valid_sentence_transformer_dir(downloaded_path):
        print(f"[ModelResolver] download complete: {downloaded_path}")
        return downloaded_path

    # 有些情况下 local_dir 才是实际可直接加载目录
    if target_dir and is_valid_sentence_transformer_dir(target_dir):
        print(f"[ModelResolver] download complete: {target_dir}")
        return target_dir

    raise FileNotFoundError(f"模型下载后仍未找到有效目录：{repo_id}")