import os
from typing import Optional


HF_LOCAL_HUB = r"C:\Users\86151\.cache\huggingface\hub"


def _normalize_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.abspath(os.path.expanduser(path))


def _find_snapshot_dir(repo_root: str) -> Optional[str]:
    if not os.path.exists(repo_root):
        return None

    snapshots_dir = os.path.join(repo_root, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    snapshot_names = [
        name for name in os.listdir(snapshots_dir)
        if os.path.isdir(os.path.join(snapshots_dir, name))
    ]
    if not snapshot_names:
        return None

    snapshot_names.sort()
    return os.path.join(snapshots_dir, snapshot_names[-1])


def _repo_root_from_model_name(base_dir: str, model_name: str) -> str:
    repo_folder = "models--" + model_name.replace("/", "--")
    return os.path.join(base_dir, repo_folder)


def resolve_or_download_model(
    model_name: Optional[str] = None,
    local_dir: Optional[str] = None,
    cache_dir: str = "./hf_cache",
    repo_id: Optional[str] = None,
    offline: bool = False,
) -> str:
    if model_name is None:
        model_name = repo_id

    if not model_name:
        raise ValueError("resolve_or_download_model() 缺少 model_name 或 repo_id")

    # 1) 显式本地目录
    local_dir = _normalize_path(local_dir)
    if local_dir and os.path.exists(local_dir):
        print(f"[ModelResolver] use local dir: {local_dir}")
        return local_dir

    # 2) 固定查 HF 默认缓存
    hf_default_hub = _normalize_path(HF_LOCAL_HUB)
    repo_root_default = _repo_root_from_model_name(hf_default_hub, model_name)
    snapshot_default = _find_snapshot_dir(repo_root_default)
    if snapshot_default:
        print(f"[ModelResolver] use HF default cache: {snapshot_default}")
        return snapshot_default

    # 3) 不再使用 project cache，也不联网
    raise RuntimeError(
        "\n"
        f"[ModelResolver] 未找到可用本地模型: {model_name}\n"
        f"已检查位置：\n"
        f"1. local_dir = {local_dir}\n"
        f"2. HF 默认缓存 = {hf_default_hub}\n\n"
        f"当前已禁用 project cache 和联网下载。\n"
        f"请确认模型存在于上述路径之一。\n"
    )