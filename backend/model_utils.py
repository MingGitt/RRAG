import os
from typing import Optional

from huggingface_hub import snapshot_download


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
    """
    优先级：
    1. 显式 local_dir
    2. HF 默认缓存 / 当前项目 cache
    3. 若 offline=False，则自动 snapshot_download
    """
    if model_name is None:
        model_name = repo_id

    if not model_name:
        raise ValueError("resolve_or_download_model() 缺少 model_name 或 repo_id")

    local_dir = _normalize_path(local_dir)
    cache_dir = _normalize_path(cache_dir) or os.path.abspath("./hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1) 显式本地目录
    if local_dir and os.path.exists(local_dir):
        print(f"[ModelResolver] use local dir: {local_dir}")
        return local_dir

    # 2) HF_HOME / 默认缓存
    candidate_hubs = []

    env_hf_home = os.environ.get("HF_HOME")
    env_hf_hub_cache = os.environ.get("HF_HUB_CACHE")

    if env_hf_hub_cache:
        candidate_hubs.append(_normalize_path(env_hf_hub_cache))
    if env_hf_home:
        candidate_hubs.append(os.path.join(_normalize_path(env_hf_home), "hub"))

    # 当前项目 cache 也检查
    candidate_hubs.append(cache_dir)

    # Linux/macOS 默认缓存
    candidate_hubs.append(os.path.expanduser("~/.cache/huggingface/hub"))
    # Windows 默认缓存
    candidate_hubs.append(os.path.expanduser("~/.cache/huggingface/hub"))
    candidate_hubs.append(os.path.expanduser("~/AppData/Local/huggingface/hub"))
    candidate_hubs.append(os.path.expanduser("~/AppData/Local/Packages/huggingface/hub"))

    checked = []
    for hub_dir in candidate_hubs:
        if not hub_dir:
            continue
        repo_root = _repo_root_from_model_name(hub_dir, model_name)
        snapshot = _find_snapshot_dir(repo_root)
        checked.append(repo_root)
        if snapshot:
            print(f"[ModelResolver] use cached snapshot: {snapshot}")
            return snapshot

    # 3) 自动下载
    if not offline:
        print(f"[ModelResolver] downloading model from HF: {model_name}")
        path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=None,
            local_dir_use_symlinks=False,
        )
        print(f"[ModelResolver] downloaded to: {path}")
        return path

    raise RuntimeError(
        "\n"
        f"[ModelResolver] 未找到可用模型: {model_name}\n"
        f"已检查缓存路径：\n- " + "\n- ".join(checked) + "\n\n"
        f"当前 offline=True，已禁止自动下载。\n"
        f"请设置 SELF_RAG_MODEL_LOCAL_DIR 或关闭离线模式。\n"
    )