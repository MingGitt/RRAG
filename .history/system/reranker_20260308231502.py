import os
import torch
from sentence_transformers import CrossEncoder

from config import (
    RERANK_MODEL_NAME,
    RERANK_MODEL_LOCAL_DIR,
    HF_CACHE_DIR,
    HF_HUB_ETAG_TIMEOUT,
    HF_HUB_DOWNLOAD_TIMEOUT,
)
from model_utils import resolve_or_download_model


class Reranker:

    def __init__(self, model_name=None, batch_size=32, max_length=512):
        os.environ["HF_HOME"] = HF_CACHE_DIR
        os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
        os.environ["HF_HUB_ETAG_TIMEOUT"] = HF_HUB_ETAG_TIMEOUT
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = HF_HUB_DOWNLOAD_TIMEOUT

        resolved_model = resolve_or_download_model(
            repo_id=RERANK_MODEL_NAME,
            local_dir=RERANK_MODEL_LOCAL_DIR,
            cache_dir=HF_CACHE_DIR,
            offline=False,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            resolved_model,
            device=device,
            max_length=max_length
        )
        self.batch_size = batch_size

    def rerank_texts(self, query, texts, ids, top_k=None):
        texts = [t[:1200] for t in texts]
        pairs = [(query, t) for t in texts]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        ranked = sorted(zip(ids, texts, scores), key=lambda x: x[2], reverse=True)
        if top_k:
            ranked = ranked[:top_k]
        return ranked