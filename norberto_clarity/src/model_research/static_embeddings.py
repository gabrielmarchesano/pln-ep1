"""CPU-friendly multilingual embeddings distilled from BGE-M3."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

from clarity.data import normalize_text

STATIC_MODEL_ID = "minishlab/potion-multilingual-128M"
STATIC_REVISION = "73908c3438cf03b6a01bcb9611d62b23d0726f08"


@lru_cache(maxsize=1)
def load_static_encoder(model_name, revision, cache_dir):
    try:
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel
    except ImportError as error:
        raise ImportError("Instale requirements/research/static.txt para usar Model2Vec.") from error
    snapshot = snapshot_download(
        model_name, revision=revision, cache_dir=cache_dir,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "README.md"],
    )
    return StaticModel.from_pretrained(snapshot, normalize=True)


class StaticEmbeddings(TransformerMixin, BaseEstimator):
    def __init__(self, model_name=STATIC_MODEL_ID, revision=STATIC_REVISION,
                 cache_dir=".cache/huggingface"):
        self.model_name = model_name
        self.revision = revision
        self.cache_dir = cache_dir

    def fit(self, X, y=None):
        if not self.revision or self.revision == "main":
            raise ValueError("Use uma revisão fixa do modelo semântico.")
        return self

    def transform(self, X):
        self.fit(X)
        encoder = load_static_encoder(self.model_name, self.revision, self.cache_dir)
        # Keep the entire answer; static embeddings don't have a context window.
        vectors = encoder.encode([normalize_text(text) for text in X],
                                 max_length=None, use_multiprocessing=False)
        vectors = np.asarray(vectors, dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("O encoder produziu embeddings não finitos.")
        return normalize(vectors)
