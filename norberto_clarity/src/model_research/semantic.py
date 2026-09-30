"""Frozen multilingual representations with a label-independent, persistent cache."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize

from clarity.data import normalize_text

MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@lru_cache(maxsize=1)
def load_encoder(model_name, revision, model_cache, device):
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise ImportError("Instale requirements/research/semantic.txt para usar embeddings.") from error
    torch.set_num_threads(2)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("PyTorch cannot access a GPU. Check the driver and CUDA/ROCm runtime.")
    encoder = SentenceTransformer(
        model_name, revision=revision, cache_folder=model_cache,
        device=device, trust_remote_code=False,
    )
    encoder.eval()
    return encoder


def chunk_token_ids(tokens, budget, max_chunks):
    """Choose evenly spaced windows, always including the beginning and end."""
    windows = [tokens[index:index + budget] for index in range(0, len(tokens), budget)] or [[]]
    if len(windows) > max_chunks:
        indices = np.linspace(0, len(windows) - 1, max_chunks, dtype=int)
        windows = [windows[index] for index in indices]
    return windows


def encode_documents(encoder, texts, *, batch_size, max_chunks):
    """Pool selected token windows without decoding and re-tokenizing boundaries."""
    import torch

    tokenizer = encoder.tokenizer
    if tokenizer.num_special_tokens_to_add(pair=False) != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("O encoder deve usar uma sequência com tokens CLS e SEP.")
    budget = encoder.max_seq_length - 2
    documents = tokenizer(list(texts), add_special_tokens=False, truncation=False,
                          verbose=False)["input_ids"]
    windows, owners = [], []
    for owner, tokens in enumerate(documents):
        for chunk in chunk_token_ids(tokens, budget, max_chunks):
            # This model uses the XLM-R fast tokenizer with BERT weights.
            # prepare_for_model() on its generic fast tokenizer does not apply
            # the backend postprocessor; add its CLS/SEP explicitly instead.
            ids = [tokenizer.cls_token_id, *chunk, tokenizer.sep_token_id]
            windows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
            owners.append(owner)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            features = tokenizer.pad(windows[start:start + batch_size],
                                     padding=True, return_tensors="pt")
            features = {key: value.to(encoder.device) for key, value in features.items()}
            vectors = encoder(features)["sentence_embedding"]
            chunks.append(vectors.cpu().numpy())
    vectors = np.concatenate(chunks).astype(np.float32)
    pooled = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    np.add.at(pooled, owners, vectors)
    pooled /= np.bincount(owners)[:, None]
    return normalize(pooled).astype(np.float32)


class FrozenEmbeddings(TransformerMixin, BaseEstimator):
    """The encoder never sees labels or fits on the assignment corpus.

    The classifier is still fitted separately in each CV fold. Only the fixed
    document representations may be shared across folds.
    """

    def __init__(self, model_name=MODEL_ID, revision=MODEL_REVISION,
                 cache_path=".cache/embeddings.sqlite3", model_cache=".cache/huggingface",
                 batch_size=16, max_chunks=3, device="auto"):
        self.model_name = model_name
        self.revision = revision
        self.cache_path = cache_path
        self.model_cache = model_cache
        self.batch_size = batch_size
        self.max_chunks = max_chunks
        self.device = device

    def fit(self, X, y=None):
        if self.batch_size < 1 or self.max_chunks < 1:
            raise ValueError("batch_size e max_chunks devem ser positivos.")
        if not self.revision or self.revision == "main":
            raise ValueError("Use uma revisão fixa do encoder para reprodução e cache.")
        return self

    def transform(self, X):
        self.fit(X)
        texts = [normalize_text(text) for text in X]
        if not texts:
            raise ValueError("Não há textos para codificar.")
        config = json.dumps({"model": self.model_name, "revision": self.revision,
                             "max_chunks": self.max_chunks, "pooling_version": 1}, sort_keys=True)
        keys = [hashlib.sha256((config + "\n" + text).encode()).hexdigest() for text in texts]
        cache_path = Path(self.cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(cache_path, timeout=60)
        try:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings "
                               "(key TEXT PRIMARY KEY, vector BLOB NOT NULL)")
            vectors, missing = {}, {}
            for key, text in zip(keys, texts):
                if key in vectors or key in missing:
                    continue
                row = connection.execute("SELECT vector FROM embeddings WHERE key = ?", (key,)).fetchone()
                if row:
                    vectors[key] = np.frombuffer(row[0], dtype=np.float32).copy()
                else:
                    missing[key] = text
            if missing:
                encoder = load_encoder(self.model_name, self.revision, self.model_cache, self.device)
                entries = list(missing.items())
                print(f"Embeddings: {len(entries)} textos únicos ainda sem cache.", flush=True)
                for start in range(0, len(entries), 128):
                    batch = entries[start:start + 128]
                    encoded = encode_documents(encoder, [text for _, text in batch],
                                               batch_size=self.batch_size, max_chunks=self.max_chunks)
                    for (key, _), vector in zip(batch, encoded):
                        vectors[key] = vector
                        connection.execute("INSERT OR REPLACE INTO embeddings VALUES (?, ?)",
                                           (key, vector.tobytes()))
                    connection.commit()
                    print(f"  Codificados: {min(start + 128, len(entries))}/{len(entries)}", flush=True)
            result = np.stack([vectors[key] for key in keys])
            if not np.isfinite(result).all():
                raise ValueError("Cache de embeddings contém valores não finitos.")
            return result
        finally:
            connection.close()
