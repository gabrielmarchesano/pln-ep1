from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from clarity.data import normalize_text

MODEL_NAMES = ("majority", "baseline", "candidate", "lexical", "semantic", "hybrid",
               "static", "static_hybrid", "word_svc", "linear_svc", "semantic_mlp")


def build_model(name: str, *, seed: int = 42, cache_dir=None) -> Pipeline:
    def word(**kwargs):
        options = dict(
            preprocessor=normalize_text, ngram_range=(1, 2), min_df=2,
            max_df=0.98, sublinear_tf=True, max_features=60_000,
            dtype=np.float32,
        )
        options.update(kwargs)
        return TfidfVectorizer(**options)

    def chars(**kwargs):
        options = dict(
            preprocessor=normalize_text, analyzer="char_wb", ngram_range=(3, 5),
            min_df=3, sublinear_tf=True, max_features=50_000, dtype=np.float32,
        )
        options.update(kwargs)
        return TfidfVectorizer(**options)

    def logistic():
        return LogisticRegression(C=2.0, max_iter=1500, solver="lbfgs", random_state=seed)

    def linear_svc():
        return LinearSVC(C=0.1, dual="auto", max_iter=10_000, tol=1e-4,
                         random_state=seed)

    if name == "majority":
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    if name == "baseline":
        return Pipeline([("tfidf", word()), ("classifier", logistic())], memory=cache_dir)
    if name == "word_svc":
        return Pipeline([("tfidf", word()), ("classifier", linear_svc())], memory=cache_dir)
    if name == "candidate":
        features = FeatureUnion([
            ("word", word(min_df=3, max_features=25_000)),
            ("char", chars(min_df=5, max_features=20_000)),
        ])
        classifier = LinearSVC(C=0.5, max_iter=5000, random_state=seed)
    elif name in {"lexical", "hybrid", "linear_svc"}:
        parts = [("word", word()), ("char", chars())]
        if name == "hybrid":
            from .semantic import FrozenEmbeddings
            parts.append(("semantic", FrozenEmbeddings()))
        features = FeatureUnion(parts)
        classifier = linear_svc() if name == "linear_svc" else logistic()
    elif name in {"semantic", "semantic_mlp"}:
        from .semantic import FrozenEmbeddings
        features = FrozenEmbeddings()
        classifier = logistic() if name == "semantic" else MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu", solver="adam",
            alpha=0.1, batch_size=128, learning_rate_init=0.001,
            max_iter=200, tol=1e-4, n_iter_no_change=15,
            # sklearn's automatic validation split is not group-aware. Stop
            # only on training loss (or the epoch cap), never on outer labels.
            early_stopping=False, shuffle=True, random_state=seed,
        )
    elif name in {"static", "static_hybrid"}:
        from .static_embeddings import StaticEmbeddings
        features = StaticEmbeddings()
        if name == "static_hybrid":
            features = FeatureUnion([("word", word()), ("semantic", features)],
                                    transformer_weights={"word": 1.0, "semantic": 3.0})
        classifier = logistic()
    else:
        raise ValueError(f"Modelo desconhecido: {name}. Opções: {MODEL_NAMES}")
    return Pipeline([("features", features), ("classifier", classifier)], memory=cache_dir)
