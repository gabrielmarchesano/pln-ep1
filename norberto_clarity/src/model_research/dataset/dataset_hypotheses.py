"""Prospective, leakage-aware experiments on dataset hypotheses (no production changes)."""
from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from clarity.data import LABELS, file_digest, grouped_splits, load_training_data, normalize_text
from model_research.experiment import environment, metrics, write_json
from model_research.models import build_model

MODEL_C = 0.05
TOKEN = r"[A-Za-zÀ-ÿ]+"
IDENTIFIER_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I), " emailtoken "),
    (re.compile(r"(?:https?://|www\.)[^\s<>]+", re.I), " urltoken "),
    (re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"), " cnpjtoken "),
    (re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"), " cpftoken "),
    (re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"), " datetoken "),
    (re.compile(r"\b(?:r\$|brl)\s*\d[\d.,]*", re.I), " valuetoken "),
    (re.compile(r"\b(?:protocolo|processo|identificador|id)\s*(?:n[º°o.]?\s*)?[:#-]?\s*[\d./-]{4,}\b", re.I), " protocoltoken "),
    (re.compile(r"\b\d+(?:[.,]\d+)*\b"), " numbertoken "),
)


def normalize_identifiers(text: object) -> str:
    """Replace observed identifier formats without modifying the source workbook."""
    value = normalize_text(text)
    for pattern, replacement in IDENTIFIER_PATTERNS:
        value = pattern.sub(replacement, value)
    return re.sub(r"\s+", " ", value).strip()


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Compute bounded, text-only style features for a sparse feature union."""

    def fit(self, texts, labels=None):
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            value = str(text)
            words = re.findall(TOKEN, value.lower())
            sentences = [part for part in re.split(r"[.!?]+", value) if part.strip()]
            paragraphs = [part for part in re.split(r"\n\s*\n", value) if part.strip()]
            count = len(words)
            rows.append([
                np.log1p(count), np.log1p(len(sentences)), np.log1p(len(paragraphs)),
                np.log1p(count / max(len(sentences), 1)),
                len(set(words)) / max(count, 1),
                sum(len(word) for word in words) / max(count, 1),
                value.count("?"), value.count("!"), value.count(","),
                value.count(";"), value.count(":"),
                sum(char.isdigit() for char in value) / max(len(value), 1),
                sum(char.isupper() for char in value) / max(sum(char.isalpha() for char in value), 1),
            ])
        return np.asarray(rows, dtype=np.float32)


class SparseStyleFeatures(BaseEstimator, TransformerMixin):
    """Scale style dimensions using training rows only."""

    def fit(self, texts, labels=None):
        self.scaler_ = StandardScaler()
        self.scaler_.fit(StyleFeatures().transform(texts))
        return self

    def transform(self, texts):
        values = self.scaler_.transform(StyleFeatures().transform(texts))
        return sparse.csr_matrix(np.clip(values, -5, 5) * 0.5)


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Model P(y>=c234) and P(y>=c5), then project to ordered class probabilities."""

    def __init__(self, C: float = MODEL_C, seed: int = 42):
        self.C = C
        self.seed = seed

    def fit(self, features, labels):
        labels = np.asarray(labels)
        if set(labels) != set(LABELS):
            raise ValueError("Ordinal training needs all three classes.")
        self.lower_ = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
        self.upper_ = LogisticRegression(C=self.C, max_iter=1000, random_state=self.seed)
        self.lower_.fit(features, labels != LABELS[0])
        self.upper_.fit(features, labels == LABELS[2])
        self.classes_ = np.asarray(LABELS)
        return self

    def predict_proba(self, features):
        lower = self.lower_.predict_proba(features)[:, 1]
        upper = np.minimum(self.upper_.predict_proba(features)[:, 1], lower)
        return np.column_stack((1 - lower, lower - upper, upper))

    def predict(self, features):
        return self.classes_[np.argmax(self.predict_proba(features), axis=1)]


def build_candidate(*, ordinal: bool, style: bool, seed: int):
    model = build_model("linear_svc", seed=seed)
    model.set_params(classifier__C=MODEL_C)
    if style:
        model.set_params(features=FeatureUnion([
            ("lexical", model.named_steps["features"]), ("style", SparseStyleFeatures()),
        ]))
    if ordinal:
        model.set_params(classifier=OrdinalClassifier(seed=seed))
    return model


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size)

    def find(self, row: int) -> int:
        while self.parent[row] != row:
            self.parent[row] = self.parent[self.parent[row]]
            row = int(self.parent[row])
        return row

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def similarity_groups(texts: np.ndarray, *, threshold: float = 0.93,
                      working_memory: int = 96) -> tuple[np.ndarray, dict]:
    """Exhaustively link canonical equivalents and cosine-near lexical copies."""
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must be in (0, 1].")
    union = UnionFind(len(texts))
    first = {}
    for row, text in enumerate(texts):
        key = normalize_identifiers(text)
        if key in first:
            union.union(first[key], row)
        else:
            first[key] = row
    exact_components = len({union.find(row) for row in range(len(texts))})
    vectorizer = TfidfVectorizer(preprocessor=normalize_text, analyzer="char_wb",
                                 ngram_range=(3, 5), min_df=1, dtype=np.float32,
                                 norm="l2")
    features = vectorizer.fit_transform(texts)
    linked_pairs = 0

    def link_chunk(distances, start):
        nonlocal linked_pairs
        for local, row_distances in enumerate(distances):
            row = start + local
            neighbors = np.flatnonzero(row_distances[row + 1:] <= 1 - threshold) + row + 1
            for neighbor in neighbors:
                union.union(row, int(neighbor))
            linked_pairs += len(neighbors)
        return None

    for _ in pairwise_distances_chunked(features, metric="cosine", reduce_func=link_chunk,
                                        working_memory=working_memory, n_jobs=1):
        pass
    groups = np.asarray([union.find(row) for row in range(len(texts))], dtype=np.int32)
    return groups, {"threshold": threshold, "canonical_groups": exact_components,
                    "similarity_pairs": int(linked_pairs),
                    "final_groups": int(len(np.unique(groups))),
                    "method": "canonical identifier masking plus exhaustive char 3-5 TF-IDF cosine"}


def majority_relabel(train: np.ndarray, labels: np.ndarray,
                     keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Resolve exact-text label conflicts only inside the current training split."""
    counts = defaultdict(Counter)
    for row in train:
        counts[keys[row]][labels[row]] += 1
    retained, corrected = [], []
    ties = changes = 0
    for row in train:
        ranking = counts[keys[row]].most_common()
        if len(ranking) > 1 and ranking[0][1] == ranking[1][1]:
            ties += 1
            continue
        label = ranking[0][0]
        changes += label != labels[row]
        retained.append(row)
        corrected.append(label)
    if set(corrected) != set(LABELS):
        raise ValueError("Majority policy removed a class from the training split.")
    return np.asarray(retained, dtype=np.int64), np.asarray(corrected), {
        "tie_rows_removed": ties, "labels_reassigned": changes,
    }


def consensus_disagreements(texts: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                            *, seed: int, folds: int = 3,
                            min_probability: float = 0.65) -> tuple[np.ndarray, dict]:
    """Use train-only grouped OOF predictions from three independent classifiers."""
    splits = grouped_splits(texts, labels, groups, folds, seed)
    votes = np.empty((len(labels), 3), dtype=object)
    confidence = np.zeros(len(labels))
    for train, valid in splits:
        lexical = build_candidate(ordinal=False, style=False, seed=seed)
        lexical.fit(texts[train], labels[train])
        votes[valid, 0] = lexical.predict(texts[valid])
        logistic = clone(lexical)
        logistic.set_params(classifier=LogisticRegression(C=0.25, max_iter=1000,
                                                         random_state=seed))
        logistic.fit(texts[train], labels[train])
        probabilities = logistic.predict_proba(texts[valid])
        votes[valid, 1] = logistic.classes_[probabilities.argmax(axis=1)]
        confidence[valid] = probabilities.max(axis=1)
        naive_bayes = clone(lexical)
        naive_bayes.set_params(classifier=MultinomialNB(alpha=1.0))
        naive_bayes.fit(texts[train], labels[train])
        votes[valid, 2] = naive_bayes.predict(texts[valid])
    suspect = ((votes[:, 0] == votes[:, 1]) & (votes[:, 1] == votes[:, 2]) &
               (votes[:, 0] != labels) & (confidence >= min_probability))
    return suspect, {"suspect_rows": int(suspect.sum()),
                     "confidence_threshold": min_probability,
                     "models": ["linear_svc", "logistic_regression", "multinomial_nb"]}


POLICIES = {
    "raw": (False, False, False, False, False),
    "majority": (False, False, False, True, False),
    "normalized": (True, False, False, False, False),
    "style": (False, True, False, False, False),
    "ordinal": (False, False, True, False, False),
    "consensus_filter": (False, False, False, False, True),
    "combined_svc": (True, True, False, True, True),
    "combined_ordinal": (True, True, True, True, True),
}


def run_experiment(train_path: Path, output_dir: Path, *, seed: int = 42,
                   folds: int = 3, holdout_folds: int = 5,
                   similarity_threshold: float = 0.93) -> dict:
    """Freeze a prospective holdout; compare predefined policies on development folds."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be new or empty; existing results are immutable.")
    if folds < 2 or holdout_folds < 2:
        raise ValueError("Fold counts must be at least two.")
    source_digest = file_digest(train_path)
    frame = load_training_data(train_path)
    texts = frame.resp_text.to_numpy()
    labels = frame.clarity.to_numpy()
    started = time.monotonic()
    groups, grouping = similarity_groups(texts, threshold=similarity_threshold)
    holdout_splitter = StratifiedGroupKFold(n_splits=holdout_folds, shuffle=True,
                                            random_state=seed)
    development, holdout = next(holdout_splitter.split(texts, labels, groups))
    if set(groups[development]) & set(groups[holdout]):
        raise RuntimeError("Holdout group leakage.")
    development_splits = grouped_splits(texts[development], labels[development],
                                        groups[development], folds, seed + 1)
    role = np.full(len(texts), "holdout", dtype=object)
    role[development] = "development"
    fold_id = np.zeros(len(texts), dtype=np.int8)
    for fold, (_, valid) in enumerate(development_splits, 1):
        fold_id[development[valid]] = fold
    output_dir.mkdir(parents=True, exist_ok=True)
    component_frame = pd.DataFrame({
        "group": groups[development], "label": labels[development],
        "exact_text": [normalize_text(texts[row]) for row in development],
    })
    components = component_frame.groupby("group")
    conflict_groups = components.label.nunique().gt(1)
    near_conflicts = conflict_groups & components.exact_text.nunique().gt(1)
    grouping["development_conflicting_groups"] = int(conflict_groups.sum())
    grouping["development_conflicting_rows"] = int(
        components.size()[conflict_groups].sum())
    grouping["development_near_text_conflicting_groups"] = int(near_conflicts.sum())
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "holdout_frozen", "dataset_sha256": source_digest,
        "protocol": "prospective_holdout_grouped_development_cv",
        "holdout_used_for_modeling": False, "holdout_evaluated": False,
        "historical_caveat": "All source rows were present in earlier exploratory analyses; this holdout is prospective, not historically untouched.",
        "rows": len(texts), "development_rows": len(development),
        "holdout_rows": len(holdout), "seed": seed,
        "folds": folds, "holdout_folds": holdout_folds,
        "grouping": grouping, "policies": POLICIES,
        "environment": environment(),
        "experiment_source_sha256": file_digest(Path(__file__)),
    }
    write_json(output_dir / "manifest.json", manifest)
    pd.DataFrame({"excel_row": np.arange(len(texts)) + 2, "group_id": groups,
                  "role": role, "development_fold": fold_id}).to_csv(
                      output_dir / "splits.csv", index=False)
    if file_digest(train_path) != source_digest:
        raise RuntimeError("Source workbook changed while freezing splits.")
    normalized = np.asarray([normalize_identifiers(text) for text in texts])
    exact_keys = np.asarray([hashlib.sha256(normalize_text(text).encode()).hexdigest()
                             for text in texts])
    predictions = {name: np.empty(len(development), dtype=object) for name in POLICIES}
    diagnostics = []
    for fold, (train_local, valid_local) in enumerate(development_splits, 1):
        train, valid = development[train_local], development[valid_local]
        print(f"Development fold {fold}/{folds}: train={len(train)} valid={len(valid)}", flush=True)
        suspects, noise_info = consensus_disagreements(
            texts[train], labels[train], groups[train], seed=seed + fold,
            folds=min(folds, 3))
        keep_noise = ~suspects
        fold_info = {"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                     "noise_detection": noise_info, "policies": {}}
        for name, (use_normalized, use_style, use_ordinal, use_majority,
                   use_filter) in POLICIES.items():
            selected = train[keep_noise] if use_filter else train
            targets = labels[selected]
            info = {}
            if use_majority:
                selected, targets, info = majority_relabel(selected, labels, exact_keys)
            if set(targets) != set(LABELS):
                raise ValueError(f"Policy {name} removed a class in fold {fold}.")
            values = normalized if use_normalized else texts
            model = build_candidate(ordinal=use_ordinal, style=use_style, seed=seed + fold)
            fit_started = time.monotonic()
            model.fit(values[selected], targets)
            predicted = model.predict(values[valid])
            predictions[name][valid_local] = predicted
            fold_info["policies"][name] = {
                **metrics(labels[valid], predicted), "train_rows_used": len(selected),
                "seconds": time.monotonic() - fit_started, **info,
            }
            print(f"  {name}: accuracy={fold_info['policies'][name]['accuracy']:.4f} "
                  f"macro_f1={fold_info['policies'][name]['f1_macro']:.4f}", flush=True)
        diagnostics.append(fold_info)
        write_json(output_dir / f"fold-{fold}.json", fold_info)
    dev_labels = labels[development]
    results = {}
    for name, values in predictions.items():
        per_fold = [dict(fold=fold, **metrics(dev_labels[fold_id[development] == fold],
                                              values[fold_id[development] == fold]))
                    for fold in range(1, folds + 1)]
        results[name] = {"pooled": metrics(dev_labels, values),
                         "fold_results": per_fold,
                         "mean_accuracy": float(np.mean([item["accuracy"] for item in per_fold])),
                         "mean_f1_macro": float(np.mean([item["f1_macro"] for item in per_fold]))}
    oof = pd.DataFrame({"excel_row": development + 2, "group_id": groups[development],
                        "fold": fold_id[development], "true_label": dev_labels,
                        **{f"pred_{name}": values for name, values in predictions.items()}})
    if not oof.groupby("group_id").fold.nunique().eq(1).all():
        raise RuntimeError("Similar-text group crossed development folds.")
    oof.to_csv(output_dir / "development-oof.csv", index=False)
    if file_digest(train_path) != source_digest:
        raise RuntimeError("Source workbook changed during the experiment.")
    report = {**manifest, "status": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
              "seconds": time.monotonic() - started, "holdout_evaluated": False,
              "fold_diagnostics": diagnostics, "models": results,
              "limitations": ["Development CV is exploratory; the prospective holdout was not scored.",
                              "Identifier canonicalization may merge distinct answers into one group.",
                              "Consensus disagreement does not prove an annotation error."]}
    write_json(output_dir / "results.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--holdout-folds", type=int, default=5)
    parser.add_argument("--similarity-threshold", type=float, default=0.93)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    with threadpool_limits(limits=args.threads):
        report = run_experiment(args.train, args.output_dir, seed=args.seed,
                                folds=args.folds, holdout_folds=args.holdout_folds,
                                similarity_threshold=args.similarity_threshold)
    for name, row in report["models"].items():
        print(f"{name}: accuracy={row['pooled']['accuracy']:.4f} "
              f"macro_f1={row['pooled']['f1_macro']:.4f}", flush=True)


if __name__ == "__main__":
    main()
