"""Evaluate leakage-safe dataset cleaning with a fixed lexical LinearSVC."""
from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from clarity.data import LABELS, file_digest, grouped_splits, load_training_data, normalize_text, text_groups
from model_research.experiment import environment, metrics, paired_group_bootstrap, write_json
from model_research.models import build_model

URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
LONG_NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")
MODEL_C = 0.05


@dataclass(frozen=True)
class CleaningPolicy:
    name: str
    mask_identifiers: bool
    exclude_conflicts: bool


POLICIES = (
    CleaningPolicy("raw", False, False),
    CleaningPolicy("masked", True, False),
    CleaningPolicy("conflicts_excluded", False, True),
    CleaningPolicy("masked_conflicts_excluded", True, True),
)


def mask_identifiers(text: object) -> str:
    """Mask noisy identifiers without deleting surrounding answer content."""
    value = normalize_text(text)
    value = EMAIL_PATTERN.sub(" emailtoken ", value)
    value = URL_PATTERN.sub(" urltoken ", value)
    value = LONG_NUMBER_PATTERN.sub(" longnumbertoken ", value)
    return re.sub(r"\s+", " ", value).strip()


def retained_training_rows(indices: np.ndarray, labels: np.ndarray,
                           raw_groups: np.ndarray, exclude_conflicts: bool) -> np.ndarray:
    """Detect label conflicts only among rows available in this training split."""
    if not exclude_conflicts:
        return indices
    group_labels = defaultdict(set)
    for row in indices:
        group_labels[raw_groups[row]].add(labels[row])
    retained = np.asarray([row for row in indices if len(group_labels[raw_groups[row]]) == 1],
                          dtype=np.int64)
    if len(retained) == 0 or set(labels[retained]) != set(LABELS):
        raise ValueError("Conflict filtering removed all training rows or a class.")
    return retained


def fit_predict(policy: CleaningPolicy, raw_texts: np.ndarray, masked_texts: np.ndarray,
                labels: np.ndarray, raw_groups: np.ndarray, train: np.ndarray,
                valid: np.ndarray, seed: int) -> tuple[np.ndarray, int, float]:
    kept = retained_training_rows(train, labels, raw_groups, policy.exclude_conflicts)
    texts = masked_texts if policy.mask_identifiers else raw_texts
    model = build_model("linear_svc", seed=seed)
    model.set_params(classifier__C=MODEL_C)
    started = time.monotonic()
    model.fit(texts[kept], labels[kept])
    predictions = model.predict(texts[valid])
    return predictions, len(train) - len(kept), time.monotonic() - started


def summarize(labels: np.ndarray, predictions: np.ndarray, fold_results: list[dict]) -> dict:
    scores = [row["accuracy"] for row in fold_results]
    return {
        "pooled": metrics(labels, predictions),
        "fold_results": fold_results,
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist(),
        "classification_report": classification_report(
            labels, predictions, labels=LABELS, output_dict=True, zero_division=0),
    }


def run_cleaning_experiment(train_path: Path, output_dir: Path, *, outer_folds: int = 3,
                            inner_folds: int = 3, seed: int = 42) -> dict:
    """Select cleaning inside outer training folds and evaluate unchanged labels OOF."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Output directory must be new or empty.")
    if outer_folds < 2 or inner_folds < 2:
        raise ValueError("Both fold counts must be at least two.")
    frame = load_training_data(train_path)
    raw_texts = frame.resp_text.to_numpy()
    labels = frame.clarity.to_numpy()
    masked_texts = np.asarray([mask_identifiers(text) for text in raw_texts])
    raw_groups = text_groups(raw_texts)
    # All policies share these conservative folds: cleaned copies cannot cross folds.
    groups = text_groups(masked_texts)
    outer = grouped_splits(raw_texts, labels, groups, outer_folds, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "nested_grouped_cleaning_selection",
        "dataset_sha256": file_digest(train_path),
        "environment": environment(),
        "experiment_source_sha256": file_digest(Path(__file__)),
        "rows": len(frame),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "seed": seed,
        "model": "linear_svc_word_char_tfidf",
        "classifier_C": MODEL_C,
        "grouping": "sha256(mask_identifiers(resp_text))",
        "policies": [policy.__dict__ for policy in POLICIES],
        "label_policy": "original labels retained for scoring; conflict groups removed only from training splits",
    }
    write_json(output_dir / "manifest.json", manifest)
    predictions = {policy.name: np.empty(len(labels), dtype=object) for policy in POLICIES}
    predictions["selected"] = np.empty(len(labels), dtype=object)
    fold_ids = np.zeros(len(labels), dtype=np.int32)
    folds_report = []
    for fold, (outer_train, outer_valid) in enumerate(outer, 1):
        print(f"OUTER FOLD {fold}/{outer_folds}: train={len(outer_train)}, valid={len(outer_valid)}", flush=True)
        inner = grouped_splits(raw_texts[outer_train], labels[outer_train],
                               groups[outer_train], inner_folds, seed + fold)
        inner_scores = {}
        for policy in POLICIES:
            scores = []
            for inner_train, inner_valid in inner:
                train_rows = outer_train[inner_train]
                valid_rows = outer_train[inner_valid]
                predicted, _, _ = fit_predict(policy, raw_texts, masked_texts, labels,
                                              raw_groups, train_rows, valid_rows, seed + fold)
                scores.append(metrics(labels[valid_rows], predicted)["accuracy"])
            inner_scores[policy.name] = scores
            print(f"  {policy.name}: inner_mean={np.mean(scores):.4f}", flush=True)
        # A tie favors the uncleaned control because it is first in POLICIES.
        selected = max(POLICIES, key=lambda policy: np.mean(inner_scores[policy.name])).name
        fold_ids[outer_valid] = fold
        fold_predictions = {}
        outer_fits = {}
        for policy in POLICIES:
            predicted, removed, seconds = fit_predict(
                policy, raw_texts, masked_texts, labels, raw_groups,
                outer_train, outer_valid, seed + fold,
            )
            predictions[policy.name][outer_valid] = predicted
            fold_predictions[policy.name] = predicted
            outer_fits[policy.name] = {
                "excluded_training_rows": removed,
                "seconds": seconds,
                **metrics(labels[outer_valid], predicted),
            }
        predictions["selected"][outer_valid] = fold_predictions[selected]
        folds_report.append({
            "fold": fold,
            "train_rows": len(outer_train),
            "valid_rows": len(outer_valid),
            "selected_policy": selected,
            "inner_accuracy": inner_scores,
            "outer_fits": outer_fits,
        })
        write_json(output_dir / f"fold-{fold}.json", folds_report[-1])
        print(f"  selected={selected}; raw={outer_fits['raw']['accuracy']:.4f}; "
              f"selected_accuracy={outer_fits[selected]['accuracy']:.4f}", flush=True)
    if np.any(fold_ids == 0):
        raise RuntimeError("Missing out-of-fold predictions.")
    oof = pd.DataFrame({
        "excel_row": np.arange(len(labels)) + 2,
        "group_id": groups,
        "fold": fold_ids,
        "true_label": labels,
        **{f"pred_{name}": values for name, values in predictions.items()},
    })
    if not oof.groupby("group_id").fold.nunique().eq(1).all():
        raise RuntimeError("Cleaned-text group crossed outer folds.")
    oof.to_csv(output_dir / "oof.csv", index=False)
    models = {}
    comparisons = {}
    for name, values in predictions.items():
        fold_metrics = [
            {"fold": row["fold"], **metrics(labels[fold_ids == row["fold"]],
                                           values[fold_ids == row["fold"]])}
            for row in folds_report
        ]
        models[name] = summarize(labels, values, fold_metrics)
        if name != "raw":
            comparisons[name] = paired_group_bootstrap(
                labels, values, predictions["raw"], groups, seed=seed,
            )
    report = {
        **manifest,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic() - started,
        "status": "complete",
        "folds": folds_report,
        "models": models,
        "comparisons_with_raw": comparisons,
        "limitations": [
            "The previously selected SVC and seed make this an exploratory comparison.",
            "The cleaned-text grouping changes outer folds relative to historical reports.",
            "Internal selection occurs only within each outer training partition.",
            "No labels are changed; conflict exclusion uses training labels only.",
            "Descriptive OOF intervals do not account for adaptive prior research.",
        ],
    }
    write_json(output_dir / "results.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive.")
    with threadpool_limits(limits=args.threads):
        result = run_cleaning_experiment(args.train, args.output_dir,
                                         outer_folds=args.outer_folds,
                                         inner_folds=args.inner_folds, seed=args.seed)
    for name, row in result["models"].items():
        print(f"{name}: {row['pooled']['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
