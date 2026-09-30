"""Recompute experiment metrics and check saved prediction/split integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .data import LABELS, file_digest, grouped_splits, load_training_data, text_groups
from .delivery import require_new_file
from .reporting import metrics, paired_group_bootstrap, write_json
from .finetune_data import prepare_training_corpus
from .result_paths import resolve_result_input


def check(condition, message):
    if not condition:
        raise ValueError(message)


def check_probabilities(predictions, name):
    columns = [f"prob_{name}_{label}" for label in LABELS]
    present = [column in predictions for column in columns]
    if not any(present):
        return False  # Historical reports stored labels only.
    check(all(present), f"Incomplete probability columns: {name}")
    values = predictions[columns].to_numpy(dtype=float)
    check(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
          and np.allclose(values.sum(axis=1), 1, atol=1e-6), f"Invalid probabilities: {name}")
    check(np.array_equal(np.asarray(LABELS)[values.argmax(axis=1)], predictions[f"pred_{name}"]),
          f"Probability argmax differs from predictions: {name}")
    return True


def audit_result_dir(root: Path, train_path: Path) -> dict:
    root = resolve_result_input(root)
    report = json.loads((root / "results.json").read_text(encoding="utf-8"))
    check(report["dataset_sha256"] == file_digest(train_path), "Dataset digest mismatch.")
    pilot = report.get("mode") == "pilot"
    supervised = "mode" in report
    check(not supervised or report["mode"] in {"pilot", "experiment"},
          "This audit supports pilot/experiment predictions only.")
    predictions = pd.read_csv(root / ("inner-predictions.csv" if pilot else "oof.csv"),
                              dtype={"group_id": str})
    if report.get("split_sha256"):
        frame, groups, excel_rows, _ = prepare_training_corpus(
            train_path, Path(report["split_path"]), report["split_sha256"])
        check(report.get("holdout_evaluated") is False, "Development CV evaluated the holdout.")
    else:
        frame = load_training_data(train_path)
        groups = text_groups(frame.resp_text)
        excel_rows = np.arange(len(frame)) + 2
    x, y = frame.resp_text.to_numpy(), frame.clarity.to_numpy()
    row_index = pd.Index(excel_rows)
    check(not predictions.isna().any().any(), "Missing OOF values.")
    check(predictions.excel_row.is_unique, "Duplicate OOF row identifiers.")
    check(np.issubdtype(predictions.excel_row.dtype, np.integer), "Noninteger Excel row identifiers.")
    rows = row_index.get_indexer(predictions.excel_row.to_numpy())
    check(((rows >= 0) & (rows < len(frame))).all(), "Invalid Excel row identifiers.")
    check(np.array_equal(predictions.true_label, y[rows]), "Ground truth mismatch.")
    check(np.array_equal(predictions.group_id, groups[rows]), "Group identifier mismatch.")
    check(predictions.groupby("group_id").fold.nunique().eq(1).all(), "Group crosses evaluated folds.")
    if not pilot:
        check(len(rows) == len(frame), "Incomplete outer predictions.")
    config = report["config"]
    folds = config["outer_folds"] if supervised else report["outer_folds"]
    seed = config["seed"] if supervised else report["seed"]
    splits = grouped_splits(x, y, groups, folds, seed)
    for fold in predictions.fold.unique():
        check(1 <= fold <= folds, "Invalid fold id.")
        outer_train, outer_valid = splits[fold - 1]
        target = rows[predictions.fold.to_numpy() == fold]
        if supervised:
            saved = json.loads((root / f"split-{fold}.json").read_text())
            train = row_index.get_indexer(saved["train_excel_rows"])
            valid = row_index.get_indexer(saved["inner_valid_excel_rows"])
            check((train >= 0).all() and (valid >= 0).all(), "Training/validation includes reserved rows.")
            check(set(train) | set(valid) <= set(outer_train), "Inner split outside outer training.")
            check(set(row_index.get_indexer(saved["outer_valid_excel_rows"])) == set(outer_valid),
                  "Saved outer split differs from declared split.")
            for left, right in [(train, valid), (train, outer_valid), (valid, outer_valid)]:
                check(not set(groups[left]) & set(groups[right]), "Group leakage in saved split.")
            check(set(target) == set(valid if pilot else outer_valid), "Evaluated rows differ from saved split.")
        else:
            check(set(target) == set(outer_valid), "OOF fold differs from declared split.")
    names = list(report["models"])
    probability_models = []
    for name, summary in report["models"].items():
        pred = predictions[f"pred_{name}"]
        check(set(pred) <= set(LABELS), f"Unknown predicted class: {name}")
        if check_probabilities(predictions, name):
            probability_models.append(name)
        for key, value in metrics(predictions.true_label, pred).items():
            check(np.isclose(value, summary["pooled"][key]), f"Pooled {name}/{key} mismatch.")
        check(confusion_matrix(predictions.true_label, pred, labels=LABELS).tolist()
              == summary["confusion_matrix"], f"Confusion matrix mismatch: {name}")
        check(len(summary["fold_results"]) == predictions.fold.nunique(), "Fold count mismatch.")
        for record in summary["fold_results"]:
            mask = predictions.fold == record["fold"]
            for key, value in metrics(predictions.loc[mask, "true_label"], pred[mask]).items():
                check(np.isclose(value, record[key]), f"Fold {record['fold']}/{name}/{key} mismatch.")
        for key in ("accuracy", "f1_macro", "balanced_accuracy"):
            values = [row[key] for row in summary["fold_results"]]
            check(np.isclose(np.mean(values), summary[f"mean_{key}"]), f"Mean {name}/{key} mismatch.")
            check(np.isclose(np.std(values), summary[f"std_{key}"]), f"Std {name}/{key} mismatch.")
    comparisons = {}
    reference = "lexical_matched" if pilot else "lexical_full" if supervised else "baseline"
    for name in names:
        if name not in {reference, "majority"}:
            comparisons[name] = paired_group_bootstrap(
                predictions.true_label.to_numpy(), predictions[f"pred_{name}"].to_numpy(),
                predictions[f"pred_{reference}"].to_numpy(), predictions.group_id.to_numpy(), seed=seed)
    if "selected_per_fold" in report:
        for choice in report["selected_per_fold"]:
            mask = predictions.fold == choice["fold"]
            check(np.array_equal(predictions.loc[mask, "pred_selected"],
                                 predictions.loc[mask, "pred_" + choice["name"]]), "Selected predictions mismatch.")
    return {"status": "passed", "result_dir": str(root), "evaluated_rows": len(rows),
            "evaluated_folds": int(predictions.fold.nunique()), "is_inner_selection_only": pilot,
            "dataset_sha256": report["dataset_sha256"], "reference": reference,
            "probability_models_checked": probability_models,
            "reserved_holdout_rows": report.get("reserved_holdout_rows", 0),
            "split_sha256": report.get("split_sha256"),
            "comparisons": comparisons,
            "note": "Checks saved data, splits and recomputed metrics, not all code paths or statistical independence of research rounds."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        require_new_file(args.output)
    result = audit_result_dir(args.result_dir, args.train)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
