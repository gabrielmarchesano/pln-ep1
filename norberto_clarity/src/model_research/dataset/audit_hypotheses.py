"""Independently audit saved prospective dataset-hypothesis experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clarity.data import LABELS, file_digest, load_training_data
from .dataset_hypotheses import normalize_identifiers
from clarity.delivery import require_new_file
from clarity.result_paths import resolve_result_input
from model_research.experiment import metrics, paired_group_bootstrap, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_hypotheses(root: Path, train_path: Path) -> dict:
    root = resolve_result_input(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((root / "results.json").read_text(encoding="utf-8"))
    splits = pd.read_csv(root / "splits.csv")
    oof = pd.read_csv(root / "development-oof.csv")
    source_hash = file_digest(train_path)
    require(report["status"] == "complete" and manifest["status"] == "holdout_frozen",
            "Incomplete experiment.")
    require(report["dataset_sha256"] == manifest["dataset_sha256"] == source_hash,
            "Dataset digest mismatch.")
    require(report["holdout_evaluated"] is False and manifest["holdout_evaluated"] is False,
            "Holdout was evaluated.")
    frame = load_training_data(train_path)
    require(len(frame) == len(splits) == report["rows"], "Source/split row count mismatch.")
    require(splits.excel_row.is_unique and oof.excel_row.is_unique,
            "Duplicate Excel row identifiers.")
    require(set(splits.excel_row) == set(range(2, len(frame) + 2)),
            "Split rows do not cover the complete source.")
    require(set(splits.role) == {"development", "holdout"}, "Unexpected split role.")
    development = splits.loc[splits.role == "development"]
    holdout = splits.loc[splits.role == "holdout"]
    require(len(development) == len(oof) == report["development_rows"] and
            len(holdout) == report["holdout_rows"], "Development/holdout size mismatch.")
    require(set(oof.excel_row) == set(development.excel_row),
            "OOF rows do not exactly match development rows.")
    require(not set(development.group_id) & set(holdout.group_id),
            "A text group crosses the holdout boundary.")
    require(holdout.development_fold.eq(0).all(), "Holdout has a development fold.")
    require(set(development.development_fold) == set(range(1, report["folds"] + 1)),
            "Missing development fold.")
    require(development.groupby("group_id").development_fold.nunique().eq(1).all(),
            "Text group crosses development folds.")
    require(not splits.isna().any().any() and not oof.isna().any().any(),
            "Missing split or prediction value.")
    indexed = splits.set_index("excel_row")
    require(np.array_equal(oof.fold.to_numpy(),
                           indexed.loc[oof.excel_row, "development_fold"].to_numpy()),
            "Saved OOF fold differs from frozen split.")
    require(np.array_equal(oof.group_id.to_numpy(),
                           indexed.loc[oof.excel_row, "group_id"].to_numpy()),
            "Saved OOF group differs from frozen split.")
    rows = oof.excel_row.to_numpy() - 2
    require(np.array_equal(oof.true_label.to_numpy(), frame.clarity.to_numpy()[rows]),
            "Saved development labels differ from source.")
    canonical_groups = {}
    for row, group in enumerate(splits.group_id):
        key = normalize_identifiers(frame.resp_text.iloc[row])
        previous = canonical_groups.setdefault(key, group)
        require(previous == group, "Canonical duplicate crosses groups.")
    require(set(report["models"]) == set(manifest["policies"]),
            "Model list differs from registered policies.")
    comparisons = {}
    labels = oof.true_label.to_numpy()
    baseline = oof.pred_raw.to_numpy()
    for name, summary in report["models"].items():
        column = f"pred_{name}"
        require(column in oof, f"Missing prediction column: {name}")
        predictions = oof[column].to_numpy()
        require(set(predictions) <= set(LABELS), f"Unknown class in {name} predictions.")
        for key, value in metrics(labels, predictions).items():
            require(np.isclose(value, summary["pooled"][key], atol=1e-12),
                    f"Pooled metric mismatch: {name}/{key}")
        require(len(summary["fold_results"]) == report["folds"],
                f"Incorrect fold count: {name}")
        for fold_record in summary["fold_results"]:
            fold = fold_record["fold"]
            selected = oof.fold.to_numpy() == fold
            require(selected.any(), f"Empty fold: {fold}")
            for key, value in metrics(labels[selected], predictions[selected]).items():
                require(np.isclose(value, fold_record[key], atol=1e-12),
                        f"Fold metric mismatch: {name}/{fold}/{key}")
            require(np.isclose(fold_record["accuracy"],
                               report["fold_diagnostics"][fold - 1]["policies"][name]["accuracy"]),
                    f"Fold diagnostic mismatch: {name}/{fold}")
        if name != "raw":
            comparisons[name] = paired_group_bootstrap(
                labels, predictions, baseline, oof.group_id.to_numpy(), seed=report["seed"])
            comparisons[name]["resampling_unit"] = "similarity_component"
    return {
        "status": "passed", "dataset_sha256": source_hash,
        "development_rows": len(development), "holdout_rows": len(holdout),
        "evaluated_folds": report["folds"], "holdout_evaluated": False,
        "group_overlap": 0, "policies_checked": list(report["models"]),
        "comparisons_with_raw": comparisons,
        "limitations": [
            "This audits saved files and metrics, not every training operation.",
            "Near-similarity edges are not recomputed here; canonical equivalence is checked.",
            "Bootstrap intervals are descriptive and do not correct adaptive research selection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        require_new_file(args.output)
    result = audit_hypotheses(args.result_dir, args.train)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
