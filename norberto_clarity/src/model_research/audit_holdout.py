"""Audit a fixed-holdout result from its saved predictions and model artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from clarity.data import LABELS, LABEL_COLUMN, TEXT_COLUMN, file_digest, load_training_data
from clarity.delivery import require_new_file
from .experiment import metrics, paired_group_bootstrap, read_config, write_json
from clarity.holdout import load_fixed_holdout
from clarity.result_paths import resolve_result_input


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_fixed_holdout(metadata_path: Path, predictions_path: Path, model_path: Path,
                        train_path: Path, split_path: Path, config_path: Path) -> dict:
    metadata_path = resolve_result_input(metadata_path)
    predictions_path = resolve_result_input(predictions_path)
    split_path = resolve_result_input(split_path)
    report = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = read_config(config_path)
    frame = load_training_data(train_path)
    x = frame[TEXT_COLUMN].to_numpy()
    y = frame[LABEL_COLUMN].to_numpy()
    development, holdout, groups = load_fixed_holdout(
        train_path, split_path, y,
        expected_split_sha256=config["frozen_split_sha256"])
    require(report["dataset_sha256"] == file_digest(train_path),
            "Dataset digest mismatch.")
    require(report["fixed_validation"]["split_sha256"] == file_digest(split_path),
            "Split digest mismatch in metadata.")
    require(report["config"] == config, "Saved configuration differs from source configuration.")

    saved = pd.read_csv(predictions_path, dtype={"group_id": str})
    require(not saved.isna().any().any(), "Missing prediction values.")
    require(len(saved) == len(holdout), "Prediction row count differs from holdout.")
    require(saved.excel_row.is_unique, "Duplicate Excel row identifiers.")
    require(np.array_equal(saved.excel_row.to_numpy(), holdout + 2),
            "Prediction rows differ from frozen holdout order.")
    require(np.array_equal(saved.group_id.to_numpy(), groups[holdout]),
            "Prediction groups differ from frozen split.")
    require(np.array_equal(saved.true_label.to_numpy(), y[holdout]),
            "Saved labels differ from source workbook.")

    validation = report["fixed_validation"]
    reference = validation["comparison_reference"]
    names = [candidate["name"] for candidate in config["candidates"]]
    require(set(validation["models"]) == set(names), "Evaluated model list mismatch.")
    recomputed = {}
    for name in names:
        column = f"pred_{name}"
        require(column in saved, f"Missing prediction column: {column}.")
        predictions = saved[column].to_numpy()
        require(set(predictions) <= set(LABELS), f"Unknown predicted label: {name}.")
        values = metrics(y[holdout], predictions)
        for metric_name, value in values.items():
            require(np.isclose(value, validation["models"][name][metric_name], atol=1e-12),
                    f"Metric mismatch: {name}/{metric_name}.")
        require(confusion_matrix(y[holdout], predictions, labels=LABELS).tolist()
                == validation["models"][name]["confusion_matrix"],
                f"Confusion matrix mismatch: {name}.")
        recomputed[name] = values

    comparisons = {}
    for name in names:
        if name == reference:
            continue
        comparison = paired_group_bootstrap(
            y[holdout], saved[f"pred_{name}"].to_numpy(),
            saved[f"pred_{reference}"].to_numpy(), groups[holdout],
            seed=report["seed"])
        comparison["resampling_unit"] = "frozen_similarity_component"
        expected = validation["comparisons_with_reference"][name]
        require(np.isclose(comparison["accuracy_delta"], expected["accuracy_delta"]),
                f"Paired delta mismatch: {name}.")
        require(np.allclose(comparison["descriptive_95pct_interval"],
                            expected["descriptive_95pct_interval"]),
                f"Paired interval mismatch: {name}.")
        comparisons[name] = comparison

    artifact = joblib.load(model_path)
    require(artifact["metadata"] == report, "Artifact metadata differs from JSON metadata.")
    require(report["selected"] in names, "Selected model is not a configured candidate.")
    reproduced = artifact["model"].predict(x[holdout])
    require(np.array_equal(reproduced, saved["predicted_label"].to_numpy()),
            "Serialized model does not reproduce selected predictions.")
    require(np.array_equal(reproduced,
                           saved[f"pred_{report['selected']}"].to_numpy()),
            "Generic predictions differ from selected candidate predictions.")
    winner = max(report["searches"], key=lambda row: row["best_inner_accuracy"])["name"]
    require(winner == report["selected"], "Selected model differs from development-CV winner.")

    return {
        "status": "passed",
        "dataset_sha256": file_digest(train_path),
        "split_sha256": file_digest(split_path),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "group_overlap": len(set(groups[development]) & set(groups[holdout])),
        "selected_by_development_cv": report["selected"],
        "models_checked": names,
        "metrics": recomputed,
        "comparisons_with_reference": comparisons,
        "serialized_predictions_reproduced": True,
        "limitations": [
            "This audits saved inputs, metrics and the selected artifact, not every training operation.",
            "The reused holdout is fixed validation, not an independent final test.",
            "Bootstrap intervals are descriptive and do not correct adaptive research selection.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        require_new_file(args.output)
    result = audit_fixed_holdout(
        args.metadata, args.predictions, args.model, args.train, args.split, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
