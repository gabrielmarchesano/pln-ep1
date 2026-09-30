"""Recover a final training duration from completed cross-validation evidence."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import median

from .data import file_digest
from .result_paths import resolve_result_input


def positive_epoch(value, maximum: float) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 < value <= maximum):
        raise ValueError(f"Invalid selected epoch: {value!r} (maximum {maximum}).")
    return float(value)


def recover_best_epoch(training: dict, maximum: float) -> tuple[float, str]:
    """Use the selected checkpoint step, never the last executed epoch."""
    explicit = training.get("best_epoch")
    if explicit is not None:
        return positive_epoch(explicit, maximum), "recorded_best_epoch"
    checkpoint = str(training.get("best_checkpoint", ""))
    match = re.fullmatch(r"checkpoint-(\d+)", Path(checkpoint).name)
    if match is None:
        raise ValueError("The fold does not identify its best checkpoint step.")
    step = int(match.group(1))
    matches = [row for row in training.get("history", [])
               if row.get("step") == step and "eval_accuracy" in row]
    if len(matches) != 1:
        raise ValueError("Cannot uniquely map the best checkpoint to an evaluated epoch.")
    selected = matches[0]
    if not math.isclose(selected["eval_accuracy"], training["best_inner_accuracy"], abs_tol=1e-8):
        raise ValueError("Best checkpoint accuracy disagrees with the recorded selection.")
    epoch = positive_epoch(selected["epoch"], maximum)
    if epoch > training["epochs_executed"] + 1e-8:
        raise ValueError("Selected epoch exceeds executed training.")
    return epoch, "best_checkpoint_step_matched_to_evaluation_history"


def select_final_epochs(experiment_dir: Path, config: dict, dataset_sha256: str,
                        split_sha256: str, development_rows: int) -> dict:
    experiment_dir = resolve_result_input(experiment_dir)
    path = experiment_dir / "results.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("mode") != "experiment" or report.get("status") != "complete":
        raise ValueError("Final training requires a completed experiment.")
    if report.get("dataset_sha256") != dataset_sha256 or report.get("split_sha256") != split_sha256:
        raise ValueError("The source experiment used a different dataset or frozen split.")
    if report.get("rows") != development_rows or report.get("evaluated_rows") != development_rows:
        raise ValueError("The experiment did not cover the complete development set.")
    if report.get("holdout_evaluated") is not False:
        raise ValueError("The source experiment must leave the frozen holdout unevaluated.")
    expected_config = {key: value for key, value in config.items() if key != "description"}
    source_config = {key: value for key, value in report["config"].items() if key != "description"}
    if expected_config != source_config:
        raise ValueError("Final training must preserve the approved experiment configuration.")
    expected_folds = list(range(1, config["outer_folds"] + 1))
    if report.get("requested_folds") != expected_folds:
        raise ValueError("Source experiment fold coverage is incomplete.")
    explicit = report.get("final_training_epochs")
    nested_explicit = report.get("epoch_selection", {}).get("final_epochs")
    if explicit is not None and nested_explicit is not None and explicit != nested_explicit:
        raise ValueError("Conflicting explicit final epoch selections.")
    explicit = explicit if explicit is not None else nested_explicit
    if explicit is not None:
        chosen = positive_epoch(explicit, config["epochs"])
        return {"source_experiment": str(experiment_dir),
                "source_sha256": {"results.json": file_digest(path)},
                "fold_best_epochs": [], "rule": "explicit_final_epoch_recorded_in_experiment",
                "final_epochs": int(chosen) if chosen.is_integer() else chosen,
                "selection_uses_holdout": False}
    fold_epochs, source_hashes = [], {"results.json": file_digest(path)}
    for fold in expected_folds:
        fold_path = experiment_dir / f"fold-{fold}.json"
        record = json.loads(fold_path.read_text(encoding="utf-8"))
        if record.get("fold") != fold:
            raise ValueError("Source fold identifier mismatch.")
        epoch, origin = recover_best_epoch(record["training"], config["epochs"])
        fold_epochs.append({"fold": fold, "best_epoch": epoch, "origin": origin,
                            "best_checkpoint": record["training"].get("best_checkpoint")})
        source_hashes[fold_path.name] = file_digest(fold_path)
    # Epoch-boundary checkpoints are integral here; round down any fractional median.
    chosen = max(1, math.floor(median(row["best_epoch"] for row in fold_epochs)))
    rule = "median_of_fold_best_epochs_rounded_down_minimum_one"
    return {"source_experiment": str(experiment_dir), "source_sha256": source_hashes,
            "fold_best_epochs": fold_epochs, "rule": rule,
            "final_epochs": int(chosen) if float(chosen).is_integer() else chosen,
            "selection_uses_holdout": False}
