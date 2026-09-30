"""Run the existing grouped three-fold experiment for screened NorBERTo finalists."""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import finetune as ft
from . import finetune_grid as grid
from .data import file_digest
from .finetune_epochs import recover_best_epoch
from .finetune_final import check_destinations
from .hardware import log_selected_device
from .reporting import environment, metrics, write_json
from .result_paths import resolve_result_input

CV_COLUMNS = (
    "cv_rank", "candidate_id", "screening_rank", "learning_rate", "lora_rank",
    "lora_alpha", "screening_accuracy", "mean_accuracy", "std_accuracy",
    "mean_f1_macro", "std_f1_macro", "fold_1_accuracy", "fold_2_accuracy",
    "fold_3_accuracy", "fold_1_f1_macro", "fold_2_f1_macro", "fold_3_f1_macro",
    "fold_1_best_epoch", "fold_2_best_epoch", "fold_3_best_epoch",
    "runtime_seconds", "config_sha256",
)


def preflight(args) -> dict:
    """Validate the completed screening and frozen data before any output is created."""
    args.screening_dir = resolve_result_input(args.screening_dir)
    source = args.screening_dir
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("mode") != "grid-screening" or summary.get("status") != "complete"
            or summary.get("stage") != "screening_complete"
            or summary.get("cv_performed") is not False
            or summary.get("holdout_evaluated") is not False
            or summary.get("final_model_trained") is not False):
        raise ValueError("Source must be a completed screening without CV or holdout evaluation.")
    base = ft.read_finetune_config(grid.DEFAULT_CONFIG)
    if base != summary.get("base_config"):
        raise ValueError("Current base config differs from the completed screening.")
    context = grid.prepare_run(args, base)
    if (summary.get("dataset_sha256") != context["dataset_sha256"]
            or summary.get("holdout_split_sha256") != grid.FROZEN_SPLIT_SHA256
            or summary.get("development_rows") != len(context["frame"])
            or summary.get("reserved_holdout_rows") != context["holdout"]["reserved_holdout_rows"]
            or summary.get("screening_split_sha256") != file_digest(source / "screening-split.csv")):
        raise ValueError("Source screening hashes or development/holdout counts changed.")
    split = pd.read_csv(source / "screening-split.csv", dtype={"group_id": str})
    if not split.equals(context["screening_split"]):
        raise ValueError("Saved screening split differs from the reproducible fixed split.")
    if summary.get("isolation") != context["isolation"]:
        raise ValueError("Holdout isolation evidence differs from the screening.")
    ranking = summary.get("screening_ranking", [])
    if len(ranking) != 9 or len(summary.get("screening_results", [])) != 9:
        raise ValueError("Source screening must contain nine complete candidate results.")
    expected_ranking, _ = grid.summarize_screening(summary["screening_results"])
    if ranking != expected_ranking or summary.get("top_three") != [
            row["candidate_id"] for row in expected_ranking[:3]]:
        raise ValueError("TOP 3 or screening ranking differs from the recorded results.")
    candidates = {row["candidate_id"]: row for row in context["candidates"]}
    if summary.get("candidates") != context["candidates"]:
        raise ValueError("Candidate list differs from the frozen nine-configuration grid.")
    for row in ranking:
        candidate = candidates[row["candidate_id"]]
        config = ft.read_finetune_config(source / "configs" / f"{row['candidate_id']}.json")
        if (config != candidate["config"] or row["config_sha256"] != candidate["config_sha256"]
                or row["screening_split_sha256"] != summary["screening_split_sha256"]):
            raise ValueError(f"Config or split hash changed for {row['candidate_id']}.")
        result = json.loads((source / "screening" / row["candidate_id"] /
                             "results.json").read_text(encoding="utf-8"))
        if (result["candidate_id"] != row["candidate_id"]
                or result["accuracy"] != row["accuracy"]
                or result["f1_macro"] != row["f1_macro"]
                or result["config_sha256"] != row["config_sha256"]):
            raise ValueError(f"Screening result changed for {row['candidate_id']}.")
        predictions = pd.read_csv(source / "screening" / row["candidate_id"] /
                                  "predictions.csv")
        valid = context["screen_valid"]
        if (len(predictions) != len(valid)
                or predictions.excel_row.tolist() != context["excel_rows"][valid].tolist()
                or predictions.true_label.tolist() != context["frame"].clarity.iloc[valid].tolist()):
            raise ValueError(f"Screening predictions use different rows for {row['candidate_id']}.")
        recalculated = metrics(predictions.true_label, predictions.predicted_label)
        if any(recalculated[key] != row[key]
               for key in ("accuracy", "f1_macro", "balanced_accuracy")):
            raise ValueError(f"Screening metrics differ from predictions for {row['candidate_id']}.")
    check_destinations(args.output_dir, args.artifact_dir)
    if any(path.resolve().is_relative_to(source.resolve()) or
           source.resolve().is_relative_to(path.resolve())
           for path in (args.output_dir, args.artifact_dir)):
        raise ValueError("CV destinations must not overwrite the screening source.")
    if args.output_dir.exists() and (
            not args.output_dir.is_dir() or any(
                path.name not in {"run.log", "pid.txt"} for path in args.output_dir.iterdir())):
        raise ValueError("CV output directory must be new or contain only run.log and pid.txt.")
    if args.artifact_dir.exists() and (
            not args.artifact_dir.is_dir() or any(args.artifact_dir.iterdir())):
        raise ValueError("CV artifact directory must be new or empty.")
    return {"source": summary, "source_sha256": file_digest(summary_path),
            "context": context, "candidates": candidates, "ranking": ranking}


def initialize_directories(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ft.new_directory(args.artifact_dir)
    (args.output_dir / "candidates").mkdir()
    (args.artifact_dir / "candidates").mkdir()
    pd.DataFrame(columns=CV_COLUMNS).to_csv(args.output_dir / "cv_results.csv", index=False)


def cv_rank_key(row: dict) -> tuple:
    return (-row["mean_accuracy"], -row["mean_f1_macro"], row["lora_rank"],
            row["learning_rate"], row["candidate_id"])


def save_cv_csv(path: Path, rows: list[dict]) -> None:
    flat = []
    for row in rows:
        record = dict(row)
        for fold in row["fold_results"]:
            index = fold["fold"]
            record[f"fold_{index}_accuracy"] = fold["accuracy"]
            record[f"fold_{index}_f1_macro"] = fold["f1_macro"]
            record[f"fold_{index}_best_epoch"] = fold["best_epoch"]
        flat.append(record)
    pd.DataFrame(flat, columns=CV_COLUMNS).to_csv(path, index=False)


def evaluate_candidate(args, candidate: dict, screening_row: dict, dataset_sha256: str) -> dict:
    candidate_id = candidate["candidate_id"]
    result_dir = args.output_dir / "candidates" / candidate_id
    artifact_dir = args.artifact_dir / "candidates" / candidate_id
    started = time.monotonic()
    report = ft.run(argparse.Namespace(
        mode="experiment", train=args.train,
        config=args.screening_dir / "configs" / f"{candidate_id}.json",
        output_dir=result_dir, artifact_dir=artifact_dir,
        holdout_splits=args.holdout_splits, device="gpu"))
    if (report.get("mode") != "experiment" or report.get("status") != "complete"
            or report.get("dataset_sha256") != dataset_sha256
            or report.get("split_sha256") != grid.FROZEN_SPLIT_SHA256
            or report.get("reserved_holdout_rows") != 3999
            or report.get("rows") != 16093 or report.get("evaluated_rows") != 16093
            or report.get("requested_folds") != [1, 2, 3]
            or report.get("config") != candidate["config"]):
        raise ValueError(f"Frozen three-fold protocol failed for {candidate_id}.")
    model = report["models"]["norberto_lora"]
    scores = {row["fold"]: row for row in model["fold_results"]}
    if set(scores) != {1, 2, 3}:
        raise ValueError(f"Three fold scores were not produced for {candidate_id}.")
    fold_results = []
    for fold in (1, 2, 3):
        record = json.loads((result_dir / f"fold-{fold}.json").read_text(encoding="utf-8"))
        training = record["training"]
        if training["training_stopped_by_budget"]:
            raise RuntimeError(f"{candidate_id} fold {fold} hit the training time budget.")
        best_epoch, epoch_source = recover_best_epoch(training, candidate["config"]["epochs"])
        fold_results.append({**scores[fold], "best_epoch": best_epoch,
                             "best_epoch_source": epoch_source,
                             "best_checkpoint": training["best_checkpoint"],
                             "training_seconds": training["training_seconds"],
                             "trainable_parameters": training["trainable_parameters"]})
    return {"candidate_id": candidate_id, "screening_rank": screening_row["screening_rank"],
            "screening_accuracy": screening_row["accuracy"],
            "learning_rate": candidate["learning_rate"], "lora_rank": candidate["lora_rank"],
            "lora_alpha": candidate["lora_alpha"], "config_sha256": candidate["config_sha256"],
            "mean_accuracy": model["mean_accuracy"], "std_accuracy": model["std_accuracy"],
            "mean_f1_macro": model["mean_f1_macro"], "std_f1_macro": model["std_f1_macro"],
            "fold_results": fold_results, "runtime_seconds": time.monotonic() - started,
            "result_dir": str(result_dir), "artifact_dir": str(artifact_dir)}


def run(args) -> dict:
    print(f"GRID CV START pid={os.getpid()} source={args.screening_dir}", flush=True)
    prepared = preflight(args)
    if args.preflight:
        print("GRID CV PREFLIGHT COMPLETE", prepared["source"]["top_three"], flush=True)
        return prepared
    use_cpu, accelerator = ft.select_execution_device("gpu")
    if use_cpu:
        raise RuntimeError("Three-fold CV requires a functional GPU.")
    log_selected_device(accelerator, operation="grid finalist CV", requested="gpu")
    initialize_directories(args)
    source = prepared["source"]
    summary = {
        "mode": "grid-finalists-cv", "status": "running", "stage": "setup",
        "started_at": grid.utc_now(), "pid": os.getpid(), "git_commit": grid.git_commit(),
        "protocol": "existing_grouped_outer_three_fold_experiment_per_screening_top_three",
        "source_screening_dir": str(args.screening_dir),
        "source_screening_summary_sha256": prepared["source_sha256"],
        "source_screening_commit": source["git_commit"],
        "screening_split_sha256": source["screening_split_sha256"],
        "dataset_sha256": source["dataset_sha256"],
        "holdout_split_sha256": grid.FROZEN_SPLIT_SHA256,
        "development_rows": 16093, "reserved_holdout_rows": 3999,
        "isolation": prepared["context"]["isolation"],
        "environment": environment(), "accelerator": accelerator, "seed": source["seed"],
        "top_three_from_screening": source["top_three"],
        "cv_ranking_rule": "mean accuracy descending; mean macro-F1 descending; "
                           "LoRA rank ascending; learning rate ascending; candidate ID ascending",
        "cv_results": [], "cv_ranking": [], "final_model_trained": False,
        "holdout_evaluated": False,
        "caveat": "Candidates were selected on the same development set; this CV is exploratory, "
                  "not an independent confirmatory estimate. The frozen holdout remains unused.",
    }
    write_json(args.output_dir / "summary.json", summary)
    try:
        for screening_row in prepared["ranking"][:3]:
            candidate_id = screening_row["candidate_id"]
            candidate = prepared["candidates"][candidate_id]
            summary["stage"] = candidate_id
            write_json(args.output_dir / "summary.json", summary)
            print("GRID CV CANDIDATE START", candidate_id, flush=True)
            result = evaluate_candidate(args, candidate, screening_row, source["dataset_sha256"])
            summary["cv_results"].append(result)
            save_cv_csv(args.output_dir / "cv_results.csv", summary["cv_results"])
            write_json(args.output_dir / "summary.json", summary)
            print("GRID CV CANDIDATE COMPLETE", candidate_id, result["mean_accuracy"], flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        summary["cv_ranking"] = [
            {**row, "cv_rank": index}
            for index, row in enumerate(sorted(summary["cv_results"], key=cv_rank_key), 1)]
        save_cv_csv(args.output_dir / "cv_results.csv", summary["cv_ranking"])
        summary.update(status="complete", stage="cv_complete", finished_at=grid.utc_now())
        write_json(args.output_dir / "summary.json", summary)
        print("GRID CV COMPLETE", [row["candidate_id"] for row in summary["cv_ranking"]], flush=True)
        return summary
    except Exception as error:
        summary.update(status="failed", finished_at=grid.utc_now(),
                       error={"type": type(error).__name__, "message": str(error)})
        write_json(args.output_dir / "summary.json", summary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--holdout-splits", type=Path, default=grid.DEFAULT_SPLIT)
    parser.add_argument("--screening-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="Validate inputs without creating output.")
    args = parser.parse_args()
    base = ft.read_finetune_config(grid.DEFAULT_CONFIG)
    with threadpool_limits(limits=base["threads"]):
        run(args)


if __name__ == "__main__":
    main()
