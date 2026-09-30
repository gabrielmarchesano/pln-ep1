"""Nine-candidate NorBERTo LoRA screening on one frozen development split."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from . import finetune as ft
from .data import LABELS, file_digest, grouped_splits
from .finetune_data import prepare_training_corpus
from .finetune_epochs import recover_best_epoch
from .finetune_final import check_destinations, verify_development_rows
from .hardware import log_selected_device
from .reporting import environment, metrics, write_json
from .result_paths import resolve_result_input

DEFAULT_CONFIG = Path("configs/norberto-lora-development-ctx512.json")
DEFAULT_SPLIT = Path("results/research/dataset-hypotheses-20260923/splits.csv")
FROZEN_SPLIT_SHA256 = "0f78e7a809c7b7584db05b33cf03a73ff586d5de02ba824c16562d5070d41494"
BASE_CONTRACT = {
    "model_name": ft.NORBERTO_ID,
    "revision": ft.NORBERTO_REVISION,
    "max_length": 512,
    "truncation": "head_tail",
    "lora_rank": 4,
    "lora_alpha": 8,
    "lora_dropout": 0.05,
    "lora_layers": None,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "eval_batch_size": 8,
    "learning_rate": 0.001,
    "epochs": 3,
    "patience": 1,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "fp16": False,
    "attention": "sdpa",
    "outer_folds": 3,
    "inner_folds": 5,
    "seed": 42,
    "threads": 2,
    "train_subset_folds": 1,
    "budget_minutes": 75,
    "holdout_split_sha256": FROZEN_SPLIT_SHA256,
}
LEARNING_RATES = (1e-4, 3e-4, 1e-3)
LORA_RANKS = (4, 8, 16)
SCREENING_COLUMNS = (
    "screening_rank", "candidate_id", "learning_rate", "lora_rank", "lora_alpha",
    "lora_dropout", "accuracy", "f1_macro", "balanced_accuracy",
    "accuracy_delta_vs_baseline", "accuracy_gap_to_first",
    "accuracy_gap_to_third", "accuracy_gap_to_previous", "best_epoch",
    "best_checkpoint", "training_loss", "trainable_parameters", "train_rows",
    "validation_rows", "training_seconds", "runtime_seconds", "seed", "config_sha256",
    "screening_split_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def config_digest(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_candidates(base: dict) -> list[dict]:
    """Reject drift in the approved baseline before changing only three fields."""
    actual = {key: value for key, value in base.items() if key != "description"}
    if actual != BASE_CONTRACT:
        raise ValueError("Grid baseline differs from the approved frozen NorBERTo-512 contract.")
    candidates = []
    for rank in LORA_RANKS:
        for learning_rate in LEARNING_RATES:
            config = {**base, "learning_rate": learning_rate,
                      "lora_rank": rank, "lora_alpha": 2 * rank}
            candidate_id = f"candidate-{len(candidates) + 1:02d}"
            candidates.append({"candidate_id": candidate_id, "config": config,
                               "config_sha256": config_digest(config),
                               "learning_rate": learning_rate, "lora_rank": rank,
                               "lora_alpha": 2 * rank, "lora_dropout": config["lora_dropout"]})
    return candidates


def screening_split(frame: pd.DataFrame, groups: np.ndarray, excel_rows: np.ndarray,
                    base: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Use only grouped_splits(..., inner_folds=5, seed=42)[0] on development."""
    texts, labels = frame.resp_text.to_numpy(), frame.clarity.to_numpy()
    train, valid = grouped_splits(texts, labels, groups, base["inner_folds"], base["seed"])[0]
    if set(excel_rows[train]) & set(excel_rows[valid]) or set(groups[train]) & set(groups[valid]):
        raise ValueError("Screening train/validation rows or groups overlap.")
    roles = np.full(len(frame), "train", dtype=object)
    roles[valid] = "validation"
    split = pd.DataFrame({"excel_row": excel_rows, "group_id": groups, "role": roles})
    if len(train) + len(valid) != len(frame):
        raise ValueError("Screening split does not cover all development rows.")
    return train, valid, split


def screening_rank_key(row: dict) -> tuple:
    return (-row["accuracy"], -row["f1_macro"], row["lora_rank"],
            row["learning_rate"], row["candidate_id"])


def ranked(rows: list[dict], key, rank_field: str) -> list[dict]:
    return [{**row, rank_field: index} for index, row in enumerate(sorted(rows, key=key), 1)]


def summarize_screening(rows: list[dict]) -> tuple[list[dict], dict]:
    if len(rows) != 9 or {row["candidate_id"] for row in rows} != {
            f"candidate-{index:02d}" for index in range(1, 10)}:
        raise ValueError("Screening ranking requires exactly nine distinct candidates.")
    baseline = next(row for row in rows if row["candidate_id"] == "candidate-03")
    ordered = ranked(rows, screening_rank_key, "screening_rank")
    first, second, third, fourth = ordered[:4]
    enriched = []
    for index, row in enumerate(ordered):
        enriched.append({**row,
                         "accuracy_delta_vs_baseline": row["accuracy"] - baseline["accuracy"],
                         "accuracy_gap_to_first": first["accuracy"] - row["accuracy"],
                         "accuracy_gap_to_third": third["accuracy"] - row["accuracy"],
                         "accuracy_gap_to_previous": (
                             ordered[index - 1]["accuracy"] - row["accuracy"]
                             if index else 0.0)})
    comparison = {
        "baseline_candidate_id": "candidate-03",
        "baseline_accuracy": baseline["accuracy"],
        "first_vs_second_accuracy": first["accuracy"] - second["accuracy"],
        "second_vs_third_accuracy": second["accuracy"] - third["accuracy"],
        "third_vs_fourth_accuracy": third["accuracy"] - fourth["accuracy"],
        "first_vs_fourth_accuracy": first["accuracy"] - fourth["accuracy"],
        "best_outside_top_three": fourth["candidate_id"],
    }
    return enriched, comparison


def git_commit() -> str | None:
    provided = os.environ.get("PLN_GRID_GIT_COMMIT")
    if provided is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", provided):
            raise ValueError("PLN_GRID_GIT_COMMIT must be a 40-character Git SHA-1.")
        return provided
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def initialize_directories(output_dir: Path, artifact_dir: Path) -> None:
    check_destinations(output_dir, artifact_dir)
    if output_dir.exists():
        if not output_dir.is_dir() or any(path.name not in {"run.log", "pid.txt"}
                                          for path in output_dir.iterdir()):
            raise ValueError("Grid output directory must be new or contain only run.log and pid.txt.")
    else:
        output_dir.mkdir(parents=True)
    ft.new_directory(artifact_dir)
    for name in ("configs", "screening"):
        (output_dir / name).mkdir()
    for name in ("screening",):
        (artifact_dir / name).mkdir()
    pd.DataFrame(columns=SCREENING_COLUMNS).to_csv(output_dir / "screening_results.csv", index=False)


def prepare_run(args, base: dict) -> dict:
    args.holdout_splits = resolve_result_input(args.holdout_splits)
    candidates = build_candidates(base)
    if file_digest(args.holdout_splits) != FROZEN_SPLIT_SHA256:
        raise ValueError("The frozen holdout split SHA-256 changed.")
    frame, groups, excel_rows, holdout = prepare_training_corpus(
        args.train, args.holdout_splits, FROZEN_SPLIT_SHA256)
    isolation = verify_development_rows(args.holdout_splits, excel_rows, groups)
    if (len(frame), holdout["source_rows"], holdout["reserved_holdout_rows"]) != (16093, 20092, 3999):
        raise ValueError("Frozen development/holdout row counts changed.")
    train, valid, split = screening_split(frame, groups, excel_rows, base)
    if not np.array_equal(excel_rows, split.excel_row.to_numpy()):
        raise ValueError("Screening row IDs differ from the development set.")
    return {"frame": frame, "groups": groups, "excel_rows": excel_rows,
            "screen_train": train, "screen_valid": valid, "screening_split": split,
            "candidates": candidates, "holdout": holdout, "isolation": isolation,
            "dataset_sha256": file_digest(args.train)}


def initial_summary(args, base: dict, context: dict, accelerator: dict) -> dict:
    return {
        "mode": "grid-screening", "status": "running", "stage": "setup",
        "started_at": utc_now(), "pid": os.getpid(), "git_commit": git_commit(),
        "protocol": "nine_candidates_on_one_fixed_grouped_development_split; stop_after_screening",
        "selection_uses_holdout": False, "holdout_evaluated": False,
        "final_model_trained": False, "cv_performed": False,
        "base_config": base, "model_name": base["model_name"], "revision": base["revision"],
        "dataset_sha256": context["dataset_sha256"],
        "holdout_split_sha256": FROZEN_SPLIT_SHA256,
        "split_path": str(args.holdout_splits),
        "development_rows": len(context["frame"]),
        "reserved_holdout_rows": context["holdout"]["reserved_holdout_rows"],
        "isolation": context["isolation"], "environment": environment(),
        "accelerator": accelerator, "seed": base["seed"],
        "screening_split_rule": "grouped_splits(development, inner_folds=5, seed=42)[0]",
        "screening_train_rows": len(context["screen_train"]),
        "screening_validation_rows": len(context["screen_valid"]),
        "screening_ranking_rule": "accuracy descending; macro-F1 descending; LoRA rank ascending; "
                                  "learning rate ascending; candidate ID ascending",
        "candidates": context["candidates"], "screening_results": [],
        "screening_ranking": [], "top_three": [], "top_three_comparison": None,
        "caveat": "The ranking is based on one validation split and is exploratory. "
                  "No cross-validation or frozen-holdout evaluation is performed.",
    }


def save_screening_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=SCREENING_COLUMNS).to_csv(path, index=False)


def save_screening_report(path: Path, ranking: list[dict], comparison: dict) -> None:
    lines = [
        "# NorBERTo-512 hyperparameter screening",
        "",
        "Nine configurations were evaluated on the same fixed grouped development split. "
        "No three-fold CV or final holdout evaluation was run.",
        "",
        "Accuracy differences below are absolute proportions; 0.01 means one percentage point. "
        "The baseline is candidate-03 (learning_rate=1e-3, rank=4, alpha=8).",
        "",
        "| Rank | Candidate | LR | LoRA rank | Alpha | Accuracy | Macro-F1 | Δ baseline | "
        "Gap to #1 | Gap to #3 | Runtime (min) | Best epoch | Best checkpoint |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ranking:
        lines.append(
            f"| {row['screening_rank']} | {row['candidate_id']} "
            f"{'**TOP 3**' if row['screening_rank'] <= 3 else ''} | "
            f"{row['learning_rate']:.0e} | {row['lora_rank']} | {row['lora_alpha']} | "
            f"{row['accuracy']:.4f} | {row['f1_macro']:.4f} | "
            f"{row['accuracy_delta_vs_baseline']:+.4f} | "
            f"{row['accuracy_gap_to_first']:.4f} | "
            f"{row['accuracy_gap_to_third']:+.4f} | "
            f"{row['runtime_seconds'] / 60:.1f} | {row['best_epoch']} | "
            f"`{row['best_checkpoint']}` |"
        )
    lines.extend([
        "", "## Decision gaps", "",
        f"- #1 vs #2: {comparison['first_vs_second_accuracy']:.4f}",
        f"- #2 vs #3: {comparison['second_vs_third_accuracy']:.4f}",
        f"- #3 vs #4: {comparison['third_vs_fourth_accuracy']:.4f}",
        f"- #1 vs #4: {comparison['first_vs_fourth_accuracy']:.4f}",
        "",
        "These single-split results are exploratory. Review the size and stability of "
        "the gaps before deciding which candidates warrant three-fold CV.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_screening_candidate(candidate: dict, context: dict, tokenizer, features,
                            output_dir: Path, artifact_dir: Path, split_sha256: str) -> dict:
    candidate_id, config = candidate["candidate_id"], candidate["config"]
    result_dir, model_dir = output_dir / candidate_id, artifact_dir / candidate_id
    ft.new_directory(result_dir)
    ft.new_directory(model_dir)
    labels = np.asarray([LABELS.index(label) for label in context["frame"].clarity])
    train, valid = context["screen_train"], context["screen_valid"]
    started = time.monotonic()
    trainer, training = ft.train_one(
        config, tokenizer, features, labels, train, valid, model_dir, config["seed"], use_cpu=False)
    if training["training_stopped_by_budget"]:
        raise RuntimeError(f"Screening {candidate_id} stopped at its time budget; ranking is invalid.")
    prediction = trainer.predict(ft.TokenizedRows(features, labels, valid))
    predicted = np.asarray(LABELS)[prediction.predictions.argmax(axis=-1)]
    scores = metrics(context["frame"].clarity.iloc[valid].to_numpy(), predicted)
    if not math.isclose(scores["accuracy"], training["best_inner_accuracy"], abs_tol=1e-8):
        raise ValueError(f"Restored checkpoint accuracy disagrees with screening score for {candidate_id}.")
    best_epoch, epoch_source = recover_best_epoch(training, config["epochs"])
    loss = float(training["trainer_metrics"]["train_loss"])
    if not math.isfinite(loss):
        raise ValueError(f"Nonfinite training loss for {candidate_id}.")
    result = {**{key: candidate[key] for key in ("candidate_id", "config_sha256", "learning_rate",
                "lora_rank", "lora_alpha", "lora_dropout")},
              **scores, "best_epoch": best_epoch, "best_epoch_source": epoch_source,
              "best_checkpoint": training["best_checkpoint"], "training_loss": loss,
              "trainable_parameters": training["trainable_parameters"],
              "train_rows": len(train), "validation_rows": len(valid),
              "training_seconds": training["training_seconds"],
              "runtime_seconds": time.monotonic() - started, "seed": config["seed"],
              "screening_split_sha256": split_sha256,
              "adapter_dir": str(model_dir / "best_adapter"),
              "training_report": str(model_dir / "training.json")}
    pd.DataFrame({"excel_row": context["excel_rows"][valid],
                  "true_label": context["frame"].clarity.iloc[valid].to_numpy(),
                  "predicted_label": predicted}).to_csv(result_dir / "predictions.csv", index=False)
    write_json(result_dir / "results.json", result)
    del trainer, prediction
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_grid(args) -> dict:
    print(f"GRID START pid={os.getpid()} output={args.output_dir}", flush=True)
    base = ft.read_finetune_config(args.config)
    context = prepare_run(args, base)
    use_cpu, accelerator = ft.select_execution_device("gpu")
    if use_cpu:
        raise RuntimeError("Grid search requires a functional GPU.")
    log_selected_device(accelerator, operation="grid screening", requested="gpu")
    torch.set_num_threads(base["threads"])
    initialize_directories(args.output_dir, args.artifact_dir)
    summary = initial_summary(args, base, context, accelerator)
    split_path = args.output_dir / "screening-split.csv"
    context["screening_split"].to_csv(split_path, index=False)
    split_sha256 = file_digest(split_path)
    summary["screening_split_sha256"] = split_sha256
    for candidate in context["candidates"]:
        write_json(args.output_dir / "configs" / f"{candidate['candidate_id']}.json",
                   candidate["config"])
        loaded = ft.read_finetune_config(args.output_dir / "configs" /
                                         f"{candidate['candidate_id']}.json")
        if loaded != candidate["config"]:
            raise ValueError("Candidate config changed during serialization.")
    write_json(args.output_dir / "summary.json", summary)
    try:
        tokenizer = ft.AutoTokenizer.from_pretrained(
            base["model_name"], revision=base["revision"],
            cache_dir=".cache/huggingface", local_files_only=True, do_lower_case=False)
        if args.smoke:
            candidate = context["candidates"][0]
            model = ft.make_model(candidate["config"], base["seed"])
            parameters = sum(parameter.numel() for parameter in model.parameters()
                             if parameter.requires_grad)
            del model
            gc.collect()
            summary.update(status="smoke_complete", stage="smoke", finished_at=utc_now(),
                           smoke={"candidate_id": candidate["candidate_id"],
                                  "trainable_parameters": parameters,
                                  "model_initialized": True, "training_performed": False})
            write_json(args.output_dir / "summary.json", summary)
            print("GRID SMOKE COMPLETE", args.output_dir, flush=True)
            return summary

        texts = context["frame"].resp_text.to_numpy()
        features, tokenization = ft.tokenize_corpus(texts, tokenizer, base)
        summary.update(stage="screening", tokenization=tokenization)
        write_json(args.output_dir / "summary.json", summary)
        for candidate in context["candidates"]:
            print("SCREENING START", candidate["candidate_id"], candidate["learning_rate"],
                  candidate["lora_rank"], flush=True)
            result = run_screening_candidate(
                candidate, context, tokenizer, features, args.output_dir / "screening",
                args.artifact_dir / "screening", split_sha256)
            summary["screening_results"].append(result)
            save_screening_csv(args.output_dir / "screening_results.csv",
                               summary["screening_results"])
            write_json(args.output_dir / "summary.json", summary)
            print("SCREENING COMPLETE", candidate["candidate_id"], result["accuracy"], flush=True)
        summary["screening_ranking"], summary["top_three_comparison"] = summarize_screening(
            summary["screening_results"])
        summary["top_three"] = [row["candidate_id"] for row in summary["screening_ranking"][:3]]
        save_screening_csv(args.output_dir / "screening_results.csv",
                           summary["screening_ranking"])
        save_screening_report(args.output_dir / "screening_report.md",
                              summary["screening_ranking"], summary["top_three_comparison"])
        summary.update(status="complete", stage="screening_complete", finished_at=utc_now(),
                       best_screening_candidate_id=summary["top_three"][0])
        write_json(args.output_dir / "summary.json", summary)
        print("SCREENING COMPLETE; NO CV RUN", summary["top_three"], args.output_dir, flush=True)
        return summary
    except Exception as error:
        summary.update(status="failed", finished_at=utc_now(),
                       error={"type": type(error).__name__, "message": str(error)})
        write_json(args.output_dir / "summary.json", summary)
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--holdout-splits", type=Path, default=DEFAULT_SPLIT)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--artifact-dir", type=Path, required=True)
    command.add_argument("--smoke", action="store_true", help="Initialize one model; do not train.")
    return command


def main() -> None:
    args = parser().parse_args()
    base = ft.read_finetune_config(args.config)
    with threadpool_limits(limits=base["threads"]):
        run_grid(args)


if __name__ == "__main__":
    main()
