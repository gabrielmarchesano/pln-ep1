"""Final development-only training and an explicitly separate holdout evaluation."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import finetune as ft
from .audit import audit_result_dir
from .data import LABELS, file_digest, load_training_data
from .reporting import environment, metrics, write_json
from .finetune_data import prepare_training_corpus
from .finetune_epochs import select_final_epochs
from .holdout import load_fixed_holdout
from .hardware import log_selected_device
from .result_paths import resolve_result_input


def verify_development_rows(split_path: Path, excel_rows, groups) -> dict:
    split = pd.read_csv(split_path, dtype={"group_id": str})
    development = split.loc[split.role.eq("development")]
    holdout = split.loc[split.role.eq("holdout")]
    if (not np.array_equal(excel_rows, development.excel_row.to_numpy())
            or not np.array_equal(np.asarray(groups, dtype=str), development.group_id.to_numpy())):
        raise ValueError("Final training rows/groups differ from the frozen development set.")
    if set(excel_rows) & set(holdout.excel_row) or set(groups) & set(holdout.group_id):
        raise ValueError("Reserved holdout rows or groups leaked into final training.")
    return {"training_rows": len(development), "training_groups": development.group_id.nunique(),
            "holdout_row_overlap": 0, "holdout_group_overlap": 0}


def adapter_hashes(path: Path) -> dict:
    required = ("adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json")
    if any(not (path / name).is_file() for name in required):
        raise ValueError("The final adapter/tokenizer is incomplete.")
    return {str(file.relative_to(path)): file_digest(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def check_destinations(output_dir: Path, artifact_dir: Path) -> None:
    output, artifact = output_dir.resolve(), artifact_dir.resolve()
    if output.is_relative_to(artifact) or artifact.is_relative_to(output):
        raise ValueError("Report and artifact directories must be separate and non-nested.")


def run_final(args) -> dict:
    if not args.holdout_splits or not args.experiment_dir:
        raise ValueError("Final mode requires --holdout-splits and --experiment-dir.")
    args.holdout_splits = resolve_result_input(args.holdout_splits)
    args.experiment_dir = resolve_result_input(args.experiment_dir)
    config = ft.read_finetune_config(args.config)
    check_destinations(args.output_dir, args.artifact_dir)
    frame, groups, excel_rows, holdout_metadata = prepare_training_corpus(
        args.train, args.holdout_splits, config.get("holdout_split_sha256"))
    isolation = verify_development_rows(args.holdout_splits, excel_rows, groups)
    dataset_sha256 = file_digest(args.train)
    selection = select_final_epochs(args.experiment_dir, config, dataset_sha256,
                                    holdout_metadata["split_sha256"], len(frame))
    source_audit = audit_result_dir(args.experiment_dir, args.train)
    effective_config = {**config, "epochs": selection["final_epochs"]}
    requested_device = getattr(args, "device", "auto")
    use_cpu, accelerator = ft.select_execution_device(requested_device)
    ft.new_directory(args.output_dir)
    ft.new_directory(args.artifact_dir)
    log_selected_device(accelerator, operation="final training", requested=requested_device)
    torch.set_num_threads(config["threads"])
    rows_path = args.artifact_dir / "training-rows.csv"
    pd.DataFrame({"excel_row": excel_rows, "group_id": groups}).to_csv(rows_path, index=False)
    metadata = {
        "mode": "final", "status": "running",
        "protocol": "fresh_base_full_development_fixed_epochs_no_validation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_name": config["model_name"], "revision": config["revision"],
        "config": effective_config, "approved_config": config, "labels": list(LABELS),
        "epoch_selection": selection, "dataset_sha256": dataset_sha256,
        "training_rows_sha256": file_digest(rows_path),
        "environment": environment(), "accelerator": accelerator,
        "requested_device": requested_device,
        "source_experiment_audit": source_audit,
        "initialization": "fresh_pinned_base_checkpoint; no fold adapter loaded",
        **holdout_metadata, **isolation,
    }
    write_json(args.output_dir / "manifest.json", metadata)
    write_json(args.artifact_dir / "run.json", metadata)
    write_json(args.artifact_dir / "config.json", effective_config)
    tokenizer = ft.AutoTokenizer.from_pretrained(
        config["model_name"], revision=config["revision"],
        cache_dir=".cache/huggingface", local_files_only=False, do_lower_case=False)
    features, statistics = ft.tokenize_corpus(frame.resp_text.to_numpy(), tokenizer, config)
    write_json(args.output_dir / "tokenization.json", statistics)
    labels = np.asarray([LABELS.index(label) for label in frame.clarity])
    print(f"FINAL: train={len(frame)}; epochs={selection['final_epochs']}; "
          f"fold_best_epochs={[row['best_epoch'] for row in selection['fold_best_epochs']]}; "
          "holdout_evaluated=False", flush=True)
    try:
        trainer, training = ft.train_one(
            effective_config, tokenizer, features, labels, np.arange(len(frame)), None,
            args.artifact_dir, config["seed"], final=True, allow_download=True, use_cpu=use_cpu)
        complete = (not training["training_stopped_by_budget"]
                    and np.isclose(training["epochs_executed"], selection["final_epochs"])
                    and np.isfinite(training["training_loss"]))
        report = {**metadata, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "status": "complete" if complete else "incomplete",
                  "training": training, "tokenization": statistics,
                  "final_model": str(args.artifact_dir / "final_model"),
                  "adapter_sha256": adapter_hashes(args.artifact_dir / "final_model")}
        del trainer
    except Exception as error:
        failure = {**metadata, "status": "failed", "error": str(error)}
        write_json(args.output_dir / "manifest.json", failure)
        write_json(args.artifact_dir / "run.json", failure)
        raise
    write_json(args.output_dir / "results.json", report)
    write_json(args.output_dir / "manifest.json", report)
    write_json(args.artifact_dir / "run.json", report)
    if not complete:
        raise RuntimeError("Final training ended before completing its selected epochs.")
    print("COMPLETE FINAL", args.artifact_dir / "final_model", flush=True)
    return report


def evaluate_holdout(args) -> dict:
    """Load only the saved final adapter and run inference without optimization."""
    if not args.holdout_splits:
        raise ValueError("Holdout evaluation requires --holdout-splits.")
    args.holdout_splits = resolve_result_input(args.holdout_splits)
    check_destinations(args.output_dir, args.artifact_dir)
    config = ft.read_finetune_config(args.config)
    metadata_path = args.artifact_dir / "run.json"
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved.get("mode") != "final" or saved.get("status") != "complete":
        raise ValueError("Only a completed final model can be evaluated on the holdout.")
    if config != saved["approved_config"] or list(LABELS) != saved["labels"]:
        raise ValueError("Evaluation configuration or labels differ from the saved final model.")
    if file_digest(args.train) != saved["dataset_sha256"]:
        raise ValueError("Dataset changed since final training.")
    if config["holdout_split_sha256"] != saved["split_sha256"]:
        raise ValueError("Holdout split differs from final training.")
    final_path = args.artifact_dir / "final_model"
    hashes = adapter_hashes(final_path)
    if hashes != saved["adapter_sha256"]:
        raise ValueError("Final adapter files changed since training.")
    rows_path = args.artifact_dir / "training-rows.csv"
    if file_digest(rows_path) != saved["training_rows_sha256"]:
        raise ValueError("Saved final training row identifiers changed.")
    training_rows = pd.read_csv(rows_path, dtype={"group_id": str})
    isolation = verify_development_rows(
        args.holdout_splits, training_rows.excel_row.to_numpy(), training_rows.group_id.to_numpy())
    frame = load_training_data(args.train)
    _, holdout, groups = load_fixed_holdout(
        args.train, args.holdout_splits, frame.clarity.to_numpy(),
        expected_split_sha256=saved["split_sha256"])
    requested_device = getattr(args, "device", "auto")
    use_cpu, accelerator = ft.select_execution_device(requested_device)
    ft.new_directory(args.output_dir)
    log_selected_device(accelerator, operation="holdout evaluation",
                        requested=requested_device)
    torch.set_num_threads(config["threads"])
    metadata = {
        "mode": "evaluate-holdout", "status": "running",
        "protocol": "final_model_frozen_holdout_evaluation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False, "holdout_evaluated": True,
        "source_final_metadata_sha256": file_digest(metadata_path),
        "final_model": str(final_path), "adapter_sha256": hashes,
        "dataset_sha256": saved["dataset_sha256"], "split_sha256": saved["split_sha256"],
        "split_path": str(args.holdout_splits), "config": saved["config"],
        "labels": list(LABELS), "environment": environment(), "accelerator": accelerator,
        "requested_device": requested_device,
        "caveat": "Final-model evaluation only. This frozen holdout has previously been "
                  "consulted in lexical research; it is not historically untouched.",
        **isolation,
    }
    write_json(args.output_dir / "manifest.json", metadata)
    tokenizer = ft.AutoTokenizer.from_pretrained(final_path, local_files_only=True)
    target = frame.iloc[holdout]
    features, statistics = ft.tokenize_corpus(target.resp_text.to_numpy(), tokenizer, config)
    labels = np.asarray([LABELS.index(label) for label in target.clarity])
    model = ft.make_model(config, config["seed"], adapter_path=final_path)
    arguments = ft.TrainingArguments(
        output_dir=str(args.output_dir / "inference"),
        per_device_eval_batch_size=config["eval_batch_size"],
        fp16=config["fp16"] and not use_cpu, use_cpu=use_cpu,
        report_to=[], disable_tqdm=True, dataloader_num_workers=0,
        label_names=["labels"], eval_accumulation_steps=32,
    )
    trainer = ft.Trainer(
        model=model, args=arguments, processing_class=tokenizer,
        data_collator=ft.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8))
    started = time.monotonic()
    prediction = trainer.predict(ft.TokenizedRows(features, labels, np.arange(len(target))))
    seconds = time.monotonic() - started
    probabilities = torch.softmax(torch.from_numpy(prediction.predictions).float(), dim=-1).numpy()
    name = ft.model_spec(config)["result_name"]
    columns = ft.probability_columns(name, probabilities)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    output = pd.DataFrame({"excel_row": holdout + 2, "group_id": groups[holdout],
                           "true_label": target.clarity.to_numpy(), f"pred_{name}": predicted,
                           **columns})
    output.to_csv(args.output_dir / "holdout-predictions.csv", index=False)
    report = {**metadata, "status": "complete", "evaluated_rows": len(target),
              "finished_at": datetime.now(timezone.utc).isoformat(),
              "inference_seconds": seconds, "tokenization": statistics,
              **metrics(target.clarity, predicted),
              "classification_report": classification_report(
                  target.clarity, predicted, labels=LABELS, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(target.clarity, predicted, labels=LABELS).tolist()}
    if adapter_hashes(final_path) != hashes:
        raise ValueError("Final adapter files changed during evaluation.")
    write_json(args.output_dir / "results.json", report)
    write_json(args.output_dir / "manifest.json", report)
    print("COMPLETE HOLDOUT EVALUATION", args.output_dir, flush=True)
    return report
