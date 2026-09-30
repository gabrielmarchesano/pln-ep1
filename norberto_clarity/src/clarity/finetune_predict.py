"""Export unlabeled Excel predictions from a verified final LoRA adapter."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import finetune as ft
from .data import LABEL_COLUMN, LABELS, file_digest, read_frame
from .delivery import require_new_file, write_predictions
from .reporting import environment, write_json
from .finetune_final import adapter_hashes, check_destinations
from .hardware import log_selected_device


def load_verified_final(config_path: Path, artifact_dir: Path) -> tuple[dict, dict, Path, dict]:
    """Reject stale or incompatible final adapters before processing test data."""
    config = ft.read_finetune_config(config_path)
    metadata_path = artifact_dir / "run.json"
    saved = json.loads(metadata_path.read_text(encoding="utf-8"))
    if saved.get("mode") != "final" or saved.get("status") != "complete":
        raise ValueError("Only a completed final model can predict test rows.")
    if config != saved.get("approved_config") or list(LABELS) != saved.get("labels"):
        raise ValueError("Prediction configuration or labels differ from the saved final model.")
    final_path = artifact_dir / "final_model"
    hashes = adapter_hashes(final_path)
    if hashes != saved.get("adapter_sha256"):
        raise ValueError("Final adapter files changed since training.")
    return config, saved, final_path, hashes


def select_inference_device(requested: str) -> tuple[bool, dict]:
    """Use the same CPU/GPU selection policy as final training."""
    return ft.select_execution_device(requested)


def predict_test(args) -> dict:
    """Predict a new workbook without training or using its labels for evaluation."""
    check_destinations(args.output_dir, args.artifact_dir)
    if args.output.suffix.lower() != ".xlsx":
        raise ValueError("The labeled output must be an .xlsx workbook.")
    if args.test.resolve() == args.output.resolve():
        raise ValueError("The output cannot overwrite the test workbook.")
    if args.output.resolve().is_relative_to(args.artifact_dir.resolve()):
        raise ValueError("The delivery workbook must be outside the saved-model directory.")
    require_new_file(args.output)
    config, saved, final_path, hashes = load_verified_final(args.config, args.artifact_dir)
    frame = read_frame(args.test, training=False)
    if LABEL_COLUMN in frame and frame[LABEL_COLUMN].notna().any():
        raise ValueError("Test labels must be empty; refusing to overwrite existing labels.")
    input_sha256 = file_digest(args.test)
    use_cpu, accelerator = select_inference_device(getattr(args, "device", "auto"))
    ft.new_directory(args.output_dir)
    log_selected_device(accelerator, operation="test prediction",
                        requested=getattr(args, "device", "auto"))
    torch.set_num_threads(config["threads"])
    metadata = {
        "mode": "predict-test", "status": "running",
        "protocol": "frozen_final_adapter_unlabeled_excel_export",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "training_performed": False, "evaluation_performed": False,
        "test_path": str(args.test), "test_sha256": input_sha256,
        "test_rows": len(frame), "output_path": str(args.output),
        "source_final_metadata_sha256": file_digest(args.artifact_dir / "run.json"),
        "training_dataset_sha256": saved["dataset_sha256"],
        "final_model": str(final_path), "adapter_sha256": hashes,
        "config": saved["config"], "labels": list(LABELS),
        "environment": environment(), "accelerator": accelerator,
        "requested_device": getattr(args, "device", "auto"),
    }
    write_json(args.output_dir / "manifest.json", metadata)
    tokenizer = ft.AutoTokenizer.from_pretrained(final_path, local_files_only=True)
    features, statistics = ft.tokenize_corpus(frame.resp_text.to_numpy(), tokenizer, config)
    model = ft.make_model(config, config["seed"], adapter_path=final_path,
                          allow_download=True)
    trainer = ft.Trainer(
        model=model,
        args=ft.TrainingArguments(
            output_dir=str(args.output_dir / "inference"),
            per_device_eval_batch_size=config["eval_batch_size"],
            fp16=config["fp16"] and not use_cpu, use_cpu=use_cpu,
            report_to=[], disable_tqdm=True, dataloader_num_workers=0,
            eval_accumulation_steps=32,
        ),
        processing_class=tokenizer,
        data_collator=ft.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    started = time.monotonic()
    raw = trainer.predict(ft.TokenizedRows(features, None, np.arange(len(frame))))
    inference_seconds = time.monotonic() - started
    probabilities = torch.softmax(torch.from_numpy(raw.predictions).float(), dim=-1).numpy()
    probability_columns = ft.probability_columns("norberto_lora", probabilities)
    predicted = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    if file_digest(args.test) != input_sha256:
        raise ValueError("Test workbook changed during inference.")
    if adapter_hashes(final_path) != hashes:
        raise ValueError("Final adapter files changed during inference.")
    write_predictions(args.test, args.output, predicted)
    pd.DataFrame({"excel_row": np.arange(len(frame)) + 2,
                  "predicted_label": predicted, **probability_columns}).to_csv(
                      args.output_dir / "predictions.csv", index=False)
    report = {
        **metadata, "status": "complete",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "inference_seconds": inference_seconds, "tokenization": statistics,
        "prediction_counts": {label: int(np.count_nonzero(predicted == label)) for label in LABELS},
        "predictions_sha256": file_digest(args.output_dir / "predictions.csv"),
        "output_sha256": file_digest(args.output),
    }
    write_json(args.output_dir / "results.json", report)
    write_json(args.output_dir / "manifest.json", report)
    print("COMPLETE TEST PREDICTION", args.output, flush=True)
    return report
