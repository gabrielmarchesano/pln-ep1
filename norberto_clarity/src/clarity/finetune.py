"""Budgeted supervised encoder experiments with group-isolated validation.

Unlike experiment.py, this uses one inner grouped holdout per outer fold, not
an inner grid-search CV. The holdout selects a checkpoint; the outer fold is
used only after training. A pilot never evaluates an outer fold.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)

from .data import LABELS, file_digest, grouped_splits
from .reporting import environment, metrics, paired_group_bootstrap, write_json
from .finetune_data import prepare_training_corpus
from .hardware import log_selected_device, probe_accelerator

MODEL_ID = "neuralmind/bert-base-portuguese-cased"
MODEL_REVISION = "94d69c95f98f7d5b2a8700c420230ae10def0baa"
NORBERTO_ID = "Itau-Unibanco/NorBERTo-base"
NORBERTO_REVISION = "db73446f89c96044863ea05a39f680524b84bccb"
MODEL_SPECS = {
    MODEL_ID: {
        "revision": MODEL_REVISION,
        "architecture": "bert",
        "num_hidden_layers": 12,
        "max_length": 512,
        "lora_target_modules": ["query", "value"],
        "pretrained_options": {},
        "result_name": "bertimbau_lora",
    },
    NORBERTO_ID: {
        "revision": NORBERTO_REVISION,
        "architecture": "modernbert",
        "num_hidden_layers": 22,
        "max_length": 8192,
        "lora_target_modules": ["Wqkv"],
        "pretrained_options": {"reference_compile": False},
        "result_name": "norberto_lora",
    },
}


def model_spec(config: dict) -> dict:
    """Return the immutable experiment contract for a pinned base model."""
    spec = MODEL_SPECS.get(config.get("model_name"))
    if spec is None or config.get("revision") != spec["revision"]:
        raise ValueError("Unsupported model or revision; use a pinned model contract.")
    return spec


def clean_cased(text: str) -> str:
    """Normalization for model input, deliberately separate from grouping."""
    return re.sub(r"\s+", " ", html.unescape(str(text))).strip()


def trim_tokens(ids: list[int], budget: int, policy: str) -> list[int]:
    if budget < 1 or policy not in {"head", "head_tail"}:
        raise ValueError("Invalid token budget or truncation policy.")
    if len(ids) <= budget:
        return ids
    if policy == "head":
        return ids[:budget]
    head = (budget + 1) // 2
    tail = budget - head
    return ids[:head] + (ids[-tail:] if tail else [])


def read_finetune_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"model_name", "revision", "max_length", "truncation", "lora_rank",
                "lora_alpha", "lora_dropout", "lora_layers", "batch_size",
                "gradient_accumulation", "eval_batch_size", "learning_rate",
                "epochs", "patience", "weight_decay", "warmup_ratio", "fp16",
                "outer_folds", "inner_folds", "seed", "threads",
                "train_subset_folds", "budget_minutes", "attention"}
    if required - config.keys():
        raise ValueError(f"Missing config fields: {required - config.keys()}")
    spec = model_spec(config)
    for field in ["lora_rank", "lora_alpha", "batch_size", "gradient_accumulation", "eval_batch_size",
                  "epochs", "patience", "threads", "train_subset_folds", "max_length",
                  "outer_folds", "inner_folds"]:
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"{field} must be a positive integer.")
    if not 4 <= config["max_length"] <= spec["max_length"]:
        raise ValueError(f"max_length must be in [4, {spec['max_length']}] for this model.")
    if config["outer_folds"] < 2 or config["inner_folds"] < 2:
        raise ValueError("Both split counts must be at least two.")
    for field in ["learning_rate", "budget_minutes", "weight_decay", "warmup_ratio", "lora_dropout"]:
        if type(config[field]) not in (int, float) or not math.isfinite(config[field]):
            raise ValueError(f"{field} must be a finite number.")
    if type(config["fp16"]) is not bool or type(config["seed"]) is not int:
        raise ValueError("fp16 must be boolean and seed must be an integer.")
    if config["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative.")
    if config["learning_rate"] <= 0 or config["budget_minutes"] <= 0:
        raise ValueError("Learning rate and budget must be positive.")
    if not 0 <= config["warmup_ratio"] < 1 or not 0 <= config["lora_dropout"] < 1:
        raise ValueError("Invalid warmup ratio or dropout.")
    if config["attention"] not in {"eager", "sdpa"}:
        raise ValueError("Unknown attention implementation.")
    layers = config["lora_layers"]
    layer_count = spec["num_hidden_layers"]
    if layers is not None and (not isinstance(layers, list) or not layers
                               or len(set(layers)) != len(layers)
                               or any(type(i) is not int or not 0 <= i < layer_count
                                      for i in layers)):
        raise ValueError(
            f"lora_layers must be null or unique layer indices 0..{layer_count - 1}.")
    trim_tokens([1, 2], config["max_length"] - 2, config["truncation"])
    return config


def make_split(x, y, groups, config, fold: int):
    outer = grouped_splits(x, y, groups, config["outer_folds"], config["seed"])
    if not 1 <= fold <= len(outer):
        raise ValueError("Invalid outer fold.")
    outer_train, outer_valid = outer[fold - 1]
    inner_train, inner_valid = grouped_splits(
        x[outer_train], y[outer_train], groups[outer_train],
        config["inner_folds"], config["seed"] + fold,
    )[0]
    train, valid = outer_train[inner_train], outer_train[inner_valid]
    if config["train_subset_folds"] > 1:
        _, sample = grouped_splits(x[train], y[train], groups[train],
                                   config["train_subset_folds"], config["seed"] + 100 + fold)[0]
        train = train[sample]
    for left, right in [(train, valid), (train, outer_valid), (valid, outer_valid)]:
        if set(groups[left]) & set(groups[right]):
            raise ValueError("Group leakage in fine-tuning partitions.")
    return train, valid, outer_train, outer_valid


def tokenize_corpus(x, tokenizer, config, cache_dir=Path(".cache/finetune-tokens")):
    texts = [clean_cased(text) for text in x]
    model_input_names = list(getattr(tokenizer, "model_input_names", []))
    fingerprint = hashlib.sha256(json.dumps({
        "texts": texts, "model": config["model_name"], "revision": config["revision"],
        "max_length": config["max_length"], "truncation": config["truncation"],
        "model_input_names": model_input_names, "normalization_version": 2,
    }, ensure_ascii=False).encode()).hexdigest()
    path = cache_dir / f"{fingerprint}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        return cached["features"], cached["statistics"]
    ids = tokenizer(texts, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
    budget = config["max_length"] - tokenizer.num_special_tokens_to_add(pair=False)
    features = []
    for tokens in ids:
        inputs = tokenizer.build_inputs_with_special_tokens(trim_tokens(tokens, budget, config["truncation"]))
        feature = {"input_ids": inputs, "attention_mask": [1] * len(inputs)}
        if "token_type_ids" in model_input_names:
            feature["token_type_ids"] = [0] * len(inputs)
        features.append(feature)
    lengths = np.asarray([len(tokens) for tokens in ids])
    statistics = {"rows": len(texts), "truncated_rows": int((lengths > budget).sum()),
                  "truncated_fraction": float((lengths > budget).mean()),
                  "median_raw_tokens": float(np.median(lengths)),
                  "p95_raw_tokens": float(np.quantile(lengths, 0.95)),
                  "input_case_preserved": True, "cache_fingerprint": fingerprint}
    write_json(path, {"features": features, "statistics": statistics})
    return features, statistics


class TokenizedRows(torch.utils.data.Dataset):
    def __init__(self, features, labels, indices):
        self.features, self.labels, self.indices = features, labels, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        feature = self.features[row]
        return feature if self.labels is None else {**feature, "labels": int(self.labels[row])}


def probability_columns(name, probabilities, classes=LABELS):
    """Stable named columns; never assume a control's class order matches ours."""
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 2 or probabilities.shape[1] != len(LABELS)
            or len(classes) != len(LABELS) or set(classes) != set(LABELS)
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Invalid class probabilities.")
    return {f"prob_{name}_{label}": probabilities[:, list(classes).index(label)]
            for label in LABELS}


def make_model(config, seed, *, adapter_path=None, allow_download=False):
    set_seed(seed)
    spec = model_spec(config)
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], revision=config["revision"],
        cache_dir=".cache/huggingface", local_files_only=not allow_download,
        num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        attn_implementation=config["attention"],
        **spec["pretrained_options"],
    )
    if adapter_path is not None:
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    return get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_CLS, r=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
        target_modules=spec["lora_target_modules"],
        layers_to_transform=config["lora_layers"],
    ))


class BudgetProgress(TrainerCallback):
    def __init__(self, minutes):
        self.minutes = minutes
        self.started = time.monotonic()
        self.last = self.started
        self.exhausted = False

    def on_step_end(self, args, state, control, **kwargs):
        now = time.monotonic()
        if now - self.last >= 30:
            print(f"  step={state.global_step}/{state.max_steps}; epoch={state.epoch:.3f}; "
                  f"training_minutes={(now-self.started)/60:.2f}", flush=True)
            self.last = now
        if now - self.started > self.minutes * 60:
            self.exhausted = True
            control.should_training_stop = True
            print("  Training budget reached; finish evaluation/checkpoint, then stop.", flush=True)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        for field in ("loss", "grad_norm", "eval_loss"):
            if field in (logs or {}) and not math.isfinite(logs[field]):
                raise FloatingPointError(f"Nonfinite {field}; stopping experiment.")


def select_execution_device(requested: str) -> tuple[bool, dict]:
    """Resolve CPU or a validated GPU, detecting CUDA versus ROCm internally."""
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("Device must be auto, cpu, or gpu.")
    if requested == "cpu":
        return True, {"backend": "cpu", "device_name": "CPU", "torch_version": torch.__version__}
    if not torch.cuda.is_available():
        if requested == "gpu":
            raise RuntimeError("GPU was requested but no CUDA or ROCm device is available to PyTorch.")
        return True, {"backend": "cpu", "device_name": "CPU", "torch_version": torch.__version__}
    return False, probe_accelerator()


def train_one(config, tokenizer, features, labels, train, valid, artifact_dir, seed,
              *, final=False, allow_download=False, use_cpu: bool | None = None):
    if final and valid is not None:
        raise ValueError("Final training must not have a validation set.")
    use_accelerator = torch.cuda.is_available() if use_cpu is None else not use_cpu
    model = make_model(config, seed, allow_download=allow_download)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable}; train={len(train)}, "
          f"inner_valid={0 if final else len(valid)}", flush=True)
    callback = BudgetProgress(config["budget_minutes"])
    args = TrainingArguments(
        output_dir=str(artifact_dir / "checkpoints"),
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation"],
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"], num_train_epochs=config["epochs"],
        fp16=config["fp16"], use_cpu=not use_accelerator,
        optim="adamw_torch", max_grad_norm=1.0,
        eval_strategy="no" if final else "epoch", save_strategy="epoch", save_total_limit=1,
        save_only_model=True, load_best_model_at_end=not final,
        metric_for_best_model=None if final else "accuracy",
        greater_is_better=None if final else True,
        logging_steps=20, logging_nan_inf_filter=False,
        report_to=[], disable_tqdm=True, seed=seed, data_seed=seed,
        dataloader_num_workers=0, dataloader_pin_memory=True,
        label_names=["labels"], eval_accumulation_steps=32,
    )
    trainer = Trainer(
        model=model, args=args, processing_class=tokenizer,
        train_dataset=TokenizedRows(features, labels, train),
        eval_dataset=None if final else TokenizedRows(features, labels, valid),
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        compute_metrics=None if final else lambda output: metrics(
            np.asarray(LABELS)[output.label_ids],
            np.asarray(LABELS)[output.predictions.argmax(axis=-1)]),
        callbacks=[callback] if final else [
            callback, EarlyStoppingCallback(early_stopping_patience=config["patience"])],
    )
    if use_accelerator:
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    result = trainer.train()
    training_seconds = time.monotonic() - started
    model_path = artifact_dir / ("final_model" if final else "best_adapter")
    trainer.save_model(str(model_path))
    tokenizer.save_pretrained(model_path)
    report = {"training_seconds": training_seconds, "trainable_parameters": trainable,
              "best_checkpoint": trainer.state.best_model_checkpoint,
              "best_inner_accuracy": trainer.state.best_metric,
              "epochs_executed": trainer.state.epoch,
              "training_stopped_by_budget": callback.exhausted,
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if use_accelerator else 0.0,
              "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if use_accelerator else 0.0,
              "trainer_metrics": result.metrics, "history": trainer.state.log_history}
    if final:
        report.update(mode="final", selected_epochs=config["epochs"], training_rows=len(train),
                      training_loss=float(result.training_loss), validation_used=False)
    write_json(artifact_dir / "training.json", report)
    return trainer, report


def new_directory(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def summarize_predictions(y, pred, fold_metrics):
    report = {"pooled": metrics(y, pred), "fold_results": fold_metrics,
              "labels": list(LABELS),
              "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
              "classification_report": classification_report(
                  y, pred, labels=LABELS, output_dict=True, zero_division=0)}
    for metric in ("accuracy", "f1_macro", "balanced_accuracy"):
        values = [row[metric] for row in fold_metrics]
        report[f"mean_{metric}"] = float(np.mean(values))
        report[f"std_{metric}"] = float(np.std(values))
    return report


def run(args):
    if args.mode in {"final", "evaluate-holdout", "predict-test"}:
        from .finetune_final import evaluate_holdout, run_final
        from .finetune_predict import predict_test
        return {"final": run_final, "evaluate-holdout": evaluate_holdout,
                "predict-test": predict_test}[args.mode](args)
    config = read_finetune_config(args.config)
    if getattr(args, "device", "auto") == "cpu":
        raise ValueError("Pilot and experiment modes require GPU; use --device auto or gpu.")
    spec = model_spec(config)
    frame, groups, excel_rows, holdout_metadata = prepare_training_corpus(
        args.train, getattr(args, "holdout_splits", None), config.get("holdout_split_sha256"))
    x, y = frame.resp_text.to_numpy(), frame.clarity.to_numpy()
    labels = np.asarray([LABELS.index(label) for label in y])
    folds = [1] if args.mode == "pilot" else list(range(1, config["outer_folds"] + 1))
    prepared_splits = {fold: make_split(x, y, groups, config, fold) for fold in folds}
    accelerator = probe_accelerator()
    if args.output_dir.resolve() == args.artifact_dir.resolve():
        raise ValueError("Reports and model artifacts must use separate directories.")
    new_directory(args.output_dir)
    new_directory(args.artifact_dir)
    log_selected_device(accelerator, operation=args.mode,
                        requested=getattr(args, "device", "auto"))
    torch.set_num_threads(config["threads"])
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(),
                "config": config, "dataset_sha256": file_digest(args.train),
                "environment": environment(), "gpu": accelerator["device_name"],
                "accelerator": accelerator, "torch_cuda": torch.version.cuda,
                "torch_hip": torch.version.hip, "mode": args.mode,
                "protocol": "grouped_inner_holdout_pilot" if args.mode == "pilot" else
                            "grouped_outer_cv_with_inner_holdout_checkpoint_selection",
                "note": "One inner grouped holdout, not inner k-fold grid search; no full outer-train refit.",
                "rows": len(frame), "requested_folds": folds, **holdout_metadata}
    write_json(args.output_dir / "manifest.json", metadata)
    write_json(args.artifact_dir / "run.json", metadata)
    write_json(args.artifact_dir / "config.json", config)
    pd.DataFrame({"excel_row": excel_rows, "group_id": groups}).to_csv(
        args.output_dir / "development-rows.csv", index=False)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], revision=config["revision"],
        cache_dir=".cache/huggingface", local_files_only=True, do_lower_case=False)
    features, statistics = tokenize_corpus(x, tokenizer, config)
    print("TOKENIZATION", statistics, flush=True)
    write_json(args.output_dir / "tokenization.json", statistics)
    outputs, records = [], []
    started = time.monotonic()
    for fold in folds:
        train, valid, outer_train, outer_valid = prepared_splits[fold]
        target = valid if args.mode == "pilot" else outer_valid
        split = {"fold": fold, "train_excel_rows": excel_rows[train].tolist(),
                 "inner_valid_excel_rows": excel_rows[valid].tolist(),
                 "outer_valid_excel_rows": excel_rows[outer_valid].tolist(),
                 "train_rows": len(train), "inner_valid_rows": len(valid),
                 "outer_valid_rows": len(outer_valid), "group_overlap": 0}
        write_json(args.output_dir / f"split-{fold}.json", split)
        print(f"FOLD {fold}: train={len(train)}, inner={len(valid)}, outer={len(outer_valid)}", flush=True)
        trainer, training = train_one(config, tokenizer, features, labels, train, valid,
                                      args.artifact_dir / f"fold-{fold}", config["seed"] + fold)
        tick = time.monotonic()
        predictions = trainer.predict(TokenizedRows(features, labels, target))
        inference_seconds = time.monotonic() - tick
        encoder_name = spec["result_name"]
        predicted = {encoder_name: np.asarray(LABELS)[predictions.predictions.argmax(axis=-1)]}
        probabilities = torch.softmax(torch.from_numpy(predictions.predictions).float(), dim=-1).numpy()
        probability_data = probability_columns(encoder_name, probabilities)
        del trainer, predictions
        gc.collect()
        torch.cuda.empty_cache()
        # Matched controls use exactly the rows seen by the encoder's optimizer.
        # The full lexical reference is evaluated only on an untouched outer fold.
        from model_research.models import build_model
        specs = [("baseline_matched", "baseline", train), ("lexical_matched", "lexical", train)]
        if args.mode == "experiment":
            specs.append(("lexical_full", "lexical", outer_train))
        control_seconds = {}
        for name, model_name, rows in specs:
            tick = time.monotonic()
            control = build_model(model_name, seed=config["seed"] + fold, cache_dir=".cache/sklearn")
            if model_name == "lexical":
                control.set_params(classifier__C=0.25)
            control.fit(x[rows], y[rows])
            predicted[name] = control.predict(x[target])
            probability_data.update(probability_columns(name, control.predict_proba(x[target]), control.classes_))
            control_seconds[name] = time.monotonic() - tick
        record = {"fold": fold, "training": training, "control_seconds": control_seconds,
                  "encoder_inference_seconds": inference_seconds,
                  "metrics": {name: metrics(y[target], pred) for name, pred in predicted.items()}}
        records.append(record)
        write_json(args.output_dir / f"fold-{fold}.json", record)
        output = pd.DataFrame({"excel_row": excel_rows[target], "group_id": groups[target],
                               "fold": fold, "true_label": y[target],
                               **{f"pred_{name}": pred for name, pred in predicted.items()},
                               **probability_data})
        output.to_csv(args.output_dir / f"predictions-fold-{fold}.csv", index=False)
        outputs.append(output)
        print("FOLD RESULT", record["metrics"], flush=True)
    joined = pd.concat(outputs).sort_values("excel_row").reset_index(drop=True)
    if args.mode == "experiment" and not np.array_equal(joined.excel_row.to_numpy(), excel_rows):
        raise ValueError("OOF predictions must cover every development row exactly once.")
    prediction_names = [column.removeprefix("pred_") for column in joined if column.startswith("pred_")]
    reference = "lexical_full" if args.mode == "experiment" else "lexical_matched"
    report = {**metadata, "finished_at": datetime.now(timezone.utc).isoformat(),
              "seconds": time.monotonic() - started, "evaluated_rows": len(joined),
              "status": "complete", "tokenization": statistics,
              "models": {name: summarize_predictions(joined.true_label, joined[f"pred_{name}"],
                          [{"fold": r["fold"], **r["metrics"][name]} for r in records])
                         for name in prediction_names},
              "comparisons_with_lexical": {
                  name: paired_group_bootstrap(joined.true_label.to_numpy(), joined[f"pred_{name}"].to_numpy(),
                      joined[f"pred_{reference}"].to_numpy(), joined.group_id.to_numpy(), seed=config["seed"])
                  for name in prediction_names if name != reference},
              "limitations": ["Pilot scores select checkpoints and are not outer-test estimates.",
                              "Epochs selected by one inner holdout, not full inner CV.",
                              "No full outer-train refit; matched and full lexical controls distinguish training size.",
                              "Research configurations may change between runs; repeated comparisons are exploratory.",
                              "A global configuration chosen on fold-1 pilot labels can adapt to rows in outer folds 2/3; outer CV is exploratory, not a fresh confirmatory test.",
                              "Budget stops, when present, are recorded per fold; training may end before requested epochs."]}
    joined.to_csv(args.output_dir / ("oof.csv" if args.mode == "experiment" else "inner-predictions.csv"), index=False)
    write_json(args.output_dir / "results.json", report)
    print("COMPLETE", args.output_dir, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "experiment", "final", "evaluate-holdout", "predict-test"])
    parser.add_argument("--train", type=Path, default=Path("data/train.xlsx"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, help="Unlabeled Excel workbook for final-model prediction.")
    parser.add_argument("--output", type=Path, help="New labeled .xlsx file for predict-test mode.")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto",
                        help="Execution device; GPU detects CUDA or ROCm. Pilot/experiment require GPU.")
    parser.add_argument("--holdout-splits", type=Path,
                        help="Frozen split CSV; exclude its holdout from this entire run.")
    parser.add_argument("--experiment-dir", type=Path,
                        help="Completed experiment providing final-training epoch selection.")
    args = parser.parse_args()
    if args.mode == "final" and (not args.holdout_splits or not args.experiment_dir):
        parser.error("final requires --holdout-splits and --experiment-dir.")
    if args.mode == "evaluate-holdout" and not args.holdout_splits:
        parser.error("evaluate-holdout requires --holdout-splits.")
    if args.mode == "predict-test" and (not args.test or not args.output):
        parser.error("predict-test requires --test and --output.")
    config = read_finetune_config(args.config)
    with threadpool_limits(limits=config["threads"]):
        run(args)


if __name__ == "__main__":
    main()
