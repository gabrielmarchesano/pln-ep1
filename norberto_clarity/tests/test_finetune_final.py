import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from clarity.data import file_digest, load_training_data, text_groups
from clarity.finetune_epochs import recover_best_epoch, select_final_epochs


def write_experiment(root, config, rows=30, dataset="dataset", split="split"):
    root.mkdir()
    report = {"mode": "experiment", "status": "complete", "config": config,
              "dataset_sha256": dataset, "split_sha256": split,
              "rows": rows, "evaluated_rows": rows, "holdout_evaluated": False,
              "requested_folds": [1, 2, 3]}
    (root / "results.json").write_text(json.dumps(report))
    for fold, epoch in enumerate((2, 3, 2), 1):
        training = {"best_checkpoint": f"fold-{fold}/checkpoints/checkpoint-{epoch * 10}",
                    "best_inner_accuracy": 0.5, "epochs_executed": 3,
                    "history": [{"epoch": e, "step": e * 10,
                                 "eval_accuracy": 0.5 if e == epoch else 0.4}
                                for e in (1, 2, 3)]}
        (root / f"fold-{fold}.json").write_text(json.dumps({"fold": fold, "training": training}))
    return report


def test_epoch_selection_uses_best_checkpoint_not_last_epoch(tmp_path):
    config = {"epochs": 3, "outer_folds": 3}
    root = tmp_path / "experiment"
    write_experiment(root, config)
    selection = select_final_epochs(root, config, "dataset", "split", 30)
    assert [row["best_epoch"] for row in selection["fold_best_epochs"]] == [2, 3, 2]
    assert selection["final_epochs"] == 2
    assert "median" in selection["rule"]
    with pytest.raises(ValueError, match="configuration"):
        select_final_epochs(root, {**config, "learning_rate": 0.02}, "dataset", "split", 30)
    with pytest.raises(ValueError, match="different dataset"):
        select_final_epochs(root, config, "changed", "split", 30)
    with pytest.raises(ValueError, match="best checkpoint"):
        recover_best_epoch({"epochs_executed": 3}, 3)


def test_explicit_final_epoch_takes_priority_without_reestimating(tmp_path):
    config = {"epochs": 3, "outer_folds": 3}
    root = tmp_path / "experiment"
    report = write_experiment(root, config)
    report["final_training_epochs"] = 1
    (root / "results.json").write_text(json.dumps(report))
    # The explicit choice must not depend on recovering per-fold epochs again.
    (root / "fold-1.json").write_text("{}")
    selection = select_final_epochs(root, config, "dataset", "split", 30)
    assert selection["final_epochs"] == 1
    assert selection["rule"] == "explicit_final_epoch_recorded_in_experiment"


def test_final_training_without_validation_saves_reloadable_adapter(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from transformers import EarlyStoppingCallback, TrainingArguments
    from clarity import finetune as ft

    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=False)
    base = BertForSequenceClassification(BertConfig(
        vocab_size=8, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, num_labels=3))
    initial = copy.deepcopy(base)
    model = get_peft_model(base, LoraConfig(task_type=TaskType.SEQ_CLS, r=2,
                                           lora_alpha=4, target_modules=["query", "value"]))
    original_make_model = ft.make_model
    monkeypatch.setattr(ft, "make_model", lambda config, seed, **kwargs: model)
    monkeypatch.setattr(ft, "TrainingArguments", lambda **kwargs: TrainingArguments(
        **{**kwargs, "use_cpu": True, "dataloader_pin_memory": False}))
    for name in ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 0)
    config = ft.read_finetune_config(Path("configs/research/bertimbau-lora-pilot.json"))
    config.update(epochs=2, batch_size=2, gradient_accumulation=2)
    torch.set_num_threads(1)
    features, _ = ft.tokenize_corpus(["a b", "b c", "c a"] * 2, tokenizer, config, tmp_path / "tokens")
    labels = np.array([0, 1, 2] * 2)
    trainer, training = ft.train_one(config, tokenizer, features, labels,
                                    np.arange(6), None, tmp_path / "artifact", 42, final=True)
    assert training["epochs_executed"] == 2
    assert training["validation_used"] is False
    assert training["best_checkpoint"] is None
    assert np.isfinite(training["training_loss"])
    assert trainer.eval_dataset is None and not trainer.args.load_best_model_at_end
    assert not any(isinstance(callback, EarlyStoppingCallback)
                   for callback in trainer.callback_handler.callbacks)
    assert not any("eval_accuracy" in row for row in training["history"])
    final_path = tmp_path / "artifact/final_model"
    assert (final_path / "adapter_model.safetensors").exists()
    monkeypatch.setattr(ft.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **k: initial)
    restored = original_make_model(config, 42, adapter_path=final_path).eval()
    model.eval()
    batch = {"input_ids": torch.tensor([[2, 5, 6, 3]]),
             "attention_mask": torch.ones(1, 4, dtype=torch.long)}
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch).logits, model(**batch).logits)


def test_final_and_evaluation_use_disjoint_data_and_never_reuse_a_fold(training_path, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune as ft
    from clarity import finetune_final as final

    source = load_training_data(training_path)
    roles = np.where(np.arange(len(source)) // 6 % 4 == 0, "holdout", "development")
    split = tmp_path / "splits.csv"
    pd.DataFrame({"excel_row": np.arange(len(source)) + 2, "group_id": text_groups(source.resp_text),
                  "role": roles, "development_fold": 0}).to_csv(split, index=False)
    counts = {"development_rows": int((roles == "development").sum()),
              "holdout_rows": int((roles == "holdout").sum())}
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_sha256": file_digest(training_path), **counts}))
    config = ft.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    config["holdout_split_sha256"] = file_digest(split)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    experiment = tmp_path / "experiment"
    write_experiment(experiment, config, counts["development_rows"], file_digest(training_path), file_digest(split))
    monkeypatch.setattr(final, "audit_result_dir", lambda *args: {"status": "passed"})
    monkeypatch.setattr(ft, "probe_accelerator", lambda: {"device_name": "test"})
    monkeypatch.setattr(ft, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: None))
    tokenized = []
    monkeypatch.setattr(ft, "tokenize_corpus", lambda texts, *args: (
        tokenized.append(list(texts)) or [{"input_ids": [1]} for _ in texts], {"rows": len(texts)}))
    monkeypatch.setattr(ft, "make_split", lambda *a, **k: pytest.fail("Final mode created a fold"))

    def train(config, tokenizer, features, labels, train, valid, artifact, seed,
              *, final, allow_download, use_cpu):
        assert final and allow_download and valid is None and config["epochs"] == 2 and seed == 42
        assert use_cpu is True
        assert len(train) == len(features) == counts["development_rows"]
        path = artifact / "final_model"
        path.mkdir()
        for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"):
            (path / name).write_text("test fixture")
        return None, {"epochs_executed": 2, "training_loss": 1.0, "training_stopped_by_budget": False}

    monkeypatch.setattr(ft, "train_one", train)
    args = SimpleNamespace(mode="final", train=training_path, config=config_path,
                           holdout_splits=split, experiment_dir=experiment,
                           output_dir=tmp_path / "final-report", artifact_dir=tmp_path / "artifact",
                           device="cpu")
    report = ft.run(args)
    assert "DEVICE final training: cpu; backend=CPU; name=CPU" in capsys.readouterr().out
    assert report["mode"] == "final" and report["status"] == "complete"
    assert report["requested_device"] == "cpu"
    assert report["holdout_evaluated"] is False
    assert tokenized[0] == source.loc[roles == "development", "resp_text"].tolist()
    assert not list(args.output_dir.glob("split-*.json"))
    hashes = final.adapter_hashes(args.artifact_dir / "final_model")

    def load(config, seed, *, adapter_path):
        assert adapter_path == args.artifact_dir / "final_model"
        return object()

    class Predictor:
        def __init__(self, **kwargs):
            assert "train_dataset" not in kwargs
            assert kwargs["args"].use_cpu is True

        def train(self):
            pytest.fail("Holdout evaluation attempted training")

        def predict(self, dataset):
            assert len(dataset) == counts["holdout_rows"]
            return SimpleNamespace(predictions=np.zeros((len(dataset), 3), dtype=np.float32))

    monkeypatch.setattr(ft, "make_model", load)
    monkeypatch.setattr(ft, "Trainer", Predictor)
    monkeypatch.setattr(ft, "TrainingArguments", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(ft, "DataCollatorWithPadding", lambda *a, **k: None)
    args.mode, args.output_dir = "evaluate-holdout", tmp_path / "evaluation"
    evaluated = ft.run(args)
    assert "DEVICE holdout evaluation: cpu; backend=CPU; name=CPU" in capsys.readouterr().out
    assert evaluated["mode"] == "evaluate-holdout" and evaluated["training_performed"] is False
    assert tokenized[1] == source.loc[roles == "holdout", "resp_text"].tolist()
    assert final.adapter_hashes(args.artifact_dir / "final_model") == hashes
    predictions = pd.read_csv(args.output_dir / "holdout-predictions.csv")
    assert predictions.excel_row.tolist() == (np.flatnonzero(roles == "holdout") + 2).tolist()
    (args.artifact_dir / "final_model/adapter_config.json").write_text("changed")
    args.output_dir = tmp_path / "bad-evaluation"
    with pytest.raises(ValueError, match="adapter files changed"):
        ft.run(args)


def test_final_row_guard_rejects_reserved_rows(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity.finetune_final import verify_development_rows

    path = tmp_path / "splits.csv"
    pd.DataFrame({"excel_row": [2, 3], "group_id": ["a", "b"],
                  "role": ["development", "holdout"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ from"):
        verify_development_rows(path, np.array([2, 3]), np.array(["a", "b"]))
