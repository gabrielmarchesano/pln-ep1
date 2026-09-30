import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
from peft import PeftModel
from transformers import BertConfig, BertForSequenceClassification
from clarity.data import load_training_data, text_groups
from clarity.finetune import (BudgetProgress, TokenizedRows, clean_cased,
    make_model, make_split, model_spec, probability_columns, read_finetune_config,
    tokenize_corpus, train_one, trim_tokens)


def config():
    return read_finetune_config(Path("configs/research/bertimbau-lora-pilot.json"))


def test_cased_normalization_and_token_selection():
    assert clean_cased("  NÃO &amp; Sim\n") == "NÃO & Sim"
    assert trim_tokens(list(range(10)), 5, "head_tail") == [0, 1, 2, 8, 9]
    assert trim_tokens(list(range(10)), 5, "head") == list(range(5))
    assert trim_tokens(list(range(10)), 1, "head_tail") == [0]
    assert trim_tokens([], 5, "head_tail") == []
    with pytest.raises(ValueError):
        trim_tokens([1], 0, "head")


def test_grouped_inner_holdout_and_subsampling(training_path):
    frame = load_training_data(training_path)
    x, y = frame.resp_text.to_numpy(), frame.clarity.to_numpy()
    groups = text_groups(x)
    c = {**config(), "inner_folds": 2, "train_subset_folds": 2}
    for fold in (1, 2, 3):
        train, valid, outer_train, outer_valid = make_split(x, y, groups, c, fold)
        assert set(train) | set(valid) <= set(outer_train)
        assert not set(outer_train) & set(outer_valid)
        for left, right in ((train, valid), (train, outer_valid), (valid, outer_valid)):
            assert not set(groups[left]) & set(groups[right])
        for indices in (train, valid, outer_valid):
            assert set(y[indices]) == {"c1", "c234", "c5"}
        # Sampling does not split duplicate groups or depend on outer labels.
        assert all(set(np.flatnonzero(groups == g)) <= set(train) for g in groups[train])
        changed = y.copy()
        # Confirm deterministic repeat of the declared split.
        np.testing.assert_array_equal(make_split(x, changed, groups, c, fold)[0], train)


def test_token_cache_respects_case_and_policy(tmp_path):
    class Tokenizer:
        calls = 0
        model_input_names = ["input_ids", "attention_mask"]

        def __call__(self, texts, **kwargs):
            self.calls += 1
            return {"input_ids": [[ord(char) for char in text] for text in texts]}

        def num_special_tokens_to_add(self, **kwargs):
            return 2

        def build_inputs_with_special_tokens(self, ids):
            return [101, *ids, 102]

    tokenizer = Tokenizer()
    c = {**config(), "max_length": 8}
    features, stats = tokenize_corpus(["ABCDEFGH", "x"], tokenizer, c, tmp_path)
    assert features[0]["input_ids"] == [101, 65, 66, 67, 70, 71, 72, 102]
    assert "token_type_ids" not in features[0]
    assert stats["truncated_rows"] == 1
    assert tokenize_corpus(["ABCDEFGH", "x"], tokenizer, c, tmp_path) == (features, stats)
    assert tokenizer.calls == 1
    tokenize_corpus(["abcdefgh", "x"], tokenizer, c, tmp_path)
    tokenize_corpus(["ABCDEFGH", "x"], tokenizer, {**c, "truncation": "head"}, tmp_path)
    assert tokenizer.calls == 3
    rows = TokenizedRows(features, np.array([1, 2]), [1, 0])
    assert rows[0]["labels"] == 2 and len(rows) == 2
    assert "labels" not in features[1]


def test_lora_updates_only_adapters_and_head_and_reloads(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    base = BertForSequenceClassification(BertConfig(vocab_size=32, hidden_size=16,
        num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, num_labels=3))
    initial = copy.deepcopy(base)
    monkeypatch.setattr("clarity.finetune.AutoModelForSequenceClassification.from_pretrained",
                        lambda *args, **kwargs: base)
    model = make_model(config(), 42)
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert all("lora_" in n or "modules_to_save" in n for n, p in model.named_parameters() if p.requires_grad)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    batch = {"input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]]),
             "attention_mask": torch.ones(2, 3, dtype=torch.long), "labels": torch.tensor([0, 2])}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model.train()
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)
    assert all(torch.equal(frozen[n], p) for n, p in model.named_parameters() if n in frozen)
    model.eval()
    with torch.no_grad():
        expected = model(**batch).logits
    model.save_pretrained(tmp_path / "adapter")
    restored = PeftModel.from_pretrained(initial, tmp_path / "adapter").eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch).logits, expected)


def test_invalid_config_and_nonfinite_logs(tmp_path):
    c = {**config(), "revision": "main"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="pinned"):
        read_finetune_config(path)
    with pytest.raises(FloatingPointError):
        BudgetProgress(1).on_log(None, None, None, logs={"loss": float("nan")})


def test_norberto_model_contract_and_layer_validation(tmp_path):
    norberto = {
        **config(),
        "model_name": "Itau-Unibanco/NorBERTo-base",
        "revision": "db73446f89c96044863ea05a39f680524b84bccb",
        "lora_layers": [0, 21],
    }
    path = tmp_path / "norberto.json"
    path.write_text(json.dumps(norberto))
    loaded = read_finetune_config(path)
    assert model_spec(loaded)["architecture"] == "modernbert"
    assert model_spec(loaded)["lora_target_modules"] == ["Wqkv"]
    assert model_spec(loaded)["pretrained_options"] == {"reference_compile": False}
    path.write_text(json.dumps({**norberto, "lora_layers": [22]}))
    with pytest.raises(ValueError, match="0..21"):
        read_finetune_config(path)


@pytest.mark.parametrize("field,value", [("learning_rate", float("nan")),
    ("budget_minutes", float("inf")), ("weight_decay", -1), ("max_length", 128.5),
    ("fp16", "false"), ("lora_alpha", 0), ("outer_folds", 3.5)])
def test_config_rejects_invalid_numeric_values(tmp_path, field, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({**config(), field: value}))
    with pytest.raises(ValueError):
        read_finetune_config(path)


def test_probability_columns_check_values_and_reorder_classes():
    values = np.array([[0.1, 0.7, 0.2]])
    columns = probability_columns("model", values, ["c5", "c1", "c234"])
    assert columns["prob_model_c1"] == [0.7]
    assert columns["prob_model_c234"] == [0.2]
    for bad in [np.array([[0.1, 0.2, 0.3]]), np.array([[np.nan, 0.5, 0.5]]),
                np.array([[-0.1, 0.6, 0.5]]), np.array([[0.5, 0.5]])]:
        with pytest.raises(ValueError):
            probability_columns("model", bad)


def test_training_loop_selects_and_saves_checkpoint_on_tiny_cpu_model(tmp_path, monkeypatch):
    """Exercise the real Trainer/PEFT loop without a download or GPU allocation."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertTokenizerFast, TrainingArguments

    vocabulary = tmp_path / "vocab.txt"
    vocabulary.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n")
    tokenizer = BertTokenizerFast(vocab_file=str(vocabulary), do_lower_case=False)
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = get_peft_model(BertForSequenceClassification(BertConfig(
        vocab_size=8, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, num_labels=3)), LoraConfig(
        task_type=TaskType.SEQ_CLS, r=2, lora_alpha=4, target_modules=["query", "value"]))
    monkeypatch.setattr("clarity.finetune.make_model", lambda *args, **kwargs: model)
    monkeypatch.setattr("clarity.finetune.TrainingArguments",
        lambda **kwargs: TrainingArguments(**{**kwargs, "use_cpu": True, "dataloader_pin_memory": False}))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    features, _ = tokenize_corpus(["a b", "b c", "c a"] * 4, tokenizer, config(), tmp_path / "tokens")
    labels = np.array([0, 1, 2] * 4)
    trainer, report = train_one({**config(), "epochs": 1, "batch_size": 2,
                                "gradient_accumulation": 2}, tokenizer, features, labels,
                               np.arange(9), np.arange(9, 12), tmp_path / "model", 43)
    assert trainer.state.global_step == 3  # Includes the final partial accumulation.
    assert report["epochs_executed"] == 1
    assert report["best_checkpoint"] is not None
    assert np.isfinite(report["best_inner_accuracy"])
    assert not report["training_stopped_by_budget"]
    assert (tmp_path / "model/best_adapter/adapter_model.safetensors").exists()
    logits = trainer.predict(TokenizedRows(features, labels, np.arange(9, 12))).predictions
    assert logits.shape == (3, 3) and np.isfinite(logits).all()
