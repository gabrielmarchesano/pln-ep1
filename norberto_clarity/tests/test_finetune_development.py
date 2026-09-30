import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from clarity.data import file_digest, load_training_data, text_groups
from clarity.finetune_data import prepare_training_corpus


def make_reserved_split(training_path, tmp_path):
    frame = load_training_data(training_path)
    # Reserve interleaved groups so an incorrect row-offset mapping cannot pass.
    roles = np.where(np.arange(len(frame)) // 6 % 4 == 0, "holdout", "development")
    path = tmp_path / "splits.csv"
    pd.DataFrame({"excel_row": np.arange(len(frame)) + 2,
                  "group_id": text_groups(frame.resp_text), "role": roles,
                  "development_fold": 0}).to_csv(path, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_sha256": file_digest(training_path),
        "development_rows": int((roles == "development").sum()),
        "holdout_rows": int((roles == "holdout").sum()),
    }))
    return path, np.flatnonzero(roles == "development") + 2


def test_reserved_rows_are_removed_before_tokenization(training_path, tmp_path):
    path, expected_rows = make_reserved_split(training_path, tmp_path)
    frame, groups, rows, metadata = prepare_training_corpus(
        training_path, path, file_digest(path))
    source = load_training_data(training_path)
    np.testing.assert_array_equal(rows, expected_rows)
    assert frame.resp_text.tolist() == source.iloc[expected_rows - 2].resp_text.tolist()
    assert len(groups) == len(rows) == metadata["development_rows"]
    assert metadata["holdout_evaluated"] is False
    with pytest.raises(ValueError, match="requires --holdout-splits"):
        prepare_training_corpus(training_path, None, file_digest(path))


def test_research_modes_keep_gpu_requirement():
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune

    for mode in ("pilot", "experiment"):
        with pytest.raises(ValueError, match="require GPU"):
            finetune.run(SimpleNamespace(
                mode=mode, config=Path("configs/norberto-lora-development-ctx512.json"),
                device="cpu"))


def test_finetune_runner_keeps_original_rows_and_excludes_holdout(training_path, tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune

    path, expected_rows = make_reserved_split(training_path, tmp_path)
    config = finetune.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    config.update(inner_folds=2, holdout_split_sha256=file_digest(path))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    source = load_training_data(training_path)
    captured = []

    def tokenize(texts, tokenizer, config):
        assert list(texts) == source.iloc[expected_rows - 2].resp_text.tolist()
        return [{"input_ids": [1]} for _ in texts], {"rows": len(texts)}

    def train(config, tokenizer, features, labels, train, valid, artifact_dir, seed):
        assert not set(train) & set(valid)
        captured.append((train, valid))

        class Trainer:
            def predict(self, dataset):
                return SimpleNamespace(predictions=np.eye(3)[labels[dataset.indices]])

        return Trainer(), {}

    monkeypatch.setattr(finetune, "probe_accelerator",
                        lambda: {"backend": "rocm", "device_name": "test"})
    monkeypatch.setattr(finetune, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: None))
    monkeypatch.setattr(finetune, "tokenize_corpus", tokenize)
    monkeypatch.setattr(finetune, "train_one", train)
    monkeypatch.setattr(finetune.torch.cuda, "empty_cache", lambda: None)
    output = tmp_path / "reports"
    result = finetune.run(SimpleNamespace(
        train=training_path, config=config_path, mode="experiment", holdout_splits=path,
        output_dir=output, artifact_dir=tmp_path / "artifacts"))
    assert len(captured) == 3
    assert "DEVICE experiment: cuda; backend=ROCM; name=test" in capsys.readouterr().out
    assert result["holdout_evaluated"] is False
    assert result["evaluated_rows"] == len(expected_rows)
    np.testing.assert_array_equal(pd.read_csv(output / "oof.csv").excel_row, expected_rows)
    for fold in (1, 2, 3):
        split = json.loads((output / f"split-{fold}.json").read_text())
        for field in ("train_excel_rows", "inner_valid_excel_rows", "outer_valid_excel_rows"):
            assert set(split[field]) <= set(expected_rows)
