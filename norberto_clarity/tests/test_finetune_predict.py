import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from clarity.data import LABELS, file_digest, read_frame


def final_fixture(tmp_path, config):
    from clarity.finetune_final import adapter_hashes

    artifact_dir = tmp_path / "artifact"
    final_path = artifact_dir / "final_model"
    final_path.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"):
        (final_path / name).write_text("fixture", encoding="utf-8")
    saved = {
        "mode": "final", "status": "complete", "approved_config": config,
        "config": {**config, "epochs": 2}, "labels": list(LABELS),
        "dataset_sha256": "training-dataset", "adapter_sha256": adapter_hashes(final_path),
    }
    (artifact_dir / "run.json").write_text(json.dumps(saved), encoding="utf-8")
    return artifact_dir


def test_predict_test_exports_only_labels_without_training(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune as ft

    config = ft.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_dir = final_fixture(tmp_path, config)
    test_path = tmp_path / "test.xlsx"
    pd.DataFrame({"id": [1, 2, 3], "resp_text": ["a", "b", "c"],
                  "clarity": [None, None, None]}).to_excel(test_path, index=False)
    source_hash = file_digest(test_path)

    monkeypatch.setattr(ft, "probe_accelerator", lambda: {"device_name": "test"})
    monkeypatch.setattr(ft.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(ft, "tokenize_corpus", lambda texts, *a: (
        [{"input_ids": [index + 1]} for index in range(len(texts))], {"rows": len(texts)}))
    monkeypatch.setattr(ft, "make_model", lambda *a, **k: object())
    monkeypatch.setattr(ft, "TrainingArguments", lambda **k: SimpleNamespace(**k))
    monkeypatch.setattr(ft, "DataCollatorWithPadding", lambda *a, **k: None)

    class Predictor:
        def __init__(self, **kwargs):
            assert "train_dataset" not in kwargs
            assert kwargs["args"].use_cpu is True

        def train(self):
            pytest.fail("Test prediction attempted training")

        def predict(self, dataset):
            assert len(dataset) == 3 and "labels" not in dataset[0]
            return SimpleNamespace(predictions=np.array([
                [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=np.float32))

    monkeypatch.setattr(ft, "Trainer", Predictor)
    args = SimpleNamespace(mode="predict-test", config=config_path, artifact_dir=artifact_dir,
                           test=test_path, output=tmp_path / "delivery.xlsx",
                           output_dir=tmp_path / "report", device="cpu")
    report = ft.run(args)
    assert report["mode"] == "predict-test" and report["status"] == "complete"
    assert not report["training_performed"] and not report["evaluation_performed"]
    assert "DEVICE test prediction: cpu; backend=CPU; name=CPU; requested=cpu" in capsys.readouterr().out
    assert report["accelerator"]["backend"] == "cpu" and report["requested_device"] == "cpu"
    assert report["prediction_counts"] == {"c1": 1, "c234": 1, "c5": 1}
    assert file_digest(test_path) == source_hash
    assert read_frame(args.output, training=False).clarity.tolist() == list(LABELS)
    assert pd.read_csv(args.output_dir / "predictions.csv").excel_row.tolist() == [2, 3, 4]
    assert report["output_sha256"] == file_digest(args.output)

    with pytest.raises(ValueError, match="existe"):
        ft.run(args)
    (artifact_dir / "final_model/adapter_config.json").write_text("changed", encoding="utf-8")
    args.output, args.output_dir = tmp_path / "second.xlsx", tmp_path / "second-report"
    with pytest.raises(ValueError, match="adapter files changed"):
        ft.run(args)
    assert not args.output.exists()


def test_predict_test_refuses_existing_labels(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity.finetune_predict import predict_test
    from clarity import finetune as ft

    config = ft.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    artifact_dir = final_fixture(tmp_path, config)
    test_path = tmp_path / "labeled.xlsx"
    pd.DataFrame({"resp_text": ["a"], "clarity": ["c1"]}).to_excel(test_path, index=False)
    args = SimpleNamespace(config=config_path, artifact_dir=artifact_dir, test=test_path,
                           output=tmp_path / "delivery.xlsx", output_dir=tmp_path / "report")
    with pytest.raises(ValueError, match="labels must be empty"):
        predict_test(args)
    assert not args.output.exists() and not args.output_dir.exists()


def test_inference_device_selection_supports_cpu_and_accelerators(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune as ft
    from clarity.finetune_predict import select_inference_device

    monkeypatch.setattr(ft.torch.cuda, "is_available", lambda: False)
    use_cpu, info = select_inference_device("auto")
    assert use_cpu and info["backend"] == "cpu"
    with pytest.raises(RuntimeError, match="GPU was requested"):
        select_inference_device("gpu")
    with pytest.raises(ValueError, match="auto, cpu, or gpu"):
        select_inference_device("cuda")
    monkeypatch.setattr(ft.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ft, "probe_accelerator", lambda: {"backend": "rocm"})
    assert select_inference_device("auto") == (False, {"backend": "rocm"})
    assert select_inference_device("gpu") == (False, {"backend": "rocm"})
    monkeypatch.setattr(ft, "probe_accelerator", lambda: {"backend": "cuda"})
    assert select_inference_device("gpu") == (False, {"backend": "cuda"})
    assert select_inference_device("cpu")[0] is True
    monkeypatch.setattr(ft.torch.cuda, "is_available", lambda: pytest.fail("CPU queried CUDA"))
    assert select_inference_device("cpu")[0] is True


def test_prediction_can_download_pinned_base_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    from clarity import finetune as ft

    config = ft.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    captured = {}
    base, adapter = object(), object()

    def load_base(name, **kwargs):
        captured["name"], captured["options"] = name, kwargs
        return base

    def load_adapter(model, path, **kwargs):
        captured["base"], captured["path"], captured["adapter_options"] = model, path, kwargs
        return adapter

    monkeypatch.setattr(ft.AutoModelForSequenceClassification, "from_pretrained", load_base)
    monkeypatch.setattr(ft.PeftModel, "from_pretrained", load_adapter)
    assert ft.make_model(config, 42, adapter_path=tmp_path, allow_download=True) is adapter
    assert captured["name"] == config["model_name"]
    assert captured["options"]["revision"] == config["revision"]
    assert captured["options"]["local_files_only"] is False
    assert captured["base"] is base and captured["path"] == tmp_path
    assert captured["adapter_options"]["is_trainable"] is False
