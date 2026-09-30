"""The public entry point must route only to the approved NorBERTo model."""
from pathlib import Path

import pytest


def test_train_uses_frozen_holdout_and_fresh_destinations(monkeypatch, tmp_path):
    import main

    captured = {}
    monkeypatch.setattr(main, "run", lambda args: captured.update(vars(args)) or {"status": "complete"})
    report = main.main([
        "train", "--train", "data/train.xlsx",
        "--output-dir", str(tmp_path / "reports"),
        "--artifact-dir", str(tmp_path / "model"),
    ])
    assert report["status"] == "complete"
    assert captured["mode"] == "final"
    assert captured["train"] == Path("data/train.xlsx")
    assert captured["holdout_splits"] == main.DEFAULT_SPLITS
    assert captured["experiment_dir"] == main.DEFAULT_EXPERIMENT
    assert captured["config"] == main.DEFAULT_CONFIG
    assert captured["device"] == "auto"


def test_test_defaults_to_separate_delivery_and_requires_input(monkeypatch):
    import main

    captured = {}
    monkeypatch.setattr(main, "run", lambda args: captured.update(vars(args)) or {})
    main.main(["test", "--test", "data/new.xlsx"])
    assert captured["mode"] == "predict-test"
    assert captured["output"] == Path("deliveries/new.xlsx")
    assert captured["output_dir"] == Path("results/norberto/delivery/prediction-new")
    assert captured["artifact_dir"] == main.DEFAULT_ARTIFACT
    assert captured["device"] == "auto"

    with pytest.raises(SystemExit):
        main.main(["test"])


def test_device_choices_are_cpu_gpu_or_auto():
    import main

    train = ["train", "--output-dir", "reports", "--artifact-dir", "models"]
    test = ["test", "--test", "data/new.xlsx"]
    for choice in ("auto", "cpu", "gpu"):
        assert main.parser().parse_args([*train, "--device", choice]).device == choice
        assert main.parser().parse_args([*test, "--device", choice]).device == choice
    with pytest.raises(SystemExit):
        main.parser().parse_args([*train, "--device", "cuda"])
    with pytest.raises(SystemExit):
        main.parser().parse_args([*test, "--device", "rocm"])


def test_main_rejects_unapproved_model(monkeypatch):
    import main

    monkeypatch.setattr(main, "read_finetune_config",
                        lambda path: {"model_name": "other", "max_length": 512})
    with pytest.raises(ValueError, match="approved NorBERTo-512"):
        main.main(["test", "--test", "data/test1.xlsx"])


def test_selected_adapter_package_is_reloadable_without_training():
    import main
    from clarity.finetune_predict import load_verified_final

    config, saved, final_path, hashes = load_verified_final(
        main.DEFAULT_CONFIG, main.DEFAULT_ARTIFACT)
    assert config["max_length"] == 512
    assert saved["status"] == "complete"
    assert final_path.is_dir()
    assert hashes["adapter_model.safetensors"] == (
        "d693f6e2995d117ad04a5692817527764fcccee6bfb2dde99120baf9fdd2d385")
