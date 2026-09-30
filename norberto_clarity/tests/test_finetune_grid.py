"""The grid runner must rank fixed development-only experiments reproducibly."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from clarity import finetune as ft
from clarity import finetune_grid as grid
from clarity.data import load_training_data, text_groups


def base_config() -> dict:
    return ft.read_finetune_config(grid.DEFAULT_CONFIG)


def test_exact_grid_changes_only_learning_rate_and_lora_size():
    base = base_config()
    candidates = grid.build_candidates(base)
    assert len(candidates) == 9
    assert [(row["learning_rate"], row["lora_rank"], row["lora_alpha"])
            for row in candidates] == [
                (rate, rank, 2 * rank)
                for rank in (4, 8, 16) for rate in (1e-4, 3e-4, 1e-3)]
    for candidate in candidates:
        changed = {key for key in base if candidate["config"][key] != base[key]}
        assert changed <= {"learning_rate", "lora_rank", "lora_alpha"}
        assert candidate["config_sha256"] == grid.config_digest(candidate["config"])
    with pytest.raises(ValueError, match="baseline differs"):
        grid.build_candidates({**base, "lora_dropout": 0.1})
    with pytest.raises(ValueError, match="baseline differs"):
        grid.build_candidates({**base, "holdout_split_sha256": "wrong"})


def test_screening_uses_one_reproducible_grouped_development_split(training_path):
    frame = load_training_data(training_path)
    groups = text_groups(frame.resp_text)
    excel_rows = np.arange(len(frame)) + 2
    train, valid, split = grid.screening_split(frame, groups, excel_rows, base_config())
    other_train, other_valid, other_split = grid.screening_split(
        frame, groups, excel_rows, base_config())
    np.testing.assert_array_equal(train, other_train)
    np.testing.assert_array_equal(valid, other_valid)
    pd.testing.assert_frame_equal(split, other_split)
    assert set(groups[train]).isdisjoint(groups[valid])
    assert set(split.role) == {"train", "validation"}
    assert len(train) + len(valid) == len(frame)


def test_deterministic_screening_ranking():
    rows = [
        {"candidate_id": "candidate-09", "accuracy": 0.50, "f1_macro": 0.40,
         "lora_rank": 16, "learning_rate": 1e-3},
        {"candidate_id": "candidate-01", "accuracy": 0.50, "f1_macro": 0.40,
         "lora_rank": 4, "learning_rate": 1e-4},
        {"candidate_id": "candidate-02", "accuracy": 0.50, "f1_macro": 0.42,
         "lora_rank": 4, "learning_rate": 3e-4},
    ]
    screening = grid.ranked(rows, grid.screening_rank_key, "screening_rank")
    assert [row["candidate_id"] for row in screening] == [
        "candidate-02", "candidate-01", "candidate-09"]


def fake_context(base: dict) -> dict:
    frame = pd.DataFrame({"resp_text": ["example"], "clarity": ["c1"]})
    return {
        "frame": frame, "groups": np.array(["group"]), "excel_rows": np.array([2]),
        "screen_train": np.array([0]), "screen_valid": np.array([], dtype=int),
        "screening_split": pd.DataFrame({"excel_row": [2], "group_id": ["group"],
                                         "role": ["train"]}),
        "candidates": grid.build_candidates(base),
        "holdout": {"reserved_holdout_rows": 3999},
        "isolation": {"holdout_row_overlap": 0, "holdout_group_overlap": 0},
        "dataset_sha256": "test-dataset",
    }


def test_smoke_initializes_one_model_and_directories_without_training(tmp_path, monkeypatch):
    base = base_config()
    context = fake_context(base)
    monkeypatch.setattr(grid, "prepare_run", lambda args, config: context)
    monkeypatch.setattr(ft, "select_execution_device", lambda requested: (
        False, {"backend": "rocm", "device_name": "test GPU"}))
    monkeypatch.setattr(ft.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(ft, "make_model", lambda *args, **kwargs: torch.nn.Linear(2, 2))
    monkeypatch.setattr(ft, "train_one", lambda *args, **kwargs: pytest.fail("Smoke trained a model"))
    monkeypatch.setattr(grid, "environment", lambda: {"packages": {}})
    args = SimpleNamespace(config=grid.DEFAULT_CONFIG, train=Path("data/train.xlsx"),
                           holdout_splits=grid.DEFAULT_SPLIT, output_dir=tmp_path / "results",
                           artifact_dir=tmp_path / "artifacts", smoke=True)
    summary = grid.run_grid(args)
    assert summary["status"] == "smoke_complete"
    assert summary["smoke"]["model_initialized"] is True
    assert summary["smoke"]["training_performed"] is False
    assert len(list((args.output_dir / "configs").glob("candidate-*.json"))) == 9
    assert (args.output_dir / "screening-split.csv").is_file()
    assert (args.output_dir / "screening_results.csv").is_file()
    assert not (args.output_dir / "finalists_results.csv").exists()
    assert not (args.artifact_dir / "finalists").exists()
    with pytest.raises(ValueError, match="must be new"):
        grid.initialize_directories(args.output_dir, tmp_path / "another-artifact")


def test_screening_runs_nine_candidates_then_stops_without_cv(tmp_path, monkeypatch):
    base = base_config()
    context = fake_context(base)
    monkeypatch.setattr(grid, "prepare_run", lambda args, config: context)
    monkeypatch.setattr(ft, "select_execution_device", lambda requested: (
        False, {"backend": "rocm", "device_name": "test GPU"}))
    monkeypatch.setattr(ft.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(ft, "tokenize_corpus", lambda *args: (["tokens"], {"rows": 1}))
    monkeypatch.setattr(ft, "run", lambda *args, **kwargs: pytest.fail("CV was started"))
    monkeypatch.setattr(grid, "environment", lambda: {"packages": {}})
    accuracies = [0.71, 0.75, 0.70, 0.74, 0.78, 0.72, 0.73, 0.77, 0.69]
    calls = []

    def fake_screening(candidate, *args):
        calls.append(candidate["candidate_id"])
        accuracy = accuracies[int(candidate["candidate_id"][-2:]) - 1]
        return {**{key: candidate[key] for key in ("candidate_id", "learning_rate",
                 "lora_rank", "lora_alpha", "lora_dropout", "config_sha256")},
                "accuracy": accuracy, "f1_macro": accuracy - 0.02,
                "runtime_seconds": 12.0, "best_epoch": 2.0,
                "best_checkpoint": "checkpoint-2"}

    monkeypatch.setattr(grid, "run_screening_candidate", fake_screening)
    args = SimpleNamespace(config=grid.DEFAULT_CONFIG, train=Path("data/train.xlsx"),
                           holdout_splits=grid.DEFAULT_SPLIT, output_dir=tmp_path / "results",
                           artifact_dir=tmp_path / "artifacts", smoke=False)
    summary = grid.run_grid(args)
    assert len(calls) == 9
    assert summary["status"] == "complete"
    assert summary["stage"] == "screening_complete"
    assert summary["cv_performed"] is False
    assert summary["top_three"] == ["candidate-05", "candidate-08", "candidate-02"]
    assert summary["top_three_comparison"]["third_vs_fourth_accuracy"] == pytest.approx(0.01)
    baseline = next(row for row in summary["screening_ranking"]
                    if row["candidate_id"] == "candidate-03")
    assert baseline["accuracy_delta_vs_baseline"] == 0
    assert summary["screening_ranking"][0]["accuracy_delta_vs_baseline"] == pytest.approx(0.08)
    assert len(pd.read_csv(args.output_dir / "screening_results.csv")) == 9
    report = (args.output_dir / "screening_report.md").read_text(encoding="utf-8")
    assert "candidate-05 **TOP 3**" in report
    assert "#3 vs #4: 0.0100" in report
    assert not (args.output_dir / "finalists").exists()
