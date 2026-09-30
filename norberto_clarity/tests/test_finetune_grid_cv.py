"""The CV runner must reuse completed screening inputs and the existing experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")

from clarity import finetune as ft
from clarity import finetune_grid_cv as cv
from clarity.reporting import write_json


def test_preflight_rejects_incomplete_screening_before_creating_outputs(tmp_path):
    source = tmp_path / "screening"
    source.mkdir()
    write_json(source / "summary.json", {"mode": "grid-screening", "status": "running"})
    args = SimpleNamespace(screening_dir=source, output_dir=tmp_path / "results",
                           artifact_dir=tmp_path / "artifacts")
    with pytest.raises(ValueError, match="completed screening"):
        cv.preflight(args)
    assert not args.output_dir.exists()
    assert not args.artifact_dir.exists()


def test_candidate_uses_existing_three_fold_experiment(tmp_path, monkeypatch):
    screening = tmp_path / "screening"
    (screening / "configs").mkdir(parents=True)
    config = ft.read_finetune_config(Path("configs/norberto-lora-development-ctx512.json"))
    write_json(screening / "configs" / "candidate-05.json", config)
    scores = [{"fold": fold, "accuracy": 0.40 + fold / 100,
               "f1_macro": 0.39 + fold / 100} for fold in (1, 2, 3)]
    report = {"mode": "experiment", "status": "complete", "dataset_sha256": "dataset",
              "split_sha256": cv.grid.FROZEN_SPLIT_SHA256, "reserved_holdout_rows": 3999,
              "rows": 16093, "evaluated_rows": 16093, "requested_folds": [1, 2, 3],
              "config": config, "models": {"norberto_lora": {
                  "mean_accuracy": 0.42, "std_accuracy": 0.01,
                  "mean_f1_macro": 0.41, "std_f1_macro": 0.01, "fold_results": scores}}}

    def fake_experiment(args):
        assert args.mode == "experiment"
        assert args.device == "gpu"
        assert args.holdout_splits == Path("frozen-splits.csv")
        args.output_dir.mkdir(parents=True)
        for fold in (1, 2, 3):
            write_json(args.output_dir / f"fold-{fold}.json", {"training": {
                "training_stopped_by_budget": False, "best_checkpoint": f"checkpoint-{fold}",
                "training_seconds": 10.0, "trainable_parameters": 542979}})
        return report

    monkeypatch.setattr(ft, "run", fake_experiment)
    monkeypatch.setattr(cv, "recover_best_epoch", lambda training, epochs: (2.0, "test"))
    args = SimpleNamespace(train=Path("data/train.xlsx"), screening_dir=screening,
                           output_dir=tmp_path / "results", artifact_dir=tmp_path / "artifacts",
                           holdout_splits=Path("frozen-splits.csv"))
    candidate = {"candidate_id": "candidate-05", "config": config,
                 "config_sha256": "digest", "learning_rate": 3e-4,
                 "lora_rank": 8, "lora_alpha": 16}
    result = cv.evaluate_candidate(args, candidate,
                                   {"screening_rank": 1, "accuracy": 0.445}, "dataset")
    assert result["mean_accuracy"] == 0.42
    assert [fold["best_epoch"] for fold in result["fold_results"]] == [2.0] * 3
    assert result["screening_accuracy"] == 0.445


def test_cv_runs_only_screening_top_three_in_order(tmp_path, monkeypatch):
    top_three = ["candidate-05", "candidate-08", "candidate-02"]
    source = {"top_three": top_three, "git_commit": "source-commit",
              "screening_split_sha256": "screen-split", "dataset_sha256": "dataset",
              "seed": 42}
    ranking = [{"candidate_id": candidate_id, "screening_rank": index,
                "accuracy": 0.45 - index / 100}
               for index, candidate_id in enumerate(top_three, 1)]
    prepared = {"source": source, "source_sha256": "source-hash", "ranking": ranking,
                "context": {"isolation": {"holdout_row_overlap": 0,
                                          "holdout_group_overlap": 0}},
                "candidates": {candidate_id: {"candidate_id": candidate_id}
                               for candidate_id in top_three}}
    monkeypatch.setattr(cv, "preflight", lambda args: prepared)
    monkeypatch.setattr(cv, "environment", lambda: {"packages": {}})
    monkeypatch.setattr(ft, "select_execution_device", lambda requested: (
        False, {"backend": "rocm", "device_name": "test GPU"}))
    calls = []

    def fake_candidate(args, candidate, screening_row, dataset_sha256):
        calls.append(candidate["candidate_id"])
        return {"candidate_id": candidate["candidate_id"], "screening_rank":
                screening_row["screening_rank"], "mean_accuracy": screening_row["accuracy"],
                "mean_f1_macro": screening_row["accuracy"], "lora_rank": 4,
                "learning_rate": 3e-4, "fold_results": [{"fold": fold,
                "accuracy": 0.4, "f1_macro": 0.4, "best_epoch": 2}
                for fold in (1, 2, 3)]}

    monkeypatch.setattr(cv, "evaluate_candidate", fake_candidate)
    args = SimpleNamespace(preflight=False, screening_dir=tmp_path / "screening",
                           output_dir=tmp_path / "results", artifact_dir=tmp_path / "artifacts")
    summary = cv.run(args)
    assert calls == top_three
    assert summary["status"] == "complete"
    assert summary["holdout_evaluated"] is False
    assert summary["final_model_trained"] is False
    assert [row["candidate_id"] for row in summary["cv_ranking"]] == top_three
    assert len((args.output_dir / "cv_results.csv").read_text().splitlines()) == 4
    saved = json.loads((args.output_dir / "summary.json").read_text())
    assert saved["stage"] == "cv_complete"
