import json

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from clarity.data import file_digest, load_training_data, text_groups
from model_research.audit_holdout import audit_fixed_holdout
from clarity.holdout import load_fixed_holdout
from model_research.holdout_train import run


def fixed_split(training_path, tmp_path):
    frame = load_training_data(training_path)
    split_path = tmp_path / "splits.csv"
    roles = np.where(np.arange(len(frame)) // 6 < 14, "development", "holdout")
    pd.DataFrame({
        "excel_row": np.arange(len(frame)) + 2,
        "group_id": text_groups(frame.resp_text),
        "role": roles,
        "development_fold": np.ones(len(frame), dtype=int),
    }).to_csv(split_path, index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dataset_sha256": file_digest(training_path),
        "development_rows": int((roles == "development").sum()),
        "holdout_rows": int((roles == "holdout").sum()),
    }))
    return split_path


def test_fixed_holdout_train_excludes_validation_from_fit(training_path, tmp_path):
    split_path = fixed_split(training_path, tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "frozen_split_sha256": file_digest(split_path),
        "candidates": [
            {"name": "baseline", "model": "baseline", "grid": {}},
            {"name": "challenger", "model": "word_svc",
             "grid": {"classifier__C": [0.05]}},
        ],
    }))
    output_path = tmp_path / "model.joblib"
    with threadpool_limits(limits=1):
        result = run(training_path, split_path, config_path, output_path,
                     folds=2, cache_dir=str(tmp_path / "cache"))
    assert result["fixed_validation"]["development_rows"] == 84
    assert result["fixed_validation"]["holdout_rows"] == 24
    assert result["fixed_validation"]["accuracy"] == 1.0
    assert "item17" not in joblib.load(output_path)["model"]["tfidf"].vocabulary_
    assert len(pd.read_csv(output_path.with_suffix(".holdout.csv"))) == 24
    assert "pred_baseline" in pd.read_csv(output_path.with_suffix(".holdout.csv")).columns
    assert "pred_challenger" in pd.read_csv(output_path.with_suffix(".holdout.csv")).columns
    assert "challenger" in result["fixed_validation"]["comparisons_with_reference"]
    assert output_path.with_suffix(".json").exists()
    audit = audit_fixed_holdout(
        output_path.with_suffix(".json"), output_path.with_suffix(".holdout.csv"),
        output_path, training_path, split_path, config_path)
    assert audit["status"] == "passed"
    assert audit["serialized_predictions_reproduced"] is True
    assert (audit["comparisons_with_reference"]["challenger"]["resampling_unit"]
            == "frozen_similarity_component")
    corrupted = pd.read_csv(output_path.with_suffix(".holdout.csv"))
    corrupted.loc[0, "true_label"] = "c5" if corrupted.loc[0, "true_label"] != "c5" else "c1"
    corrupted.to_csv(output_path.with_suffix(".holdout.csv"), index=False)
    with pytest.raises(ValueError, match="Saved labels"):
        audit_fixed_holdout(
            output_path.with_suffix(".json"), output_path.with_suffix(".holdout.csv"),
            output_path, training_path, split_path, config_path)
    with pytest.raises(ValueError, match="já existe"):
        run(training_path, split_path, config_path, output_path, folds=2)


def test_fixed_holdout_rejects_changed_source_and_group_leak(training_path, tmp_path):
    split_path = fixed_split(training_path, tmp_path)
    labels = load_training_data(training_path).clarity.to_numpy()
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["dataset_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="SHA-256"):
        load_fixed_holdout(training_path, split_path, labels,
                           expected_split_sha256=file_digest(split_path))
    manifest["dataset_sha256"] = file_digest(training_path)
    manifest_path.write_text(json.dumps(manifest))
    split = pd.read_csv(split_path)
    split.loc[0, "group_id"] = split.loc[len(split) - 1, "group_id"]
    split.to_csv(split_path, index=False)
    with pytest.raises(ValueError, match="compartilhados"):
        load_fixed_holdout(training_path, split_path, labels,
                           expected_split_sha256=file_digest(split_path))


def test_fixed_holdout_rejects_changed_split_hash(training_path, tmp_path):
    split_path = fixed_split(training_path, tmp_path)
    labels = load_training_data(training_path).clarity.to_numpy()
    with pytest.raises(ValueError, match="SHA-256 congelado"):
        load_fixed_holdout(training_path, split_path, labels,
                           expected_split_sha256="0" * 64)
