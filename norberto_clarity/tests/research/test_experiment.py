import json

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from model_research.cli import execute, parser
from clarity.data import load_training_data
from model_research.experiment import read_config, run_experiment
from model_research.models import build_model
from sklearn.svm import LinearSVC


def test_vocabulary_is_fitted_only_on_training_rows():
    model = build_model("baseline")
    texts = ["negativa negada", "negativa recusa", "parcial consulta", "parcial recurso", "completa aceita", "completa sucesso"]
    model.fit(texts, ["c1", "c1", "c234", "c234", "c5", "c5"])
    model.predict(["somentevalidacao negativa"])
    assert "somentevalidacao" not in model["tfidf"].vocabulary_


def test_nested_selection_and_oof_artifacts(training_path, tmp_path):
    frame = load_training_data(training_path)
    config = {"candidates": [
        {"name": "baseline", "model": "baseline", "grid": {}},
        {"name": "lexical", "model": "lexical", "grid": {"classifier__C": [0.5, 2.0]}},
        {"name": "word_svc", "model": "word_svc", "grid": {"classifier__C": [0.05, 0.5]}},
        {"name": "word_char_svc", "model": "linear_svc", "grid": {"classifier__C": [0.05, 0.5]}},
    ]}
    with threadpool_limits(limits=1):
        report = run_experiment(frame, training_path, config, tmp_path / "run",
                                folds=3, inner_folds=2, seed=42)
    oof = pd.read_csv(tmp_path / "run" / "oof.csv")
    assert len(oof) == len(frame)
    assert oof.groupby("group_id").fold.nunique().eq(1).all()
    assert sorted(oof.excel_row) == list(range(2, len(frame) + 2))
    for choice, search in zip(report["selected_per_fold"], report["inner_searches"]):
        expected = max(search["candidates"], key=lambda row: row["best_inner_accuracy"])["name"]
        assert choice["name"] == expected
        selected = oof.fold.eq(choice["fold"])
        assert oof.loc[selected, "pred_selected"].equals(oof.loc[selected, f"pred_{expected}"])
        for candidate in search["candidates"]:
            assert candidate["refit_seconds"] >= 0
            assert "n_iter" in candidate["refit_diagnostics"]
            for trial in candidate["trials"]:
                for field in ("mean_fit_seconds", "std_fit_seconds",
                              "mean_score_seconds", "std_score_seconds"):
                    assert np.isfinite(trial[field]) and trial[field] >= 0
    assert json.loads((tmp_path / "run" / "results.json").read_text())["protocol"] == "nested_stratified_group_kfold"
    assert report["models"]["baseline"]["pooled"]["accuracy"] > 0.9
    for name in ("word_svc", "word_char_svc"):
        assert report["models"][name]["pooled"]["accuracy"] > 0.9
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["config"] == config
    assert manifest["dataset_sha256"] == report["dataset_sha256"]
    assert manifest["seed"] == report["seed"]


@pytest.mark.parametrize("model_name, reference_name, feature_step", [
    ("word_svc", "baseline", "tfidf"),
    ("linear_svc", "lexical", "features"),
])
def test_svc_shares_reference_features_and_survives_serialization(
    training_path, tmp_path, model_name, reference_name, feature_step
):
    data = load_training_data(training_path)
    model = build_model(model_name)
    reference = build_model(reference_name)
    assert isinstance(model["classifier"], LinearSVC)
    with threadpool_limits(limits=1):
        actual = model[feature_step].fit_transform(data.resp_text)
        expected = reference[feature_step].fit_transform(data.resp_text)
        assert actual.shape == expected.shape
        assert (actual != expected).nnz == 0
        model.fit(data.resp_text, data.clarity)
        predictions = model.predict(data.resp_text)
    path = tmp_path / "svm.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict(data.resp_text), predictions)
    assert model["classifier"].n_iter_ < model["classifier"].max_iter


def test_train_reload_predict_roundtrip(training_path, tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"candidates": [{"name": "baseline", "model": "baseline", "grid": {}}]}))
    artifact = tmp_path / "model.joblib"
    with threadpool_limits(limits=1):
        execute(parser().parse_args(["train", "--train", str(training_path), "--config", str(config_path),
                                     "--output", str(artifact), "--folds", "2", "--cache-dir", str(tmp_path / "cache")]))
    assert artifact.exists() and artifact.with_suffix(".json").exists()
    test_path, output_path = tmp_path / "test.xlsx", tmp_path / "labeled.xlsx"
    pd.DataFrame({"resp_text": ["recusa negativa sigilo", "atendido resposta completa"]}).to_excel(test_path, index=False)
    execute(parser().parse_args(["predict", "--artifact", str(artifact), "--test", str(test_path), "--output", str(output_path)]))
    assert pd.read_excel(output_path).clarity.tolist() == ["c1", "c5"]
    bundle = joblib.load(artifact)
    assert bundle["model"].memory is None
    assert bundle["metadata"]["selected"] == "baseline"
    with pytest.raises(ValueError, match="já existe"):
        execute(parser().parse_args(["predict", "--artifact", str(artifact), "--test", str(test_path), "--output", str(output_path)]))


def test_config_rejects_invalid_parameters(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"candidates": [{"name": "x", "model": "baseline", "grid": {"nonexistent": [1]}}]}))
    with pytest.raises(ValueError, match="desconhecidos"):
        read_config(path)
