import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from model_research import semantic
from clarity.data import load_training_data
from model_research.experiment import read_config, run_experiment, search_candidates
from model_research.models import build_model


def test_mlp_is_small_seeded_and_does_not_split_duplicate_groups():
    model = build_model("semantic_mlp", seed=17)
    classifier = model["classifier"]
    assert isinstance(classifier, MLPClassifier)
    assert classifier.hidden_layer_sizes == (64,)
    assert classifier.random_state == 17
    assert classifier.max_iter == 200
    assert classifier.early_stopping is False
    assert model["features"].get_params() == build_model("semantic")["features"].get_params()
    assert clone(model).get_params()["classifier__alpha"] == 0.1
    config = read_config(Path("configs/research/semantic-mlp.json"))
    assert config["candidates"][-1]["grid"]["classifier__alpha"] == [0.1, 1.0]


def test_search_records_all_convergence_warnings(monkeypatch):
    monkeypatch.setattr("model_research.experiment.build_model", lambda *args, **kwargs: Pipeline([
        ("classifier", MLPClassifier(hidden_layer_sizes=(2,), max_iter=1, random_state=42)),
    ]))
    y = np.array(["c1", "c234", "c5"] * 6)
    x = np.random.default_rng(42).normal(size=(18, 3))
    config = {"candidates": [{"name": "mlp", "model": "semantic_mlp", "grid": {}}]}
    with threadpool_limits(limits=1), pytest.warns(ConvergenceWarning):
        _, records, _ = search_candidates(x, y, np.arange(18), config,
                                         folds=2, seed=42, cache_dir=None)
    assert sum(warning["count"] for warning in records[0]["convergence_warnings"]) == 3
    assert records[0]["refit_diagnostics"]["n_iter"] == 1


def test_mlp_nested_selection_cache_and_serialization(training_path, tmp_path, monkeypatch):
    encoded = []
    monkeypatch.setattr(semantic, "load_encoder", lambda *args: object())

    def fake_encode(encoder, texts, **kwargs):
        encoded.extend(texts)
        # Features depend only on text; no labels are passed to the encoder.
        return np.array([[float(term in text) for term in ("recusa", "parcial", "atendido")]
                         for text in texts], dtype=np.float32)

    monkeypatch.setattr(semantic, "encode_documents", fake_encode)
    frame = load_training_data(training_path)
    params = {
        "features__cache_path": str(tmp_path / "embeddings.db"),
        "classifier__batch_size": 16,
        "classifier__max_iter": 300,
        "classifier__learning_rate_init": 0.01,
        "classifier__tol": 0.001,
    }
    grid = {key: [value] for key, value in params.items()}
    grid["classifier__alpha"] = [0.1, 1.0]
    config = {"candidates": [{"name": "mlp", "model": "semantic_mlp", "grid": grid}]}
    with threadpool_limits(limits=1):
        report = run_experiment(frame, training_path, config, tmp_path / "run",
                                folds=3, inner_folds=2, cache_dir=str(tmp_path / "pipeline"))
        model = build_model("semantic_mlp").set_params(**params)
        model.fit(frame.resp_text, frame.clarity)
        predictions = model.predict(frame.resp_text)
    assert report["models"]["mlp"]["pooled"]["accuracy"] > 0.9
    assert len(encoded) == 54  # One encoding per unique text across all folds.
    oof = pd.read_csv(tmp_path / "run" / "oof.csv")
    assert oof.groupby("group_id").fold.nunique().eq(1).all()
    for search in report["inner_searches"]:
        candidate = next(c for c in search["candidates"] if c["name"] == "mlp")
        diag = candidate["refit_diagnostics"]
        assert 0 < diag["n_iter"] <= 300
        assert len(diag["training_loss_curve"]) == diag["n_iter"]
        assert np.isfinite(diag["training_loss"])
    path = tmp_path / "mlp.joblib"
    joblib.dump(model, path)

    def no_encoder(*args):
        raise AssertionError("Persisted predictions must reuse the frozen cache.")

    monkeypatch.setattr(semantic, "load_encoder", no_encoder)
    restored = joblib.load(path)
    before = [coef.copy() for coef in restored["classifier"].coefs_]
    np.testing.assert_array_equal(restored.predict(frame.resp_text), predictions)
    for previous, current in zip(before, restored["classifier"].coefs_):
        np.testing.assert_array_equal(previous, current)
    assert json.loads((tmp_path / "run" / "results.json").read_text())["primary_metric"] == "mean_outer_accuracy"
