import json

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from clarity.data import file_digest
from model_research.dataset.dataset_hypotheses import (
    OrdinalClassifier,
    StyleFeatures,
    majority_relabel,
    normalize_identifiers,
    run_experiment,
    similarity_groups,
)


def test_extended_identifier_normalization_preserves_context():
    text = ("Não omita R$ 1.234,50; CPF 123.456.789-00; CNPJ "
            "12.345.678/0001-90; protocolo 2026/123456; data 23/09/2026; "
            "contato pessoa@example.com em https://example.org/item/100.")
    normalized = normalize_identifiers(text)
    for token in ("valuetoken", "cpftoken", "cnpjtoken", "protocoltoken",
                  "datetoken", "emailtoken", "urltoken"):
        assert token in normalized
    assert "não omita" in normalized
    assert "123.456.789" not in normalized


def test_style_features_are_finite_and_text_only():
    features = StyleFeatures().transform(["", "Uma frase? Outra frase!\n\nNovo parágrafo."])
    assert features.shape == (2, 13)
    assert np.isfinite(features).all()
    assert features[1, 0] > features[0, 0]


def test_similarity_groups_link_near_copies_and_masked_templates():
    texts = np.asarray([
        "A resposta solicitada está disponível no portal oficial para consulta completa.",
        "A resposta solicitada está disponível no portal oficial para consulta completá.",
        "Protocolo 2026/123456 contém o documento solicitado.",
        "Protocolo 2026/999999 contém o documento solicitado.",
        "A solicitação foi negada por sigilo legal.",
    ])
    groups, info = similarity_groups(texts, threshold=0.80)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[4]
    assert info["final_groups"] < len(texts)


def test_majority_relabels_only_training_and_drops_ties():
    labels = np.asarray(["c1", "c1", "c5", "c1", "c5", "c234", "c5"])
    keys = np.asarray(["majority"] * 3 + ["tie"] * 2 + ["middle", "high"])
    rows, targets, stats = majority_relabel(np.arange(7), labels, keys)
    assert rows.tolist() == [0, 1, 2, 5, 6]
    assert targets.tolist() == ["c1", "c1", "c1", "c234", "c5"]
    assert stats == {"tie_rows_removed": 2, "labels_reassigned": 1}
    training_only = np.asarray([0, 1, 3, 4, 5, 6])
    rows, _, _ = majority_relabel(training_only, labels, keys)
    assert 2 not in rows


def test_ordinal_probabilities_are_valid():
    features = np.asarray([[0], [0.1], [1], [1.1], [2], [2.1]])
    labels = np.asarray(["c1", "c1", "c234", "c234", "c5", "c5"])
    model = OrdinalClassifier().fit(features, labels)
    probabilities = model.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert (probabilities >= 0).all()
    assert set(model.predict(features)) <= set(labels)


def test_experiment_freezes_holdout_and_never_overwrites(training_path, tmp_path):
    output = tmp_path / "hypotheses"
    digest = file_digest(training_path)
    with threadpool_limits(limits=1):
        report = run_experiment(training_path, output, folds=2, holdout_folds=3,
                                similarity_threshold=0.95)
    assert file_digest(training_path) == digest
    assert report["status"] == "complete"
    assert len(report["experiment_source_sha256"]) == 64
    assert report["holdout_evaluated"] is False
    splits = pd.read_csv(output / "splits.csv")
    oof = pd.read_csv(output / "development-oof.csv")
    assert len(oof) == report["development_rows"]
    assert not set(splits.loc[splits.role == "holdout", "excel_row"]) & set(oof.excel_row)
    assert oof.groupby("group_id").fold.nunique().eq(1).all()
    assert set(oof.columns) >= {"pred_raw", "pred_combined_svc", "pred_combined_ordinal"}
    assert set(report["models"]) == set(report["policies"])
    assert json.loads((output / "manifest.json").read_text())["status"] == "holdout_frozen"
    with pytest.raises(ValueError, match="new or empty"):
        run_experiment(training_path, output, folds=2, holdout_folds=3)
