import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model_research.dataset.cleaning import mask_identifiers, retained_training_rows
from model_research.dataset.cleaning import run_cleaning_experiment
from clarity.data import load_training_data, text_groups


def test_identifier_masking_preserves_short_numbers_and_negation():
    text = "NÃO acesse https://example.org/item/123456 ou contato@example.org; art. 5, protocolo 12345678"
    cleaned = mask_identifiers(text)
    assert "não" in cleaned
    assert "art. 5" in cleaned
    assert "urltoken" in cleaned
    assert "emailtoken" in cleaned
    assert "longnumbertoken" in cleaned
    assert "example.org" not in cleaned
    assert "12345678" not in cleaned


def test_conflict_filter_uses_training_labels_only():
    labels = np.asarray(["c1", "c5", "c234", "c5", "c1"])
    groups = text_groups(["same", "SAME ", "middle", "good", "low"])
    training = np.asarray([0, 2, 3, 4])
    np.testing.assert_array_equal(
        retained_training_rows(training, labels, groups, True), training,
    )
    all_rows = np.arange(5)
    np.testing.assert_array_equal(
        retained_training_rows(all_rows, labels, groups, True), np.asarray([2, 3, 4]),
    )


def test_cleaning_experiment_is_grouped_and_preserves_labels(training_path, tmp_path):
    output = tmp_path / "cleaning"
    with threadpool_limits(limits=1):
        report = run_cleaning_experiment(training_path, output,
                                         outer_folds=2, inner_folds=2, seed=42)
    oof = pd.read_csv(output / "oof.csv")
    original = load_training_data(training_path)
    assert report["status"] == "complete"
    assert len(report["experiment_source_sha256"]) == 64
    assert len(oof) == len(original)
    assert oof.true_label.tolist() == original.clarity.tolist()
    assert oof.groupby("group_id").fold.nunique().eq(1).all()
    assert sorted(oof.excel_row) == list(range(2, len(original) + 2))
    for fold in report["folds"]:
        selected = fold["selected_policy"]
        scores = fold["inner_accuracy"]
        assert np.mean(scores[selected]) == max(np.mean(value) for value in scores.values())
        rows = oof.fold.eq(fold["fold"])
        assert oof.loc[rows, "pred_selected"].equals(oof.loc[rows, f"pred_{selected}"])
    assert json.loads((output / "results.json").read_text())["status"] == "complete"
