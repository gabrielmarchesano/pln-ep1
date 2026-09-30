import json

import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from clarity.audit import audit_result_dir, check_probabilities
from clarity.data import load_training_data
from model_research.experiment import run_experiment


def test_audit_validates_and_rejects_corrupted_predictions(training_path, tmp_path):
    root = tmp_path / "result"
    with threadpool_limits(limits=1):
        run_experiment(load_training_data(training_path), training_path,
                       {"candidates": [{"name": "baseline", "model": "baseline", "grid": {}}]},
                       root, folds=3, inner_folds=2)
    assert audit_result_dir(root, training_path)["status"] == "passed"
    predictions = pd.read_csv(root / "oof.csv")
    predictions.loc[0, "true_label"] = "invalid"
    predictions.to_csv(root / "oof.csv", index=False)
    with pytest.raises(ValueError, match="Ground truth"):
        audit_result_dir(root, training_path)


def test_probability_audit_checks_complete_normalized_and_matching_values():
    frame = pd.DataFrame({"pred_test": ["c234"], "prob_test_c1": [0.2],
                          "prob_test_c234": [0.7], "prob_test_c5": [0.1]})
    assert check_probabilities(frame, "test")
    assert not check_probabilities(frame[["pred_test"]], "test")
    with pytest.raises(ValueError, match="Incomplete"):
        check_probabilities(frame.drop(columns="prob_test_c5"), "test")
    with pytest.raises(ValueError, match="argmax"):
        check_probabilities(frame.assign(pred_test="c1"), "test")
    with pytest.raises(ValueError, match="Invalid probabilities"):
        check_probabilities(frame.assign(prob_test_c1=0.9), "test")
