import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from model_research.dataset.audit_hypotheses import audit_hypotheses
from model_research.dataset.dataset_hypotheses import run_experiment


def test_hypothesis_audit_checks_frozen_holdout_and_metrics(training_path, tmp_path):
    output = tmp_path / "experiment"
    with threadpool_limits(limits=1):
        run_experiment(training_path, output, folds=2, holdout_folds=3,
                       similarity_threshold=0.95)
    audit = audit_hypotheses(output, training_path)
    assert audit["status"] == "passed"
    assert audit["group_overlap"] == 0
    assert audit["holdout_evaluated"] is False
    oof = pd.read_csv(output / "development-oof.csv")
    oof.loc[0, "true_label"] = "c5" if oof.loc[0, "true_label"] != "c5" else "c1"
    oof.to_csv(output / "development-oof.csv", index=False)
    with pytest.raises(ValueError, match="labels differ"):
        audit_hypotheses(output, training_path)
