import json

import pytest
from threadpoolctl import threadpool_limits

from model_research.audit_artifact import audit_final_artifact
from model_research.cli import execute, parser


def test_final_artifact_audit_checks_bundle_and_rejects_metadata_corruption(
    training_path, tmp_path
):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "candidates": [{
            "name": "final", "model": "lexical",
            "grid": {"classifier__C": [0.25]},
        }],
    }))
    artifact = tmp_path / "final.joblib"
    with threadpool_limits(limits=1):
        execute(parser().parse_args([
            "train", "--train", str(training_path), "--config", str(config_path),
            "--output", str(artifact), "--folds", "2",
            "--cache-dir", str(tmp_path / "cache"),
        ]))
    audit = audit_final_artifact(
        artifact, artifact.with_suffix(".json"), training_path, config_path)
    assert audit["status"] == "passed"
    assert audit["classifier_c"] == 0.25
    assert audit["rows_fitted"] == 108
    metadata = json.loads(artifact.with_suffix(".json").read_text())
    metadata["dataset_sha256"] = "0" * 64
    artifact.with_suffix(".json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata differs"):
        audit_final_artifact(
            artifact, artifact.with_suffix(".json"), training_path, config_path)
