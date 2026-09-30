"""Audit a final classical artifact without using a validation or test set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from clarity.data import LABELS, LABEL_COLUMN, TEXT_COLUMN, file_digest, load_training_data
from clarity.delivery import require_new_file
from clarity.result_paths import resolve_result_input
from .experiment import read_config, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_final_artifact(artifact_path: Path, metadata_path: Path,
                         train_path: Path, config_path: Path) -> dict:
    metadata_path = resolve_result_input(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = read_config(config_path)
    artifact = joblib.load(artifact_path)
    require(artifact.get("format_version") == 1, "Artifact format version mismatch.")
    require(tuple(artifact.get("labels", ())) == LABELS, "Artifact label contract mismatch.")
    require(artifact.get("metadata") == metadata,
            "Artifact metadata differs from adjacent JSON metadata.")
    require(metadata.get("config") == config, "Saved configuration differs from final configuration.")
    require(len(config["candidates"]) == 1, "Final configuration must contain exactly one candidate.")
    candidate = config["candidates"][0]
    require(candidate["grid"] and all(len(values) == 1 for values in candidate["grid"].values()),
            "Final configuration must fix exactly one value per parameter.")
    require(metadata.get("selected") == candidate["name"], "Unexpected selected model.")
    require(len(metadata.get("searches", [])) == 1, "Unexpected number of final searches.")
    search = metadata["searches"][0]
    require(search["name"] == candidate["name"] and search["model"] == candidate["model"],
            "Search identity differs from final candidate.")
    require(search["best_params"] == {name: values[0] for name, values in candidate["grid"].items()},
            "Fitted parameters differ from fixed final parameters.")
    require(not search["convergence_warnings"], "Final fit recorded convergence warnings.")

    frame = load_training_data(train_path)
    require(metadata.get("dataset_sha256") == file_digest(train_path), "Dataset digest mismatch.")
    require(metadata["dataset"]["rows"] == len(frame), "Training row count mismatch.")
    require(metadata["dataset"]["class_counts"]
            == frame[LABEL_COLUMN].value_counts().sort_index().to_dict(),
            "Training class counts mismatch.")

    model = artifact["model"]
    require(model.memory is None, "Final artifact retained a cache directory.")
    require("features" in model.named_steps and "classifier" in model.named_steps,
            "Final pipeline structure mismatch.")
    classifier = model["classifier"]
    require(isinstance(classifier, LogisticRegression),
            "Final classifier is not logistic regression.")
    expected_c = candidate["grid"]["classifier__C"][0]
    require(np.isclose(classifier.C, expected_c), "Final logistic C mismatch.")
    require(np.array_equal(classifier.classes_, np.asarray(LABELS)),
            "Fitted classifier classes differ from label contract.")
    require(np.isfinite(classifier.coef_).all() and np.isfinite(classifier.intercept_).all(),
            "Final classifier contains nonfinite parameters.")
    require(np.asarray(classifier.n_iter_).max() < classifier.max_iter,
            "Final classifier reached its iteration limit.")

    texts = frame[TEXT_COLUMN].to_numpy()
    labels = frame[LABEL_COLUMN].to_numpy()
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    require(predictions.shape == (len(frame),), "Prediction shape mismatch.")
    require(probabilities.shape == (len(frame), len(LABELS)), "Probability shape mismatch.")
    require(np.isfinite(probabilities).all()
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6),
            "Invalid final probabilities.")
    require(np.array_equal(classifier.classes_[probabilities.argmax(axis=1)], predictions),
            "Probability argmax differs from final predictions.")
    prediction_digest = hashlib.sha256(
        "\n".join(predictions.tolist()).encode("utf-8")).hexdigest()
    features = model["features"]
    vocabulary_sizes = {
        name: len(vectorizer.vocabulary_)
        for name, vectorizer in features.transformer_list
    }
    require(set(vocabulary_sizes) == {"word", "char"}
            and all(size > 0 for size in vocabulary_sizes.values()),
            "Final lexical vocabularies are missing or empty.")
    return {
        "status": "passed",
        "artifact_sha256": file_digest(artifact_path),
        "metadata_sha256": file_digest(metadata_path),
        "dataset_sha256": file_digest(train_path),
        "selected": metadata["selected"],
        "rows_fitted": len(frame),
        "labels": list(LABELS),
        "classifier": "LogisticRegression",
        "classifier_c": float(classifier.C),
        "classifier_iterations": np.asarray(classifier.n_iter_).tolist(),
        "vocabulary_sizes": vocabulary_sizes,
        "training_prediction_sha256": prediction_digest,
        "training_prediction_counts": {
            label: int((predictions == label).sum()) for label in LABELS
        },
        "training_accuracy_sanity_check": float((predictions == labels).mean()),
        "probabilities_checked": True,
        "limitations": [
            "Training accuracy is only a serialization sanity check, not a generalization estimate.",
            "The complete training set was used for fitting; no internal untouched test remains.",
            "Generalization must be assessed on the official external test when available.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        require_new_file(args.output)
    result = audit_final_artifact(
        args.artifact, args.metadata, args.train, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()
