from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from threadpoolctl import threadpool_limits

from clarity.data import LABELS, LABEL_COLUMN, TEXT_COLUMN, describe_dataset, file_digest, load_training_data
from clarity.delivery import require_new_file
from .experiment import (environment, metrics, paired_group_bootstrap, read_config,
                         search_candidates, write_json)
from clarity.holdout import load_fixed_holdout
from clarity.result_paths import resolve_result_input


def run(train_path: Path, split_path: Path, config_path: Path, output_path: Path,
        *, folds: int = 3, seed: int = 42, cache_dir: str = ".cache/sklearn-holdout") -> dict:
    split_path = resolve_result_input(split_path)
    if output_path.suffix != ".joblib":
        raise ValueError("O modelo deve ser salvo como .joblib.")
    metadata_path = output_path.with_suffix(".json")
    predictions_path = output_path.with_suffix(".holdout.csv")
    for path in (output_path, metadata_path, predictions_path):
        require_new_file(path)
    frame = load_training_data(train_path)
    x, y = frame[TEXT_COLUMN].to_numpy(), frame[LABEL_COLUMN].to_numpy()
    config = read_config(config_path)
    expected_split_sha256 = config.get("frozen_split_sha256")
    if not isinstance(expected_split_sha256, str) or len(expected_split_sha256) != 64:
        raise ValueError("A configuração deve fixar frozen_split_sha256 com 64 caracteres.")
    development, holdout, groups = load_fixed_holdout(
        train_path, split_path, y, expected_split_sha256=expected_split_sha256)
    fitted, searches, winner = search_candidates(
        x[development], y[development], groups[development], config,
        folds=folds, seed=seed, cache_dir=cache_dir,
    )
    candidate_predictions = {}
    candidate_metrics = {}
    for name, candidate_model in fitted.items():
        candidate_model.memory = None
        predictions = candidate_model.predict(x[holdout])
        candidate_predictions[name] = predictions
        candidate_metrics[name] = {
            **metrics(y[holdout], predictions),
            "confusion_matrix": confusion_matrix(
                y[holdout], predictions, labels=LABELS).tolist(),
            "classification_report": classification_report(
                y[holdout], predictions, labels=LABELS,
                output_dict=True, zero_division=0),
        }
    model = fitted[winner]
    predictions = candidate_predictions[winner]
    reference = config["candidates"][0]["name"]
    comparisons = {
        name: paired_group_bootstrap(
            y[holdout], candidate_predictions[name], candidate_predictions[reference],
            groups[holdout], seed=seed)
        for name in candidate_predictions if name != reference
    }
    for comparison in comparisons.values():
        comparison["resampling_unit"] = "frozen_similarity_component"
    selected_metrics = candidate_metrics[winner]
    validation = {
        "protocol": "frozen_grouped_holdout_reused_for_validation",
        "split_sha256": file_digest(split_path),
        "split_path": str(split_path),
        "development_rows": len(development),
        "holdout_rows": len(holdout),
        "development_groups": len(np.unique(groups[development])),
        "holdout_groups": len(np.unique(groups[holdout])),
        "labels": list(LABELS),
        **selected_metrics,
        "models": candidate_metrics,
        "comparison_reference": reference,
        "comparisons_with_reference": comparisons,
        "caveat": "Este holdout será consultado repetidamente para decisões de modelagem; "
                  "é validação fixa, não teste final independente. A base inteira já foi analisada antes.",
    }
    result = {
        "selected": winner, "searches": searches, "config": config,
        "dataset_sha256": file_digest(train_path),
        "dataset": describe_dataset(frame),
        "folds": folds, "seed": seed,
        "environment": environment(),
        "fixed_validation": validation,
        "note": "A busca e o ajuste usam apenas desenvolvimento; o holdout só entra na avaliação.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"format_version": 1, "labels": LABELS, "model": model,
                 "metadata": result}, output_path, compress=3)
    write_json(metadata_path, result)
    prediction_frame = pd.DataFrame({
        "excel_row": holdout + 2, "group_id": groups[holdout],
        "true_label": y[holdout], "predicted_label": predictions,
    })
    for name, candidate_prediction in candidate_predictions.items():
        prediction_frame[f"pred_{name}"] = candidate_prediction
    prediction_frame.to_csv(predictions_path, index=False)
    print(f"Modelo selecionado: {winner}; salvo em {output_path}")
    for name, values in candidate_metrics.items():
        print(f"  {name}: acurácia={values['accuracy']:.4f}; "
              f"F1 macro={values['f1_macro']:.4f}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Treinar apenas no desenvolvimento e avaliar holdout fixo.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--cache-dir", default=".cache/sklearn-holdout")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads deve ser positivo.")
    try:
        with threadpool_limits(limits=args.threads):
            run(args.train, args.split, args.config, args.output,
                folds=args.folds, seed=args.seed, cache_dir=args.cache_dir)
    except (ValueError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"Erro: {error}\n")


if __name__ == "__main__":
    main()
