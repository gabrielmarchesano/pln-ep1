from __future__ import annotations

import importlib.metadata
import json
import platform
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, ParameterGrid

from clarity.data import LABELS, LABEL_COLUMN, TEXT_COLUMN, describe_dataset, file_digest
from clarity.data import grouped_splits, text_groups
from .models import MODEL_NAMES, build_model


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def environment() -> dict:
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "joblib",
                "threadpoolctl", "sentence-transformers", "transformers", "torch", "model2vec",
                "peft", "accelerate", "safetensors", "huggingface-hub"]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    source_root = Path(__file__).parent.parent
    source_files = sorted(source_root.rglob("*.py"))
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions,
            "source_sha256": {str(path.relative_to(source_root)): file_digest(path)
                              for path in source_files}}


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    candidates = config.get("candidates", [])
    if not candidates:
        raise ValueError("A configuração deve conter uma lista candidates não vazia.")
    names = [candidate["name"] for candidate in candidates]
    if len(set(names)) != len(names) or {"majority", "selected"} & set(names):
        raise ValueError("Nomes dos candidatos devem ser únicos e não reservados.")
    for candidate in candidates:
        model_name = candidate["model"]
        if model_name not in MODEL_NAMES or model_name == "majority":
            raise ValueError(f"Modelo inválido na configuração: {model_name}")
        params = build_model(model_name).get_params(deep=True)
        for values in ParameterGrid(candidate.get("grid", {})):
            if set(values) - set(params):
                raise ValueError(f"Parâmetros desconhecidos: {set(values) - set(params)}")
    return config


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def search_candidates(x, y, groups, config, *, folds, seed, cache_dir, verbose=0):
    splits = grouped_splits(x, y, groups, folds, seed)
    fitted, records = {}, []
    for candidate in config["candidates"]:
        name = candidate["name"]
        started = time.monotonic()
        print(f"  Busca: {name}", flush=True)
        model = build_model(candidate["model"], seed=seed, cache_dir=cache_dir)
        search = GridSearchCV(
            model, candidate.get("grid", {}), scoring="accuracy", cv=splits,
            refit=True, n_jobs=1, error_score="raise", return_train_score=False,
            verbose=verbose,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            search.fit(x, y)
        convergence_counts = Counter(str(warning.message) for warning in caught
                                     if issubclass(warning.category, ConvergenceWarning))
        # Keep warnings visible in the execution log as well as in the report.
        for warning in caught:
            warnings.warn_explicit(warning.message, warning.category,
                                   warning.filename, warning.lineno)
        fitted[name] = search.best_estimator_
        trials = []
        for index, params in enumerate(search.cv_results_["params"]):
            trials.append({
                "params": params,
                "mean_accuracy": float(search.cv_results_["mean_test_score"][index]),
                "std_accuracy": float(search.cv_results_["std_test_score"][index]),
                "mean_fit_seconds": float(search.cv_results_["mean_fit_time"][index]),
                "std_fit_seconds": float(search.cv_results_["std_fit_time"][index]),
                "mean_score_seconds": float(search.cv_results_["mean_score_time"][index]),
                "std_score_seconds": float(search.cv_results_["std_score_time"][index]),
                "fold_accuracy": [float(search.cv_results_[f"split{fold}_test_score"][index])
                                  for fold in range(folds)],
            })
        classifier = search.best_estimator_["classifier"]
        diagnostics = {}
        if hasattr(classifier, "n_iter_"):
            diagnostics["n_iter"] = np.asarray(classifier.n_iter_).tolist()
        if hasattr(classifier, "loss_"):
            diagnostics["training_loss"] = float(classifier.loss_)
        if hasattr(classifier, "loss_curve_"):
            diagnostics["training_loss_curve"] = [float(value) for value in classifier.loss_curve_]
        records.append({
            "name": name, "model": candidate["model"],
            "best_params": search.best_params_,
            "best_inner_accuracy": float(search.best_score_),
            "refit_seconds": float(search.refit_time_),
            "refit_diagnostics": diagnostics,
            "convergence_warnings": [
                {"message": message, "count": count}
                for message, count in convergence_counts.items()
            ],
            "seconds": time.monotonic() - started, "trials": trials,
        })
        print(f"    CV interna: {search.best_score_:.4f}; {search.best_params_}", flush=True)
    # Ties go to the first candidate, so order the baseline first in the config.
    best = max(records, key=lambda row: row["best_inner_accuracy"])
    return fitted, records, best["name"]


def summarize(y, predictions, folds) -> dict:
    summary = {"fold_results": folds, "pooled": metrics(y, predictions),
               "confusion_matrix": confusion_matrix(y, predictions, labels=LABELS).tolist(),
               "labels": list(LABELS),
               "classification_report": classification_report(
                   y, predictions, labels=LABELS, output_dict=True, zero_division=0)}
    for metric in ("accuracy", "f1_macro", "balanced_accuracy"):
        scores = [row[metric] for row in folds]
        summary[f"mean_{metric}"] = float(np.mean(scores))
        summary[f"std_{metric}"] = float(np.std(scores))
    summary["mean_train_accuracy"] = float(np.mean([row["train_accuracy"] for row in folds]))
    summary["train_validation_gap"] = summary["mean_train_accuracy"] - summary["mean_accuracy"]
    return summary


def paired_group_bootstrap(y, candidate, baseline, groups, *, seed, repeats=2000):
    """Descriptive paired interval; resample groups, not correlated duplicate rows."""
    unique, inverse = np.unique(groups, return_inverse=True)
    differences = (candidate == y).astype(float) - (baseline == y).astype(float)
    sums = np.bincount(inverse, weights=differences)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repeats)
    for index in range(repeats):
        sample = rng.integers(0, len(unique), size=len(unique))
        estimates[index] = sums[sample].sum() / sizes[sample].sum()
    return {"accuracy_delta": float(differences.mean()),
            "descriptive_95pct_interval": np.quantile(estimates, [0.025, 0.975]).tolist(),
            "resampling_unit": "normalized_text_group", "repeats": repeats,
            "caveat": "Intervalo descritivo condicionado às predições avaliadas; não mede toda a incerteza do treino "
                      "nem constitui garantia de melhora no teste oficial."}


def run_experiment(frame, train_path, config, output_dir, *, folds=3, inner_folds=3,
                   seed=42, cache_dir=None, verbose=0) -> dict:
    run_environment = environment()
    run_environment["source_capture"] = "experiment_start"
    x = frame[TEXT_COLUMN].to_numpy()
    y = frame[LABEL_COLUMN].to_numpy()
    groups = text_groups(x)
    splits = grouped_splits(x, y, groups, folds, seed)
    names = [row["name"] for row in config["candidates"]]
    names += ["majority", "selected"]
    # Always evaluate the fixed baseline on the same outer folds.
    baseline_spec = {"name": "baseline", "model": "baseline", "grid": {}}
    if "baseline" not in names:
        config = {**config, "candidates": [baseline_spec, *config["candidates"]]}
        names.insert(0, "baseline")
    elif config["candidates"][names.index("baseline")] != baseline_spec:
        raise ValueError("O nome baseline é reservado à referência fixa, sem busca.")
    predicted = {name: np.empty(len(frame), dtype=object) for name in names}
    results = {name: [] for name in names}
    fold_ids = np.zeros(len(frame), dtype=int)
    choices, searches = [], []
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "nested_stratified_group_kfold", "seed": seed,
        "outer_folds": folds, "inner_folds": inner_folds,
        "dataset_sha256": file_digest(train_path), "rows": len(frame),
        "config": config, "environment": run_environment,
    })
    for fold, (train, valid) in enumerate(splits, 1):
        print(f"Fold externo {fold}/{folds}: treino={len(train)}, validação={len(valid)}", flush=True)
        fitted, records, winner = search_candidates(
            x[train], y[train], groups[train], config, folds=inner_folds,
            seed=seed + fold, cache_dir=cache_dir, verbose=verbose,
        )
        fitted["majority"] = build_model("majority").fit(x[train], y[train])
        fitted["selected"] = fitted[winner]
        choices.append({"fold": fold, "name": winner})
        searches.append({"fold": fold, "candidates": records})
        fold_ids[valid] = fold
        for name, model in fitted.items():
            if name == "selected":
                # The selected estimator is the already evaluated winner. Reuse
                # its outputs instead of vectorizing the full corpus twice.
                predicted[name][valid] = predicted[winner][valid]
                results[name].append(dict(results[winner][-1]))
                print(f"  selected: {winner}", flush=True)
                continue
            pred = model.predict(x[valid])
            predicted[name][valid] = pred
            row = {"fold": fold, "train_rows": len(train), "valid_rows": len(valid),
                   "train_groups": len(np.unique(groups[train])),
                   "valid_groups": len(np.unique(groups[valid])),
                   "group_overlap": 0,
                   "train_accuracy": float(accuracy_score(y[train], model.predict(x[train]))),
                   **metrics(y[valid], pred)}
            results[name].append(row)
            print(f"  {name}: acurácia externa={row['accuracy']:.4f}", flush=True)
        write_json(output_dir / f"fold-{fold}.json", {
            "fold": fold, "selected": winner, "search": records,
            "metrics": {name: values[-1] for name, values in results.items()},
        })
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "nested_stratified_group_kfold",
        "primary_metric": "mean_outer_accuracy", "seed": seed,
        "outer_folds": folds, "inner_folds": inner_folds,
        "dataset_sha256": file_digest(train_path), "dataset": describe_dataset(frame),
        "environment": run_environment, "config": config,
        "grouping": "sha256 of HTML-unescaped, lowercase, whitespace-normalized text",
        "seconds": time.monotonic() - started,
        "selected_per_fold": choices, "inner_searches": searches,
        "models": {name: summarize(y, predicted[name], results[name]) for name in names},
        "comparisons_with_baseline": {
            name: paired_group_bootstrap(y, predicted[name], predicted["baseline"],
                                         groups, seed=seed)
            for name in names if name not in {"baseline", "majority"}
        },
        "limitations": [
            "Duplicatas aproximadas não são agrupadas.",
            "Escolher uma família após inspecionar seus folds externos gera viés; "
            "selected estima o procedimento de seleção interna completo.",
            "O desempenho no conjunto de teste oficial ainda é desconhecido.",
        ],
    }
    oof = pd.DataFrame({"excel_row": np.arange(len(frame)) + 2,
                        "group_id": groups, "fold": fold_ids, "true_label": y})
    for name in names:
        oof[f"pred_{name}"] = predicted[name]
    oof.to_csv(output_dir / "oof.csv", index=False)
    write_json(output_dir / "results.json", report)
    return report


def evaluate(frame, model_name: str, folds: int, seed=42) -> dict:
    """Compatibility mode: fixed parameters, no hyperparameter selection."""
    x, y = frame[TEXT_COLUMN].to_numpy(), frame[LABEL_COLUMN].to_numpy()
    groups = text_groups(x)
    predictions = np.empty(len(frame), dtype=object)
    results = []
    for fold, (train, valid) in enumerate(grouped_splits(x, y, groups, folds, seed), 1):
        print(f"Fold {fold}/{folds}: {model_name}", flush=True)
        model = build_model(model_name, seed=seed).fit(x[train], y[train])
        pred = model.predict(x[valid])
        predictions[valid] = pred
        results.append({"fold": fold, **metrics(y[valid], pred),
                        "train_accuracy": float(accuracy_score(y[train], model.predict(x[train])))})
    return {"model": model_name, "folds": folds, "seed": seed,
            **summarize(y, predictions, results)}
