"""Shared metrics, provenance, and JSON persistence for model runs."""
from __future__ import annotations

import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .data import LABELS, file_digest


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
    source_files = [source_root / "main.py", *sorted((source_root / "clarity").glob("*.py"))]
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions,
            "source_sha256": {str(path.relative_to(source_root)): file_digest(path)
                              for path in source_files}}


def metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=LABELS,
                                    average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def paired_group_bootstrap(y, candidate, baseline, groups, *, seed, repeats=2000):
    """Describe paired accuracy uncertainty while resampling complete text groups."""
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
