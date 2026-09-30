from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import LABELS, file_digest
from .result_paths import resolve_result_input


def load_fixed_holdout(train_path: Path, split_path: Path, labels: np.ndarray, *,
                       expected_split_sha256: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a frozen row-level split against the unchanged source workbook."""
    split_path = resolve_result_input(split_path)
    actual_split_sha256 = file_digest(split_path)
    if actual_split_sha256 != expected_split_sha256:
        raise ValueError(
            "A partição não corresponde ao SHA-256 congelado: "
            f"esperado {expected_split_sha256}, obtido {actual_split_sha256}."
        )
    manifest_path = split_path.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if file_digest(train_path) != manifest.get("dataset_sha256"):
        raise ValueError("A planilha não corresponde ao SHA-256 do holdout congelado.")
    split = pd.read_csv(split_path, dtype={"group_id": str})
    required = {"excel_row", "group_id", "role", "development_fold"}
    if not required.issubset(split.columns) or split[list(required)].isna().any().any():
        raise ValueError("A partição de holdout tem colunas obrigatórias ausentes ou vazias.")
    if len(split) != len(labels) or not np.array_equal(split.excel_row.to_numpy(), np.arange(len(labels)) + 2):
        raise ValueError("As linhas da partição não correspondem à ordem da planilha.")
    if not split.role.isin(["development", "holdout"]).all():
        raise ValueError("A partição contém papéis desconhecidos.")
    development = np.flatnonzero(split.role.eq("development").to_numpy())
    holdout = np.flatnonzero(split.role.eq("holdout").to_numpy())
    if not len(development) or not len(holdout):
        raise ValueError("A partição precisa de desenvolvimento e holdout não vazios.")
    if len(development) != manifest.get("development_rows") or len(holdout) != manifest.get("holdout_rows"):
        raise ValueError("As contagens da partição diferem do manifesto.")
    groups = split.group_id.to_numpy()
    if set(groups[development]) & set(groups[holdout]):
        raise ValueError("Há grupos compartilhados entre desenvolvimento e holdout.")
    if set(labels[development]) != set(LABELS) or set(labels[holdout]) != set(LABELS):
        raise ValueError("Uma das partições perdeu uma classe.")
    return development, holdout, groups
