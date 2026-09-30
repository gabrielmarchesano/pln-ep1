from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import file_digest, load_training_data, text_groups
from .holdout import load_fixed_holdout
from .result_paths import resolve_result_input


def prepare_training_corpus(train_path: Path, split_path: Path | None = None,
                            expected_split_sha256: str | None = None):
    """Exclude reserved rows before tokenization while preserving Excel row IDs."""
    frame = load_training_data(train_path)
    excel_rows = np.arange(len(frame)) + 2
    if split_path is None:
        if expected_split_sha256 is not None:
            raise ValueError("The configuration requires --holdout-splits.")
        return frame, text_groups(frame.resp_text), excel_rows, {}
    if not expected_split_sha256:
        raise ValueError("A frozen holdout split SHA-256 is required in the configuration.")
    split_path = resolve_result_input(split_path)
    development, holdout, groups = load_fixed_holdout(
        train_path, split_path, frame.clarity.to_numpy(),
        expected_split_sha256=expected_split_sha256,
    )
    metadata = {
        "split_path": str(split_path), "split_sha256": file_digest(split_path),
        "source_rows": len(frame), "development_rows": len(development),
        "reserved_holdout_rows": len(holdout), "holdout_evaluated": False,
        "grouping": "frozen similarity components from the holdout split",
    }
    return (frame.iloc[development].reset_index(drop=True), groups[development],
            excel_rows[development], metadata)
