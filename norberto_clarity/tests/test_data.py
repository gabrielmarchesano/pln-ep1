import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from clarity.data import describe_dataset, grouped_splits, load_training_data, normalize_text
from clarity.data import read_frame, text_groups


def test_normalization_preserves_negation_and_accents():
    assert normalize_text("  NÃO&nbsp; é\n possível! ") == "não é possível!"
    assert normalize_text(None) == ""
    assert normalize_text(float("nan")) == ""


def test_grouped_splits_are_reproducible_and_disjoint(training_path):
    data = load_training_data(training_path)
    x, y = data.resp_text.to_numpy(), data.clarity.to_numpy()
    groups = text_groups(x)
    first = grouped_splits(x, y, groups, 3, 42)
    second = grouped_splits(x, y, groups, 3, 42)
    seen = []
    for (train, valid), (train2, valid2) in zip(first, second):
        assert set(groups[train]).isdisjoint(groups[valid])
        np.testing.assert_array_equal(train, train2)
        np.testing.assert_array_equal(valid, valid2)
        seen.extend(valid)
        for inner_train, inner_valid in grouped_splits(x[train], y[train], groups[train], 2, 5):
            assert set(groups[train][inner_train]).isdisjoint(groups[train][inner_valid])
    assert sorted(seen) == list(range(len(data)))
    assert describe_dataset(data)["duplicate_extra_rows"] == len(data) // 2


def test_insufficient_groups_fail(training_path):
    data = load_training_data(training_path)
    x, y = data.resp_text.to_numpy(), data.clarity.to_numpy()
    for folds in (1, 19):
        with pytest.raises(ValueError):
            grouped_splits(x, y, text_groups(x), folds, 42)


@pytest.mark.parametrize("labels", [["c1", "c234", "c6"], ["c1", "c234", None], ["c1", "c1", "c5"]])
def test_invalid_labels_fail(tmp_path, labels):
    path = tmp_path / "invalid.xlsx"
    pd.DataFrame({"resp_text": ["x", "y", "z"], "clarity": labels}).to_excel(path, index=False)
    with pytest.raises(ValueError, match="rótulos"):
        load_training_data(path)


def test_numeric_and_empty_texts_are_preserved_and_audited(tmp_path):
    path = tmp_path / "numeric.xlsx"
    pd.DataFrame({"resp_text": [123, None, "Texto"], "clarity": ["c1", "c234", "c5"]}).to_excel(path, index=False)
    frame = load_training_data(path)
    assert frame.resp_text.tolist() == ["123", "", "Texto"]
    assert describe_dataset(frame)["coerced_text_excel_rows"] == [2]
    assert describe_dataset(frame)["empty_text_rows"] == 1


@pytest.mark.parametrize("header, rows, message", [
    (["resp_text", "resp_text"], [["x", "y"]], "duplicados"),
    (["outro"], [["x"]], "ausentes"),
    (["resp_text"], [], "exemplos"),
    (["resp_text"], [['="texto"']], "fórmulas"),
])
def test_bad_workbook_fails(tmp_path, header, rows, message):
    path = tmp_path / "bad.xlsx"
    workbook = Workbook()
    workbook.active.append(header)
    for row in rows:
        workbook.active.append(row)
    workbook.save(path)
    with pytest.raises(ValueError, match=message):
        read_frame(path, training=False)

