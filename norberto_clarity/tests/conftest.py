import pandas as pd
import pytest

from clarity.data import LABELS


@pytest.fixture
def training_path(tmp_path):
    rows = []
    terms = {"c1": "recusa negativa sigilo", "c234": "parcial consulta prazo", "c5": "atendido resposta completa"}
    for index in range(18):
        for label in LABELS:
            text = f"{terms[label]} protocolo item{index} {label}"
            rows.extend([(text, label), (text.upper() + "  ", label)])
    path = tmp_path / "train.xlsx"
    pd.DataFrame(rows, columns=["resp_text", "clarity"]).to_excel(path, index=False)
    return path

