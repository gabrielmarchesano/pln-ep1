import os

import numpy as np
import pytest
from sklearn.base import clone

from model_research import static_embeddings


def test_static_encoder_ignores_labels_and_does_not_truncate(monkeypatch):
    calls = []

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return [[3.0, 4.0] for _ in texts]

    monkeypatch.setattr(static_embeddings, "load_static_encoder", lambda *args: FakeEncoder())
    transformer = static_embeddings.StaticEmbeddings()
    first = transformer.fit_transform([" Não&nbsp;  ", "texto " * 2000], ["c1", "c5"])
    second = clone(transformer).fit_transform([" Não&nbsp;  ", "texto " * 2000], ["c5", "c1"])
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, [[0.6, 0.8], [0.6, 0.8]])
    assert calls[0][0][0] == "não"
    assert len(calls[0][0][1].split()) == 2000
    assert calls[0][1]["max_length"] is None
    assert calls[0][1]["use_multiprocessing"] is False


@pytest.mark.skipif(os.environ.get("EP1_TEST_ENCODER") != "1", reason="Exige o encoder local já baixado.")
def test_real_static_encoder_handles_empty_and_long_text():
    vectors = static_embeddings.StaticEmbeddings().fit_transform(["", "resposta " * 2000, "não foi atendido"])
    assert vectors.shape == (3, 256)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(vectors[0], np.zeros(256))
    np.testing.assert_allclose(np.linalg.norm(vectors[1:], axis=1), [1, 1], atol=1e-5)

