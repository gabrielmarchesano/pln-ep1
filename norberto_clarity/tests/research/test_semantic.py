import numpy as np
import os
import pytest
from sklearn.base import clone

from model_research import semantic


def test_chunks_cover_beginning_middle_and_end():
    assert semantic.chunk_token_ids(list(range(10)), 2, 3) == [[0, 1], [4, 5], [8, 9]]
    assert semantic.chunk_token_ids([], 2, 3) == [[]]
    assert semantic.chunk_token_ids([1, 2, 3], 2, 3) == [[1, 2], [3]]


def test_cache_deduplicates_and_invalidates_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(semantic, "load_encoder", lambda *args: object())

    def fake_encode(encoder, texts, **kwargs):
        calls.extend(texts)
        return np.asarray([[len(text), 1, 2] for text in texts], dtype=np.float32)

    monkeypatch.setattr(semantic, "encode_documents", fake_encode)
    transformer = semantic.FrozenEmbeddings(cache_path=str(tmp_path / "cache.db"))
    first = transformer.fit_transform(["Olá", " OLÁ  ", "outro"], ["ignored", "labels", "here"])
    assert calls == ["olá", "outro"]
    np.testing.assert_array_equal(first[0], first[1])
    second = clone(transformer).fit_transform(["outro", "Olá"])
    np.testing.assert_array_equal(second, first[[2, 0]])
    assert len(calls) == 2
    clone(transformer).set_params(revision="another-fixed-revision").fit_transform(["Olá"])
    assert len(calls) == 3


@pytest.mark.skipif(os.environ.get("EP1_TEST_ENCODER") != "1", reason="Exige o encoder local já baixado.")
def test_real_encoder_matches_official_encode_for_short_documents():
    encoder = semantic.load_encoder(semantic.MODEL_ID, semantic.MODEL_REVISION,
                                    ".cache/huggingface", os.environ.get("EP1_TEST_DEVICE", "cpu"))
    texts = ["não foi possível atender à solicitação.", "resposta completa disponível.", ""]
    expected = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    actual = semantic.encode_documents(encoder, texts, batch_size=2, max_chunks=3)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
    long = semantic.encode_documents(encoder, ["resposta muito longa " * 1000], batch_size=2, max_chunks=3)
    assert long.shape == (1, 384) and np.isfinite(long).all()
    np.testing.assert_allclose(np.linalg.norm(long, axis=1), [1.0], atol=1e-5)
