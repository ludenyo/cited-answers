from typing import Any

import numpy as np
import pytest

from cited_answers import embeddings


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        self.calls.append(texts)
        assert kwargs["normalize_embeddings"] is True
        return np.ones((len(texts), embeddings.EMBEDDING_DIM), dtype=np.float32)


def test_embed_texts_returns_one_384_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    monkeypatch.setattr(embeddings, "_model", lambda: model)

    vectors = embeddings.embed_texts(["a", "b", "c"])

    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)
    assert len(model.calls) == 1  # a single batched call


def test_embed_texts_empty_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> None:
        raise AssertionError("model should not load for empty input")

    monkeypatch.setattr(embeddings, "_model", fail)
    assert embeddings.embed_texts([]) == []
