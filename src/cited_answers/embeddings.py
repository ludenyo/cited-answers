"""Text embeddings with BAAI/bge-small-en-v1.5 (384 dimensions)."""

from functools import lru_cache
from typing import Any

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


@lru_cache(maxsize=1)
def _model() -> Any:
    # Imported here so modules that only need the pure helpers do not pay the
    # torch import cost, and tests can run without loading the model.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed texts in batches. Vectors are normalised, which suits cosine search."""
    if not texts:
        return []
    vectors = _model().encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]
