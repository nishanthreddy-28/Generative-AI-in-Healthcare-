"""
Embeddings
==========
Wraps BAAI/bge-base-en-v1.5 for document and query encoding.

BGE-specific behaviour (from model card):
  - Documents: encoded without a prefix
  - Queries:   prefixed with "Represent this sentence for searching relevant passages: "

Vectors are L2-normalized to enable cosine similarity via FAISS inner-product search.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Module-level singleton — loaded once
_model: "SentenceTransformer | None" = None


def _get_model() -> "SentenceTransformer":
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded. Dimension: %d", get_embedding_dim())
    return _model


def get_embedding_dim() -> int:
    """Return the embedding vector dimension."""
    model = _get_model()
    # Use new API (sentence-transformers >= 3.x) with fallback
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()


def encode_documents(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of document texts.
    No prefix applied (per BGE model card for passage encoding).
    Returns L2-normalized float32 array of shape (N, dim).
    """
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2 normalize
    )
    return embeddings.astype(np.float32)


def encode_query(query: str) -> np.ndarray:
    """
    Encode a single query string.
    Applies BGE query prefix for retrieval (per model card).
    Returns L2-normalized float32 array of shape (1, dim).
    """
    model = _get_model()
    prefixed = BGE_QUERY_PREFIX + query
    embedding = model.encode(
        [prefixed],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.astype(np.float32)
