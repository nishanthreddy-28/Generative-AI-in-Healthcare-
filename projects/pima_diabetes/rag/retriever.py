"""
Retriever
=========
Performs FAISS similarity search over the persisted vector store.

Design:
  - Queries are encoded with the BGE query prefix (see embeddings.py)
  - FAISS IndexFlatIP returns inner-product scores (= cosine similarity
    for L2-normalized vectors; range [-1, 1])
  - Results are filtered by a configurable similarity threshold
  - If no result meets the threshold, empty lists are returned — the caller
    must handle insufficient context gracefully

Security note:
  Retrieved document text is treated as untrusted data.
  It must never override application or system-level instructions.

Configuration (via environment or constructor):
  TOP_K=5
  SIMILARITY_THRESHOLD=0.35   (calibrated against downloaded corpus)
"""

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_TOP_K     = int(os.getenv("TOP_K", "5"))
DEFAULT_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))


class Retriever:
    """
    Wraps the FAISS index and provides a retrieve() method.
    The index is loaded lazily on first use.
    """

    def __init__(self):
        self._index     = None
        self._metadata: list[dict[str, Any]] = []

    def _ensure_loaded(self) -> None:
        if self._index is None:
            from rag.vector_store import load
            self._index, self._metadata = load()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> dict[str, list]:
        """
        Retrieve the top-K chunks most similar to the query.

        Args:
            query:     free-text query string
            top_k:     number of candidates to retrieve (default from env/5)
            threshold: minimum cosine similarity to include (default from env/0.35)

        Returns:
            {
              "documents": [str, ...],     # chunk text
              "scores":    [float, ...],   # cosine similarity scores
              "metadata":  [dict, ...],    # chunk metadata (chunk_id, source, page, ...)
            }
            All lists are parallel and ordered by descending score.
            Returns empty lists if no result meets the threshold.
        """
        self._ensure_loaded()

        k         = top_k     if top_k     is not None else DEFAULT_TOP_K
        min_score = threshold if threshold is not None else DEFAULT_THRESHOLD

        from rag.embeddings import encode_query
        query_vec = encode_query(query)   # shape (1, dim), L2-normalized

        # Search — FAISS returns (scores, indices) arrays of shape (1, k)
        k_search = min(k, self._index.ntotal)
        if k_search == 0:
            logger.warning("FAISS index is empty; returning no results.")
            return {"documents": [], "scores": [], "metadata": []}

        scores_arr, idx_arr = self._index.search(query_vec, k_search)
        scores  = scores_arr[0].tolist()
        indices = idx_arr[0].tolist()

        documents = []
        filtered_scores = []
        filtered_meta   = []

        for score, idx in zip(scores, indices):
            if idx < 0:
                continue  # FAISS sentinel for unfilled slots
            if score < min_score:
                continue  # Below threshold — do not force irrelevant docs into LLM context
            meta = self._metadata[idx]
            documents.append(meta.get("text", ""))
            filtered_scores.append(round(float(score), 4))
            filtered_meta.append({
                "chunk_id":   meta.get("chunk_id", ""),
                "document_id": meta.get("document_id", ""),
                "source":     meta.get("source", ""),
                "publisher":  meta.get("publisher", "Unknown"),
                "url":        meta.get("url", ""),
                "page":       meta.get("page", 1),
                "text":       meta.get("text", ""),
                "score":      round(float(score), 4),
            })

        if not documents:
            logger.info(
                "RAG retrieval: 0 results above threshold=%.3f for query (%.60r...)",
                min_score, query
            )
        else:
            logger.info(
                "RAG retrieval: %d results (top score=%.4f, threshold=%.3f)",
                len(documents), filtered_scores[0], min_score
            )

        return {
            "documents": documents,
            "scores":    filtered_scores,
            "metadata":  filtered_meta,
        }


# Module-level singleton
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
