"""
Chunker
=======
Splits document units into overlapping text chunks.
Each chunk retains the full provenance from its source document.

Chunk ID format: {document_id}:p{page}:c{index:02d}

Configuration via environment variables:
  CHUNK_SIZE    default 500 characters
  CHUNK_OVERLAP default 50 characters
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def chunk_documents(
    document_units: list[dict[str, Any]],
    chunk_size:    int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict[str, Any]]:
    """
    Split document units into overlapping chunks.

    Args:
        document_units: list of units from document_loader.load_documents()
        chunk_size:     max characters per chunk (default from env / 500)
        chunk_overlap:  overlap characters between consecutive chunks (default from env / 50)

    Returns:
        List of chunk dicts, each containing:
          chunk_id, document_id, source, publisher, url, page, text
    """
    size    = chunk_size    if chunk_size    is not None else DEFAULT_CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else DEFAULT_CHUNK_OVERLAP

    if overlap >= size:
        raise ValueError(f"chunk_overlap ({overlap}) must be < chunk_size ({size})")

    all_chunks: list[dict[str, Any]] = []

    for unit in document_units:
        text        = unit.get("text", "")
        doc_id      = unit.get("document_id", "unknown")
        page        = unit.get("page", 1)
        source      = unit.get("source", "")
        publisher   = unit.get("publisher", "Unknown")
        url         = unit.get("url", "")

        if not text.strip():
            continue

        # Sliding-window chunking over character offsets
        start = 0
        chunk_index = 0
        while start < len(text):
            end  = min(start + size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = f"{doc_id}:p{page}:c{chunk_index:02d}"
                all_chunks.append({
                    "chunk_id":   chunk_id,
                    "document_id": doc_id,
                    "source":     source,
                    "publisher":  publisher,
                    "url":        url,
                    "page":       page,
                    "text":       chunk_text,
                })
                chunk_index += 1

            if end == len(text):
                break
            start += size - overlap

    logger.info(
        "Chunking complete: %d units → %d chunks (size=%d, overlap=%d)",
        len(document_units), len(all_chunks), size, overlap
    )
    return all_chunks
