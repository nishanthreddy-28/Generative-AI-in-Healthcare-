"""
Vector Store
============
Wraps FAISS IndexFlatIP for cosine similarity search (inner product on
L2-normalized vectors = cosine similarity).

Persists:
  vector_store/faiss.index      — FAISS binary index
  vector_store/metadata.json    — chunk metadata list (parallel to index rows)
  vector_store/index_manifest.json — build provenance for freshness checking

Freshness check: at load time, the source_manifest hash and embedding
configuration are compared against what was recorded at build time.
If they differ, a StaleIndexError is raised so the caller knows to rebuild.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from rag import embeddings as emb_module

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR  = PROJECT_ROOT / "vector_store"
INDEX_PATH        = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH     = VECTOR_STORE_DIR / "metadata.json"
MANIFEST_PATH     = VECTOR_STORE_DIR / "index_manifest.json"
SOURCE_MANIFEST   = PROJECT_ROOT / "data" / "medical_docs" / "source_manifest.json"


class StaleIndexError(RuntimeError):
    """Raised when the loaded index was built from a different source manifest or config."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk_bytes in iter(lambda: f.read(65536), b""):
            h.update(chunk_bytes)
    return h.hexdigest()


def _source_manifest_hash() -> str:
    if not SOURCE_MANIFEST.exists():
        return "missing"
    return _sha256_file(SOURCE_MANIFEST)


def _current_config() -> dict:
    return {
        "embedding_model":   emb_module.MODEL_NAME,
        "embedding_dim":     emb_module.get_embedding_dim(),
        "normalization":     "L2",
        "chunk_size":        int(os.getenv("CHUNK_SIZE",    "500")),
        "chunk_overlap":     int(os.getenv("CHUNK_OVERLAP", "50")),
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(chunks: list[dict[str, Any]], chunk_embeddings: np.ndarray) -> None:
    """
    Build and persist a FAISS IndexFlatIP from pre-computed embeddings.

    Args:
        chunks:           list of chunk dicts from chunker.chunk_documents()
        chunk_embeddings: float32 ndarray of shape (N, dim), L2-normalized
    """
    n, dim = chunk_embeddings.shape
    assert n == len(chunks), "Mismatch between chunks and embeddings count"

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # Build FAISS inner-product index (= cosine similarity for L2-normed vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings)
    faiss.write_index(index, str(INDEX_PATH))
    logger.info("FAISS index built: %d vectors, dim=%d", n, dim)

    # Save chunk metadata
    with open(METADATA_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    # Save index manifest for freshness checking
    manifest = {
        "build_timestamp":         datetime.now(timezone.utc).isoformat(),
        "embedding_model":         emb_module.MODEL_NAME,
        "embedding_dim":           dim,
        "normalization":           "L2",
        "chunk_size":              int(os.getenv("CHUNK_SIZE",    "500")),
        "chunk_overlap":           int(os.getenv("CHUNK_OVERLAP", "50")),
        "source_manifest_hash":    _source_manifest_hash(),
        "num_chunks":              n,
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Vector store saved to %s", VECTOR_STORE_DIR)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load() -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """
    Load the persisted FAISS index and chunk metadata.

    Returns:
        (faiss_index, chunk_metadata_list)

    Raises:
        FileNotFoundError: if index files are missing
        StaleIndexError:   if source manifest hash or embedding config differs
    """
    for path in (INDEX_PATH, METADATA_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Vector store file missing: {path}\n"
                "Run: python scripts/build_index.py"
            )

    # Load saved manifest
    with open(MANIFEST_PATH) as f:
        saved_manifest = json.load(f)

    # Freshness check 1: source manifest hash
    current_sm_hash = _source_manifest_hash()
    saved_sm_hash   = saved_manifest.get("source_manifest_hash", "")
    if current_sm_hash != saved_sm_hash:
        raise StaleIndexError(
            "The FAISS index was built from a different source_manifest.json.\n"
            "Source manifest has changed since the index was built.\n"
            "Run: python scripts/build_index.py  to rebuild."
        )

    # Freshness check 2: embedding configuration
    current_cfg = _current_config()
    for key in ("embedding_model", "normalization", "chunk_size", "chunk_overlap"):
        saved_val   = saved_manifest.get(key)
        current_val = current_cfg.get(key)
        if saved_val != current_val:
            raise StaleIndexError(
                f"Index config mismatch on '{key}': "
                f"saved={saved_val!r}, current={current_val!r}.\n"
                "Run: python scripts/build_index.py  to rebuild."
            )

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH) as f:
        chunk_meta = json.load(f)

    # Integrity check 1: vector count must equal metadata count
    if index.ntotal != len(chunk_meta):
        raise StaleIndexError(
            f"FAISS index has {index.ntotal} vectors but metadata has {len(chunk_meta)} entries. "
            "The index and metadata are out of sync.\n"
            "Run: python scripts/build_index.py  to rebuild."
        )

    # Integrity check 2: chunk_ids must be unique
    chunk_ids = [m.get("chunk_id", "") for m in chunk_meta]
    if len(chunk_ids) != len(set(chunk_ids)):
        from collections import Counter
        dupes = [cid for cid, n in Counter(chunk_ids).items() if n > 1]
        raise StaleIndexError(
            f"Metadata contains {len(dupes)} duplicate chunk_id(s): {dupes[:5]}. "
            "Run: python scripts/build_index.py  to rebuild."
        )

    logger.info(
        "Vector store loaded: %d vectors, model=%s",
        index.ntotal, saved_manifest.get("embedding_model")
    )
    return index, chunk_meta
