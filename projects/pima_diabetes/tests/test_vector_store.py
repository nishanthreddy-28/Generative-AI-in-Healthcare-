"""
Test: Vector Store & Retriever
==============================
Tests FAISS build/load, stale index detection, and retriever threshold behaviour.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def make_chunks(n: int = 5):
    return [
        {
            "chunk_id": f"doc-a:p1:c{i:02d}",
            "document_id": "doc-a",
            "source": "doc-a.txt",
            "publisher": "Test Publisher",
            "url": "https://example.com",
            "page": 1,
            "text": f"Chunk {i} about diabetes and blood glucose.",
        }
        for i in range(n)
    ]


def make_embeddings(n: int = 5, dim: int = 768):
    rng  = np.random.default_rng(42)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms   # L2-normalized


class TestVectorStoreBuildLoad:
    def test_build_creates_files(self, tmp_path, monkeypatch):
        """build() should create faiss.index, metadata.json, index_manifest.json."""
        import rag.vector_store as vs
        monkeypatch.setattr(vs, "VECTOR_STORE_DIR", tmp_path)
        monkeypatch.setattr(vs, "INDEX_PATH",      tmp_path / "faiss.index")
        monkeypatch.setattr(vs, "METADATA_PATH",   tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH",   tmp_path / "index_manifest.json")
        monkeypatch.setattr(vs, "SOURCE_MANIFEST", tmp_path / "source_manifest.json")
        # Write a fake source manifest so hash is deterministic
        (tmp_path / "source_manifest.json").write_text('[]')

        chunks = make_chunks()
        embs   = make_embeddings(len(chunks))
        vs.build(chunks, embs)

        assert (tmp_path / "faiss.index").exists()
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "index_manifest.json").exists()

    def test_load_after_build(self, tmp_path, monkeypatch):
        """load() should return the same index that was built."""
        import rag.vector_store as vs
        import rag.embeddings as emb_module
        monkeypatch.setattr(vs, "VECTOR_STORE_DIR", tmp_path)
        monkeypatch.setattr(vs, "INDEX_PATH",      tmp_path / "faiss.index")
        monkeypatch.setattr(vs, "METADATA_PATH",   tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH",   tmp_path / "index_manifest.json")
        monkeypatch.setattr(vs, "SOURCE_MANIFEST", tmp_path / "source_manifest.json")
        monkeypatch.setattr(emb_module, "MODEL_NAME", "BAAI/bge-base-en-v1.5")
        monkeypatch.setattr(emb_module, "_model", MagicMock(**{
            "get_sentence_embedding_dimension.return_value": 768
        }))
        (tmp_path / "source_manifest.json").write_text('[]')

        chunks = make_chunks(5)
        embs   = make_embeddings(5)
        vs.build(chunks, embs)

        # Patch _current_config to match what was saved
        saved_manifest = json.loads((tmp_path / "index_manifest.json").read_text())
        def fake_config():
            return {
                "embedding_model": saved_manifest["embedding_model"],
                "embedding_dim":   saved_manifest["embedding_dim"],
                "normalization":   saved_manifest["normalization"],
                "chunk_size":      saved_manifest["chunk_size"],
                "chunk_overlap":   saved_manifest["chunk_overlap"],
            }
        monkeypatch.setattr(vs, "_current_config", fake_config)

        index, meta = vs.load()
        assert index.ntotal == 5
        assert len(meta) == 5

    def test_stale_index_raises(self, tmp_path, monkeypatch):
        """load() must raise StaleIndexError when source manifest has changed."""
        import rag.vector_store as vs
        import rag.embeddings as emb_module
        monkeypatch.setattr(vs, "VECTOR_STORE_DIR", tmp_path)
        monkeypatch.setattr(vs, "INDEX_PATH",      tmp_path / "faiss.index")
        monkeypatch.setattr(vs, "METADATA_PATH",   tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH",   tmp_path / "index_manifest.json")
        monkeypatch.setattr(vs, "SOURCE_MANIFEST", tmp_path / "source_manifest.json")
        monkeypatch.setattr(emb_module, "MODEL_NAME", "BAAI/bge-base-en-v1.5")
        monkeypatch.setattr(emb_module, "_model", MagicMock(**{
            "get_sentence_embedding_dimension.return_value": 768
        }))
        (tmp_path / "source_manifest.json").write_text('[]')

        chunks = make_chunks(3)
        embs   = make_embeddings(3)
        vs.build(chunks, embs)

        # Simulate source manifest change
        (tmp_path / "source_manifest.json").write_text('[{"changed": true}]')

        saved_manifest = json.loads((tmp_path / "index_manifest.json").read_text())
        def fake_config():
            return {
                "embedding_model": saved_manifest["embedding_model"],
                "embedding_dim":   saved_manifest["embedding_dim"],
                "normalization":   saved_manifest["normalization"],
                "chunk_size":      saved_manifest["chunk_size"],
                "chunk_overlap":   saved_manifest["chunk_overlap"],
            }
        monkeypatch.setattr(vs, "_current_config", fake_config)

        with pytest.raises(vs.StaleIndexError, match="changed"):
            vs.load()

    def test_missing_index_file_raises(self, tmp_path, monkeypatch):
        import rag.vector_store as vs
        monkeypatch.setattr(vs, "INDEX_PATH", tmp_path / "nonexistent.index")
        monkeypatch.setattr(vs, "METADATA_PATH", tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH", tmp_path / "index_manifest.json")
        with pytest.raises(FileNotFoundError, match="build_index"):
            vs.load()


class TestRetriever:
    def _make_retriever(self, chunks, embs):
        """Build an in-memory retriever for testing."""
        import faiss, rag.vector_store as vs, rag.retriever as ret
        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)
        r = ret.Retriever()
        r._index    = index
        r._metadata = chunks
        return r

    def test_retrieval_returns_structured_result(self):
        from rag.retriever import Retriever
        from rag.embeddings import encode_documents, encode_query

        texts  = ["Diabetes and blood glucose levels.", "BMI and obesity risk factors."]
        chunks = [{"chunk_id": f"d:p1:c{i:02d}", "document_id": "d",
                   "source": "d.txt", "publisher": "P", "url": "", "page": 1, "text": t}
                  for i, t in enumerate(texts)]
        embs = encode_documents([c["text"] for c in chunks])
        r = self._make_retriever(chunks, embs)

        result = r.retrieve("diabetes glucose", top_k=2, threshold=0.0)
        assert "documents" in result
        assert "scores"    in result
        assert "metadata"  in result
        assert len(result["documents"]) <= 2

    def test_threshold_filters_low_scores(self):
        """With threshold=1.0, nothing should pass."""
        from rag.retriever import Retriever
        from rag.embeddings import encode_documents

        texts  = ["Completely unrelated random text about trains and weather."]
        chunks = [{"chunk_id": "d:p1:c00", "document_id": "d",
                   "source": "d.txt", "publisher": "P", "url": "", "page": 1, "text": t}
                  for t in texts]
        embs = encode_documents([c["text"] for c in chunks])
        r = self._make_retriever(chunks, embs)

        result = r.retrieve("diabetes glucose insulin BMI", top_k=5, threshold=1.0)
        assert result["documents"] == []
        assert result["scores"]    == []
        assert result["metadata"]  == []

    def test_empty_index_returns_empty(self):
        from rag.retriever import Retriever
        import faiss
        r = Retriever()
        r._index    = faiss.IndexFlatIP(768)
        r._metadata = []
        result = r.retrieve("diabetes", top_k=5, threshold=0.0)
        assert result["documents"] == []


class TestVectorStoreIntegrity:
    """Tests for Issue 16 \u2014 vector count and chunk_id uniqueness checks."""

    def test_count_mismatch_raises_stale_index(self, tmp_path, monkeypatch):
        """load() must raise StaleIndexError when index.ntotal != len(metadata)."""
        import rag.vector_store as vs
        import rag.embeddings as emb_module
        monkeypatch.setattr(vs, "VECTOR_STORE_DIR", tmp_path)
        monkeypatch.setattr(vs, "INDEX_PATH",      tmp_path / "faiss.index")
        monkeypatch.setattr(vs, "METADATA_PATH",   tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH",   tmp_path / "index_manifest.json")
        monkeypatch.setattr(vs, "SOURCE_MANIFEST", tmp_path / "source_manifest.json")
        monkeypatch.setattr(emb_module, "MODEL_NAME", "BAAI/bge-base-en-v1.5")
        monkeypatch.setattr(emb_module, "_model", MagicMock(**{
            "get_sentence_embedding_dimension.return_value": 768
        }))
        (tmp_path / "source_manifest.json").write_text('[]')

        chunks = make_chunks(5)
        embs   = make_embeddings(5)
        vs.build(chunks, embs)

        # Corrupt metadata: add extra entry so count != index.ntotal
        saved_meta = json.loads((tmp_path / "metadata.json").read_text())
        saved_meta.append({"chunk_id": "extra:p1:c99", "text": "extra"})
        (tmp_path / "metadata.json").write_text(json.dumps(saved_meta))

        saved_manifest = json.loads((tmp_path / "index_manifest.json").read_text())
        def fake_config():
            return {
                "embedding_model": saved_manifest["embedding_model"],
                "embedding_dim":   saved_manifest["embedding_dim"],
                "normalization":   saved_manifest["normalization"],
                "chunk_size":      saved_manifest["chunk_size"],
                "chunk_overlap":   saved_manifest["chunk_overlap"],
            }
        monkeypatch.setattr(vs, "_current_config", fake_config)

        with pytest.raises(vs.StaleIndexError, match="out of sync"):
            vs.load()

    def test_duplicate_chunk_ids_raises_stale_index(self, tmp_path, monkeypatch):
        """load() must raise StaleIndexError when chunk_ids are not unique."""
        import rag.vector_store as vs
        import rag.embeddings as emb_module
        monkeypatch.setattr(vs, "VECTOR_STORE_DIR", tmp_path)
        monkeypatch.setattr(vs, "INDEX_PATH",      tmp_path / "faiss.index")
        monkeypatch.setattr(vs, "METADATA_PATH",   tmp_path / "metadata.json")
        monkeypatch.setattr(vs, "MANIFEST_PATH",   tmp_path / "index_manifest.json")
        monkeypatch.setattr(vs, "SOURCE_MANIFEST", tmp_path / "source_manifest.json")
        monkeypatch.setattr(emb_module, "MODEL_NAME", "BAAI/bge-base-en-v1.5")
        monkeypatch.setattr(emb_module, "_model", MagicMock(**{
            "get_sentence_embedding_dimension.return_value": 768
        }))
        (tmp_path / "source_manifest.json").write_text('[]')

        # Build with 5 chunks but manually make metadata have duplicate chunk_ids
        chunks = make_chunks(5)
        embs   = make_embeddings(5)
        vs.build(chunks, embs)

        # Corrupt metadata: duplicate the first chunk_id
        saved_meta = json.loads((tmp_path / "metadata.json").read_text())
        saved_meta[1]["chunk_id"] = saved_meta[0]["chunk_id"]  # create duplicate
        (tmp_path / "metadata.json").write_text(json.dumps(saved_meta))

        saved_manifest = json.loads((tmp_path / "index_manifest.json").read_text())
        def fake_config():
            return {
                "embedding_model": saved_manifest["embedding_model"],
                "embedding_dim":   saved_manifest["embedding_dim"],
                "normalization":   saved_manifest["normalization"],
                "chunk_size":      saved_manifest["chunk_size"],
                "chunk_overlap":   saved_manifest["chunk_overlap"],
            }
        monkeypatch.setattr(vs, "_current_config", fake_config)

        with pytest.raises(vs.StaleIndexError, match="duplicate"):
            vs.load()


class TestDocumentLoaderProvenance:
    """Tests for Issues 10 and 11 \u2014 URL field resolution and unregistered doc rejection."""

    def test_url_prefers_final_url(self, tmp_path):
        """_make_unit must prefer final_url over initial_url and legacy url."""
        from rag.document_loader import _make_unit
        entry = {
            "id": "test-doc",
            "publisher": "Test",
            "final_url": "https://final.example.com",
            "initial_url": "https://initial.example.com",
            "url": "https://legacy.example.com",
        }
        unit = _make_unit("text", "test.txt", entry, page=1)
        assert unit["url"] == "https://final.example.com"

    def test_url_falls_back_to_initial_url(self, tmp_path):
        """If final_url is absent, initial_url is used."""
        from rag.document_loader import _make_unit
        entry = {
            "id": "test-doc",
            "publisher": "Test",
            "initial_url": "https://initial.example.com",
        }
        unit = _make_unit("text", "test.txt", entry, page=1)
        assert unit["url"] == "https://initial.example.com"

    def test_url_falls_back_to_legacy_url(self, tmp_path):
        """If neither final_url nor initial_url is present, legacy url is used."""
        from rag.document_loader import _make_unit
        entry = {
            "id": "test-doc",
            "publisher": "Test",
            "url": "https://legacy.example.com",
        }
        unit = _make_unit("text", "test.txt", entry, page=1)
        assert unit["url"] == "https://legacy.example.com"

    def test_unregistered_file_rejected(self, tmp_path):
        """load_documents() must reject files not in source_manifest.json."""
        from rag.document_loader import load_documents
        import json

        # Create an unregistered .txt file
        unregistered = tmp_path / "unregistered-doc.txt"
        unregistered.write_text("This document is not in the manifest.", encoding="utf-8")

        # Create a minimal valid manifest (empty \u2014 no entries registered)
        manifest = tmp_path / "source_manifest.json"
        manifest.write_text("[]", encoding="utf-8")

        # Should raise RuntimeError because no registered docs were loaded
        with pytest.raises(RuntimeError, match="No registered documents"):
            load_documents(docs_dir=tmp_path)

    def test_registered_file_loaded(self, tmp_path):
        """load_documents() must load files that are registered in source_manifest.json."""
        from rag.document_loader import load_documents
        import json

        # Create a registered .txt file
        doc = tmp_path / "test-doc.txt"
        doc.write_text("Registered medical content.", encoding="utf-8")

        # Create manifest with this file registered
        manifest = tmp_path / "source_manifest.json"
        manifest.write_text(json.dumps([{
            "id": "test-doc",
            "publisher": "Test Publisher",
            "local_filename": "test-doc.txt",
            "final_url": "https://example.com/test",
        }]), encoding="utf-8")

        units = load_documents(docs_dir=tmp_path)
        assert len(units) == 1
        assert units[0]["document_id"] == "test-doc"
        assert units[0]["url"] == "https://example.com/test"
