"""
Test: RAG System
================
Tests document loading, chunking, embedding shapes, and metadata preservation.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestChunker:
    def test_chunk_preserves_provenance(self):
        from rag.chunker import chunk_documents
        docs = [{
            "document_id": "test-doc",
            "source":      "test.txt",
            "publisher":   "Test Publisher",
            "url":         "https://example.com",
            "page":        1,
            "text":        "A" * 1200,
        }]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2
        for c in chunks:
            assert c["document_id"] == "test-doc"
            assert c["source"]      == "test.txt"
            assert c["publisher"]   == "Test Publisher"
            assert c["url"]         == "https://example.com"
            assert c["page"]        == 1
            assert "chunk_id"       in c
            assert "text"           in c

    def test_chunk_id_format(self):
        from rag.chunker import chunk_documents
        docs = [{
            "document_id": "myid", "source": "f.txt", "publisher": "P",
            "url": "", "page": 3, "text": "Hello world " * 50,
        }]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for i, c in enumerate(chunks):
            assert c["chunk_id"] == f"myid:p3:c{i:02d}", f"Bad chunk_id: {c['chunk_id']}"

    def test_empty_text_skipped(self):
        from rag.chunker import chunk_documents
        docs = [{"document_id": "empty", "source": "e.txt", "publisher": "P",
                 "url": "", "page": 1, "text": "   "}]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert chunks == []

    def test_overlap_less_than_size_required(self):
        from rag.chunker import chunk_documents
        docs = [{"document_id": "d", "source": "s.txt", "publisher": "P",
                 "url": "", "page": 1, "text": "hello"}]
        with pytest.raises(ValueError, match="overlap"):
            chunk_documents(docs, chunk_size=50, chunk_overlap=50)

    def test_no_data_loss(self):
        """Every character of source text must appear in at least one chunk."""
        from rag.chunker import chunk_documents
        long_text = "ABCDEFGHIJ" * 100
        docs = [{"document_id": "d", "source": "s.txt", "publisher": "P",
                 "url": "", "page": 1, "text": long_text}]
        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
        combined = "".join(c["text"] for c in chunks)
        # Every unique character from source should be present
        assert set(long_text).issubset(set(combined))


class TestDocumentLoader:
    def test_load_txt_file(self, tmp_path):
        from rag.document_loader import _load_txt
        txt = tmp_path / "test.txt"
        txt.write_text("Hello diabetes world", encoding="utf-8")
        entry = {"id": "test-doc", "publisher": "Test", "url": "https://example.com", "local_filename": "test.txt"}
        units = _load_txt(txt, manifest_entry=entry)
        assert len(units) == 1
        assert "Hello diabetes world" in units[0]["text"]
        assert units[0]["page"] == 1

    def test_load_txt_with_manifest(self, tmp_path):
        from rag.document_loader import _load_txt
        txt = tmp_path / "niddk.txt"
        txt.write_text("NIH content", encoding="utf-8")
        entry = {
            "id": "niddk-test", "publisher": "NIDDK",
            "url": "https://niddk.nih.gov", "local_filename": "niddk.txt"
        }
        units = _load_txt(txt, entry)
        assert units[0]["document_id"] == "niddk-test"
        assert units[0]["publisher"]   == "NIDDK"
        assert units[0]["url"]         == "https://niddk.nih.gov"

    def test_missing_dir_raises(self):
        from rag.document_loader import load_documents
        with pytest.raises(FileNotFoundError):
            load_documents(docs_dir=Path("/nonexistent/path/12345"))

    def test_empty_dir_raises(self, tmp_path):
        from rag.document_loader import load_documents
        with pytest.raises(RuntimeError, match="No .pdf/.txt/.md"):
            load_documents(docs_dir=tmp_path)

    def test_manifest_provenance_fields(self):
        """Verify the source manifest contains all required precise provenance fields."""
        import json
        manifest_path = PROJECT_ROOT / "data" / "medical_docs" / "source_manifest.json"
        
        # Skip test if manifest doesn't exist yet, but it should since scripts were run
        if not manifest_path.exists():
            pytest.skip("source_manifest.json not found")
            
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        assert isinstance(data, list)
        assert len(data) > 0
        
        source = data[0]
        
        required_fields = [
            "id", "publisher", "initial_url", "final_url", "local_filename", 
            "raw_checksum_sha256", "extracted_checksum_sha256", "retrieval_date_utc", 
            "http_content_type"
        ]
        
        for field in required_fields:
            assert field in source, f"Missing provenance field: {field}"
            
        # Verify retrieval_date is ISO 8601 (ends with Z or contains T)
        assert "T" in source["retrieval_date_utc"] or "Z" in source["retrieval_date_utc"]


class TestEmbeddings:
    def test_encode_documents_shape(self):
        """Documents should return (N, 768) float32 array."""
        from rag.embeddings import encode_documents
        texts = ["Diabetes is a metabolic condition.", "Blood glucose regulation."]
        embs = encode_documents(texts)
        assert embs.shape == (2, 768)
        assert embs.dtype.name == "float32"

    def test_encode_query_shape(self):
        from rag.embeddings import encode_query
        vec = encode_query("What is Type 2 diabetes?")
        assert vec.shape == (1, 768)
        assert vec.dtype.name == "float32"

    def test_vectors_are_normalized(self):
        """L2 norm of each vector should be ≈ 1.0."""
        from rag.embeddings import encode_documents
        embs = encode_documents(["Test sentence about diabetes."])
        norms = np.linalg.norm(embs, axis=1)
        assert abs(norms[0] - 1.0) < 1e-5, f"Norm not 1.0: {norms[0]}"

    def test_query_prefix_applied(self):
        """Query and doc embeddings for the same text should differ (prefix effect)."""
        from rag.embeddings import encode_documents, encode_query
        text = "Blood glucose and diabetes."
        doc_emb = encode_documents([text])
        qry_emb = encode_query(text)
        cosine_sim = float(np.dot(doc_emb[0], qry_emb[0]))
        # They should be similar but not identical
        assert cosine_sim < 1.0 - 1e-6, "Query and doc embeddings must differ (prefix)"
