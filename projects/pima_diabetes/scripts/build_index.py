"""
Build FAISS Index
=================
Run this after downloading medical source documents:
  python scripts/download_medical_sources.py
  python scripts/build_index.py

Flow:
  1. Verify source_manifest.json exists and has entries
  2. Load documents (with full provenance)
  3. Chunk documents
  4. Generate embeddings (BAAI/bge-base-en-v1.5)
  5. Build FAISS IndexFlatIP
  6. Save index + metadata + index_manifest.json
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import json

    print("=" * 60)
    print("Building FAISS Medical Knowledge Index")
    print("=" * 60)

    # Step 1: Verify source manifest
    manifest_path = PROJECT_ROOT / "data" / "medical_docs" / "source_manifest.json"
    if not manifest_path.exists():
        print(
            "\n[BLOCKED] source_manifest.json not found.\n"
            "Run: python scripts/download_medical_sources.py"
        )
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    if not manifest:
        print(
            "\n[BLOCKED] source_manifest.json is empty.\n"
            "No approved medical documents are available.\n"
            "Add sources to source_registry.json and run the downloader."
        )
        sys.exit(1)

    print(f"\nSource manifest: {len(manifest)} documents")
    for entry in manifest:
        print(f"  [{entry['id']}] {entry['title']} ({entry['local_filename']})")

    # Step 2: Load documents
    print("\n[1/4] Loading documents...")
    from rag.document_loader import load_documents
    try:
        documents = load_documents()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    print(f"  Loaded {len(documents)} document units")

    # Step 3: Chunk
    print("\n[2/4] Chunking documents...")
    from rag.chunker import chunk_documents
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    if not chunks:
        print("\n[ERROR] No chunks produced. Check document content.")
        sys.exit(1)

    # Step 4: Generate embeddings
    print("\n[3/4] Generating embeddings (BAAI/bge-base-en-v1.5)...")
    from rag.embeddings import encode_documents, get_embedding_dim
    texts = [c["text"] for c in chunks]
    embeddings = encode_documents(texts)
    print(f"  Embeddings: {embeddings.shape} (dim={get_embedding_dim()})")

    # Step 5: Build and save FAISS index
    print("\n[4/4] Building and saving FAISS index...")
    from rag.vector_store import build
    build(chunks, embeddings)

    print("\n[DONE] FAISS index built successfully.")
    print(f"  vector_store/faiss.index")
    print(f"  vector_store/metadata.json")
    print(f"  vector_store/index_manifest.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
