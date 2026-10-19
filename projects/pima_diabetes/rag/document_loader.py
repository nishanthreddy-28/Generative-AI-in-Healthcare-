"""
Document Loader
===============
Loads medical reference documents from data/medical_docs/.
Supports .pdf, .txt, .md.

Every extracted unit preserves full provenance from source_manifest.json:
  document_id, source, publisher, url, page, text

Only documents registered in source_manifest.json are accepted.
Unregistered files are rejected to enforce the trusted-source-only policy.
Unreadable PDFs are logged by document_id only — patient data is never
included in error logs.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
DOCS_DIR      = PROJECT_ROOT / "data" / "medical_docs"
MANIFEST_PATH = DOCS_DIR / "source_manifest.json"


def _load_manifest(docs_dir: Path | None = None) -> dict[str, dict]:
    """Return a dict keyed by local_filename -> manifest entry."""
    manifest_path = (docs_dir or DOCS_DIR) / "source_manifest.json"
    if not manifest_path.exists():
        logger.warning("source_manifest.json not found at %s", manifest_path)
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["local_filename"]: e for e in entries}


def _load_txt(path: Path, manifest_entry: dict) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [_make_unit(text, path.name, manifest_entry, page=1)]


def _load_pdf(path: Path, manifest_entry: dict) -> list[dict[str, Any]]:
    doc_id = manifest_entry["id"] if manifest_entry else path.stem
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        units = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                # Log only document ID and technical error — no patient data
                logger.warning(
                    "PDF page extraction error [doc_id=%s, page=%d]: %s",
                    doc_id, page_num, type(e).__name__
                )
                text = ""
            if text.strip():
                units.append(_make_unit(text, path.name, manifest_entry, page=page_num))
        return units
    except Exception as e:
        logger.error(
            "Cannot read PDF [doc_id=%s]: %s", doc_id, type(e).__name__
        )
        return []


def _make_unit(text: str, filename: str, manifest_entry: dict,
               page: int) -> dict[str, Any]:
    """Build a document unit from manifest entry — entry is always required."""
    # Prefer final_url (new downloader format), then initial_url, then legacy url
    url = (
        manifest_entry.get("final_url")
        or manifest_entry.get("initial_url")
        or manifest_entry.get("url", "")
    )
    return {
        "document_id": manifest_entry["id"],
        "source":      filename,
        "publisher":   manifest_entry.get("publisher", "Unknown"),
        "url":         url,
        "page":        page,
        "text":        text,
    }


def load_documents(docs_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Load all supported documents from docs_dir.
    Only documents registered in source_manifest.json are accepted.
    Unregistered files are skipped with a warning — they cannot enter the
    knowledge base under the trusted-source-only policy.

    Returns a list of document units with full provenance metadata.

    Raises FileNotFoundError if docs_dir does not exist.
    Raises RuntimeError if no .pdf/.txt/.md files are found in docs_dir.
    Raises RuntimeError if no registered documents are successfully loaded.
    """
    docs_dir = docs_dir or DOCS_DIR
    if not docs_dir.exists():
        raise FileNotFoundError(f"Medical docs directory not found: {docs_dir}")

    manifest = _load_manifest(docs_dir)
    all_units: list[dict[str, Any]] = []

    supported = (
        list(docs_dir.glob("*.pdf")) +
        list(docs_dir.glob("*.txt")) +
        list(docs_dir.glob("*.md"))
    )
    # Only document extensions — JSON/other files already excluded by glob
    supported = [p for p in supported if p.suffix in {".pdf", ".txt", ".md"}]

    if not supported:
        raise RuntimeError(
            f"No .pdf/.txt/.md files found in {docs_dir}. "
            "Run: python scripts/download_medical_sources.py"
        )

    rejected_count = 0
    for path in sorted(supported):
        entry = manifest.get(path.name)
        if entry is None:
            # Enforce trusted-source-only: reject unregistered files
            logger.warning(
                "REJECTED unregistered document: %s — not in source_manifest.json. "
                "Only manifest-registered sources may enter the knowledge base.",
                path.name,
            )
            rejected_count += 1
            continue

        logger.info("Loading document: %s (doc_id=%s)", path.name, entry["id"])
        if path.suffix == ".pdf":
            units = _load_pdf(path, entry)
        else:
            # Both .txt and .md are plain-text formats
            units = _load_txt(path, entry)
        all_units.extend(units)

    if rejected_count:
        logger.warning(
            "%d file(s) rejected as unregistered. "
            "Add them to source_registry.json and re-run the downloader.",
            rejected_count,
        )

    if not all_units:
        raise RuntimeError(
            f"No registered documents were loaded from {docs_dir}. "
            "Ensure source_manifest.json is populated and files exist. "
            "Run: python scripts/download_medical_sources.py"
        )

    logger.info("Loaded %d document units from %d files", len(all_units), len(supported) - rejected_count)
    return all_units
