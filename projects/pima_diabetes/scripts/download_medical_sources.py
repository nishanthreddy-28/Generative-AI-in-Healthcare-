"""
Medical Source Downloader
=========================
Downloads only registry-approved sources from data/medical_docs/source_registry.json.

Security constraints:
- Only downloads URLs listed in source_registry.json (no arbitrary URLs)
- Enforces HTTPS-only
- Validates domain against per-entry allowlist (both initial AND final redirect URLs)
- Enforces 10 MB file size cap
- Enforces 30 s connection timeout
- For PDF: saves as-is; for HTML pages: extracts main text content deterministically
- Preserves HTTP Content-Type and both raw and extracted checksums
- Derives precise UTC retrieval timestamp
- Writes source_manifest.json with full provenance
- Exits with error if zero documents are retrieved
"""

import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH   = PROJECT_ROOT / "data" / "medical_docs" / "source_registry.json"
DOCS_DIR        = PROJECT_ROOT / "data" / "medical_docs"
MANIFEST_PATH   = DOCS_DIR / "source_manifest.json"

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB
REQUEST_TIMEOUT_S   = 30
RETRY_ATTEMPTS      = 2
RETRY_DELAY_S       = 3

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk_bytes in iter(lambda: f.read(65536), b""):
            h.update(chunk_bytes)
    return h.hexdigest()


def validate_url(url: str, allowed_domains: list[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"URL must use HTTPS: {url}")
    hostname = parsed.hostname or ""
    if not any(hostname == d or hostname.endswith("." + d) for d in allowed_domains):
        raise ValueError(
            f"Domain '{hostname}' not in allowed list {allowed_domains} for URL {url}"
        )


def verify_copyright_terms(soup, url: str) -> str:
    """Attempt to extract exact page copyright or terms."""
    # Check for specific strings or tags indicating terms/copyright on the exact page.
    text = soup.get_text().lower()
    if "public domain" in text:
        return "Found 'public domain' in page text"
    if "copyright" in text or "©" in text:
        return "Page contains explicit copyright/© notice"
    
    return "Terms/Copyright undetermined from exact page"


def extract_main_text_from_html(html: str, source_url: str, publisher: str,
                                 title: str, retrieval_date: str) -> str:
    """
    Extract readable text deterministically from HTML pages.
    No rewriting of medical content — only whitespace normalisation and
    removal of boilerplate nav/script/style tags.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "noscript", "form", "button", "iframe"]):
        tag.decompose()

    # Try to find main content area deterministically
    main = soup.find("main")
    if not main:
        main = soup.find(id=re.compile(r"main|content|body", re.I))
    if not main:
        main = soup.find(class_=re.compile(r"main|content|body", re.I))
    if not main:
        main = soup.body
    if not main:
        main = soup

    raw_text = main.get_text(separator="\n")
    # Normalise whitespace deterministically
    lines = [line.strip() for line in raw_text.splitlines()]
    lines = [l for l in lines if l]
    text = "\n".join(lines)
    
    terms_note = verify_copyright_terms(soup, source_url)

    # Prepend provenance header — clearly labels the source
    header = (
        f"SOURCE: {title}\n"
        f"PUBLISHER: {publisher}\n"
        f"URL: {source_url}\n"
        f"RETRIEVAL DATE: {retrieval_date}\n"
        f"EXACT PAGE TERMS: {terms_note}\n"
        f"{'=' * 70}\n\n"
    )
    return header + text


def download_with_retries(session, url: str) -> tuple[bytes, str, str]:
    import requests

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_S, stream=True)
            response.raise_for_status()

            # Check Content-Length header if present
            cl = response.headers.get("Content-Length")
            if cl and int(cl) > MAX_FILE_SIZE_BYTES:
                raise ValueError(f"Content-Length {cl} exceeds 10 MB limit")

            chunks = []
            total = 0
            for chunk_bytes in response.iter_content(chunk_size=65536):
                total += len(chunk_bytes)
                if total > MAX_FILE_SIZE_BYTES:
                    raise ValueError(f"Download exceeded 10 MB size limit")
                chunks.append(chunk_bytes)
                
            return b"".join(chunks), response.url, response.headers.get("Content-Type", "unknown")

        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_ATTEMPTS} failed: {e}")
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"All {RETRY_ATTEMPTS} download attempts failed for: {url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import requests

    print("=" * 65)
    print("Medical Source Downloader")
    print("Only downloads entries in source_registry.json")
    print("=" * 65)

    if not REGISTRY_PATH.exists():
        print(f"[ERROR] source_registry.json not found at {REGISTRY_PATH}")
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        registry = json.load(f)

    print(f"\nRegistry contains {len(registry)} approved sources.")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "PimaDiabetesMLResearch/1.0 (educational project; "
            "contact: research@example.com)"
        )
    })

    manifest_entries = []
    success_count = 0
    # Precise UTC runtime timestamp
    retrieval_date = datetime.now(timezone.utc).isoformat()

    for entry in registry:
        doc_id    = entry["id"]
        publisher = entry["publisher"]
        title     = entry["title"]
        url       = entry["url"]
        allowed   = entry["allowed_domains"]
        exp_type  = entry["expected_type"]

        print(f"\n[{doc_id}]")
        print(f"  Publisher : {publisher}")
        print(f"  Title     : {title}")
        print(f"  URL       : {url}")

        try:
            # Validate initial URL
            validate_url(url, allowed)
        except ValueError as e:
            print(f"  [SKIP] Initial URL validation failed: {e}")
            continue

        try:
            raw_bytes, final_url, content_type = download_with_retries(session, url)
            
            # Validate final redirect URL
            validate_url(final_url, allowed)
        except Exception as e:
            print(f"  [FAIL] Download or redirect error: {e}")
            continue

        raw_checksum = sha256_bytes(raw_bytes)

        # Determine output filename and content
        if exp_type == "pdf":
            local_filename = f"{doc_id}.pdf"
            output_path = DOCS_DIR / local_filename
            output_path.write_bytes(raw_bytes)
        else:  # html_to_text
            local_filename = f"{doc_id}.txt"
            output_path = DOCS_DIR / local_filename
            html = raw_bytes.decode("utf-8", errors="replace")
            text_content = extract_main_text_from_html(
                html, final_url, publisher, title, retrieval_date
            )
            output_path.write_text(text_content, encoding="utf-8")

        extracted_checksum = sha256_file(output_path)
        file_size = output_path.stat().st_size

        print(f"  [OK] Saved: {local_filename} ({file_size:,} bytes)")
        print(f"       Extracted SHA-256: {extracted_checksum[:16]}...")

        manifest_entries.append({
            "id":                  doc_id,
            "publisher":           publisher,
            "title":               title,
            "initial_url":         url,
            "final_url":           final_url,
            "http_content_type":   content_type,
            "local_filename":      local_filename,
            "expected_type":       exp_type,
            "raw_checksum_sha256": raw_checksum,
            "extracted_checksum_sha256": extracted_checksum,
            "file_size_bytes":     file_size,
            "retrieval_date_utc":  retrieval_date,
        })
        success_count += 1

    # Write manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"\n{'=' * 65}")
    print(f"Downloaded: {success_count}/{len(registry)} sources")
    print(f"Manifest  : {MANIFEST_PATH}")

    if success_count == 0:
        print(
            "\n[INDEXING BLOCKED] No approved medical documents were successfully downloaded.\n"
            "Check your internet connection and approved URLs in source_registry.json.\n"
            "Indexing cannot proceed without medical source documents."
        )
        sys.exit(1)

    print("\n[DONE] Medical sources downloaded. Run: python scripts/build_index.py")


if __name__ == "__main__":
    main()
