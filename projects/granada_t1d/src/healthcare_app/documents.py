"""Bounded local extraction. Temporary OCR files are removed on success/failure."""
from dataclasses import dataclass, asdict
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import warnings

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 20
MAX_TEXT = 100_000
MAX_PIXELS = 16_000_000


class DocumentError(ValueError):
    pass


@dataclass
class Document:
    label: str
    name: str
    digest: str
    pages: list[dict]
    method: str
    warnings: list[str]

    def to_dict(self):
        return asdict(self)


def _run(args):
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run(args, capture_output=True, timeout=60, **options)
    except subprocess.TimeoutExpired as exc:
        raise DocumentError("Local OCR timed out. Use a clearer, smaller report or paste its text.") from exc
    if result.returncode:
        # Never expose subprocess output, which may include sensitive content/paths.
        raise DocumentError("Local OCR could not read this document. Paste its text or use a clearer image.")
    return result.stdout.decode("utf-8-sig", errors="replace")


def ocr_capabilities():
    return {
        "windows_ocr": os.name == "nt" and bool(shutil.which("powershell.exe")),
        "tesseract": bool(shutil.which("tesseract")),
        "pdf_renderer": bool(shutil.which("pdftoppm")),
    }


def _ocr(data, extension, page_numbers=None):
    with tempfile.TemporaryDirectory(prefix="health_report_") as temp:
        path = Path(temp) / ("report" + extension)
        path.write_bytes(data)
        if ocr_capabilities()["windows_ocr"]:
            output = _run([
                "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
                str(Path(__file__).with_name("windows_ocr.ps1")), "-InputPath", str(path.resolve()),
                "-Pages", ",".join(str(p - 1) for p in (page_numbers or [])),
            ])
            try:
                result = json.loads(output)
                if not isinstance(result, list):
                    result = [result]
                return result
            except (ValueError, TypeError) as exc:
                raise DocumentError("OCR returned an unreadable response. Paste the report text instead.") from exc
        if not shutil.which("tesseract"):
            raise DocumentError("Image/scanned-PDF reading needs local OCR. On this host OCR is unavailable; paste report text instead.")
        if extension != ".pdf":
            return [{"page": 1, "text": _run(["tesseract", str(path), "stdout", "-l", "eng", "--psm", "6"])}]
        if not shutil.which("pdftoppm"):
            raise DocumentError("Scanned PDF reading needs pdftoppm and Tesseract on this host. Paste report text instead.")
        pages = []
        for number in page_numbers or []:
            prefix = str(Path(temp) / f"page_{number}")
            _run(["pdftoppm", "-f", str(number), "-l", str(number), "-r", "150", "-scale-to", "2000", "-png", "-singlefile", str(path), prefix])
            pages.append({"page": number, "text": _run(["tesseract", prefix + ".png", "stdout", "-l", "eng", "--psm", "6"])})
        return pages


def extract_document(data: bytes, name: str, label="Report 1") -> Document:
    if not isinstance(data, bytes) or not data:
        raise DocumentError("The report is empty.")
    if len(data) > MAX_BYTES:
        raise DocumentError("Each report must be at most 10 MB.")
    extension = Path(name).suffix.lower()
    notes = []
    method = "text"
    if extension == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding.") from exc
        if "\x00" in text:
            raise DocumentError("This file does not look like plain text.")
        pages = [{"page": 1, "text": text}]
    elif extension == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise DocumentError("The uploaded file is not a valid PDF.")
        from pypdf import PdfReader
        try:
            reader = PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise DocumentError("Password-protected PDFs are unsupported. Upload an unlocked copy.")
            if not 0 < len(reader.pages) <= MAX_PAGES:
                raise DocumentError("Reports must contain 1 to 20 pages.")
            pages = [{"page": i + 1, "text": page.extract_text(extraction_mode="layout") or ""} for i, page in enumerate(reader.pages)]
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("The PDF is damaged or unsupported. Upload another copy or paste the text.") from exc
        # Low-text pages may be scanned. Preserve every page and never silently omit one.
        scanned = [p["page"] for p in pages if len(p["text"].strip()) < 30]
        method = "PDF text"
        if scanned:
            replacements = {p["page"]: p["text"] for p in _ocr(data, extension, scanned)}
            for p in pages:
                if p["page"] in replacements:
                    p["text"] = replacements[p["page"]]
            method = "PDF text + local OCR"
            notes.append("OCR was used. Check every value, unit, decimal point and reference range against the original.")
        # Mixed image/text pages can still have missed information.
        notes.append("Check the extracted text against all pages; tables and text embedded in images may need manual correction.")
    elif extension in {".png", ".jpg", ".jpeg"}:
        from PIL import Image
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(data)) as image:
                    if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                        raise DocumentError("Use a single image with at most 16 million pixels.")
                    if image.format not in {"PNG", "JPEG"}:
                        raise DocumentError("Only genuine PNG and JPEG images are supported.")
                    image.verify()
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError("The image is damaged, too large, or unsupported.") from exc
        pages = _ocr(data, extension)
        method = "local OCR"
        notes.append("OCR was used. Check every value, unit, decimal point and reference range against the original.")
    else:
        raise DocumentError("Supported formats are PDF, PNG, JPEG and UTF-8 TXT.")
    text = "\n".join(str(p.get("text", "")) for p in pages)
    if len(text) > MAX_TEXT:
        raise DocumentError("The report exceeds 100,000 extracted characters. Split it into smaller reports.")
    if not text.strip():
        raise DocumentError("No readable text was found. Use a clearer report or paste its text.")
    return Document(label, Path(name).name, hashlib.sha256(data).hexdigest(), pages, method, notes)
