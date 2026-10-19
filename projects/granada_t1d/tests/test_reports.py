import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, DictionaryObject
from healthcare_app.documents import extract_document, DocumentError, MAX_BYTES
from healthcare_app.records import parse_date, extract_findings
from healthcare_app.analysis import build_analysis, answer_question, export_markdown
from healthcare_app.knowledge import KnowledgeBase
from healthcare_app.llm import select_explanations, LLMError


def document(text, label="Report 1"):
    return extract_document(text.encode(), "report.txt", label).to_dict()


def test_export_distinguishes_unreviewed_cli_from_reviewed_ui():
    docs = [document("Glucose 146 mg/dL 70-99")]
    unreviewed = build_analysis(docs)
    assert unreviewed["user_reviewed"] is False
    assert "NOT user-reviewed" in export_markdown(unreviewed)
    assert "User-reviewed extraction" not in export_markdown(unreviewed)
    reviewed = build_analysis(docs, user_reviewed=True)
    assert reviewed["user_reviewed"] is True
    assert "User-reviewed extraction" in export_markdown(reviewed)
    assert "NOT user-reviewed" not in export_markdown(reviewed)


def pdf_bytes(text=None, encrypted=False, pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(600, 800)
        if text:
            font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
            content = DecodedStreamObject()
            lines = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").splitlines()
            content.set_data(("BT /F1 14 Tf 40 750 Td " + " 0 -24 Td ".join(f"({line}) Tj" for line in lines) + " ET").encode())
            page[NameObject("/Contents")] = writer._add_object(content)
    if encrypted:
        writer.encrypt("test-password")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("data,name", [(b"", "x.txt"), (b"bad", "x.pdf"), (b"bad", "x.png"), (b"\xff", "x.txt"), (b"abc\x00", "x.txt"), (b"abc", "x.exe"), (b"x" * (MAX_BYTES + 1), "x.txt")], ids=["empty", "invalid-pdf", "invalid-image", "invalid-encoding", "binary-text", "unsupported-type", "oversized"])
def test_reject_invalid_documents(data, name):
    with pytest.raises(DocumentError):
        extract_document(data, name)


def test_pdf_actual_text_and_provenance():
    text = "Report date: 2026-10-01\nGlucose 146 mg/dL 70-99\nHbA1c 8.2 % 4.0-5.6"
    doc = extract_document(pdf_bytes(text), "test.pdf").to_dict()
    findings, _ = extract_findings([doc])
    assert len(findings) == 2
    assert findings[0]["value"] == 146 and findings[0]["page"] == 1
    assert findings[1]["ref_high"] == 5.6


@pytest.mark.parametrize("data", [pdf_bytes(encrypted=True), pdf_bytes(pages=21), b"%PDF-broken"])
def test_reject_protected_large_or_broken_pdf(data):
    with pytest.raises(DocumentError):
        extract_document(data, "test.pdf")


@pytest.mark.parametrize("text,date", [("2026-10-01", "2026-10-01"), ("13/10/2026", "2026-10-13"), ("1 Oct 2026", "2026-10-01"), ("10/01/2026", None), ("2026-02-30", None), ("02/30/2026", None)])
def test_unambiguous_dates(text, date):
    assert parse_date(text) == date


@pytest.mark.parametrize("line,status", [
    ("Glucose 99 mg/dL 70-99", "Within report range"),
    ("Glucose 100 mg/dL 70-99", "Above report range"),
    ("Glucose 60 mg/dL 70-99", "Below report range"),
    ("Glucose 100 mg/dL <100", "Above report range"),
    ("Glucose 100 mg/dL <=100", "Within report range"),
    ("Glucose 100 70-99", "Uncertain"),
    ("Glucose <50 mg/dL 70-99", "Uncertain"),
    ("HbA1c 8.2 mg/dL 4-6", "Uncertain"),
    ("Glucose 146 mg/dL 3.9-5.5 mmol/L", "Uncertain"),
    ("Glucose -4 mg/dL 70-99", "Needs review"),
    ("Glucose 90 mg/dL 99-70", "Needs review"),
    ("Glucose 90 mg/dL", "Range unavailable"),
])
def test_range_comparisons(line, status):
    findings, _ = extract_findings([document(line)])
    assert len(findings) == 1 and findings[0]["status"] == status


def test_snapshot_no_fabricated_history_or_forecast():
    result = build_analysis([document("HbA1c 8.2 % 4-6")])
    assert result["trends"] == []
    assert result["findings"][0]["date"] is None
    assert "NLM-A1C" in result["explanations"][0]["evidence_ids"]
    assert "No prediction" in export_markdown(result)


def test_narrative_and_negation_do_not_create_diagnoses():
    result = build_analysis([document("No nephropathy. No hypertension. MRI impression: nonspecific findings.")])
    assert result["findings"] == [] and result["warnings"]


def test_nonfinite_and_scientific_notation_never_become_false_values():
    result = build_analysis([document("Glucose " + "9" * 2000 + " mg/dL 70-99\nGlucose 1e5 mg/dL 70-99")])
    assert result["findings"] == [] and result["warnings"]
    json.dumps(result, allow_nan=False)


def test_numeric_comparison_and_unit_mismatch():
    a = document("Report date: 2026-01-01\nGlucose 146 mg/dL 70-99", "Report 1")
    b = document("Report date: 2026-10-01\nGlucose 120 mg/dL 70-99", "Report 2")
    result = build_analysis([a, b])
    assert len(result["trends"]) == 1
    assert "decreased" in result["trends"][0]["text"] and "improvement" in result["trends"][0]["text"]
    b = document("Report date: 2026-10-01\nGlucose 6 mmol/L 3.9-5.5", "Report 2")
    result = build_analysis([a, b])
    assert result["trends"] == [] and any("units differ" in w for w in result["warnings"])


def test_duplicate_date_abstains():
    a = document("Report date: 2026-01-01\nGlucose 146 mg/dL 70-99\nGlucose 145 mg/dL 70-99\nReport date: 2026-10-01\nGlucose 120 mg/dL 70-99")
    assert build_analysis([a])["trends"] == []


def test_retrieval_abstention_and_source_ids():
    kb = KnowledgeBase()
    assert kb.retrieve("zxqv planet football") == []
    assert kb.retrieve("HbA1c average blood glucose")[0]["id"] == "NLM-A1C"
    with pytest.raises(ValueError):
        kb.retrieve("glucose", 0)


def test_chat_safety_and_oov():
    result = build_analysis([document("Glucose 146 mg/dL 70-99")])
    assert "146" in answer_question("Explain glucose", result)
    assert "cannot" in answer_question("How much insulin should I take?", result)
    assert "could not find" in answer_question("zxqv football", result)


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.response)))])


def test_llm_explicit_key_consent_and_data_minimization():
    result = build_analysis([document("Patient: Private Name\nReport date: 2026-10-01\nGlucose 146 mg/dL 70-99")])
    client = FakeClient({"items": [{"finding_id": "F1", "evidence_id": "NLM-GLUCOSE"}]})
    with pytest.raises(LLMError):
        select_explanations(result, "test-key", False, client)
    assert client.calls == []
    plan = select_explanations(result, "test-key", True, client)
    payload = str(client.calls)
    assert "Private Name" not in payload and "2026-10-01" not in payload and "report.txt" not in payload
    assert plan[0]["finding_id"] == "F1"


@pytest.mark.parametrize("response", [
    {"items": [{"finding_id": "F1", "evidence_id": "FAKE"}]},
    {"items": [{"finding_id": "F1", "evidence_id": "NLM-LIVER"}]},
    {"items": []},
    {"items": [{"finding_id": "F1", "evidence_id": "NLM-GLUCOSE", "dose": "10 units"}]},
    {"items": [{"finding_id": "F1", "evidence_id": "NLM-GLUCOSE"}], "text": "diagnosis"},
    {"items": [{"finding_id": "F1", "evidence_id": None}]},
    {"items": [{"finding_id": [], "evidence_id": None}]},
])
def test_llm_rejects_invalid_citations_and_free_medical_claims(response):
    result = build_analysis([document("Glucose 146 mg/dL 70-99")])
    with pytest.raises(LLMError):
        select_explanations(result, "fake-key", True, FakeClient(response))


def test_image_validation_and_ocr_failure_cleanup(monkeypatch):
    from PIL import Image
    from healthcare_app import documents
    stream = io.BytesIO()
    Image.new("RGB", (100, 100), "white").save(stream, format="PNG")
    before = set(Path(__import__('tempfile').gettempdir()).glob("health_report_*"))
    monkeypatch.setattr(documents, "_run", lambda args: (_ for _ in ()).throw(DocumentError("expected")))
    with pytest.raises(DocumentError):
        extract_document(stream.getvalue(), "x.png")
    assert set(Path(__import__('tempfile').gettempdir()).glob("health_report_*")) == before
