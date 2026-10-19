import json
from types import SimpleNamespace
import pytest
from healthcare_app.documents import extract_document
from healthcare_app.history import extract_events
from healthcare_app.analysis import build_analysis, export_markdown, answer_question
from healthcare_app.summary import local_summary, synthesize_summary, render_summary, claim_library
from healthcare_app.llm import LLMError


def doc(text, label="Report 1"):
    return extract_document(text.encode(), "fictional.txt", label).to_dict()


@pytest.mark.parametrize("text,name,status", [
    ("Diagnosis: Type 1 diabetes", "type 1 diabetes", "Documented in report"),
    ("Diagnosis: No nephropathy", "nephropathy", "Negated in report"),
    ("Diagnosis: Possible neuropathy", "neuropathy", "Uncertain in report"),
    ("Diagnosis: Rule out anemia", "anemia", "Uncertain in report"),
    ("Diagnosis: nephropathy not excluded", "nephropathy", "Uncertain in report"),
    ("Diagnosis: hypertension ruled out", "hypertension", "Negated in report"),
    ("Past medical history: history of diabetic ketoacidosis", "diabetic ketoacidosis", "Historical in report"),
    ("Family history: diabetes", "diabetes", "Family history"),
    ("Family history:\n- hypertension", "hypertension", "Family history"),
    ("Medication: stopped metformin", "metformin", "Stopped/held in report"),
    ("Medication: not taking insulin", "insulin", "Negated in report"),
    ("Medication: insulin glargine 12 units nightly", "insulin glargine", "Listed in report; current use unconfirmed"),
])
def test_documented_assertion_scope(text, name, status):
    events, _ = extract_events([doc(text)])
    assert len(events) == 1 and events[0]["name"] == name and events[0]["status"] == status
    assert events[0]["page"] == 1 and events[0]["quote"]


def test_negation_ends_at_contrast_and_clause():
    events, _ = extract_events([doc("Diagnosis: no nephropathy; hypertension\nDiagnosis: no anemia but hypothyroidism")])
    states = {e["name"]: e["status"] for e in events}
    assert states == {"nephropathy": "Negated in report", "hypertension": "Documented in report", "anemia": "Negated in report", "hypothyroidism": "Documented in report"}


def test_no_diagnosis_from_educational_text_or_abnormal_lab():
    events, _ = extract_events([doc("Diabetes can damage kidneys.\nGlucose 240 mg/dL 70-99\nInsulin 20 uIU/mL")])
    assert events == []


def test_medication_doses_do_not_cross_drug_boundaries():
    events, _ = extract_events([doc("Medication: insulin glargine 12 units nightly; metformin 500 mg twice")])
    assert [e["documented_dose"] for e in events] == ["12 units nightly", "500 mg twice"]
    complex_doses, _ = extract_events([doc("Medication: insulin glargine 12-16 units nightly; metformin 500 mg twice daily")])
    assert [e["documented_dose"] for e in complex_doses] == ["12-16 units nightly", "500 mg twice daily"]


def test_unknown_labelled_entities_remain_reviewable():
    events, _ = extract_events([doc("Diagnosis: rare undocumented condition\nMedication: unusualbrand 20 mg")])
    assert len(events) == 2
    assert all(not e["recognized_name"] and e["issues"] for e in events)
    assert events[1]["documented_dose"] == "20 mg"
    mixed, _ = extract_events([doc("Diagnosis: diabetes; rarecondition\nMedication: metformin 500 mg; unusualbrand 20 mg")])
    assert len(mixed) == 4 and sum(not e["recognized_name"] for e in mixed) == 2
    unknown, _ = extract_events([doc("Diagnosis: no pneumonia; possible rarecondition")])
    assert [e["status"] for e in unknown] == ["Negated in report", "Uncertain in report"]


def test_timeline_conflicts_unknown_dates_and_duplicate_sources():
    a = doc("Report date: 2026-01-01\nDiagnosis: hypertension\nGlucose 140 mg/dL 70-99", "Report 1")
    b = doc("Report date: 2026-01-01\nDiagnosis: no hypertension\nMedication: insulin glargine 12 units nightly", "Report 2")
    c = doc("Diagnosis: Type 1 diabetes", "Report 3")
    analysis = build_analysis([a, b, c], user_reviewed=True)
    assert len(analysis["timeline"]) == 5 and analysis["timeline"][-1]["date"] is None
    assert any("Conflicting" in w for w in analysis["warnings"])
    assert "onset" in export_markdown(analysis).lower()
    assert analysis["clinical_events"][0]["date_basis"].startswith("Document date")


class FakeClient:
    def __init__(self, plan):
        self.plan, self.calls = plan, []
        self.chat = SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps(self.plan)))])


def reviewed_analysis():
    return build_analysis([doc("Patient: Private Person\nReport date: 2026-10-01\nGlucose 146 mg/dL 70-99\nDiagnosis: Type 1 diabetes; no nephropathy\nMedication: insulin glargine 12 units nightly\nDiagnosis: PrivateCustomText")], user_reviewed=True)


def test_llm_summary_complete_schema_minimized_payload_and_citations():
    analysis = reviewed_analysis()
    plan = {"sections": local_summary(analysis)["sections"]}
    client = FakeClient(plan)
    summary = synthesize_summary(analysis, "fake-key", True, client)
    request = client.calls[0]
    assert request["model"] == "gpt-6-astra" and request["reasoning_effort"] == "low"
    assert "temperature" not in request and request["store"] is False
    assert request["response_format"]["json_schema"]["strict"] is True
    payload = request["messages"][1]["content"]
    for sensitive in ["Private Person", "2026-10-01", "PrivateCustomText", "12 units", "fictional.txt"]:
        assert sensitive not in payload
    text = render_summary(analysis, summary)
    assert "Negated in report" in text and "NIDDK-T1D" in text and "What is uncertain" in text
    assert "146" in text and "no individual clinical risk" in text


@pytest.mark.parametrize("mutation", ["extra_prose", "unknown_claim", "duplicate", "omission", "wrong_section"])
def test_generated_summary_rejects_unsupported_and_missing_claims(mutation):
    analysis = reviewed_analysis()
    plan = {"sections": local_summary(analysis)["sections"]}
    if mutation == "extra_prose":
        plan["text"] = "Increase insulin"
    elif mutation == "unknown_claim":
        plan["sections"][0]["claim_ids"].append("FAKE")
    elif mutation == "duplicate":
        plan["sections"][0]["claim_ids"].append(plan["sections"][0]["claim_ids"][0])
    elif mutation == "omission":
        plan["sections"][0]["claim_ids"].pop()
    else:
        plan["sections"][1]["claim_ids"].append(plan["sections"][0]["claim_ids"].pop())
    with pytest.raises(LLMError):
        synthesize_summary(analysis, "fake-key", True, FakeClient(plan))


def test_summary_review_and_api_consent_fail_before_call():
    client = FakeClient({})
    with pytest.raises(LLMError):
        synthesize_summary(reviewed_analysis(), "fake-key", False, client)
    with pytest.raises(LLMError):
        synthesize_summary(build_analysis([doc("Glucose 100 mg/dL")]), "fake-key", True, client)
    assert client.calls == []


def test_missing_optional_sdk_has_safe_local_fallback(monkeypatch):
    import builtins
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("SDK absent in core environment")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(LLMError, match="no API request"):
        synthesize_summary(reviewed_analysis(), "fake-key", True)


def test_medication_education_preserves_documented_status_and_no_dosing():
    answer = answer_question("Explain insulin medication", reviewed_analysis())
    assert "insulin glargine" in answer and "current use unconfirmed" in answer
    assert "12 units" not in answer
    assert "cannot" in answer_question("Should I stop taking insulin?", reviewed_analysis())
