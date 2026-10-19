"""Facts, provenance, bounded explanations and non-diagnostic dated comparisons."""
import hashlib
import json
import re
from .knowledge import KnowledgeBase
from .records import extract_findings, normalize_unit
from .history import extract_events, build_timeline


def input_fingerprint(documents):
    return hashlib.sha256(json.dumps(documents, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def format_fact(finding):
    return f"{finding['name']}: {finding['comparator']}{finding['value']:g} {finding['unit'] or '(unit missing)'} — {finding['status']}. [{finding['id']}]"


def attach_simulation(analysis, result):
    """Attach only after UI same-patient confirmation and fingerprint match."""
    evidence = {d["id"]: d for d in analysis["evidence"]}
    evidence.update({d["id"]: d for d in result["evidence"]})
    summary = {"id": "S1", "kind": "experimental_simulation", "selected_model": result["selected_model"],
               "observed_mean_glucose": result["observed_mean_glucose"],
               "simulated_mean_glucose": float(result["point"].glucose_mean.mean()),
               "artifact_fingerprint": result["artifact_fingerprint"], "evidence_ids": [d["id"] for d in result["evidence"]]}
    summary["text"] = f"The linked CGM history has an observed average of daily mean glucose of {summary['observed_mean_glucose']:.2f} mg/dL. The {summary['selected_model']} research simulation gives a next-60-day central average of {summary['simulated_mean_glucose']:.2f} mg/dL. Future values are simulated, not measured; this does not establish individual accuracy, clinical risk or a basis for changing medicine. [S1]"
    return {**analysis, "evidence": list(evidence.values()), "simulation_summary": summary}


def build_analysis(documents, knowledge=None, *, user_reviewed=False):
    kb = knowledge or KnowledgeBase()
    if not documents or len(documents) > 5:
        raise ValueError("Analyze between one and five reports for the same patient.")
    if sum(len(p["text"]) for d in documents for p in d["pages"]) > 200_000:
        raise ValueError("Combined report text exceeds 200,000 characters.")
    findings, warnings = extract_findings(documents)
    events, event_warnings = extract_events(documents)
    timeline, history_warnings = build_timeline(findings, events)
    warnings.extend(event_warnings + history_warnings)
    warnings.extend(f"{event['id']}: {issue}" for event in events for issue in event["issues"])
    if len(findings) > 250:
        raise ValueError("Too many results for one review. Use smaller reports (at most 250 results).")
    evidence = {"NLM-LAB": kb.by_id["NLM-LAB"].copy()}
    explanations = []
    for finding in findings:
        references = kb.for_test(finding["test"])
        for reference in references:
            evidence[reference["id"]] = reference
        explanations.append({"finding_id": finding["id"], "evidence_ids": [d["id"] for d in references], "fact": format_fact(finding)})
    event_explanations = []
    for event in events:
        references = kb.for_event(event)
        for reference in references:
            evidence[reference["id"]] = reference
        event_explanations.append({"event_id": event["id"], "evidence_ids": [d["id"] for d in references]})
    trends = []
    for test in sorted({f["test"] for f in findings}):
        dated = [f for f in findings if f["test"] == test and f["date"] and f["unit"] and not f["comparator"] and f["status"] not in {"Needs review", "Uncertain"}]
        units = {normalize_unit(f["unit"]) for f in dated}
        if len(units) > 1:
            warnings.append(f"{dated[0]['name']}: units differ between reports; values were not compared or converted.")
            continue
        dated.sort(key=lambda f: (f["date"], f["id"]))
        unique_dates = {f["date"] for f in dated}
        if len(unique_dates) < 2:
            continue
        if len(dated) != len(unique_dates):
            warnings.append(f"{dated[0]['name']}: multiple results on the same date; resolve them before comparing.")
            continue
        first, last = dated[0], dated[-1]
        difference = last["value"] - first["value"]
        direction = "increased" if difference > 0 else "decreased" if difference < 0 else "was unchanged"
        trends.append({"test": test, "name": first["name"], "finding_ids": [first["id"], last["id"]], "text": f"{first['name']} {direction} from {first['value']:g} to {last['value']:g} {last['unit']} between {first['date']} and {last['date']}. [{first['id']}, {last['id']}] This is a numeric comparison, not a conclusion about improvement or deterioration; methods and fasting status may differ."})
    questions = ["Which findings matter in the context of my symptoms, history and medicines?", "Do any measurements need repeating, and under what preparation conditions?", "What reference ranges or personal targets apply to me?"]
    if any(f["status"] in {"Above report range", "Below report range"} for f in findings):
        questions.insert(0, "How should the results outside the report's ranges be followed up?")
    warnings.extend(issue for f in findings for issue in f["issues"])
    return {
        "schema_version": 2, "user_reviewed": bool(user_reviewed), "fingerprint": input_fingerprint(documents), "knowledge_version": kb.fingerprint,
        "report_count": len(documents), "findings": findings, "explanations": explanations,
        "clinical_events": events, "event_explanations": event_explanations, "timeline": timeline,
        "retrieval_method": kb.retrieval_method,
        "evidence": list(evidence.values()), "trends": trends, "questions": questions,
        "warnings": list(dict.fromkeys(warnings)),
        "limits": ["Explicit diagnoses and medications are documented mentions, not independently verified diagnoses or instructions to take medicine. Review negation, uncertainty, family history, names and doses against the originals.", "Event dates are document dates; onset, medication start and adherence are unconfirmed. Unsupported narrative and imaging require clinical interpretation.", "One dated report is a snapshot. Multiple dated values permit limited comparisons; neither supplies the full daily history required by the diabetes research models.", "No prediction is generated from sparse laboratory reports. The corrected glucose simulation requires 30 consecutive days of sufficiently covered measured CGM. The separate legacy models require complete multimodal observations. Both remain experimental; no individual clinical risk is estimated."],
    }


def export_markdown(analysis):
    reviewed = analysis.get("user_reviewed", False)
    review_label = "User-reviewed extraction" if reviewed else "Extraction NOT user-reviewed: verify values against the original report."
    lines = ["# Medical report explanation", "", "Educational research output; review with a qualified clinician.", "", review_label, "", "## Findings", ""]
    lines.extend("- " + e["fact"] for e in analysis["explanations"])
    if not analysis["findings"]:
        lines.append("No supported numeric results were recognized. No medical conclusion was inferred.")
    lines += ["", "## Source locations", ""]
    for f in analysis["findings"]:
        lines.append(f"- [{f['id']}] {f['source']}, page {f['page']}, line {f['line']}; date: {f['date'] or 'unknown'}. {review_label}")
    lines += ["", "## Documented clinical events", ""]
    for event in analysis.get("clinical_events", []):
        lines.append(f"- [{event['id']}] {event['kind']}: {event['name']} — {event['status']}. Documented dose: {event['documented_dose'] or 'not recorded'}. {event['source']}, page {event['page']}, line {event['line']}; document date: {event['date'] or 'unknown'}. Onset/current use unconfirmed.")
    lines += ["", "## Patient timeline", ""]
    for event in analysis.get("timeline", []):
        lines.append(f"- {event['date'] or 'Undated'} [{event['id']}] {event['name']}: {event['detail']} — {event['status']}.")
    lines += ["", "## Evidence and explanations", ""]
    for d in analysis["evidence"]:
        lines.append(f"- [{d['id']}] [{d['title']}]({d['url']}): {d['content']} (Source checked {d['reviewed']}; locally authored educational paraphrase.)")
    lines += ["", "## Dated comparisons", ""] + [t["text"] for t in analysis["trends"]]
    if analysis.get("simulation_summary"):
        lines += ["", "## Linked CGM research simulation", "", analysis["simulation_summary"]["text"],
                  "Artifact fingerprint: " + analysis["simulation_summary"]["artifact_fingerprint"]]
    lines += ["", "## Questions for your clinician", ""] + ["- " + q for q in analysis["questions"]]
    lines += ["", "## Extraction warnings", ""] + ["- " + q for q in analysis["warnings"]]
    lines += ["", "## Limits", ""] + ["- " + q for q in analysis["limits"]]
    return "\n".join(lines)


def answer_question(question, analysis, knowledge=None):
    if not isinstance(question, str) or not question.strip() or len(question) > 2000:
        raise ValueError("Enter a question of 1 to 2,000 characters.")
    if re.search(r"\b(dose|dosing|titrate|titration|prescribe|prescription|emergency|diagnos\w*)\b|\b(?:should i|how much|can i|increase|decrease|stop|start|change)\b.*\b(?:insulin|medication|medicine|take)\b", question, re.I):
        return "This application cannot diagnose conditions, assess emergencies or recommend medication changes. Discuss treatment and follow-up with your clinician. If you feel acutely unwell, seek medical care rather than relying on this report explanation."
    kb = knowledge or KnowledgeBase()
    relevant = kb.retrieve(question, top_k=3)
    if not relevant:
        return "I could not find sufficiently relevant evidence in the local knowledge base. Rephrase using a test name, or ask your clinician. No answer was inferred."
    test_ids = {test for d in relevant for test in d["tests"]}
    selected = [f for f in analysis["findings"] if f["test"] in test_ids]
    lines = [format_fact(f) for f in selected]
    for event in analysis.get("clinical_events", []):
        if event["recognized_name"] and any(event["name"] in d.get("events", []) or event["kind"] == "medication" and d["id"] == "NLM-MEDICINES" for d in relevant):
            lines.append(f"[{event['id']}] Report documents {event['name']}: {event['status']}. This mention does not establish current treatment or confirm a diagnosis.")
    if not selected:
        lines.append("No matching measurement was recognized in your reviewed reports. The following is general educational context.")
    lines.extend(f"{d['content']} [{d['id']}: {d['title']}]({d['url']})" for d in relevant)
    lines.append("Interpret these facts with your clinician; this answer does not infer a diagnosis or a future outcome.")
    return "\n\n".join(lines)
