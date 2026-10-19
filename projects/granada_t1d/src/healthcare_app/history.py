"""Explicit documented clinical events, assertion scope and provenance.

This is a bounded high-precision extractor, not a diagnostic classifier. Unknown
entities are preserved only in explicitly labelled diagnosis/medication sections.
"""
import re
from .records import parse_date

CONDITIONS = {
    "type 1 diabetes": ["type 1 diabetes", "type i diabetes", "t1dm", "t1d"],
    "type 2 diabetes": ["type 2 diabetes", "type ii diabetes", "t2dm"],
    "diabetes": ["diabetes mellitus", "diabetes"],
    "hypertension": ["hypertension", "high blood pressure"],
    "hypothyroidism": ["hypothyroidism"], "hyperthyroidism": ["hyperthyroidism"],
    "dyslipidemia": ["dyslipidemia", "hyperlipidemia"],
    "nephropathy": ["diabetic nephropathy", "nephropathy"],
    "retinopathy": ["diabetic retinopathy", "retinopathy"],
    "neuropathy": ["diabetic neuropathy", "neuropathy"],
    "chronic kidney disease": ["chronic kidney disease", "ckd"],
    "anemia": ["anaemia", "anemia"],
    "diabetic ketoacidosis": ["diabetic ketoacidosis", "dka"],
}
MEDICINES = {
    "insulin glargine": ["insulin glargine", "glargine", "lantus", "basaglar"],
    "insulin aspart": ["insulin aspart", "aspart", "novorapid", "novolog"],
    "insulin lispro": ["insulin lispro", "lispro", "humalog"],
    "insulin degludec": ["insulin degludec", "degludec", "tresiba"],
    "insulin": ["insulin"], "metformin": ["metformin"],
    "levothyroxine": ["levothyroxine"], "atorvastatin": ["atorvastatin"],
    "rosuvastatin": ["rosuvastatin"], "lisinopril": ["lisinopril"],
    "losartan": ["losartan"], "amlodipine": ["amlodipine"],
    "aspirin": ["aspirin"], "semaglutide": ["semaglutide"],
    "empagliflozin": ["empagliflozin"], "dapagliflozin": ["dapagliflozin"],
}
HEADER = re.compile(r"^(diagnos(?:is|es)|assessment|past medical history|medical history|family history|medications?|medicines?|current medications?|prescriptions?)\s*[:=]\s*(.*)$", re.I)
DATE = re.compile(r"(?:report date|collection date|sample date|collected(?: on)?|date)\s*[:=]\s*(.+)", re.I)


def _matches(text, vocabulary):
    found, spans = [], []
    aliases = sorted(((alias, name) for name, aliases in vocabulary.items() for alias in aliases), key=lambda p: -len(p[0]))
    for alias, name in aliases:
        for match in re.finditer(r"\b" + re.escape(alias) + r"\b", text, re.I):
            if any(match.start() < end and match.end() > start for start, end in spans):
                continue
            found.append((name, match.start(), match.end()))
            spans.append(match.span())
    return sorted(found, key=lambda item: item[1])


def assertion(text, start, end, kind, family=False):
    # Negation/uncertainty scope ends at a semicolon, full stop or contrast.
    prefix = re.split(r"[;.\n]|\b(?:but|however)\b", text[:start], flags=re.I)[-1]
    suffix = re.split(r"[;.\n]|\b(?:but|however)\b", text[end:], flags=re.I)[0]
    scope = prefix + " " + suffix
    if family or re.search(r"\b(family history|mother|father|sister|brother|parent)\b", prefix, re.I):
        return "Family history"
    if re.search(r"\b(no|not|denies|denied|without|negative for|ruled out)\b", prefix, re.I) or re.match(r"\s*(?:is |was )?(?:absent|ruled out|negative)\b", suffix, re.I):
        return "Negated in report"
    if re.search(r"\b(rule out|r/o|possible|suspected|query|consider|uncertain|unlikely)\b", prefix, re.I) or re.match(r"\s*\?", suffix) or re.search(r"\b(?:not excluded|cannot exclude)\b", scope, re.I):
        return "Uncertain in report"
    if kind == "medication":
        if re.search(r"\b(stopped|stop|discontinued|held|hold)\b", scope, re.I):
            return "Stopped/held in report"
        if re.search(r"\b(start|started|continue|taking|takes|current|prescribed)\b", scope, re.I):
            return "Listed in report"
        return "Listed in report; current use unconfirmed"
    if re.search(r"\b(history of|previous|prior|resolved|past)\b", prefix, re.I):
        return "Historical in report"
    return "Documented in report"


def extract_events(documents):
    events, warnings = [], []
    for document in documents:
        current_date, section = None, None
        for page in document["pages"]:
            for line_no, original in enumerate(page["text"].splitlines(), 1):
                text = re.sub(r"\s+", " ", original).strip().strip("| ")
                if not text:
                    section = None
                    continue
                dated = DATE.fullmatch(text)
                if dated:
                    current_date = parse_date(dated.group(1))
                    section = None
                    continue
                header = HEADER.fullmatch(text)
                labelled = bool(header)
                if header:
                    heading, body = header.groups()
                    section = "medication" if re.search("medic|prescri", heading, re.I) and "history" not in heading.lower() else "diagnosis"
                    family = heading.lower() == "family history"
                    section = "family" if family else section
                    text_body = body
                elif re.match(r"^(?:[-*•]|\d+[.)])\s+", text) and section:
                    text_body = re.sub(r"^(?:[-*•]|\d+[.)])\s+", "", text)
                    labelled = True
                else:
                    text_body = text
                    section = None
                if not text_body:
                    continue
                # Only names in explicit clinical statements are processed. A bare
                # disease mention in education is not a patient diagnosis.
                clinical = labelled or bool(re.search(r"\b(diagnosed with|history of|denies|no evidence of|no|suspected|possible|rule out|taking|takes|stopped|continue)\b", text_body, re.I))
                if not clinical:
                    continue
                for kind, vocabulary in [("diagnosis", CONDITIONS), ("medication", MEDICINES)]:
                    if labelled and section in {"medication", "diagnosis", "family"} and kind != ("diagnosis" if section == "family" else section):
                        continue
                    matches = _matches(text_body, vocabulary)
                    if labelled and text_body.lower() not in {"none", "nil", "none reported", "no known diagnoses", "no medications"}:
                        # Preserve unknown names from clear labels, never classify
                        # arbitrary free text as a confirmed disease/drug.
                        offset = 0
                        for clause in re.split(r";", text_body):
                            clause_start, clause_end = offset, offset + len(clause)
                            offset = clause_end + 1
                            if any(clause_start <= match[1] < clause_end for match in matches):
                                continue
                            name = clause.strip(" -*•")[:120]
                            if name:
                                start = clause_start + len(clause) - len(clause.lstrip(" -*•"))
                                assertion_prefix = re.match(r"^(?:no evidence of|no|denies|suspected|possible|rule out|history of|not taking|stopped|continue)\s+", name, re.I)
                                if assertion_prefix:
                                    start += assertion_prefix.end()
                                    name = name[assertion_prefix.end():]
                                if kind == "medication":
                                    name = re.split(r"\s+\d", name, maxsplit=1)[0]
                                matches.append((name, start, start + len(name)))
                        matches.sort(key=lambda match: match[1])
                    for name, start, end in matches:
                        scope_end = min([m[1] for m in matches if m[1] > start] or [len(text_body)])
                        state = assertion(text_body[:scope_end], start, end, kind, section == "family")
                        dose = None
                        if kind == "medication":
                            next_start = min([m[1] for m in matches if m[1] > start] or [len(text_body)])
                            dose_match = re.search(r"\b\d+(?:\.\d+)?(?:\s*(?:-|–|to)\s*\d+(?:\.\d+)?)?\s*(?:mcg|µg|mg|g|units?|iu|ml)\b(?:\s*/\s*(?:\d+(?:\.\d+)?\s*)?(?:mcg|mg|g|units?|iu|ml)\b)?(?:\s*(?:(?:once|twice|three times)(?:\s+(?:a|per))?\s*(?:daily|day|weekly)?|daily|nightly|weekly|at bedtime|bid|tid|od|qhs))?", text_body[end:next_start], re.I)
                            if dose_match:
                                dose = dose_match.group(0)
                        known = name in vocabulary
                        events.append({"id": f"E{len(events) + 1}", "kind": kind, "name": name,
                                       "status": state, "date": current_date, "date_basis": "Document date; onset/start date unconfirmed",
                                       "documented_dose": dose, "recognized_name": known,
                                       "source": document["label"], "page": page["page"], "line": line_no,
                                       "quote": original[:1000], "issues": ([] if known else ["Unrecognized labelled entry: manually verify name and meaning."])})
                        if len(events) > 250:
                            raise ValueError("Too many clinical events. Use smaller reports (at most 250 events).")
    return events, warnings


def build_timeline(findings, events):
    timeline, warnings = [], []
    for finding in findings:
        timeline.append({"id": finding["id"], "kind": "laboratory", "date": finding["date"],
                         "name": finding["name"], "detail": f"{finding['comparator']}{finding['value']:g} {finding['unit']}",
                         "status": finding["status"], "source": finding["source"], "page": finding["page"], "line": finding["line"]})
    for event in events:
        timeline.append({k: event[k] for k in ("id", "kind", "date", "name", "status", "source", "page", "line")})
        timeline[-1]["detail"] = event["documented_dose"] or event["date_basis"]
    # Preserve every source. Detect opposing assertions rather than overwriting.
    grouped = {}
    for event in events:
        grouped.setdefault((event["kind"], event["name"].lower(), event["date"]), []).append(event)
    for group in grouped.values():
        states = {e["status"] for e in group}
        if "Negated in report" in states and states & {"Documented in report", "Listed in report", "Listed in report; current use unconfirmed"}:
            warnings.append(f"Conflicting {group[0]['kind']} assertions for {group[0]['name']} on {group[0]['date'] or 'unknown date'}: " + ", ".join(e["id"] for e in group) + ". Verify against source reports.")
        if len(group) > 1 and len(states) == 1:
            warnings.append(f"Repeated mention of {group[0]['name']}: source entries retained; this does not establish separate episodes.")
    return sorted(timeline, key=lambda e: (e["date"] is None, e["date"] or "", e["id"])), warnings
