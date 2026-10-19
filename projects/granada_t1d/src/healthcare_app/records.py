"""Conservative, reviewable laboratory extraction; no inferred patient diagnosis."""
from dataclasses import dataclass, asdict
from datetime import date
import math
import re

# Ordered longest-first at matching time. All comparisons use the report's range.
ALIASES = {
    "hba1c": ["hemoglobin a1c", "haemoglobin a1c", "glycated hemoglobin", "glycosylated hemoglobin", "hba1c", "hb a1c", "a1c"],
    "glucose": ["fasting blood glucose", "fasting blood sugar", "random blood glucose", "random blood sugar", "blood glucose", "serum glucose", "fasting glucose", "glucose fasting", "glucose", "fbs", "rbs"],
    "creatinine": ["serum creatinine", "creatinine"],
    "egfr": ["estimated glomerular filtration rate", "estimated gfr", "egfr"],
    "hemoglobin": ["haemoglobin", "hemoglobin", "hgb", "hb"],
    "wbc": ["white blood cell count", "white blood cells", "total leukocyte count", "white cell count", "wbc", "tlc"],
    "platelets": ["platelet count", "platelets"],
    "rbc": ["red blood cell count", "red blood cells", "rbc"],
    "hematocrit": ["hematocrit", "haematocrit", "hct", "pcv"],
    "mcv": ["mean corpuscular volume", "mcv"],
    "tsh": ["thyroid stimulating hormone", "thyroid-stimulating hormone", "tsh"],
    "cholesterol": ["total cholesterol", "cholesterol total", "cholesterol"],
    "hdl": ["hdl cholesterol", "hdl-cholesterol", "hdl"],
    "ldl": ["ldl cholesterol", "ldl-cholesterol", "ldl"],
    "triglycerides": ["triglycerides", "triglyceride"],
    "alt": ["alanine aminotransferase", "alt (sgpt)", "sgpt", "alt"],
    "ast": ["aspartate aminotransferase", "ast (sgot)", "sgot", "ast"],
    "bilirubin": ["total bilirubin", "bilirubin total", "bilirubin"],
    "albumin": ["serum albumin", "albumin"],
    "sodium": ["serum sodium", "sodium"],
    "potassium": ["serum potassium", "potassium"],
    "calcium": ["serum calcium", "calcium"],
    "urea": ["blood urea nitrogen", "blood urea", "urea", "bun"],
}
LABELS = {
    "hba1c": "HbA1c", "glucose": "Glucose", "egfr": "eGFR", "wbc": "White blood cells",
    "rbc": "Red blood cells", "tsh": "TSH", "hdl": "HDL cholesterol", "ldl": "LDL cholesterol",
    "alt": "ALT", "ast": "AST", "mcv": "MCV",
}
NUMBER = r"[-+]?\d+(?:\.\d+)?"
UNITS = re.compile(r"^(?:H\b|L\b|HIGH\b|LOW\b|[*↑↓])?\s*([a-zA-Zµμ%][a-zA-Z0-9µμ%/^.²*\-]*|(?:x\s*)?10\s*[\^*]\s*\d+\s*/\s*[a-zA-Zµμ]+|/\s*[a-zA-Zµμ]+)", re.I)
EXPECTED_UNITS = {
    "hba1c": {"%", "mmol/mol"}, "glucose": {"mg/dl", "mmol/l"},
    "creatinine": {"mg/dl", "umol/l"}, "egfr": {"ml/min/1.73m2", "ml/min/1.73m²", "ml/min"},
    "hemoglobin": {"g/dl", "g/l"}, "hematocrit": {"%", "l/l"}, "mcv": {"fl"},
    "wbc": {"10^9/l", "x10^9/l", "10*9/l", "cells/ul", "/ul", "k/ul"},
    "rbc": {"10^12/l", "x10^12/l", "million/ul", "m/ul"},
    "platelets": {"10^9/l", "x10^9/l", "cells/ul", "/ul", "k/ul"},
    "tsh": {"miu/l", "uiu/ml", "mu/l"}, "cholesterol": {"mg/dl", "mmol/l"},
    "hdl": {"mg/dl", "mmol/l"}, "ldl": {"mg/dl", "mmol/l"}, "triglycerides": {"mg/dl", "mmol/l"},
    "alt": {"u/l", "iu/l"}, "ast": {"u/l", "iu/l"}, "bilirubin": {"mg/dl", "umol/l"},
    "albumin": {"g/dl", "g/l"}, "sodium": {"mmol/l", "meq/l"}, "potassium": {"mmol/l", "meq/l"},
    "calcium": {"mg/dl", "mmol/l"}, "urea": {"mg/dl", "mmol/l"},
}


@dataclass
class Finding:
    id: str
    test: str
    name: str
    value: float
    unit: str
    comparator: str
    ref_low: float | None
    ref_high: float | None
    ref_operator: str
    date: str | None
    source: str
    page: int
    line: int
    quote: str
    status: str
    issues: list[str]

    def to_dict(self):
        return asdict(self)


def normalize_unit(unit):
    return re.sub(r"\s+", "", unit.lower().replace("μ", "u").replace("µ", "u"))


def parse_date(text):
    """ISO dates and unambiguous day-first dates only; month names accepted."""
    import datetime
    value = text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None
    for pattern in ("%d %b %Y", "%d %B %Y", "%d-%b-%Y"):
        try:
            return datetime.datetime.strptime(value, pattern).date().isoformat()
        except ValueError:
            pass
    match = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", value)
    if match:
        day, month, year = map(int, match.groups())
        if day > 12 and month <= 12:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass
    return None


def classify(value, comparator, low, high, operator, unit):
    if not unit or comparator:
        return "Uncertain"
    if value < 0 or (low is not None and high is not None and low > high):
        return "Needs review"
    if low is None and high is None:
        return "Range unavailable"
    if low is not None and (value < low or (operator == ">" and value == low)):
        return "Below report range"
    if high is not None and (value > high or (operator == "<" and value == high)):
        return "Above report range"
    return "Within report range"


def extract_findings(documents):
    findings, warnings = [], []
    aliases = sorted(((alias, key) for key, values in ALIASES.items() for alias in values), key=lambda p: -len(p[0]))
    for document in documents:
        current_date = None
        for page in document["pages"]:
            for line_number, original in enumerate(page["text"].splitlines(), 1):
                line = re.sub(r"\s+", " ", original).strip().strip("| ")
                date_match = re.match(r"(?:report date|collection date|sample date|collected(?: on)?|date)\s*[:=]\s*(.+)", line, re.I)
                if date_match:
                    current_date = parse_date(date_match.group(1))
                    if not current_date:
                        warnings.append(f"{document['label']}, page {page['page']}: ambiguous or invalid date; use YYYY-MM-DD.")
                    continue
                for alias, key in aliases:
                    match = re.match(r"^" + re.escape(alias) + r"(?![a-zA-Z])\s*[:|=]?\s*(<=|>=|<|>)?\s*(" + NUMBER + r")(?![\d.,eE])", line, re.I)
                    if not match:
                        continue
                    comparator, number = match.groups()
                    value = float(number)
                    if not math.isfinite(value):
                        warnings.append(f"{document['label']}, page {page['page']}, line {line_number}: numeric value is too large; result withheld until corrected.")
                        break
                    rest = line[match.end():].strip(" |")
                    unit_match = UNITS.match(rest)
                    unit = unit_match.group(1).strip() if unit_match else ""
                    if unit.lower() in {"reference", "ref", "range", "normal", "high", "low", "h", "l"}:
                        unit = ""
                        unit_match = None
                    ref_text = rest[unit_match.end():] if unit_match else rest
                    ref_text = ref_text.replace("–", "-").replace("—", "-")
                    interval = re.search(r"(" + NUMBER + r")\s*(?:-|\bto\b)\s*(" + NUMBER + r")", ref_text, re.I)
                    bound = re.search(r"(<=|>=|<|>)\s*(" + NUMBER + r")", ref_text)
                    low = high = None
                    operator = "range"
                    if interval:
                        low, high = map(float, interval.groups())
                    elif bound:
                        operator, val = bound.groups()
                        if operator.startswith("<"):
                            high = float(val)
                        else:
                            low = float(val)
                    issues = []
                    invalid_reference = any(v is not None and not math.isfinite(v) for v in (low, high))
                    if invalid_reference:
                        low = high = None
                        issues.append("Reference range is not finite: comparison was withheld.")
                    if not unit:
                        issues.append("Unit missing or unsupported: verify before interpreting.")
                    known_unit = bool(unit) and normalize_unit(unit) in EXPECTED_UNITS[key]
                    unit_valid = not invalid_reference and known_unit
                    if unit and not known_unit:
                        issues.append("Unit is not supported for this test: range interpretation is withheld.")
                    if interval or bound:
                        reference_end = interval.end() if interval else bound.end()
                        reference_unit = UNITS.match(ref_text[reference_end:].strip(" |)"))
                        if reference_unit and normalize_unit(reference_unit.group(1)) in EXPECTED_UNITS[key] and normalize_unit(reference_unit.group(1)) != normalize_unit(unit):
                            unit_valid = False
                            issues.append("Result and reference range use different units: comparison was withheld.")
                    if comparator:
                        issues.append("Result is a bound rather than an exact value; range classification is withheld.")
                    if value < 0:
                        issues.append("Negative laboratory value: verify the extraction.")
                    if low is not None and high is not None and low > high:
                        issues.append("Reference range is reversed: correct it.")
                    if not current_date:
                        issues.append("No unambiguous report date; excluded from timeline comparisons.")
                    if not math.isfinite(value):
                        issues.append("Non-finite value: correct it.")
                    status = classify(value, comparator or "", low, high, operator, unit)
                    if not unit_valid:
                        status = "Uncertain"
                    if value < 0 or (low is not None and high is not None and low > high):
                        status = "Needs review"
                    findings.append(Finding(f"F{len(findings) + 1}", key, LABELS.get(key, key.capitalize()), value, unit, comparator or "", low, high, operator, current_date, document["label"], int(page["page"]), line_number, line, status, issues).to_dict())
                    break
    if not findings:
        warnings.append("No supported numeric lab results were recognized. Narrative/imaging reports need clinician interpretation; this parser does not diagnose their content.")
    return findings, list(dict.fromkeys(warnings))
