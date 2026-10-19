"""Bounded educational retrieval and research summaries. No implicit API calls.

The legacy ADA index is retained unchanged. Current vectors are built in memory
from sourced NLM summaries, with the same vocabulary used for query encoding.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from healthcare_app.knowledge import KnowledgeBase


def locate_file(filename, search_dirs=None):
    directories = [ROOT, *[p for p in ROOT.iterdir() if p.is_dir()], *[Path(p) for p in (search_dirs or [])]]
    for directory in directories:
        candidate = directory / filename
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Required project artifact is missing: {filename}")


class ADAGuidelinesRAG:
    """Legacy name; sources are now NLM educational summaries, not ADA chapters."""
    def __init__(self, guidelines_path=None, index_path=None):
        self.guidelines_path = guidelines_path
        self.index_path = index_path
        if guidelines_path is not None and not Path(guidelines_path).is_file():
            raise FileNotFoundError("The supplied knowledge source is missing.")
        active_path = guidelines_path if guidelines_path is not None and Path(guidelines_path).name != "ada_diabetes_clinical_guidelines.json" else None
        self.kb = KnowledgeBase(active_path)
        self.guidelines = self.kb.documents
        self.vectorizer = self.kb.vectorizer
        self.build_faiss_index()

    def build_faiss_index(self):
        import faiss
        self.index = faiss.IndexFlatIP(self.kb.matrix.shape[1])
        self.index.add(self.kb.matrix.copy())

    def retrieve(self, query_text, top_k=3):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if not isinstance(query_text, str) or not query_text.strip() or len(query_text) > 2000:
            return []
        query = self.vectorizer.transform([query_text]).toarray().astype(np.float32)
        if not query.any():
            return []
        scores, indices = self.index.search(query, min(top_k, len(self.guidelines)))
        return [{**self.guidelines[int(i)], "score": round(float(s), 4), "rank": n + 1} for n, (s, i) in enumerate(zip(scores[0], indices[0])) if i >= 0 and s >= 0.12]


class ClinicalDecisionSynthesizer:
    def __init__(self, rag_engine):
        self.rag = rag_engine

    def construct_clinical_query(self, summary):
        return "blood glucose diabetes measurement HbA1c laboratory interpretation"

    def _summary(self, values):
        columns = ["glucose_mean", "tir_70_180", "tbr_below_70", "tar_above_180", "glucose_cv", "gmi_estimated_a1c"]
        if isinstance(values, dict):
            result = {c: float(values[c]) for c in columns if c in values}
        else:
            array = np.asarray(values, dtype=float)
            if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != 6:
                raise ValueError("A trajectory must be a nonempty horizon-by-six array.")
            result = dict(zip(columns, array.mean(axis=0).astype(float)))
        if not result or not all(np.isfinite(v) for v in result.values()):
            raise ValueError("Simulated metrics must contain finite values.")
        for col, value in result.items():
            if col != "glucose_mean" and not 0 <= value <= 100:
                raise ValueError("Percentage outputs are out of bounds.")
        return result

    def generate_doctor_report(self, patient_profile, trajectory_simulation, use_llm=False):
        summary = self._summary(trajectory_simulation)
        evidence = self.rag.retrieve(self.construct_clinical_query(summary))
        labels = {"glucose_mean": "Mean glucose (mg/dL)", "tir_70_180": "Time in range (%)", "tbr_below_70": "Time below range (%)", "tar_above_180": "Time above range (%)", "glucose_cv": "Glucose variability CV (%)", "gmi_estimated_a1c": "Derived GMI (%) — not measured HbA1c"}
        lines = ["EXPERIMENTAL DIABETES RESEARCH SUMMARY", "These are model-generated outcomes, not observed report results or clinical predictions.", f"Age at model window: {patient_profile.get('age', 'not supplied')}", ""]
        lines.extend(f"{labels[c]}: {value:.2f}" for c, value in summary.items())
        lines += ["", "Important limitations:", "The source artifacts retain preprocessing leakage. Latent-perturbation spread is not calibrated clinical uncertainty.", "Daily aggregates do not establish when a low-glucose event happened, its cause, or a suitable insulin adjustment.", "No medication recommendation or diagnosis is generated. Discuss observed measurements with your clinician.", "", "Educational evidence:"]
        lines.extend(f"[{d['id']}] {d['title']}: {d['content']} Source: {d['url']}" for d in evidence)
        if not evidence:
            lines.append("No sufficiently relevant source was retrieved; evidence-based inference was withheld.")
        if use_llm:
            lines.append("Live generation requires the application's explicit key-and-consent control; no API was called here.")
        return "\n".join(lines), evidence, "Offline sourced research summary"

    def generate_patient_guidance_report(self, patient_profile, trajectory_simulation, use_llm=False):
        report, evidence, engine = self.generate_doctor_report(patient_profile, trajectory_simulation, use_llm)
        return report + "\n\nQuestions for your clinician:\n- Which observed results need follow-up?\n- What additional history or monitoring is needed?\n- What personal targets apply to me?", evidence, engine

    def call_llm(self, patient_summary, retrieved_guidelines, baseline_report):
        return baseline_report, "Offline; use the app for explicit opt-in LLM evidence selection"

    def chat_with_clinical_assistant(self, patient_profile, trajectory_simulation, user_query, chat_history=None, use_llm=False):
        import re
        if not isinstance(user_query, str) or not 0 < len(user_query.strip()) <= 2000:
            raise ValueError("Enter a question of 1 to 2,000 characters.")
        if re.search(r"\b(insulin|dose|dosing|titration|medication|prescribe|diagnos\w*|night|nocturnal)\b", user_query, re.I):
            return "Daily model aggregates cannot establish nocturnal events or a medication decision. This research system cannot recommend treatment changes or diagnose a condition; discuss observed results with your clinician.", [], "Offline bounded research response"
        evidence = self.rag.retrieve(user_query)
        if not evidence:
            return "No sufficiently relevant evidence was found. No answer was inferred.", [], "Retrieval abstention"
        report, _, engine = self.generate_doctor_report(patient_profile, trajectory_simulation, False)
        return report, evidence, engine


def run_rag_demo():
    for item in ADAGuidelinesRAG().retrieve("blood glucose laboratory report"):
        print(item["id"], item["title"], item["url"])


if __name__ == "__main__":
    run_rag_demo()
