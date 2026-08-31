"""
RAG Pipeline
============
Generates meaningful retrieval queries from ML results and patient features,
then runs FAISS retrieval to return grounded medical context.

Query generation avoids simple keyword queries like "diabetes".
Instead it builds a semantically rich query focusing on the specific
clinical features present in this patient's input.

The LLM receives chunk_ids (not source names) — the backend resolves citations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Features that map to meaningful medical topics for retrieval
FEATURE_TOPICS: dict[str, str] = {
    "Glucose":                  "blood glucose levels and diabetes risk",
    "BMI":                      "body mass index obesity and diabetes risk",
    "Age":                      "age-related diabetes risk factors",
    "Pregnancies":              "pregnancy gestational diabetes and diabetes risk",
    "BloodPressure":            "blood pressure hypertension and diabetes",
    "Insulin":                  "insulin levels and insulin resistance",
    "DiabetesPedigreeFunction": "family history genetic diabetes pedigree",
    "SkinThickness":            "skin thickness adiposity body fat diabetes",
}


def generate_rag_query(
    patient_features: dict[str, float],
    sensitivity: dict[str, dict[str, float]] | None = None,
) -> str:
    """
    Generate a neutral, conditional retrieval query based on the ML output,
    patient feature values, and local sensitivity.

    The query focuses on medical concepts relevant to the prediction —
    it does NOT state that the patient has diabetes or elevated values.
    """
    if sensitivity:
        sorted_feats = sorted(
            sensitivity.items(),
            key=lambda x: max(abs(v) for v in x[1].values()),
            reverse=True
        )
        notable = [f for f, _ in sorted_feats[:3]]
    else:
        notable = [f for f in patient_features if patient_features[f] != 0][:3]

    if not notable:
        notable = ["Glucose", "BMI", "Age"]

    topic_phrases = [FEATURE_TOPICS.get(f, f) for f in notable]
    topics_str = ", ".join(topic_phrases)

    query = (
        f"General medical evidence relating {topics_str} and diabetes risk. "
        f"Include clinical assessment guidelines, associations with Type 2 diabetes, "
        f"and relevant demographic risk factors."
    )

    logger.info("Generated RAG query (%d chars)", len(query))
    return query


def retrieve_medical_context(
    patient_features: dict[str, float],
    sensitivity: dict[str, dict[str, float]] | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """
    Full RAG pipeline: generate query → embed → FAISS → filter → return context.

    Returns:
        {
          "query":     str,
          "documents": [str, ...],
          "scores":    [float, ...],
          "metadata":  [dict, ...],   # chunk_id, document_id, source, publisher, url, page
          "sufficient": bool,         # False if 0 results above threshold
        }
    """
    from rag.retriever import get_retriever

    query = generate_rag_query(patient_features, sensitivity)
    retriever = get_retriever()

    result = retriever.retrieve(query, top_k=top_k, threshold=threshold)
    sufficient = len(result["documents"]) > 0

    if not sufficient:
        logger.warning(
            "RAG returned 0 results above threshold. "
            "LLM will receive insufficient context flag."
        )

    return {
        "query":      query,
        "documents":  result["documents"],
        "scores":     result["scores"],
        "metadata":   result["metadata"],
        "sufficient": sufficient,
    }
