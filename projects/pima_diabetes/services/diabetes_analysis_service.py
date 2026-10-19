"""
Diabetes Analysis Service
=========================
Orchestrates the full ML → Sensitivity → RAG → LLM pipeline.

The ML prediction is always returned first.
RAG and LLM failures are handled gracefully — the ML result is always preserved.

Architecture:
  patient_input
    → PredictionService.predict()           (always runs)
    → FeatureExplanationService.compute()   (always runs if prediction succeeds)
    → rag.pipeline.retrieve_medical_context()
    → llm.openai_client.explain()
    → structured response
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def analyze(patient_input: dict[str, Any]) -> dict[str, Any]:
    """
    Run the full analysis pipeline.

    Args:
        patient_input: validated dict with 8 Pima feature values

    Returns:
        Full structured response dict (see API response schema in app.py).
        ML prediction is always present.
        explanation_status is "success" or "unavailable".
    """
    from services.prediction_service import PredictionService

    ps = PredictionService.get()
    ml_result = ps.predict(patient_input)

    prediction       = ml_result["prediction"]
    model_probability = ml_result["model_probability"]
    risk_label       = ml_result["risk_label"]

    # --- Feature sensitivity (deterministic) ---
    sensitivity = None
    try:
        from services.feature_explanation_service import compute_sensitivity
        sensitivity = compute_sensitivity(ps, patient_input)
    except Exception as e:
        logger.warning("Feature sensitivity computation failed: %s", type(e).__name__)

    # --- RAG retrieval ---
    retrieval_result = None
    try:
        from rag.pipeline import retrieve_medical_context
        retrieval_result = retrieve_medical_context(
            patient_features = {k: float(v) for k, v in patient_input.items()},
            sensitivity      = sensitivity,
        )
        logger.info(
            "RAG retrieval: %d documents (sufficient=%s)",
            len(retrieval_result.get("documents", [])),
            retrieval_result.get("sufficient", False)
        )
    except Exception as e:
        logger.warning("RAG retrieval failed: %s", type(e).__name__)

    # --- LLM explanation ---
    explanation       = None
    explanation_status  = "unavailable"
    explanation_message = (
        "The ML prediction was generated successfully, "
        "but the AI explanation service is temporarily unavailable."
    )

    if retrieval_result is not None:
        if not retrieval_result.get("sufficient", False):
            explanation_status = "unavailable"
            explanation_message = "Insufficient medical context retrieved to generate a grounded explanation."
        else:
            try:
                from llm.groq_client import explain, SAFE_FALLBACK_EXPLANATION
                explanation = explain(
                    patient_features  = {k: float(v) for k, v in patient_input.items()},
                    prediction        = prediction,
                    model_probability = model_probability,
                    risk_label        = risk_label,
                    sensitivity       = sensitivity,
                    retrieval_result  = retrieval_result,
                )
                if explanation.summary == SAFE_FALLBACK_EXPLANATION:
                    explanation_status  = "fallback"
                    explanation_message = SAFE_FALLBACK_EXPLANATION
                else:
                    explanation_status  = "success"
                    explanation_message = None
            except Exception as e:
                logger.warning("LLM explanation failed: %s", type(e).__name__)
    else:
        explanation_message = (
            "The ML prediction was generated successfully, "
            "but the medical knowledge base is not initialized. "
            "Run: python scripts/build_index.py"
        )

    # --- Build response ---
    response: dict[str, Any] = {
        "prediction":        prediction,
        "model_probability": model_probability,
        "risk_label":        risk_label,
    }

    if explanation_status == "success" and explanation is not None:
        response["explanation_status"] = "success"
        response["explanation"] = {
            "summary":                explanation.summary,
            "prediction_explanation": explanation.prediction_explanation,
            "important_factors": [
                {
                    "factor":      f.factor,
                    "explanation": f.explanation,
                    "citation_chunk_ids": f.citation_chunk_ids,
                }
                for f in explanation.important_factors
            ],
            "medical_context": explanation.medical_context,
            "recommendation":  explanation.recommendation,
            "sources": [
                {
                    "chunk_id":  s.chunk_id,
                    "source":    s.source,
                    "publisher": s.publisher,
                    "page":      s.page,
                    "url":       s.url,
                }
                for s in explanation.sources
            ],
            "disclaimer":  explanation.disclaimer,
        }
    else:
        response["explanation_status"]  = explanation_status
        response["explanation_message"] = explanation_message
        response["sources"] = []
        
    if retrieval_result:
        response["rag_query"] = retrieval_result.get("query", "")
        response["retrieved_sources"] = [
            {
                "chunk_id":   m.get("chunk_id", ""),
                "source":     m.get("source", ""),
                "publisher":  m.get("publisher", "Unknown"),
                "page":       m.get("page", 1),
                "url":        m.get("url", ""),
                "text":       m.get("text", ""),
                "score":      m.get("score", 0.0),
            }
            for m in retrieval_result.get("metadata", [])
        ]

    if sensitivity:
        response["feature_sensitivity"] = {
            "values": sensitivity,
            "label":  "Local model sensitivity estimate; not proof of medical causation.",
        }

    return response
