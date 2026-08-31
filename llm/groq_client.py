"""
Groq Client
===========
Isolated Groq API client with structured JSON output and citation validation.
"""

import json
import logging
import os
from typing import Any

from pydantic import ValidationError
from groq import Groq

from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.response_schema import (
    AnalysisExplanation,
    LLMResponse,
    ResolvedSource,
    LLM_RESPONSE_JSON_SCHEMA,
)

logger = logging.getLogger(__name__)

SAFE_FALLBACK_EXPLANATION = (
    "Explanation could not be generated due to a validation error "
    "in the AI response. The ML prediction above is valid and unaffected."
)
DISCLAIMER_TEXT = (
    "This system provides an AI-assisted explanation of a machine-learning "
    "prediction for educational and research purposes only. It is not a medical "
    "diagnosis and does not replace advice from a qualified healthcare professional. "
    "Always consult a licensed medical provider for health decisions."
)


def _get_client() -> Groq:
    """Return a Groq client initialized from environment variables."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )
    return Groq(api_key=api_key)


def _validate_citations(
    llm_response: LLMResponse,
    retrieved_chunk_ids: set[str],
) -> LLMResponse:
    """
    Validate all chunk_id references in the LLM response against
    the set of actually retrieved chunk IDs.
    """
    def clean_ids(ids: list[str]) -> list[str]:
        cleaned   = [cid for cid in ids if cid in retrieved_chunk_ids]
        removed   = set(ids) - set(cleaned)
        if removed:
            logger.warning(
                "LLM cited %d non-retrieved chunk_id(s) — removed: %s",
                len(removed), removed
            )
        return cleaned

    # Clean medical_context citations
    llm_response.medical_context_citation_chunk_ids = clean_ids(
        llm_response.medical_context_citation_chunk_ids
    )
    if not llm_response.medical_context_citation_chunk_ids:
        raise ValueError("medical_context lacks valid citations.")

    # Clean per-factor citations
    for factor in llm_response.important_factors:
        original_factor_ids = list(factor.citation_chunk_ids)
        factor.citation_chunk_ids = clean_ids(factor.citation_chunk_ids)
        # Only raise ValueError if it originally cited something but they were all fake!
        if original_factor_ids and not factor.citation_chunk_ids:
            raise ValueError(f"important_factor '{factor.factor}' lacks valid citations.")

    return llm_response


def _resolve_sources(
    llm_response: LLMResponse,
    retrieval_metadata: list[dict[str, Any]],
) -> list[ResolvedSource]:
    """
    Convert all valid chunk_ids cited in the LLM response into
    ResolvedSource objects using retrieval metadata.
    """
    meta_by_id = {m["chunk_id"]: m for m in retrieval_metadata}

    cited_ids: set[str] = set(llm_response.medical_context_citation_chunk_ids)
    for factor in llm_response.important_factors:
        cited_ids.update(factor.citation_chunk_ids)

    sources = []
    seen: set[str] = set()
    for chunk_id in cited_ids:
        if chunk_id in meta_by_id and chunk_id not in seen:
            m = meta_by_id[chunk_id]
            sources.append(ResolvedSource(
                chunk_id  = chunk_id,
                source    = m.get("source",    ""),
                publisher = m.get("publisher", "Unknown"),
                page      = m.get("page",      1),
                url       = m.get("url",       ""),
                text      = m.get("text",      ""),
                score     = m.get("score",     0.0),
            ))
            seen.add(chunk_id)

    return sources


def explain(
    patient_features:   dict[str, float],
    prediction:         int,
    model_probability:  float,
    risk_label:         str,
    sensitivity:        dict[str, float] | None,
    retrieval_result:   dict[str, Any],
) -> AnalysisExplanation:
    """
    Call the Groq API with structured JSON output.
    """
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    documents   = retrieval_result.get("documents",  [])
    meta        = retrieval_result.get("metadata",   [])
    sufficient  = retrieval_result.get("sufficient", False)

    retrieved_chunk_ids = {m["chunk_id"] for m in meta}

    prompt_chunks = [
        {"chunk_id": m["chunk_id"], "text": doc}
        for m, doc in zip(meta, documents)
    ]

    user_prompt = build_user_prompt(
        patient_features   = patient_features,
        prediction         = prediction,
        model_probability  = model_probability,
        risk_label         = risk_label,
        sensitivity        = sensitivity,
        retrieved_chunks   = prompt_chunks,
        context_sufficient = sufficient,
    )

    logger.info(
        "Calling Groq API (model=%s, chunks=%d, sufficient=%s)",
        model_name, len(prompt_chunks), sufficient
    )

    client = _get_client()

    # Append JSON schema instruction to system prompt for Groq JSON mode
    example_response = {
        "summary": "The ML model predicted a positive class (probability 78%), indicating an elevated risk.",
        "prediction_explanation": "This model is trained on the Pima dataset. A positive prediction indicates...",
        "important_factors": [
            {
                "factor": "Glucose",
                "explanation": "Your glucose is 140 mg/dL, which is associated with prediabetes...",
                "citation_chunk_ids": ["niddk-diabetes:p1:c00"]
            }
        ],
        "medical_context": "Elevated fasting blood glucose levels and high BMI are key indicators...",
        "medical_context_citation_chunk_ids": ["cdc-risk:p2:c01"],
        "recommendation": "Consult a healthcare provider for a formal test and lifestyle counseling.",
        "disclaimer": DISCLAIMER_TEXT
    }

    full_system_prompt = (
        SYSTEM_PROMPT + 
        "\n\nYou MUST return a JSON object that matches this schema exactly:\n" +
        json.dumps(LLM_RESPONSE_JSON_SCHEMA["schema"]) +
        "\n\nExample output structure:\n" +
        json.dumps(example_response)
    )

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        raw_text = completion.choices[0].message.content
    except Exception as e:
        logger.error("Groq API call failed: %s", type(e).__name__)
        raise RuntimeError(f"Groq API call failed: {type(e).__name__}") from e

    logger.info("Groq API call completed successfully.")

    try:
        raw_dict = json.loads(raw_text)
        llm_response = LLMResponse(**raw_dict)
        llm_response = _validate_citations(llm_response, retrieved_chunk_ids)
    except (json.JSONDecodeError, ValidationError, ValueError) as e:
        logger.error("LLM response failed validation: %s", type(e).__name__)
        try:
            import traceback
            with open("validation_error.log", "w") as f:
                f.write(f"Exception: {type(e).__name__}\n")
                f.write(f"Message: {str(e)}\n\n")
                f.write("Traceback:\n")
                traceback.print_exc(file=f)
                f.write("\n\nRaw Text:\n")
                f.write(raw_text)
        except Exception:
            pass
        return AnalysisExplanation(
            summary                 = SAFE_FALLBACK_EXPLANATION,
            prediction_explanation  = SAFE_FALLBACK_EXPLANATION,
            important_factors       = [],
            medical_context         = "Insufficient or invalid AI-generated context.",
            recommendation          = "Please consult a qualified healthcare professional.",
            sources                 = [],
            disclaimer              = DISCLAIMER_TEXT,
        )

    sources = _resolve_sources(llm_response, meta)

    return AnalysisExplanation(
        summary                 = llm_response.summary,
        prediction_explanation  = llm_response.prediction_explanation,
        important_factors       = llm_response.important_factors,
        medical_context         = llm_response.medical_context,
        recommendation          = llm_response.recommendation,
        sources                 = sources,
        disclaimer              = llm_response.disclaimer or DISCLAIMER_TEXT,
    )
