"""
LLM Response Schema
===================
Pydantic models for validating the LLM structured response.
Used both for OpenAI JSON Schema structured output and server-side validation.

The LLM returns chunk_ids — the backend resolves them to source metadata.
"""

from pydantic import BaseModel, Field


class ImportantFactor(BaseModel):
    factor:             str  = Field(..., description="Name of the Pima feature or clinical concept")
    explanation:        str  = Field(..., description="Medical explanation of this factor's relevance")
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs from the retrieved context that support this explanation"
    )


class LLMResponse(BaseModel):
    summary:                         str   = Field(..., description="Brief summary of the ML prediction and overall context")
    prediction_explanation:          str   = Field(..., description="Explanation of what the ML model's prediction means")
    important_factors:               list[ImportantFactor] = Field(
        default_factory=list,
        description="Key Pima features discussed with their medical context"
    )
    medical_context:                 str   = Field(..., description="Relevant medical context retrieved from reference documents")
    medical_context_citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs supporting the medical_context field"
    )
    recommendation:                  str   = Field(..., description="General recommendation (must encourage professional consultation)")
    disclaimer:                      str   = Field(..., description="Educational disclaimer")


# JSON Schema for OpenAI structured output
LLM_RESPONSE_JSON_SCHEMA = {
    "name": "DiabetesMLExplanation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Brief summary of the ML prediction and overall context"
            },
            "prediction_explanation": {
                "type": "string",
                "description": "Explanation of what the ML model output means (not a diagnosis)"
            },
            "important_factors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "factor": {"type": "string"},
                        "explanation": {"type": "string"},
                        "citation_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["factor", "explanation", "citation_chunk_ids"],
                    "additionalProperties": False
                }
            },
            "medical_context": {
                "type": "string",
                "description": "Medical context grounded in retrieved reference documents"
            },
            "medical_context_citation_chunk_ids": {
                "type": "array",
                "items": {"type": "string"}
            },
            "recommendation": {
                "type": "string",
                "description": "General guidance; must encourage consulting a healthcare professional"
            },
            "disclaimer": {
                "type": "string",
                "description": "Educational/research disclaimer"
            }
        },
        "required": [
            "summary",
            "prediction_explanation",
            "important_factors",
            "medical_context",
            "medical_context_citation_chunk_ids",
            "recommendation",
            "disclaimer"
        ]
    }
}


class ResolvedSource(BaseModel):
    chunk_id:  str
    source:    str
    publisher: str
    page:      int
    url:       str
    text:      str = ""
    score:     float = 0.0


class AnalysisExplanation(BaseModel):
    """Final structured explanation returned to the API consumer."""
    summary:                str
    prediction_explanation: str
    important_factors:      list[ImportantFactor]
    medical_context:        str
    recommendation:         str
    sources:                list[ResolvedSource]
    disclaimer:             str
