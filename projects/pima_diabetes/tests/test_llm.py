"""
Test: LLM Client — Citation Validation
=======================================
Tests that the LLM client:
  1. Blocks fabricated chunk_ids not in the retrieved set
  2. Resolves valid chunk_ids to source metadata
  3. Handles API failure gracefully
  4. Validates structured output schema
  5. Returns safe fallback on validation failure

No real OpenAI calls are made.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_RETRIEVED_CHUNKS = {
    "niddk-diabetes:p1:c00",
    "cdc-risk:p2:c01",
}

VALID_RETRIEVAL_METADATA = [
    {
        "chunk_id":   "niddk-diabetes:p1:c00",
        "document_id": "niddk-diabetes",
        "source":     "niddk-diabetes.txt",
        "publisher":  "NIDDK",
        "url":        "https://niddk.nih.gov",
        "page":       1,
    },
    {
        "chunk_id":   "cdc-risk:p2:c01",
        "document_id": "cdc-risk",
        "source":     "cdc-risk.txt",
        "publisher":  "CDC",
        "url":        "https://cdc.gov",
        "page":       2,
    },
]


class TestCitationValidation:
    def test_valid_citations_kept(self):
        from llm.groq_client import _validate_citations
        from llm.response_schema import LLMResponse, ImportantFactor

        llm_resp = LLMResponse(
            summary="Test",
            prediction_explanation="Test",
            important_factors=[
                ImportantFactor(
                    factor="Glucose",
                    explanation="...",
                    citation_chunk_ids=["niddk-diabetes:p1:c00"]
                )
            ],
            medical_context="...",
            medical_context_citation_chunk_ids=["cdc-risk:p2:c01"],
            recommendation="Consult a doctor.",
            disclaimer="Educational only.",
        )

        validated = _validate_citations(llm_resp, VALID_RETRIEVED_CHUNKS)
        assert "niddk-diabetes:p1:c00" in validated.important_factors[0].citation_chunk_ids
        assert "cdc-risk:p2:c01" in validated.medical_context_citation_chunk_ids

    def test_fabricated_citation_blocked(self):
        """Chunk IDs not in retrieved set must be removed."""
        from llm.groq_client import _validate_citations
        from llm.response_schema import LLMResponse, ImportantFactor

        llm_resp = LLMResponse(
            summary="Test",
            prediction_explanation="Test",
            important_factors=[
                ImportantFactor(
                    factor="BMI",
                    explanation="...",
                    citation_chunk_ids=[
                        "niddk-diabetes:p1:c00",   # valid
                        "FABRICATED-SOURCE:p99:c99",  # FABRICATED
                    ]
                )
            ],
            medical_context="...",
            medical_context_citation_chunk_ids=[
                "cdc-risk:p2:c01", # valid
                "WHO-GUIDELINE-2024:p1:c00" # FABRICATED
            ],
            recommendation="...",
            disclaimer="...",
        )

        validated = _validate_citations(llm_resp, VALID_RETRIEVED_CHUNKS)
        assert "FABRICATED-SOURCE:p99:c99" not in validated.important_factors[0].citation_chunk_ids
        assert "WHO-GUIDELINE-2024:p1:c00" not in validated.medical_context_citation_chunk_ids
        assert "niddk-diabetes:p1:c00" in validated.important_factors[0].citation_chunk_ids
        assert "cdc-risk:p2:c01" in validated.medical_context_citation_chunk_ids

    def test_all_fabricated_results_in_empty_citations(self):
        from llm.groq_client import _validate_citations
        from llm.response_schema import LLMResponse, ImportantFactor

        llm_resp = LLMResponse(
            summary="Test",
            prediction_explanation="Test",
            important_factors=[
                ImportantFactor(
                    factor="Age",
                    explanation="...",
                    citation_chunk_ids=["FAKE:p1:c00", "ANOTHER-FAKE:p2:c01"],
                )
            ],
            medical_context="...",
            medical_context_citation_chunk_ids=["YET-ANOTHER-FAKE:p3:c02"],
            recommendation="...",
            disclaimer="...",
        )

        with pytest.raises(ValueError, match="medical_context lacks valid citations."):
            _validate_citations(llm_resp, VALID_RETRIEVED_CHUNKS)


class TestSourceResolution:
    def test_resolves_valid_chunk_ids(self):
        from llm.groq_client import _resolve_sources
        from llm.response_schema import LLMResponse, ImportantFactor

        llm_resp = LLMResponse(
            summary="Test",
            prediction_explanation="Test",
            important_factors=[
                ImportantFactor(
                    factor="Glucose",
                    explanation="...",
                    citation_chunk_ids=["niddk-diabetes:p1:c00"]
                )
            ],
            medical_context="...",
            medical_context_citation_chunk_ids=["cdc-risk:p2:c01"],
            recommendation="...",
            disclaimer="...",
        )

        sources = _resolve_sources(llm_resp, VALID_RETRIEVAL_METADATA)
        chunk_ids = {s.chunk_id for s in sources}
        assert "niddk-diabetes:p1:c00" in chunk_ids
        assert "cdc-risk:p2:c01"       in chunk_ids

    def test_unresolvable_chunk_ids_excluded(self):
        from llm.groq_client import _resolve_sources
        from llm.response_schema import LLMResponse, ImportantFactor

        llm_resp = LLMResponse(
            summary="Test",
            prediction_explanation="Test",
            important_factors=[
                ImportantFactor(
                    factor="BMI",
                    explanation="...",
                    citation_chunk_ids=["nonexistent:p0:c00"]
                )
            ],
            medical_context="...",
            medical_context_citation_chunk_ids=[],
            recommendation="...",
            disclaimer="...",
        )

        sources = _resolve_sources(llm_resp, VALID_RETRIEVAL_METADATA)
        assert sources == []


class TestLLMFallback:
    def test_malformed_json_returns_fallback(self):
        """Invalid JSON from LLM should return safe fallback, not crash."""
        from llm.groq_client import explain, SAFE_FALLBACK_EXPLANATION
        import os
        from unittest.mock import patch, MagicMock

        os.environ["GROQ_API_KEY"] = "gsk_test-fake-key"
        os.environ["GROQ_MODEL"]   = "openai/gpt-oss-120b"

        with patch("llm.groq_client.Groq") as MockClient:
            mock_instance = MagicMock()
            mock_response = MagicMock()
            
            # Mock the chat completions creation structure
            mock_choice = MagicMock()
            mock_choice.message.content = "this is not json at all"
            mock_response.choices = [mock_choice]
            
            mock_instance.chat.completions.create.return_value = mock_response
            MockClient.return_value = mock_instance

            explanation = explain(
                patient_features  = {"Glucose": 148, "BMI": 33.6, "Age": 50, "Pregnancies": 2, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 120, "DiabetesPedigreeFunction": 0.627},
                prediction        = 1,
                model_probability = 0.78,
                risk_label        = "Model predicts positive diabetes class",
                sensitivity       = None,
                retrieval_result  = {
                    "documents":  ["Doc 1"],
                    "scores":     [0.9],
                    "metadata":   [{"chunk_id": "test:p1:c1", "source": "test.txt", "publisher": "Test", "page": 1, "url": "http://test"}],
                    "sufficient": True,
                },
            )

            assert explanation.summary == SAFE_FALLBACK_EXPLANATION


class TestMockedLLMAPICall:
    def test_api_failure_raises_runtime_error(self):
        """API failure should raise RuntimeError, not expose key."""
        import os
        os.environ["GROQ_API_KEY"] = "gsk_test-fake-key"
        os.environ["GROQ_MODEL"]   = "openai/gpt-oss-120b"

        from llm import groq_client as client_mod
        with patch("llm.groq_client.Groq") as MockClient:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create.side_effect = Exception("connection refused")
            MockClient.return_value = mock_instance

            with pytest.raises(RuntimeError, match="Groq API call failed"):
                client_mod.explain(
                    patient_features  = {"Glucose": 148, "BMI": 33.6, "Age": 50,
                                         "Pregnancies": 2, "BloodPressure": 72,
                                         "SkinThickness": 35, "Insulin": 120,
                                         "DiabetesPedigreeFunction": 0.627},
                    prediction        = 1,
                    model_probability = 0.78,
                    risk_label        = "Model predicts positive diabetes class",
                    sensitivity       = None,
                    retrieval_result  = {
                        "documents":  [],
                        "scores":     [],
                        "metadata":   [],
                        "sufficient": False,
                    },
                )
