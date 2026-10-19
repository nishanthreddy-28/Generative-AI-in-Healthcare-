"""
Test: End-to-End API Analysis Flow
====================================
Tests the full FastAPI endpoint with mocked prediction service and LLM.
Does NOT make real OpenAI calls.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VALID_BODY = {
    "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
    "SkinThickness": 35, "Insulin": 120, "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627, "Age": 50,
}


@pytest.fixture(scope="module")
def client():
    """Test client with mocked ML artifacts."""
    import os
    os.environ.setdefault("GROQ_API_KEY", "gsk-test-fake")
    os.environ.setdefault("GROQ_MODEL",   "openai/gpt-oss-120b")

    from fastapi.testclient import TestClient

    # Patch PredictionService to avoid needing real artifacts
    with patch("services.prediction_service.PredictionService.get") as mock_ps_get:
        mock_ps = MagicMock()
        mock_ps.predict.return_value = {
            "prediction":        1,
            "model_probability": 0.78,
            "risk_label":        "Model predicts positive diabetes class",
        }
        mock_ps.feature_order = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
        ]
        mock_ps._zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        mock_ps._imputer = MagicMock()
        mock_ps._scaler  = MagicMock()
        mock_ps._model   = MagicMock()
        mock_ps_get.return_value = mock_ps

        from app import app
        yield TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "ml_ready" in data
        assert "rag_ready" in data

    def test_health_no_sensitive_info(self, client):
        """Health response must not expose keys, paths, or stack traces."""
        response = client.get("/api/health")
        text = response.text.lower()
        assert "api_key" not in text
        assert "openai_api" not in text
        assert "traceback" not in text


class TestAnalyzeEndpoint:
    def test_valid_input_returns_prediction(self, client):
        with patch("services.diabetes_analysis_service.analyze") as mock_analyze:
            mock_analyze.return_value = {
                "prediction":        1,
                "model_probability": 0.78,
                "risk_label":        "Model predicts positive diabetes class",
                "explanation_status": "unavailable",
                "explanation_message": "AI explanation unavailable.",
                "sources": [],
            }
            response = client.post("/api/diabetes/analyze", json=VALID_BODY)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] in (0, 1)
        assert "model_probability" in data
        assert "risk_label" in data

    def test_risk_label_does_not_say_you_have_diabetes(self, client):
        with patch("services.diabetes_analysis_service.analyze") as mock_analyze:
            mock_analyze.return_value = {
                "prediction": 1,
                "model_probability": 0.78,
                "risk_label": "Model predicts positive diabetes class",
                "explanation_status": "unavailable",
                "explanation_message": "unavailable",
                "sources": [],
            }
            response = client.post("/api/diabetes/analyze", json=VALID_BODY)

        data = response.json()
        label = data.get("risk_label", "").lower()
        assert "you have diabetes" not in label
        assert "you do not have" not in label

    def test_missing_field_returns_400(self, client):
        body = {k: v for k, v in VALID_BODY.items() if k != "Glucose"}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400  # Updated from 422 to 400

    def test_extra_field_returns_400(self, client):
        body = {**VALID_BODY, "Outcome": 1}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400

    def test_string_value_returns_400(self, client):
        body = {**VALID_BODY, "Glucose": "high"}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400

    def test_numeric_string_returns_400(self, client):
        """Numeric strings like '148' must be rejected (not coerced to 148.0)."""
        body = {**VALID_BODY, "Glucose": "148"}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400

    def test_boolean_value_returns_400(self, client):
        """Boolean true must be rejected (not coerced to 1.0)."""
        body = {**VALID_BODY, "Glucose": True}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400

    def test_out_of_range_glucose_returns_400(self, client):
        body = {**VALID_BODY, "Glucose": 999}
        response = client.post("/api/diabetes/analyze", json=body)
        assert response.status_code == 400

    def test_oversized_chunked_body_returns_413(self, client):
        def generate_large_body():
            yield b'{"Pregnancies": 2, "Glucose": 148, "BloodPressure": 72, "SkinThickness": 35, "Insulin": 120, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50, "Padding": "'
            yield b'A' * 20000
            yield b'"}'
            
        # Send chunked request (no Content-Length header is set automatically by httpx for generators)
        response = client.post("/api/diabetes/analyze", content=generate_large_body(), headers={"Content-Type": "application/json"})
        print(response.json())
        assert response.status_code == 413
        assert "too large" in response.json()["error"].lower()

    def test_llm_skipped_on_insufficient_retrieval(self, client):
        with patch("services.prediction_service.PredictionService.get") as mock_ps_get:
            mock_ps = MagicMock()
            mock_ps.predict.return_value = {"prediction": 1, "model_probability": 0.78, "risk_label": "positive"}
            mock_ps_get.return_value = mock_ps
            
            with patch("services.feature_explanation_service.compute_sensitivity") as mock_sens:
                mock_sens.return_value = {}
                
                with patch("rag.pipeline.retrieve_medical_context") as mock_retrieve:
                    mock_retrieve.return_value = {"documents": [], "sufficient": False}
                    
                    with patch("llm.groq_client.explain") as mock_explain:
                        from services.diabetes_analysis_service import analyze
                        result = analyze(VALID_BODY)
                        
                        mock_explain.assert_not_called()
                        assert result["explanation_status"] == "unavailable"
                        assert "Insufficient medical context" in result["explanation_message"]

    def test_citation_fallback_has_non_success_status(self, client):
        with patch("services.prediction_service.PredictionService.get") as mock_ps_get:
            mock_ps = MagicMock()
            mock_ps.predict.return_value = {"prediction": 1, "model_probability": 0.78, "risk_label": "positive"}
            mock_ps_get.return_value = mock_ps
            
            with patch("services.feature_explanation_service.compute_sensitivity") as mock_sens:
                mock_sens.return_value = {}
                
                with patch("rag.pipeline.retrieve_medical_context") as mock_retrieve:
                    mock_retrieve.return_value = {"documents": ["Test"], "sufficient": True}
                    
                    with patch("llm.groq_client.explain") as mock_explain:
                        from llm.response_schema import AnalysisExplanation
                        from llm.groq_client import SAFE_FALLBACK_EXPLANATION
                        
                        fallback_resp = AnalysisExplanation(
                            summary=SAFE_FALLBACK_EXPLANATION,
                            prediction_explanation="fallback",
                            important_factors=[],
                            medical_context="fallback",
                            recommendation="fallback",
                            sources=[],
                            disclaimer="fallback"
                        )
                        mock_explain.return_value = fallback_resp
                        
                        from services.diabetes_analysis_service import analyze
                        result = analyze(VALID_BODY)
                        
                        assert result["explanation_status"] == "fallback"
                        assert result["explanation_message"] == SAFE_FALLBACK_EXPLANATION

    def test_llm_failure_still_returns_prediction(self, client):
        """ML result must be preserved even when LLM fails."""
        with patch("services.diabetes_analysis_service.analyze") as mock_analyze:
            mock_analyze.return_value = {
                "prediction":         0,
                "model_probability":  0.22,
                "risk_label":         "Model predicts negative diabetes class",
                "explanation_status": "unavailable",
                "explanation_message": "AI explanation service temporarily unavailable.",
                "sources": [],
            }
            response = client.post("/api/diabetes/analyze", json=VALID_BODY)

        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == 0
        assert data["explanation_status"] == "unavailable"
        assert "explanation_message" in data

    def test_no_stack_trace_in_error_response(self, client):
        """Server errors must not expose stack traces."""
        with patch("services.diabetes_analysis_service.analyze",
                   side_effect=RuntimeError("Internal failure")):
            response = client.post("/api/diabetes/analyze", json=VALID_BODY)

        assert response.status_code in (500, 503)
        text = response.text.lower()
        assert "traceback" not in text
        assert "file " not in text
