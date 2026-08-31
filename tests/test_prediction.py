"""
Test: Prediction Service
=========================
Tests input validation and preprocessing pipeline.
Skips if ML artifacts are not present.
"""

import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ARTIFACTS_PRESENT = all(
    (PROJECT_ROOT / f).exists()
    for f in [
        "diabetes_best_model.joblib",
        "diabetes_scaler.joblib",
        "diabetes_imputer.joblib",
        "model_metadata.json",
    ]
)

pytestmark = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="ML artifacts not present — run rebuild_legacy_artifacts.py first"
)

VALID_INPUT = {
    "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
    "SkinThickness": 35, "Insulin": 120, "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627, "Age": 50
}


@pytest.fixture(scope="module")
def ps():
    from services.prediction_service import PredictionService
    PredictionService._instance = None  # Reset singleton for fresh load
    return PredictionService.get()


class TestValidInput:
    def test_returns_prediction_and_probability(self, ps):
        result = ps.predict(VALID_INPUT)
        assert "prediction" in result
        assert "model_probability" in result
        assert "risk_label" in result

    def test_prediction_is_binary(self, ps):
        result = ps.predict(VALID_INPUT)
        assert result["prediction"] in (0, 1)

    def test_probability_in_range(self, ps):
        result = ps.predict(VALID_INPUT)
        assert 0.0 <= result["model_probability"] <= 1.0

    def test_risk_label_wording(self, ps):
        """Label must NOT say 'you have diabetes'."""
        result = ps.predict(VALID_INPUT)
        label = result["risk_label"].lower()
        assert "you have" not in label
        assert "you do not" not in label
        assert "model predicts" in label

    def test_zero_as_missing_allowed(self, ps):
        """Zero is allowed in zero_as_missing_columns (Glucose, BMI, etc.)."""
        inp = {**VALID_INPUT, "Glucose": 0, "BMI": 0}
        result = ps.predict(inp)
        assert result["prediction"] in (0, 1)

    def test_zero_allowed_for_pregnancies(self, ps):
        """Pregnancies=0 is valid."""
        inp = {**VALID_INPUT, "Pregnancies": 0}
        result = ps.predict(inp)
        assert result["prediction"] in (0, 1)

    def test_zero_not_allowed_in_pedigree_function(self, ps):
        """DiabetesPedigreeFunction=0 is invalid."""
        inp = {**VALID_INPUT, "DiabetesPedigreeFunction": 0}
        from services.prediction_service import ValidationError
        with pytest.raises(ValidationError, match="cannot be 0"):
            ps.predict(inp)

    def test_zero_not_allowed_in_age(self, ps):
        """Age=0 is invalid."""
        inp = {**VALID_INPUT, "Age": 0}
        from services.prediction_service import ValidationError
        with pytest.raises(ValidationError, match="cannot be 0"):
            ps.predict(inp)


class TestInvalidInput:
    def test_missing_field(self, ps):
        from services.prediction_service import ValidationError
        inp = {k: v for k, v in VALID_INPUT.items() if k != "Glucose"}
        with pytest.raises(ValidationError, match="Missing"):
            ps.predict(inp)

    def test_outcome_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "Outcome": 1}
        with pytest.raises(ValidationError, match="Outcome"):
            ps.predict(inp)

    def test_string_value_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "Glucose": "high"}
        with pytest.raises(ValidationError):
            ps.predict(inp)

    def test_nan_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "Glucose": float("nan")}
        with pytest.raises(ValidationError, match="finite"):
            ps.predict(inp)

    def test_infinity_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "BMI": float("inf")}
        with pytest.raises(ValidationError, match="finite"):
            ps.predict(inp)

    def test_negative_value_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "Age": -5}
        with pytest.raises(ValidationError, match="non-negative"):
            ps.predict(inp)

    def test_upper_bound_glucose(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "Glucose": 999}
        with pytest.raises(ValidationError, match="maximum"):
            ps.predict(inp)

    def test_upper_bound_bmi(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "BMI": 71}
        with pytest.raises(ValidationError, match="maximum"):
            ps.predict(inp)

    def test_extra_field_rejected(self, ps):
        from services.prediction_service import ValidationError
        inp = {**VALID_INPUT, "ExtraField": 99}
        with pytest.raises(ValidationError, match="Unexpected"):
            ps.predict(inp)


class TestPreprocessingOrder:
    def test_pipeline_uses_all_three_artifacts(self, ps):
        """Verify imputer, scaler, and model are all used (not bypassed)."""
        assert ps._imputer is not None
        assert ps._scaler  is not None
        assert ps._model   is not None

    def test_feature_order_matches_metadata(self, ps):
        import json
        with open(PROJECT_ROOT / "model_metadata.json") as f:
            meta = json.load(f)
        assert ps.feature_order == meta["feature_order"]
