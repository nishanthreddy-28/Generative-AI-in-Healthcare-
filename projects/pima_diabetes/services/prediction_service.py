"""
Prediction Service
==================
Loads and uses the saved ML artifacts (read-only) to produce predictions.

Preprocessing pipeline (matches Notebook Cell 1 exactly):
  Raw patient input
    → validate fields
    → replace zeros in zero_as_missing_columns with NaN
    → KNNImputer (loaded from diabetes_imputer.joblib)
    → StandardScaler (loaded from diabetes_scaler.joblib)
    → SVC predict + predict_proba (loaded from diabetes_best_model.joblib)

The artifacts are loaded once at startup; the service is a singleton.

Input validation:
  - All 8 Pima features required (Outcome is NOT accepted as input)
  - Values must be numeric, finite, non-negative
  - Zeros allowed only for zero_as_missing_columns (from metadata)
  - Documented biological upper bounds are enforced
"""

import json
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_PATHS = {
    "model":    PROJECT_ROOT / "diabetes_best_model.joblib",
    "scaler":   PROJECT_ROOT / "diabetes_scaler.joblib",
    "imputer":  PROJECT_ROOT / "diabetes_imputer.joblib",
    "metadata": PROJECT_ROOT / "model_metadata.json",
}

# Biological upper bounds — values above these are rejected as implausible
FEATURE_UPPER_BOUNDS: dict[str, float] = {
    "Pregnancies":              20.0,
    "Glucose":                 500.0,
    "BloodPressure":           200.0,
    "SkinThickness":           100.0,
    "Insulin":                 900.0,
    "BMI":                      70.0,
    "DiabetesPedigreeFunction":  3.0,
    "Age":                     120.0,
}


class ArtifactLoadError(RuntimeError):
    """Raised when required ML artifacts are missing or incompatible."""


class ValidationError(ValueError):
    """Raised for invalid patient input."""


class PredictionService:
    """Singleton prediction service."""

    _instance: "PredictionService | None" = None

    def __init__(self):
        self._model        = None
        self._scaler       = None
        self._imputer      = None
        self._metadata: dict[str, Any] = {}
        self._feature_order: list[str] = []
        self._zero_as_missing: list[str] = []
        self._loaded = False

    @classmethod
    def get(cls) -> "PredictionService":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self) -> None:
        """Load and verify all artifacts."""
        missing = [
            name for name, path in ARTIFACT_PATHS.items()
            if not path.exists()
        ]
        if missing:
            raise ArtifactLoadError(
                f"ML artifacts are missing: {', '.join(missing)}.\n"
                "Run: python scripts/rebuild_legacy_artifacts.py"
            )

        self._model   = joblib.load(ARTIFACT_PATHS["model"])
        self._scaler  = joblib.load(ARTIFACT_PATHS["scaler"])
        self._imputer = joblib.load(ARTIFACT_PATHS["imputer"])

        with open(ARTIFACT_PATHS["metadata"]) as f:
            self._metadata = json.load(f)

        # ── Verify model type ──────────────────────────────────────────────
        model_cls = type(self._model).__name__
        if model_cls not in ("SVC",):
            raise ArtifactLoadError(
                f"Expected SVC model, got {model_cls}. Artifacts may be from wrong run."
            )

        # ── Verify predict_proba is available ─────────────────────────────
        if not hasattr(self._model, "predict_proba"):
            raise ArtifactLoadError("Loaded SVC model does not expose predict_proba.")

        # ── Verify metadata best_model field ──────────────────────────────
        meta_best = self._metadata.get("best_model", "")
        if meta_best != "SVM (RBF)":
            raise ArtifactLoadError(
                f"metadata best_model is '{meta_best}', expected 'SVM (RBF)'. "
                "Artifacts may be from a different training run."
            )

        # ── Verify model classes are exactly [0, 1] ───────────────────────
        if hasattr(self._model, "classes_"):
            classes = list(self._model.classes_)
            if classes != [0, 1]:
                raise ArtifactLoadError(
                    f"Model classes {classes} != [0, 1]. "
                    "The artifact was not trained on the expected binary target."
                )
        else:
            raise ArtifactLoadError("Model does not expose classes_ attribute.")

        # ── Verify exact 8-feature order from metadata ────────────────────
        self._feature_order   = self._metadata.get("feature_order", [])
        if len(self._feature_order) != 8:
            raise ArtifactLoadError(
                f"Expected 8 features in feature_order, got {len(self._feature_order)}."
            )
        expected_features = [
            "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
        ]
        if self._feature_order != expected_features:
            raise ArtifactLoadError(
                f"feature_order mismatch: {self._feature_order} != {expected_features}."
            )

        self._zero_as_missing = self._metadata.get("zero_as_missing_columns", [])

        # ── Verify artifact hashes match manifest if present ─────────────
        manifest_path = ARTIFACT_PATHS.get("manifest") or PROJECT_ROOT / "artifacts_manifest.json"
        if manifest_path and Path(manifest_path).exists():
            try:
                with open(manifest_path) as f:
                    saved_manifest = json.load(f)
                saved_hashes = saved_manifest.get("artifact_sha256", {})
                if saved_hashes:
                    for key in ("model", "scaler", "imputer", "metadata"):
                        p = ARTIFACT_PATHS.get(key)
                        if p and p.exists() and key in saved_hashes:
                            import hashlib
                            h = hashlib.sha256()
                            with open(p, "rb") as fh:
                                for chunk_bytes in iter(lambda: fh.read(65536), b""):
                                    h.update(chunk_bytes)
                            actual = h.hexdigest()
                            if actual != saved_hashes[key]:
                                raise ArtifactLoadError(
                                    f"Artifact hash mismatch for '{key}': "
                                    f"current={actual[:16]}... expected={saved_hashes[key][:16]}... "
                                    "Artifact may have been modified after verification."
                                )
            except ArtifactLoadError:
                raise
            except Exception as e:
                logger.warning("Could not verify artifact hashes: %s", type(e).__name__)

        logger.info(
            "Prediction service loaded: model=%s, features=%s",
            self._metadata.get("best_model"), self._feature_order
        )
        self._loaded = True

    def _validate(self, patient_input: dict[str, Any]) -> None:
        """Validate patient input. Raises ValidationError with a descriptive message."""
        # Must not include Outcome
        if "Outcome" in patient_input:
            raise ValidationError("'Outcome' must not be included as an input feature.")

        # All 8 features required
        missing_fields = [f for f in self._feature_order if f not in patient_input]
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        # No extra fields
        extra_fields = [k for k in patient_input if k not in self._feature_order]
        if extra_fields:
            raise ValidationError(f"Unexpected input fields: {extra_fields}")

        for feat in self._feature_order:
            val = patient_input[feat]

            # Must be numeric
            if not isinstance(val, (int, float)):
                raise ValidationError(
                    f"'{feat}' must be a number, got {type(val).__name__!r}."
                )

            # Must be finite
            if not math.isfinite(val):
                raise ValidationError(f"'{feat}' must be a finite number (got {val}).")

            # Must be non-negative
            if val < 0:
                raise ValidationError(f"'{feat}' must be non-negative (got {val}).")

            # Zero value validation
            if val == 0:
                if feat in self._zero_as_missing:
                    pass  # Zero means missing and is imputed
                elif feat == "Pregnancies":
                    pass  # Zero is a valid number of pregnancies
                else:
                    # DiabetesPedigreeFunction and Age cannot be zero
                    raise ValidationError(
                        f"'{feat}' cannot be 0. Biologically implausible for this feature."
                    )

            # Upper bound check
            upper = FEATURE_UPPER_BOUNDS.get(feat)
            if upper is not None and val > upper:
                raise ValidationError(
                    f"'{feat}' value {val} exceeds maximum plausible value {upper}."
                )

    def predict(self, patient_input: dict[str, Any]) -> dict[str, Any]:
        """
        Run the full preprocessing + prediction pipeline.

        Args:
            patient_input: dict with exactly the 8 Pima feature names

        Returns:
            {
              "prediction":        int (0 or 1),
              "model_probability": float (probability of positive class),
              "risk_label":        str (model-safe label),
            }

        Raises:
            ValidationError on invalid input
            ArtifactLoadError if artifacts are unavailable
        """
        self._validate(patient_input)

        # Build DataFrame in the exact feature order from metadata
        raw = pd.DataFrame([{f: patient_input[f] for f in self._feature_order}])

        # Replace impossible zeros with NaN (zero_as_missing_columns)
        for col in self._zero_as_missing:
            raw[col] = raw[col].replace(0, np.nan)

        # Apply saved KNNImputer
        imputed_arr = self._imputer.transform(raw)
        imputed = pd.DataFrame(imputed_arr, columns=self._feature_order)

        # Apply saved StandardScaler
        scaled_arr = self._scaler.transform(imputed)
        scaled = pd.DataFrame(scaled_arr, columns=self._feature_order)

        # Predict — use model.classes_ to identify the positive-class index
        prediction      = int(self._model.predict(scaled)[0])
        probability_arr = self._model.predict_proba(scaled)[0]
        classes         = list(self._model.classes_)
        pos_idx         = classes.index(1)   # position of class 1 in classes_
        model_probability = float(probability_arr[pos_idx])

        risk_label = (
            "Model predicts positive diabetes class"
            if prediction == 1
            else "Model predicts negative diabetes class"
        )

        logger.info(
            "Prediction generated: class=%d, model_probability=%.4f",
            prediction, model_probability
        )

        return {
            "prediction":        prediction,
            "model_probability": round(model_probability, 4),
            "risk_label":        risk_label,
        }

    @property
    def feature_order(self) -> list[str]:
        return self._feature_order

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
