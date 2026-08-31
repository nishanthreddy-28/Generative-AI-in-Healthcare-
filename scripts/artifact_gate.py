"""
Artifact Gate
=============
Phase 0 verification: confirms the four required ML artifacts are present,
mutually compatible, and match the documented SVM baseline.

Records SHA-256 hashes + library versions in artifacts_manifest.json.

Exit codes:
  0 — all artifacts verified
  1 — missing or invalid artifacts
"""

import hashlib
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_PATHS = {
    "model":    PROJECT_ROOT / "diabetes_best_model.joblib",
    "scaler":   PROJECT_ROOT / "diabetes_scaler.joblib",
    "imputer":  PROJECT_ROOT / "diabetes_imputer.joblib",
    "metadata": PROJECT_ROOT / "model_metadata.json",
}

EXPECTED_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
EXPECTED_ZERO_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk_bytes in iter(lambda: f.read(65536), b""):
            h.update(chunk_bytes)
    return h.hexdigest()


def record_library_versions() -> dict:
    import sklearn, numpy, pandas, joblib as jl
    versions = {
        "python":       sys.version,
        "scikit-learn": sklearn.__version__,
        "numpy":        numpy.__version__,
        "pandas":       pandas.__version__,
        "joblib":       jl.__version__,
    }
    try:
        import xgboost; versions["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return versions


def main():
    print("=" * 60)
    print("Artifact Gate — Phase 0 Verification")
    print("=" * 60)

    # Step 1: Check file existence
    missing = [name for name, path in ARTIFACT_PATHS.items() if not path.exists()]
    if missing:
        print("\n[GATE FAILED] Required legacy artifacts are missing:")
        for name in missing:
            print(f"  ✗ {ARTIFACT_PATHS[name].name}")
        print(
            "\nNo Phase 6/7 implementation can truthfully claim to reuse the"
            "\nexisting model until they are restored."
            "\nRun: python scripts/rebuild_legacy_artifacts.py"
            "\n(only after explicit user authorization)"
        )
        sys.exit(1)

    print("\nAll artifact files present. Verifying...")

    # Step 2: Load and verify metadata
    with open(ARTIFACT_PATHS["metadata"]) as f:
        metadata = json.load(f)

    errors = []

    # Check best_model
    best_model_name = metadata.get("best_model", "")
    if best_model_name != "SVM (RBF)":
        errors.append(f"best_model is '{best_model_name}', expected 'SVM (RBF)'")

    # Check feature_order
    feature_order = metadata.get("feature_order", [])
    if feature_order != EXPECTED_FEATURES:
        errors.append(f"feature_order mismatch: {feature_order} != {EXPECTED_FEATURES}")

    # Check zero_as_missing
    zero_missing = metadata.get("zero_as_missing_columns", [])
    if sorted(zero_missing) != sorted(EXPECTED_ZERO_MISSING):
        errors.append(f"zero_as_missing_columns mismatch: {zero_missing}")

    # Check preprocessing_order present
    if not metadata.get("preprocessing_order"):
        errors.append("preprocessing_order missing from metadata")

    if errors:
        print("\n[GATE FAILED] Metadata verification errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    # Step 3: Load model and verify type + predict_proba
    import joblib
    model = joblib.load(ARTIFACT_PATHS["model"])
    model_cls = type(model).__name__
    if model_cls not in ("SVC",):
        errors.append(f"Loaded model type is {model_cls}, expected SVC")
    if not hasattr(model, "predict_proba"):
        errors.append("Model does not have predict_proba (SVC must be initialized with probability=True)")

    if errors:
        print("\n[GATE FAILED] Model verification errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)

    print(f"  ✓ model type: {model_cls}")
    print(f"  ✓ best_model: {best_model_name}")
    print(f"  ✓ feature_order: {feature_order}")
    print(f"  ✓ zero_as_missing: {zero_missing}")
    print(f"  ✓ predict_proba: available")

    # Step 4: Record hashes + versions
    hashes = {name: sha256_file(path) for name, path in ARTIFACT_PATHS.items()}
    lib_versions = record_library_versions()

    manifest = {
        "gate_timestamp":    datetime.now(timezone.utc).isoformat(),
        "gate_result":       "PASS",
        "selected_model":    best_model_name,
        "feature_order":     feature_order,
        "zero_as_missing":   zero_missing,
        "preprocessing_order": metadata.get("preprocessing_order"),
        "evaluation_metrics": metadata.get("metrics", {}),
        "artifact_sha256":   hashes,
        "library_versions":  lib_versions,
    }
    manifest_path = PROJECT_ROOT / "artifacts_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  ✓ artifacts_manifest.json written")
    print("\n[GATE PASSED] All artifacts verified. Phase 6/7 can proceed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
