"""
REPRODUCIBILITY REBUILD — NOT RETRAINING
=========================================
This script is an authorized reconstruction of Notebook Cell 1 (Steps 1–6)
of Pima.ipynb. It uses the IDENTICAL code, dataset, feature order, model
definitions, selection rule, and RANDOM_STATE=42.

Authorization: Explicitly granted by user (2026-08-27) to reconstruct
missing legacy artifacts. This is not a new experiment or model selection.

Safety guard:
  The script refuses to overwrite existing .joblib files unless --force is
  passed. Use --force only after reviewing the rebuild motivation.

Output files:
  artifacts_rebuild_manifest.json  — full rebuild provenance (dataset hash,
                                       random state, tolerances, authorization)
  (artifacts_manifest.json is written by artifact_gate.py after gate check)

If the selected model is not SVM (RBF), or if metrics deviate beyond
defined tolerances, this script aborts — it does NOT silently accept a
different model.
"""

import argparse
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Tolerances — abort if actual metrics fall outside these bands
# ---------------------------------------------------------------------------
BASELINE_METRICS = {
    "Test_Acc":   (0.844, 0.02),  # 0.824 – 0.864
    "Precision":  (0.727, 0.03),  # 0.697 – 0.757
    "Recall":     (0.889, 0.02),  # 0.869 – 0.909
    "Test_F1":    (0.800, 0.02),  # 0.780 – 0.820
    "ROC_AUC":    (0.898, 0.02),  # 0.878 – 0.918
}

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "diabetes.csv"
OUTPUT_PATHS = {
    "model":    PROJECT_ROOT / "diabetes_best_model.joblib",
    "scaler":   PROJECT_ROOT / "diabetes_scaler.joblib",
    "imputer":  PROJECT_ROOT / "diabetes_imputer.joblib",
    "metadata": PROJECT_ROOT / "model_metadata.json",
    # Rebuild provenance goes here — NOT into artifacts_manifest.json
    # (artifact_gate.py writes artifacts_manifest.json after gate verification)
    "rebuild_manifest": PROJECT_ROOT / "artifacts_rebuild_manifest.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record_library_versions() -> dict:
    import sklearn
    import xgboost
    return {
        "python": sys.version,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    }


def verify_metrics(actual: dict) -> None:
    """Abort if any metric deviates beyond tolerance."""
    violations = []
    for metric, (expected, tol) in BASELINE_METRICS.items():
        val = actual.get(metric)
        if val is None:
            violations.append(f"  {metric}: missing from results")
        elif abs(val - expected) > tol:
            violations.append(
                f"  {metric}: got {val:.4f}, expected {expected:.4f} ± {tol:.4f}"
            )
    if violations:
        print("\n[ABORT] Metric verification FAILED — tolerances exceeded:")
        for v in violations:
            print(v)
        print(
            "\nThe selected model's metrics deviate from the documented SVM baseline.\n"
            "Artifacts have NOT been saved. Review diabetes.csv and library versions."
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Reproducibility rebuild of Pima ML artifacts."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Allow overwriting existing .joblib artifacts. Use with caution."
    )
    args = parser.parse_args()

    # ── Safety guard: refuse to overwrite existing artifacts without --force ──
    joblib_paths = [
        p for k, p in OUTPUT_PATHS.items()
        if k not in ("metadata", "rebuild_manifest") and p.suffix == ".joblib"
    ]
    existing = [p for p in joblib_paths if p.exists()]
    if existing and not args.force:
        print("\n[ABORT] The following artifact files already exist:")
        for p in existing:
            print(f"  {p.name}")
        print(
            "\nTo protect against accidental overwrites, this script requires --force "
            "when artifacts already exist.\n"
            "  python scripts/rebuild_legacy_artifacts.py --force\n"
            "\nReview the rebuild motivation before using --force."
        )
        sys.exit(1)

    if args.force and existing:
        print(f"\n[WARNING] --force flag set. Overwriting {len(existing)} existing artifact(s).")

    print("=" * 65)
    print("REPRODUCIBILITY REBUILD — NOT RETRAINING")
    print("=" * 65)
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")

    lib_versions = record_library_versions()
    print(f"Python    : {lib_versions['python'].splitlines()[0]}")
    print(f"sklearn   : {lib_versions['scikit-learn']}")
    print(f"xgboost   : {lib_versions['xgboost']}")
    print(f"numpy     : {lib_versions['numpy']}")

    # -----------------------------------------------------------------------
    # Verify dataset
    # -----------------------------------------------------------------------
    if not DATA_PATH.exists():
        print(f"\n[ABORT] Dataset not found: {DATA_PATH}")
        sys.exit(1)

    dataset_hash = sha256_file(DATA_PATH)
    print(f"\nDataset   : {DATA_PATH.name}")
    print(f"SHA-256   : {dataset_hash}")

    # -----------------------------------------------------------------------
    # STEP 1: Load dataset  (exact Cell 1 code)
    # -----------------------------------------------------------------------
    COLUMNS = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    df = pd.read_csv(DATA_PATH, header=0, names=COLUMNS)
    print(f"\n[Step 1] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # -----------------------------------------------------------------------
    # STEP 2: EDA + preprocessing  (exact Cell 1 code)
    # -----------------------------------------------------------------------
    ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    missing_report = {}
    for col in ZERO_AS_MISSING:
        n_zero = int((df[col] == 0).sum())
        missing_report[col] = n_zero
        df[col] = df[col].replace(0, np.nan)

    class_counts = df["Outcome"].value_counts().to_dict()
    print(f"[Step 2] Class balance -> Non-diabetic(0): {class_counts.get(0,0)}, "
          f"Diabetic(1): {class_counts.get(1,0)}")

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Step 2] Stratified split -> train: {X_train.shape[0]}, test: {X_test.shape[0]}")

    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test), columns=X.columns, index=X_test.index
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_imputed), columns=X.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_imputed), columns=X.columns, index=X_test.index
    )

    # -----------------------------------------------------------------------
    # STEP 3: Train models  (exact Cell 1 code)
    # -----------------------------------------------------------------------
    scale_pos_weight = class_counts.get(0, 1) / class_counts.get(1, 1)

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=RANDOM_STATE, n_estimators=300, max_depth=4, learning_rate=0.05
        ),
        "SVM (RBF)": SVC(
            class_weight="balanced", probability=True, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=11),
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_train_pred = model.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1  = f1_score(y_train, y_train_pred)

        y_pred  = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        test_acc  = accuracy_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_rec  = recall_score(y_test, y_pred)
        test_f1   = f1_score(y_test, y_pred)
        test_auc  = roc_auc_score(y_test, y_proba)

        results.append({
            "Model": name,
            "Train_Acc": train_acc, "Test_Acc": test_acc,
            "Train_F1":  train_f1,  "Test_F1":  test_f1,
            "Precision": test_prec, "Recall":   test_rec, "ROC_AUC": test_auc,
        })
        fitted_models[name] = model
        print(f"[Step 3] {name:20s} | Train Acc={train_acc:.3f} Test Acc={test_acc:.3f} | "
              f"Recall={test_rec:.3f} F1={test_f1:.3f} AUC={test_auc:.3f}")

    # -----------------------------------------------------------------------
    # STEP 4–5: Select best model  (exact Cell 1 code — Recall-first, F1 tie)
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(results).sort_values(
        by=["Recall", "Test_F1"], ascending=False
    ).reset_index(drop=True)

    best_row  = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = fitted_models[best_name]

    print(f"\n[Step 5] Selected model: {best_name} "
          f"(Recall={best_row['Recall']:.3f}, F1={best_row['Test_F1']:.3f}, "
          f"Acc={best_row['Test_Acc']:.3f}, AUC={best_row['ROC_AUC']:.3f})")

    # -----------------------------------------------------------------------
    # Verify: model must be SVM (RBF)
    # -----------------------------------------------------------------------
    if best_name != "SVM (RBF)":
        print(f"\n[ABORT] Selected model is '{best_name}', not 'SVM (RBF)'.")
        print("This does not match the documented baseline. Artifacts NOT saved.")
        sys.exit(1)

    # Verify: must have predict_proba
    if not hasattr(best_model, "predict_proba"):
        print("\n[ABORT] SVM model does not expose predict_proba. Artifacts NOT saved.")
        sys.exit(1)

    # Verify: metrics within tolerance
    actual_metrics = {
        "Test_Acc":  float(best_row["Test_Acc"]),
        "Precision": float(best_row["Precision"]),
        "Recall":    float(best_row["Recall"]),
        "Test_F1":   float(best_row["Test_F1"]),
        "ROC_AUC":   float(best_row["ROC_AUC"]),
    }
    verify_metrics(actual_metrics)
    print("[Verify] Metrics within tolerance. [OK]")

    # -----------------------------------------------------------------------
    # STEP 6: Save artifacts
    # -----------------------------------------------------------------------
    joblib.dump(best_model, OUTPUT_PATHS["model"])
    joblib.dump(scaler,     OUTPUT_PATHS["scaler"])
    joblib.dump(imputer,    OUTPUT_PATHS["imputer"])

    metadata = {
        "best_model":               best_name,
        "feature_order":            list(X.columns),
        "zero_as_missing_columns":  ZERO_AS_MISSING,
        "metrics": {k: float(v) for k, v in best_row.items() if k != "Model"},
        "class_distribution":       {str(k): int(v) for k, v in class_counts.items()},
        "scale_pos_weight_used_for_xgboost": float(scale_pos_weight),
        "preprocessing_order": [
            "train_test_split(stratified)",
            "KNNImputer(fit on train)",
            "StandardScaler(fit on train)"
        ],
        "_rebuild_note": (
            "REPRODUCIBILITY REBUILD — authorized 2026-08-27. "
            "Identical to Notebook Cell 1. Not a new training run."
        ),
    }
    with open(OUTPUT_PATHS["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Step 6] Artifacts saved:")
    for k, p in OUTPUT_PATHS.items():
        if k == "manifest":
            continue
        print(f"  {p.name}")

    # -----------------------------------------------------------------------
    # Record rebuild manifest (separate from artifacts_manifest.json)
    # -----------------------------------------------------------------------
    artifact_hashes = {
        k: sha256_file(p)
        for k, p in OUTPUT_PATHS.items()
        if k not in ("rebuild_manifest",) and p.exists()
    }

    rebuild_manifest = {
        "build_timestamp":      datetime.now(timezone.utc).isoformat(),
        "build_type":           "REPRODUCIBILITY_REBUILD",
        "authorization_note":   "Explicitly authorized by user 2026-08-27 to reconstruct missing legacy artifacts.",
        "dataset":              {"filename": DATA_PATH.name, "sha256": dataset_hash},
        "library_versions":     lib_versions,
        "selected_model":       best_name,
        "feature_order":        list(X.columns),
        "zero_as_missing":      ZERO_AS_MISSING,
        "preprocessing_order":  metadata["preprocessing_order"],
        "random_state":         RANDOM_STATE,
        "evaluation_metrics":   actual_metrics,
        "metric_tolerances":    {k: {"expected": v[0], "tolerance": v[1]}
                                  for k, v in BASELINE_METRICS.items()},
        "artifact_sha256":      artifact_hashes,
    }
    with open(OUTPUT_PATHS["rebuild_manifest"], "w") as f:
        json.dump(rebuild_manifest, f, indent=2)

    print(f"\n[Manifest] artifacts_rebuild_manifest.json saved (full rebuild provenance).")
    print("  Run python scripts/artifact_gate.py to generate artifacts_manifest.json")
    print("\n[REBUILD COMPLETE] All artifacts verified and saved.")
    print("=" * 65)


if __name__ == "__main__":
    main()
