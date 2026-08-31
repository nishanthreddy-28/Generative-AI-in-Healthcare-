"""
Feature Explanation Service
============================
Deterministic local model-sensitivity analysis.

For each feature, we perturb its raw value by ±1 standard deviation
(using the scaler's standard deviation), then reapply the preprocessing pipeline.
Values are clipped to biologically valid bounds before computing the counterfactual.
If the clipped value equals the original (e.g. already at the boundary), the
delta will be near-zero but is still recorded.

Output is clearly labeled:
  "Local model sensitivity, not causal importance."
  "These are exploratory sensitivity estimates only."

This is passed to the LLM as structured input.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_sensitivity(
    prediction_service,  # PredictionService instance (avoids circular import)
    patient_input: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Compute local model sensitivity for each feature.

    For each feature f:
      1. Perturb the raw value by +1 std and -1 std (using the scaler's scale_).
      2. Clip to valid bounds. If invalid, skip.
      3. Reapply preprocessing (imputer + scaler).
      4. Compute delta_prob = P(positive | perturbed) - P(positive | baseline)

    Args:
        prediction_service: loaded PredictionService singleton
        patient_input:      validated dict of 8 Pima feature values

    Returns:
        dict mapping feature name -> {"positive_delta": float, "negative_delta": float}
    """
    ps = prediction_service
    feat_order = ps.feature_order
    zero_missing = ps._zero_as_missing
    
    from services.prediction_service import FEATURE_UPPER_BOUNDS

    # Baseline prediction
    raw = pd.DataFrame([{f: patient_input[f] for f in feat_order}])
    for col in zero_missing:
        raw[col] = raw[col].replace(0, np.nan)

    imputed_arr = ps._imputer.transform(raw)
    imputed = pd.DataFrame(imputed_arr, columns=feat_order)
    scaled_arr = ps._scaler.transform(imputed)
    baseline_scaled = pd.DataFrame(scaled_arr, columns=feat_order)

    baseline_prob = float(ps._model.predict_proba(baseline_scaled)[0][1])

    sensitivity: dict[str, dict[str, float]] = {}

    for i, feat in enumerate(feat_order):
        raw_val = patient_input[feat]
        std_val = float(ps._scaler.scale_[i])

        deltas = {}
        
        for direction, sign in [("positive_delta", 1), ("negative_delta", -1)]:
            new_val = raw_val + sign * std_val
            
            # Clip bounds
            if new_val < 0:
                new_val = 0
            if feat in FEATURE_UPPER_BOUNDS and new_val > FEATURE_UPPER_BOUNDS[feat]:
                new_val = FEATURE_UPPER_BOUNDS[feat]
            
            # Note: For Pregnancies=0 it's valid, others are handled by the imputer
            # but if it was exactly 0 after clipping, and it's a zero-as-missing feature,
            # we shouldn't skip it, the imputer will just handle it (impute it).
            # So no need to skip `continue` anywhere now.
                
            # Compute counterfactual
            cf_raw = patient_input.copy()
            cf_raw[feat] = new_val
            
            cf_df = pd.DataFrame([{f: cf_raw[f] for f in feat_order}])
            for col in zero_missing:
                cf_df[col] = cf_df[col].replace(0, np.nan)
                
            cf_imputed = ps._imputer.transform(cf_df)
            cf_scaled = ps._scaler.transform(pd.DataFrame(cf_imputed, columns=feat_order))
            
            cf_prob = float(ps._model.predict_proba(cf_scaled)[0][1])
            deltas[direction] = round(cf_prob - baseline_prob, 4)

        if deltas:
            sensitivity[feat] = deltas

    logger.info(
        "Sensitivity computed for %d features (baseline_prob=%.4f)",
        len(sensitivity), baseline_prob
    )
    return sensitivity
