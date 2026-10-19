# Phase 3 · Generation and forecasting

This folder contains the original generative and forecasting experiments:

- `phase3_ctgan.py`: conditional tabular generation of the next 30-day summary state.
- `phase3_4_trajectory_forecasting.py`: GRU forecasts for six daily targets over 60 days.
- `train_comparative_models.py`: LSTM and temporal-GAN comparisons.

The saved result JSONs describe historical runs. Weights and patient-derived inputs are local-only. The tabular generator and daily forecaster are separate models.

Before reusing results, resolve checkpoint-selection/evaluation mismatches, feature clipping, count normalization, upstream imputation leakage, and clustered evaluation. The [research audit](../docs/reports/RESEARCH_AUDIT_2026-10-19.md) documents those issues. The active corrected CGM release and its baseline comparison are described in the [model card](../docs/models/CGM_MODEL_CARD.md).
