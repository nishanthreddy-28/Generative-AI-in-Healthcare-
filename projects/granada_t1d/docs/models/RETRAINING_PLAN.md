# Proposed corrected diabetes research release — approval required to execute

**Execution update:** The user subsequently replied “do whats best” to the specific isolated-retraining approval request. A new CGM-only release was trained and evaluated under `Phase_5_Application/models/cgm/release_001`. The lab/undated-diagnosis inputs were excluded because their contracts remain unresolved; original multimodal artifacts were preserved. [CGM_MODEL_CARD.md](CGM_MODEL_CARD.md) records actual preparation, models, selection, test performance and limitations. The proposal below is retained as the earlier plan; broader multimodal correction/external clinical validation remains future work.

This is a concrete plan for the scientific work still needed after application completion. **No training has been executed.** The user's original instruction requires permission before retraining. No original dataset, weight, scaler, index or result will be replaced by this proposed release.

## Artifacts and scope

Create a new `Phase_5_Application/models/cgm/` directory with a data/split manifest, train-fitted preprocessing, model checkpoints, evaluation results and model cards. Preserve all existing `Phase_*` artifacts. Use the authorized local T1DiabetesGranada files only. No paid API, dependency installation, model download or public publication is included in retraining approval. Existing installed CPU/GPU availability must be measured before setting execution budgets; no duration or accuracy is promised.

## Execution order

1. Resolve source biomarker units and diagnostic-code conventions from the data documentation. Treat missing diagnostic information as unknown rather than an automatic negative. Block incompatible patient-upload mappings until these contracts are established.
2. Correct per-biomarker backward-asof assembly using observation timestamps; handle exact duplicates explicitly, record missingness/staleness, and prevent any future observation from informing a prior day's input. Preserve raw data and build a versioned derived dataset.
3. Save reproducible, sorted, patient-disjoint train/validation/test manifests. Reserve a temporal evaluation within held-out patient histories. Fit imputation/scaling only on training patients and freeze them before validation/test transforms. Verify no patient/window overlap and no future-feature access.
4. Establish last-value, rolling-mean and appropriately fitted linear/autoregressive baselines. Compare the existing two-branch VAE/GRU design with a modest sequence model under equal data/split/tuning budgets. A larger model is not automatically selected. Use validation results for selection and early stopping; do not tune on the final test set.
5. Train a new VAE if retained, audit reconstruction, latent stability and subgroup behavior; train the selected forecaster. Evaluate conditional generation separately for fidelity, diversity, constraints and allowed privacy checks. Separate conditional tabular samples from longitudinal forecasts and from causal treatment scenarios.
6. Calibrate any predictive intervals on a reserved calibration set using a method compatible with temporal dependence; test empirical coverage and interval width across horizons/subgroups. Latent-noise spread alone is not predictive uncertainty.
7. Run patient-held-out and temporal evaluation, report MAE/RMSE per horizon and target, compare against baselines, and explicitly document failure modes. Require finite outputs, meaningful baseline performance and uncertainty evidence before enabling a new model release. A failed evaluation leaves the experimental feature labelled/withheld; it does not justify a clinical reliability claim.
8. Register only evaluated artifact hashes, update the upload preprocessing contract and model cards, execute regression/end-to-end checks, and prepare a reviewable release diff. External validation and clinician review remain prerequisites for expanding intended use.

## Approval decision

The requested authorization is **local retraining and evaluation into a new versioned artifact directory, preserving originals**. Further approval will be needed for installing anything, downloading new pretrained models, paid APIs or publishing. Clinical readiness cannot be granted merely by approving this training run.
