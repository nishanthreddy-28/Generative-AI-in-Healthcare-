# Corrected CGM research release 001

Trained locally on 2026-10-11 after the user's approval to proceed with isolated retraining. Original datasets, models, scalers and results remain unchanged. Artifacts are private local files under `Phase_5_Application/models/cgm/release_001`; they are excluded from Git and the proposed core Docker context.

## Intended use

Educational simulation of **daily glucose patterns in type 1 diabetes**, using 30 consecutive days of measured CGM history. The output horizon is 60 days after the selected observation window. This is not a next-reading insulin controller, a validated personal prognosis, a disease-risk model or a causal treatment simulator. Report labs, diagnoses and medication mentions remain in the separate cited clinical-history explanation; they are not used to manufacture missing forecasting inputs.

## Preparation and model

- Raw source: `Phase_1_Data_Preparation/raw_data/Glucose_measurements.csv`, T1DiabetesGranada. **22,671,708 raw rows** processed in bounded chunks. The original cleaned multimodal table and its scalers were not used to train this release.
- Identical timestamps with different values are discarded during research curation; uploaded conflicts require user resolution. **22,026 conflicting timestamps** were discarded. Observed readings are averaged into occupied 15-minute bins; **354,937 readings** were combined in bins. No missing reading is interpolated. Days require at least 64 occupied bins; **23,602 low-coverage days** were excluded. The first overly strict preparation attempt failed before fitting models; its files/log are retained separately.
- Only measured daily mean glucose, time in 70–180 mg/dL, time below 70 mg/dL and coefficient of variation are modeled. Time above range and GMI are derived coherently. GMI is not measured HbA1c. Unresolved source lab units and undated diagnostic histories are excluded rather than guessed.
- Patient-disjoint groups: **187 training, 47 validation, 39 calibration and 39 test patients**; 312 patients with a qualifying 90-day segment. Training/validation contain 1,948/557 windows. Calibration/test use one window per patient to avoid treating overlapping patient windows as independent interval scores.
- Means and standard deviations are fitted on training patients only. No target or future observation enters the 30-day input. Sorted IDs, seed 42 and deterministic CPU algorithms define reproducible splits/training. Data selection still favors patients with adequate sensor coverage.
- A GRU VAE represents observed patterns in 16 dimensions. A separate GRU/direct head predicts the next 60 daily four-metric vectors. Each model trained for a maximum of 20 epochs; checkpoint selection used validation loss. The release took approximately **106.6 seconds** on this host, including fresh raw preparation. This is one local measurement, not a deployment budget.

## Selection and independent test results

Candidate selection used validation patient-averaged, training-standardized absolute error across all four targets. The validation scores were last day 0.78419, seven-day mean 0.65262, 30-day mean 0.62877 and VAE/GRU 0.62630. The neural model was selected **before using the test metrics**. Its validation advantage over the 30-day mean is only about **0.39%**; no statistically established superiority is claimed.

| Candidate | Mean glucose MAE, mg/dL | TIR MAE, percentage points | TBR MAE, percentage points | CV MAE, percentage points |
|---|---:|---:|---:|---:|
| Last observed day | 32.16 | 20.21 | 4.46 | 8.56 |
| Past seven-day mean | 25.53 | 16.55 | 3.43 | 6.77 |
| Past 30-day mean | **23.33** | **15.27** | 3.68 | **6.54** |
| Validation-selected VAE/GRU | 23.97 | 15.58 | **3.65** | 6.57 |

These are errors over 60-day daily predictions for one held-out window per test patient. The neural model did **not** improve test glucose error over the 30-day mean comparator. Test glucose RMSE is 31.49 mg/dL; day-60 glucose MAE is 28.77 mg/dL. Both curves are displayed in the app. No tuning was performed using the test result and no clinical-reliability claim follows from model complexity or test-suite success.

The VAE's test normalized reconstruction error and latent dispersion are recorded in `release.json`. These demonstrate executed encoding/reconstruction evaluation, not a comprehensive latent clinical validity assessment. Clinical subgroup/external-site/drift validation has not been performed.

## Calibration and generation

Split-conformal absolute residual radii are estimated per horizon/target from one window per calibration patient, targeting nominal 80% marginal coverage. On test patients, aggregate daily coverage was **77.44% glucose, 77.39% TIR, 77.22% TBR and 79.15% CV**. Patient exchangeability is an assumption; observations within a trajectory are dependent. These numbers do not establish simultaneous 60-day coverage, conditional coverage for an individual, or clinical safety. Bounds are clipped to feasible marginal ranges, and bounds for different metrics are not a joint feasible patient state.

Possible paths use a zero-mean Gaussian residual model with a low-rank temporal/cross-metric factor and remaining diagonal variance fitted on calibration residuals. This supplies correlated research sensitivity paths. It is **not** a cGAN treatment-response model or a guarantee that generated physiology is clinically plausible. Constraints bound values, keep TIR/TBR/TAR totals consistent and derive GMI. Raw patient trajectories are not saved in the inference checkpoint; no privacy guarantee or memorization audit is claimed.

## Serving and safeguards

Input CSV is `timestamp,glucose_mg_dl`, one person, ISO local timestamps and measured mg/dL. The user must confirm type 1 diabetes, source/unit review and (when linking a report) the same patient. Invalid dates/values, conflicting timestamps, insufficient consecutive coverage and extreme scaler mismatch withhold simulation. Older qualifying windows are explicitly dated and the displayed historical curve uses that exact input window. Forecast dates start after that window, which may be before today.

The release manifest verifies four required artifacts before restricted tensor loading. NumPy calibration loading disables pickle. Output is deterministic for seed 42 and bounded path count. Unknown/changed artifact bytes fail closed. Data or artifact changes need a new reviewed release and server restart.

## Remaining limits

Only 39 test/calibration patients, selected coverage, one regional type 1 diabetes dataset from 2018–2022, no external validation, no illness/meal/insulin context, limited baseline/model search, no clinical subgroup testing and modest predictive performance. This is a corrected **research** release, not a proven best medical model. Public deployment requires operational/security/data-rights validation; clinical use remains **Not Ready**.

Executed training evidence: training log (retained locally under `evidence/implementation_20261011/retraining_v2_release001.log`). Regression and serving evidence: [current workflow report](../reports/WORKFLOW_COMPLETION_REPORT.md). The publication describes source-data access under specific permission: [T1DiabetesGranada publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC10733323/).
