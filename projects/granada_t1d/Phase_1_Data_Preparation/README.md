# Phase 1: Ingestion, Audit, Clinical NLP & Feature Engineering

Raw clinical CSVs are in `raw_data/`; curation and exploration notebooks are in `notebooks/`. The ingestion script and prepared outputs stay in this phase folder.

> Current location: `Phase_1_Data_Preparation`. From the BTP root, run `python scripts/run_phase_1.py`. These commands regenerate original research outputs. See [project structure](../docs/PROJECT_STRUCTURE.md) and [the audit](../docs/reports/PROJECT_AUDIT_REPORT.md).

## Overview
Phase 1 serves as the **single source of truth** for data ingestion, missing-value auditing, cohort selection, Clinical NLP extraction, and temporal feature alignment. It converts raw heterogeneous clinical observations into a high-quality, standardized longitudinal multi-modal daily dataset with **0 missing values**.

---

## Pipeline Components & Methodologies

The entire Phase 1 pipeline is consolidated within `phase1_ingestion.py`:

### 1. Streaming Missing Value Audit
* Audits ~22.6 million CGM rows via chunked streaming without out-of-memory overhead.
* Matches laboratory biomarkers with case-insensitive and alias-aware lookups.
* Quantifies patient-level missingness and verifies imputation bounds.
* Generates audit report: `missing_values_audit_report.json`.

### 2. Clinical NLP Entity Extraction
* Extracts multi-hot comorbidity vectors from free-text electronic medical record descriptions using clinical keyword matching and medical taxonomy mappings.
* Identifies 7 key diabetes-related conditions:
  - Diabetic Retinopathy (`comorbidity_retinopathy`)
  - Diabetic Nephropathy (`comorbidity_nephropathy`)
  - Diabetic Neuropathy (`comorbidity_neuropathy`)
  - Cardiovascular Disease (`comorbidity_cardiovascular`)
  - Hypertension (`comorbidity_hypertension`)
  - Dyslipidemia (`comorbidity_dyslipidemia`)
  - Thyroid Disorders (`comorbidity_thyroid`)

### 3. Cohort Selection & Multi-Modal Eligibility
* Enforces strict inclusion criteria:
  - Continuous sensor wear duration $\\ge 90$ active days.
  - Multi-modal completeness: presence of synchronized laboratory biomarker blood tests.
* Retains a high-confidence cohort of **593 patients** (`cleaned_cohort_metadata.csv`).

### 4. Daily Glycemic Aggregation & Consensus Metrics
* Processes raw 15-minute sensor data into daily summaries:
  - Physiological plausibility bounds: $[40, 500]$ mg/dL.
  - Sensor adherence threshold: $\\ge 70\\%$ valid wear per 24 hours ($\\ge 67$ measurements).
* International consensus glycemic metrics computed per patient-day:
  - **Mean Glucose** ($\\text{mg/dL}$), **Standard Deviation** ($\\text{SD}$), and **Coefficient of Variation** ($\\text{CV} = \\frac{\\text{SD}}{\\text{Mean}}$).
  - **Time-in-Range (TIR)**: Glucose between 70 and 180 mg/dL.
  - **Time-Above-Range (TAR L1 & L2)**: $>180$ mg/dL and $>250$ mg/dL.
  - **Time-Below-Range (TBR L1 & L2)**: $<70$ mg/dL and $<54$ mg/dL.
  - **Glucose Management Indicator (GMI)**: Clinically estimated HbA1c ($3.31 + 0.02392 \\times \\text{Mean Glucose}$).
  - **Diurnal metrics**: Daytime vs. Nocturnal (00:00–06:00) mean glucose and TBR.

### 5. Strict Zero-Leakage Prospective Laboratory Merging
* Performs backward `merge_asof` with a maximum lookback window of 180 days.
* **Strict anti-leakage guarantee**: Future laboratory values are never merged backward into earlier timepoints.
* Per-column binary imputation flags (`*_imputed`) recorded for all 14 laboratory parameters to maintain transparent auditability of observed vs. imputed states.
* Dynamic rolling age computed per date (`record_year - birth_year`) reflecting true longitudinal aging.

---

## Dataset Schema: 50 Canonical Features

The primary output `cleaned_daily_multimodal_dataset.csv` contains 221,366 clean daily records with **zero null values**, structured as:

1. **27 Pure Dynamic Daily Metrics**:
   - Glycemic (13): `mean_glucose`, `glucose_std`, `cv`, `min_glucose`, `max_glucose`, `tir_70_180`, `tar_180_250`, `tar_above_250`, `tbr_54_70`, `tbr_below_54`, `gmi`, `daytime_mean_glucose`, `nighttime_mean_glucose`, `nighttime_tbr_below_70`
   - Synchronized Laboratory Values (14): `lab_hba1c`, `lab_glucose`, `total_cholesterol`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`, `creatinine`, `egfr`, `urea`, `uric_acid`, `microalbuminuria`, `ast`, `alt`, `bmi`
2. **14 Imputation Indicators**:
   - `lab_hba1c_imputed`, `lab_glucose_imputed`, etc. ($1$ if imputed via population median, $0$ if clinically measured).
3. **9 Static Baseline Features**:
   - Demographics: `sex`, `rolling_age`
   - Comorbidities (7): `comorbidity_retinopathy`, `comorbidity_nephropathy`, `comorbidity_neuropathy`, `comorbidity_cardiovascular`, `comorbidity_hypertension`, `comorbidity_dyslipidemia`, `comorbidity_thyroid`

---

## Execution

To run Phase 1:
```bash
# Navigate to Phase 1 folder
cd Phase_1_Ingestion_and_Preprocessing
python3 phase1_ingestion.py
```

### Outputs Produced
* `missing_values_audit_report.json`
* `cleaned_cohort_metadata.csv` (593 patients)
* `cleaned_daily_multimodal_dataset.csv` (221,366 rows, 50 columns)
