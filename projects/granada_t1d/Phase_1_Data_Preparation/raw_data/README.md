# Phase 1: Raw clinical data

Original source CSVs live here. Exploration notebooks moved to `../notebooks/exploration`. The documentation below describes the original dataset.

# Phase 0: Raw Dataset Repository

## Overview
This folder contains the original, multi-modal hospital dataset **`T1DiabetesGranada`** published in *Nature Scientific Data* (2023). It spans 4 years of monitoring across 736 human patients with Type 1 Diabetes from the University of Granada / Hospital Universitario San Cecilio.

---

## File Manifest & Schema

### 1. `Glucose_measurements.csv` (789 MB, 22,670,972 rows)
* **Sampling Rate**: Every 15 minutes (96 points/day) from Abbott FreeStyle Libre CGM sensors.
* **Columns**:
  - `Patient_ID`: Identifier (e.g. `LIB193263`)
  - `Measurement_date`: Date (`YYYY-MM-DD`)
  - `Measurement_time`: Clock time (`HH:MM:SS`)
  - `Measurement`: Blood glucose concentration ($40 - 500\text{ mg/dL}$)

### 2. `Biochemical_parameters.csv` (3.8 MB, 87,482 rows)
* **Role**: Hospital laboratory blood and urine tests.
* **Columns**:
  - `Patient_ID`: Identifier
  - `Reception_date`: Lab sample test date
  - `Name`: Biological test name (17 biomarkers: HbA1c, Fasting Glucose, Creatinine, Urine Albumin, Lipids, Liver enzymes, TSH, IA2 Autoantibodies, Insulin, Electrolytes)
  - `Value`: Quantitative laboratory result

### 3. `Patient_info.csv` (59 KB, 736 rows)
* **Role**: Master demographic profile and sensor coverage summary table.
* **Columns**: `Patient_ID`, `Sex`, `Birth_year`, `Initial_measurement_date`, `Final_measurement_date`, `Number_of_days_with_measures`, `Number_of_measurements`, `Number_of_biochemical_parameters`, `Number_of_diagnostics`.

### 4. `Diagnostics.csv` (106 KB, 1,757 rows)
* **Role**: ICD diagnostic records and comorbidities.
* **Columns**: `Patient_ID`, `Code` (ICD code, e.g. `250.11`, `250.51`, `401.9`), `Description` (Text description).
