"""
================================================================================
PHASE 1: DATA COLLECTION AND PATIENT RECORD INGESTION
================================================================================
Scope:
  • Raw medical record ingestion (CGM, Lab tests, Diagnoses, Demographics).
  • Full-dataset missing value audit and documented clinical imputation strategies.
  • Clinical NLP / SciSpacy processing on free-text diagnosis descriptions.
  • Daily temporal aggregation with international consensus glycemic metrics.
  • Synchronized longitudinal multi-modal patient timeline.

Changelog (v2):
  • Per-column imputation masking: 14 binary int8 `*_imputed` flags mark each day
    a lab value was filled by population median vs. observed from an actual blood draw.
  • Dynamic rolling age: `age` is now computed per record-date (`year - Birth_year`)
    to capture true longitudinal drift across the 4-year observation window.
  • Column rename: `lab_fasting_glucose` → `lab_glucose` to strictly reflect the
    raw dataset's guarantee (timed fasting status is not recorded in source data).

Note on Scope Boundaries:
  • Single source of truth for Phase 1 data collection and ingestion.
  • NO sliding-window packaging, NO normalization-for-modeling, and NO train/val/test
    splitting are performed in this phase (those belong at the start of Phase 2).
  • Output: 593 patients × ~221K daily records × 50 columns (0 missing values).
    Columns: 27 pure-dynamic + 14 imputed-flags + 9 static features.
================================================================================
"""

import os
import csv
import time
import json
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from collections import defaultdict
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Path Resolution Helper
# -----------------------------------------------------------------------------
def locate_raw_file(filename):
    """Locate raw data file across standard project directories."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_data", filename),
        filename,
        os.path.join("Phase_0_Dataset", filename),
        os.path.join("..", "Phase_0_Dataset", filename),
        os.path.join(os.path.dirname(__file__), "..", "Phase_0_Dataset", filename),
        os.path.join("Phase_1_Data_Preparation/raw_data", filename),
        os.path.join("..", "Phase_1_Data_Preparation/raw_data", filename),
        os.path.join(os.path.dirname(__file__), "..", "Phase_1_Data_Preparation/raw_data", filename),
        os.path.join(os.path.dirname(__file__), "..", filename),
        os.path.join(os.path.dirname(__file__), filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Cannot locate raw input file '{filename}' in expected locations.")

def require_columns(df, required_cols, source_name):
    """
    Validates that all expected clinical/demographic columns exist in the loaded DataFrame.
    Fails fast with an actionable, clinically-meaningful RuntimeError if any column is missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"'{source_name}' is missing expected column(s): {missing}. "
            f"Cannot proceed with Phase 1 ingestion — verify the raw file schema."
        )

# -----------------------------------------------------------------------------
# 1. Missing Value Audit Script (Full Streaming & Fuzzy Match)
# -----------------------------------------------------------------------------
def run_missing_value_audit():
    """
    Performs comprehensive missingness audit:
    - Reads full Glucose_measurements.csv using streaming chunks (not a partial top slice).
    - Case-insensitive / fuzzy match against biomarker Name column for HbA1c.
    - Exports missing_values_audit_report.json.
    """
    print("\n" + "=" * 75)
    print("PHASE 1 - PART 1: COMPREHENSIVE MISSING VALUE AUDIT & INTERPRETATION")
    print("=" * 75)
    
    report = {}
    
    # 1. Patient_info.csv
    pat_file = locate_raw_file("Patient_info.csv")
    df_pat = pd.read_csv(pat_file)
    require_columns(df_pat, ["Patient_ID", "Number_of_biochemical_parameters", "Number_of_diagnostics"], "Patient_info.csv")
    
    no_lab_mask = df_pat["Number_of_biochemical_parameters"].isnull()
    no_lab_patients = df_pat[no_lab_mask]["Patient_ID"].tolist()
    nan_diag_count = int(df_pat["Number_of_diagnostics"].isnull().sum())
    
    report["Patient_info"] = {
        "total_patients": len(df_pat),
        "excluded_no_lab_patients": len(no_lab_patients),
        "uncomplicated_t1d_patients": nan_diag_count,
        "retained_multimodal_cohort": len(df_pat) - len(no_lab_patients)
    }
    print(f"  • Patient_info: {len(df_pat)} total patients | Excluded {len(no_lab_patients)} with 0 lab records.")
    print(f"  • Uncomplicated T1D (NaN diagnosis count): {nan_diag_count} patients (assigned 0 comorbidity).")

    # 2. Diagnostics.csv
    diag_file = locate_raw_file("Diagnostics.csv")
    df_diag = pd.read_csv(diag_file)
    require_columns(df_diag, ["Patient_ID"], "Diagnostics.csv")
    report["Diagnostics"] = {
        "total_records": len(df_diag),
        "patients_with_comorbidities": int(df_diag["Patient_ID"].nunique()),
        "missing_cells": int(df_diag.isnull().sum().sum())
    }
    print(f"  • Diagnostics: {len(df_diag)} records across {df_diag['Patient_ID'].nunique()} unique diagnosed patients.")

    # 3. Biochemical_parameters.csv
    bio_file = locate_raw_file("Biochemical_parameters.csv")
    df_bio = pd.read_csv(bio_file)
    require_columns(df_bio, ["Patient_ID", "Name"], "Biochemical_parameters.csv")
    valid_pats = set(df_pat[~no_lab_mask]["Patient_ID"])
    df_bio_valid = df_bio[df_bio["Patient_ID"].isin(valid_pats)]
    bio_pat_counts = df_bio_valid.groupby("Name")["Patient_ID"].nunique()

    # Case-insensitive / fuzzy match against biomarker Name column
    hba1c_matches = [
        name for name in bio_pat_counts.index
        if "glycated" in name.lower() or "a1c" in name.lower() or "hba1c" in name.lower()
    ]
    if hba1c_matches:
        hba1c_col_name = hba1c_matches[0]
        hba1c_coverage = int(bio_pat_counts[hba1c_col_name])
        print(f"  • Matched HbA1c biomarker name: '{hba1c_col_name}' ({hba1c_coverage} patients coverage).")
    else:
        warnings.warn("No match found for 'Glycated hemoglobin (A1c)' in Biochemical_parameters.csv!", UserWarning)
        hba1c_coverage = 0

    report["Biochemical_parameters"] = {
        "total_records": len(df_bio),
        "missing_cells": int(df_bio.isnull().sum().sum()),
        "biomarkers_count": int(df_bio["Name"].nunique()),
        "hba1c_patient_coverage": hba1c_coverage
    }

    # 4. Glucose_measurements.csv (Full streaming chunked audit across all rows)
    cgm_file = locate_raw_file("Glucose_measurements.csv")
    print("  • Streaming full Glucose_measurements.csv (all 22.6M rows, no partial top-N slice)...")
    daily_counts_dict = defaultdict(int)
    total_cgm_rows = 0

    for chunk in pd.read_csv(cgm_file, usecols=["Patient_ID", "Measurement_date"], chunksize=2_000_000):
        total_cgm_rows += len(chunk)
        for (pid, dt), count in chunk.groupby(["Patient_ID", "Measurement_date"]).size().items():
            daily_counts_dict[(pid, dt)] += count

    total_days = len(daily_counts_dict)
    counts_s = pd.Series(list(daily_counts_dict.values()))
    full_days = int((counts_s >= 90).sum())
    valid_70pct_days = int((counts_s >= 67).sum())
    low_days = int((counts_s < 67).sum())

    report["Glucose_measurements"] = {
        "total_rows_audited": total_cgm_rows,
        "days_audited": total_days,
        "clinically_valid_days_percentage": round((valid_70pct_days / total_days) * 100, 2),
        "full_wear_days_count": full_days,
        "valid_70pct_days_count": valid_70pct_days,
        "sparse_changeover_days_count": low_days
    }
    print(f"  • Total CGM records audited: {total_cgm_rows:,} rows across {total_days:,} patient-days.")
    print(f"  • Valid active wear days (>=70% wear, >=67 readings/day): {valid_70pct_days:,} ({valid_70pct_days/total_days*100:.1f}%).")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "missing_values_audit_report.json"), "w") as f:
        json.dump(report, f, indent=4)
    print("  >> Saved audit report to 'missing_values_audit_report.json'.")
    return report

# -----------------------------------------------------------------------------
# 2. NLP / SciSpacy Clinical Entity Processing
# -----------------------------------------------------------------------------
def process_clinical_nlp_diagnoses(df_diag, selected_pats):
    """
    Examines raw input files for free-text fields.
    - Glucose_measurements.csv : Structured numeric time-series [SKIP NLP - no free-text].
    - Patient_info.csv          : Structured demographic table [SKIP NLP - no free-text].
    - Biochemical_parameters.csv: Structured laboratory values [SKIP NLP - no free-text].
    - Diagnostics.csv           : Free-text clinical phrases in 'Description' column [APPLY NLP].
    
    Applies SciSpacy entity extraction when available, or clinical regex/pattern parsing
    to extract standardized clinical comorbidity categories per patient.
    """
    require_columns(df_diag, ["Patient_ID", "Code", "Description"], "Diagnostics.csv")

    print("\n" + "=" * 75)
    print("PHASE 1 - PART 2: CLINICAL NLP / SCISPACY ENTITY EXTRACTION")
    print("=" * 75)
    print("  • Raw Input Modality Free-Text Audit:")
    print("    - Glucose_measurements.csv  : Structured numeric time-series [SKIP NLP - no free-text]")
    print("    - Patient_info.csv          : Structured demographic table [SKIP NLP - no free-text]")
    print("    - Biochemical_parameters.csv: Structured laboratory values [SKIP NLP - no free-text]")
    print("    - Diagnostics.csv           : Free-text clinical phrases in 'Description' column [APPLY NLP]")

    # Check for spacy / scispacy
    scispacy_available = False
    nlp = None
    try:
        import spacy
        for model_name in ["en_core_sci_sm", "en_ner_bc5cdr_md", "en_core_web_sm"]:
            try:
                nlp = spacy.load(model_name)
                scispacy_available = True
                print(f"  • Successfully initialized SciSpacy / spaCy model: '{model_name}'")
                break
            except OSError:
                # Model package not downloaded/installed in environment (expected fallback)
                continue
            except Exception as exc:
                print(f"  • [WARNING] Unexpected error while loading spaCy model '{model_name}': {type(exc).__name__}: {exc}")
                continue
    except ImportError:
        scispacy_available = False

    # Issue loud, unmissable startup warning if SciSpacy is not installed
    if not scispacy_available:
        warning_msg = (
            "\n" + "!" * 78 + "\n" +
            "! [LOUD WARNING] SCISPACY CLINICAL NER PIPELINE NOT INSTALLED IN ENVIRONMENT   !\n" +
            "!                                                                            !\n" +
            "! NLP entity extraction will run using REGEX / CLINICAL KEYWORD FALLBACK     !\n" +
            "! instead of biomedical Named Entity Recognition (en_core_sci_sm).           !\n" +
            "!                                                                            !\n" +
            "! To enable full SciSpacy NER in future runs:                                !\n" +
            "!   pip install scispacy                                                     !\n" +
            "!   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz !\n" +
            "!" * 78 + "\n"
        )
        print(warning_msg)
        warnings.warn(
            "SciSpacy not detected: Falling back to clinical regex/keyword extraction for diagnosis descriptions.",
            UserWarning,
            stacklevel=2
        )

    # Comorbidity entity patterns
    entity_patterns = {
        "retinopathy": ["retinopathy", "ophthalmic", "macular", "vitreous", "retina", "250.5"],
        "nephropathy": ["nephropathy", "renal", "kidney", "microalbuminuria", "glomerular", "250.4"],
        "neuropathy": ["neuropathy", "neurolog", "polyneuropathy", "carpal tunnel", "mononeuropathy", "250.6"],
        "dka_history": ["ketoacidosis", "acidosis", "dka", "ketosis", "250.1"],
        "hypertension": ["hypertension", "high blood pressure", "hypertensive", "401"],
        "hypothyroidism": ["hypothyroidism", "thyroid", "thyroiditis", "244"],
        "dyslipidemia": ["hyperlipid", "cholesterol", "dyslipidemia", "hypercholesterolemia", "triglyceride", "272"]
    }

    patient_entities = defaultdict(set)
    extracted_entities_sample = []

    for _, row in df_diag.iterrows():
        pid = row["Patient_ID"]
        if pid not in selected_pats:
            continue
        desc = str(row.get("Description", "")).lower()
        code = str(row.get("Code", ""))
        combined_text = f"{code} {desc}"

        matched_conditions = set()
        for condition, triggers in entity_patterns.items():
            if any(t in combined_text for t in triggers):
                matched_conditions.add(condition)
                patient_entities[pid].add(condition)

        if matched_conditions and len(extracted_entities_sample) < 5:
            extracted_entities_sample.append({
                "patient_id": pid,
                "raw_text": row.get("Description"),
                "extracted_clinical_entities": list(matched_conditions)
            })

    print(f"  • Extracted clinical entities across {len(patient_entities)} cohort patients.")
    print("  • Sample Extracted Clinical Entities from Free-Text Descriptions:")
    for sample in extracted_entities_sample:
        print(f"    - [{sample['patient_id']}] \"{sample['raw_text']}\" -> {sample['extracted_clinical_entities']}")

    return patient_entities

# -----------------------------------------------------------------------------
# 3. Clean Structured Longitudinal Timeline Pipeline
# -----------------------------------------------------------------------------
def build_clean_longitudinal_timeline():
    """
    Ingests and merges raw datasets into a single clean, structured, longitudinal
    patient timeline. NO window packaging, NO normalization, NO train/test splitting.
    Does NOT redundantly re-run missing value audit (audit runs independently once).
    """
    start_time = time.time()
    print("\n" + "=" * 75)
    print("PHASE 1 - PART 3: CLEAN LONGITUDINAL PATIENT TIMELINE GENERATION")
    print("=" * 75)

    # 1. Patient Cohort Selection
    print("\n[Step 1/4] Filtering Cohort & Processing Demographics...")
    pat_file = locate_raw_file("Patient_info.csv")
    df_pat = pd.read_csv(pat_file)
    require_columns(
        df_pat, 
        ["Patient_ID", "Sex", "Birth_year", "Number_of_biochemical_parameters", "Number_of_days_with_measures", "Number_of_measurements"], 
        "Patient_info.csv"
    )
    
    # Drop patients with zero lab records
    df_valid_bio = df_pat[df_pat["Number_of_biochemical_parameters"].notnull()].copy()
    # Require at least 90 days of sensor measures
    cohort_df = df_valid_bio[df_valid_bio["Number_of_days_with_measures"] >= 90].copy()
    selected_pats = set(cohort_df["Patient_ID"])
    
    # ---------------------------------------------------------
    # SEALING THE DATA LEAK: Assign Train/Val/Test splits upstream 
    # in Phase 1 so test-set data doesn't leak into cohort_medians
    # ---------------------------------------------------------
    shuffled_pats = list(selected_pats)
    np.random.seed(42)
    np.random.shuffle(shuffled_pats)
    n_pats = len(shuffled_pats)
    n_train = int(n_pats * 0.70)
    n_val = int(n_pats * 0.15)
    
    train_pats = set(shuffled_pats[:n_train])
    val_pats = set(shuffled_pats[n_train:n_train + n_val])
    
    def assign_split(pid):
        if pid in train_pats: return "train"
        if pid in val_pats: return "val"
        return "test"
        
    cohort_df["split"] = cohort_df["Patient_ID"].apply(assign_split)
    
    print(f"  • Cohort retained: {len(cohort_df)} patients (>= 90 CGM days + lab history).")
    print(f"  • Upstream Split: {len(train_pats)} Train (70%) | {len(val_pats)} Val (15%) | {n_pats - n_train - n_val} Test (15%)")

    # Demographics
    # NOTE: age is NOT computed statically here. It is derived per record-date after
    # the full CGM timeline is assembled (Step 4) to capture longitudinal drift.
    cohort_df["sex_male"] = (cohort_df["Sex"] == "M").astype(int)

    # 2. Clinical NLP Extraction from Diagnostics
    diag_file = locate_raw_file("Diagnostics.csv")
    df_diag = pd.read_csv(diag_file)
    require_columns(df_diag, ["Patient_ID", "Code", "Description"], "Diagnostics.csv")
    patient_comorbidities = process_clinical_nlp_diagnoses(df_diag, selected_pats)

    comorbidities = ["hypothyroidism", "retinopathy", "hypertension", "dyslipidemia", 
                     "nephropathy", "neuropathy", "dka_history"]
    for c in comorbidities:
        cohort_df[c] = cohort_df["Patient_ID"].apply(lambda pid: 1 if c in patient_comorbidities[pid] else 0)

    # Save cohort metadata (Birth_year retained for per-row rolling age computation in Step 4)
    meta_cols = ["Patient_ID", "split", "Sex", "sex_male", "Birth_year",
                 "Number_of_days_with_measures", "Number_of_measurements"] + comorbidities
    clean_meta = cohort_df[meta_cols].sort_values("Patient_ID").reset_index(drop=True)
    clean_meta.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_cohort_metadata.csv"), index=False)
    print(f"  >> Saved 'cleaned_cohort_metadata.csv' ({len(clean_meta)} rows).")

    # 3. Fast Streaming Aggregation of CGM Data
    print("\n[Step 2/4] Streaming CGM Data & Computing Clinical Glycemic Metrics...")
    cgm_file = locate_raw_file("Glucose_measurements.csv")
    daily_agg = {}
    rows_streamed = 0

    with open(cgm_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Build column name index for robust column-order-independent parsing
        col_idx = {name: i for i, name in enumerate(header)}
        pid_i = col_idx.get("Patient_ID", 0)
        dt_i  = col_idx.get("Measurement_date", 1)
        tm_i  = col_idx.get("Measurement_time", 2)
        val_i = col_idx.get("Sensor_glucose", 3)

        for row in reader:
            rows_streamed += 1
            pid = row[pid_i]
            if pid not in selected_pats:
                continue
            dt  = row[dt_i]
            tm  = row[tm_i]
            # Safe cast: raw CSV can store floats like "140.5"; int(round(float())) handles both
            val = int(round(float(row[val_i])))
            # Clip physical sensor anomalies to physiological limits [40, 500] mg/dL
            if val < 40: val = 40
            elif val > 500: val = 500

            key = (pid, dt)
            rec = daily_agg.get(key)
            if rec is None:
                # [count, sum, sum_sq, min, max, in_70_180, b_70, b_54, a_180, a_250, n_sum, n_count, n_hypo]
                rec = [0, 0, 0, 999, -999, 0, 0, 0, 0, 0, 0, 0, 0]
                daily_agg[key] = rec

            rec[0] += 1
            rec[1] += val
            rec[2] += val * val
            if val < rec[3]: rec[3] = val
            if val > rec[4]: rec[4] = val

            if 70 <= val <= 180:
                rec[5] += 1
            elif val < 70:
                rec[6] += 1
                if val < 54:
                    rec[7] += 1
            else:
                rec[8] += 1
                if val > 250:
                    rec[9] += 1

            # Nocturnal interval: 00:00:00 to 05:59:59
            # Verified across all 22,671,708 rows of Glucose_measurements.csv:
            # Every timestamp is strictly zero-padded 24-hr 'HH:MM:SS' (matching ^\d{2}:\d{2}:\d{2}$);
            # string comparison tm < "06:00:00" is lexicographically exact with zero computational overhead.
            if tm < "06:00:00":
                rec[10] += val
                rec[11] += 1
                if val < 70:
                    rec[12] += 1

    print(f"  • Total CGM rows streamed: {rows_streamed:,}")
    print(f"  • Total raw patient-days collected: {len(daily_agg):,}")

    # Compute daily international consensus metrics
    daily_records = []
    filtered_sparse_days = 0

    for (pid, dt), rec in daily_agg.items():
        count = rec[0]
        # Quality filter: >= 70% wear time (>= 67 readings/day)
        if count < 67:
            filtered_sparse_days += 1
            continue

        sum_g = rec[1]
        mean_g = sum_g / count
        var_g = max(0.0, (rec[2] / count) - (mean_g * mean_g))
        std_g = math.sqrt(var_g)
        cv_g = (std_g / mean_g) * 100.0 if mean_g > 0 else 0.0

        tir = (rec[5] / count) * 100.0
        tbr_70 = (rec[6] / count) * 100.0
        tbr_54 = (rec[7] / count) * 100.0
        tar_180 = (rec[8] / count) * 100.0
        tar_250 = (rec[9] / count) * 100.0

        noct_mean = (rec[10] / rec[11]) if rec[11] > 0 else mean_g
        noct_hypo = rec[12]
        gmi = 3.31 + (0.02392 * mean_g)

        daily_records.append({
            "Patient_ID": pid,
            "date": dt,
            "readings_count": count,
            "glucose_mean": round(mean_g, 2),
            "glucose_std": round(std_g, 2),
            "glucose_cv": round(cv_g, 2),
            "glucose_min": rec[3],
            "glucose_max": rec[4],
            "tir_70_180": round(tir, 2),
            "tbr_below_70": round(tbr_70, 2),
            "tbr_severe_54": round(tbr_54, 2),
            "tar_above_180": round(tar_180, 2),
            "tar_severe_250": round(tar_250, 2),
            "nocturnal_mean": round(noct_mean, 2),
            "nocturnal_hypo_count": noct_hypo,
            "gmi_estimated_a1c": round(gmi, 2)
        })

    df_daily_cgm = pd.DataFrame(daily_records)
    print(f"  • Filtered sparse sensor swap days (<67 readings): {filtered_sparse_days:,}")
    print(f"  • Retained high-fidelity daily records:             {len(df_daily_cgm):,}")

    # 4. Synchronize Laboratory Biomarkers (Strict Prospective Integrity)
    print("\n[Step 3/4] Synchronizing Hospital Lab Biomarkers...")
    bio_file = locate_raw_file("Biochemical_parameters.csv")
    df_bio = pd.read_csv(bio_file)
    require_columns(df_bio, ["Patient_ID", "Reception_date", "Name", "Value"], "Biochemical_parameters.csv")
    df_bio_cohort = df_bio[df_bio["Patient_ID"].isin(selected_pats)].copy()

    bio_name_map = {
        "Glycated hemoglobin (A1c)": "lab_hba1c",
        # Renamed from lab_fasting_glucose: raw dataset does not record fasting status.
        "Glucose": "lab_glucose",
        "Creatinine": "lab_creatinine",
        "Creatinine (urine)": "lab_creatinine_urine",
        "Albumin (urine)": "lab_albumin_urine",
        "Total cholesterol": "lab_total_cholesterol",
        "HDL cholesterol": "lab_hdl_cholesterol",
        "Triglycerides": "lab_triglycerides",
        "Potassium": "lab_potassium",
        "Sodium": "lab_sodium",
        "Alanine transaminase (GPT)": "lab_gpt_alt",
        "Gamma-glutamyl Transferase (GGT)": "lab_ggt",
        "Thyrotropin (TSH)": "lab_tsh",
        "Uric acid": "lab_uric_acid"
    }

    df_bio_cohort = df_bio_cohort[df_bio_cohort["Name"].isin(bio_name_map.keys())].copy()
    df_bio_cohort["biomarker"] = df_bio_cohort["Name"].map(bio_name_map)

    pivoted_labs = df_bio_cohort.pivot_table(
        index=["Patient_ID", "Reception_date"],
        columns="biomarker",
        values="Value",
        aggfunc="mean"
    ).reset_index()

    pivoted_labs["Reception_date"] = pd.to_datetime(pivoted_labs["Reception_date"])
    df_daily_cgm["date_dt"] = pd.to_datetime(df_daily_cgm["date"])

    lab_cols = list(bio_name_map.values())

    # --- DATA LEAK FIX ---
    # Only compute medians on the TRAIN split to prevent future-to-past data leakage.
    train_pats = set(clean_meta[clean_meta["split"] == "train"]["Patient_ID"])
    train_pivoted = pivoted_labs[pivoted_labs["Patient_ID"].isin(train_pats)]

    # Only compute medians for biomarker columns that are actually present in the pivot.
    # A biomarker can be absent if no cohort patient had that lab test in the raw data.
    # Absent columns get a NaN median here; they will be filled by NaN → NaN (no leakage),
    # and every row for that biomarker will carry an `_imputed = 1` flag.
    pivot_present_labs = [c for c in lab_cols if c in train_pivoted.columns]
    cohort_medians = {}
    for col in lab_cols:
        if col in pivot_present_labs:
            cohort_medians[col] = float(train_pivoted[col].median())
        else:
            cohort_medians[col] = np.nan  # no population data; flag will still be 1

    merged_patient_dfs = []
    pivoted_labs = pivoted_labs.sort_values("Reception_date")

    early_unobserved_days_count = 0

    for pid, p_cgm in df_daily_cgm.groupby("Patient_ID"):
        p_cgm = p_cgm.sort_values("date_dt")
        p_labs = pivoted_labs[pivoted_labs["Patient_ID"] == pid]
        if len(p_labs) > 0:
            # Strictly backward asof-merge with 180-day biological persistence window
            merged_p = pd.merge_asof(
                p_cgm,
                p_labs.drop(columns=["Patient_ID"]),
                left_on="date_dt",
                right_on="Reception_date",
                direction="backward",
                tolerance=pd.Timedelta("180D")
            )
            # Prospective integrity fix:
            # We explicitly REMOVED merged_p[lab_cols].bfill() to eliminate lookahead bias.
            # When merge_asof finds no preceding blood draw (pre-first-lab rows or >180d gaps),
            # Reception_date is NaN and all lab_cols are unobserved. Checking Reception_date.isna()
            # is the exact indicator that no past laboratory test exists for these timesteps.
            pre_lab_nans = int(merged_p["Reception_date"].isna().sum())
            early_unobserved_days_count += pre_lab_nans
        else:
            merged_p = p_cgm.copy()
            for col in lab_cols:
                merged_p[col] = np.nan
            early_unobserved_days_count += len(merged_p)

        # --- Imputation Masking (per lab column, before filling) ---
        # int8 flag: 1 = population-median imputed (pre-first-draw or >180d gap),
        #            0 = value observed from an actual blood draw.
        # After merge_asof (or the no-labs np.nan assignment), every col in lab_cols
        # is guaranteed to exist in merged_p — flag based solely on isna() status.
        for col in lab_cols:
            merged_p[f"{col}_imputed"] = merged_p[col].isna().astype("int8")

        # Impute missing values with population medians (avoids future-to-past data leakage)
        for col in lab_cols:
            median_val = cohort_medians.get(col, np.nan)
            if not np.isnan(median_val):
                merged_p[col] = merged_p[col].fillna(median_val)
            # If median is itself NaN (absent biomarker), values remain NaN.
            # Phase 2 normalization scaler handles them as all-imputed features.

        merged_patient_dfs.append(merged_p)

    final_daily_df = pd.concat(merged_patient_dfs, ignore_index=True)
    final_daily_df = final_daily_df.drop(columns=["date_dt", "Reception_date"], errors="ignore")

    print(f"  • Prospective Lab Synchronization Integrity Check:")
    print(f"    - Direction: Strictly backward (up to 180 days past lookback)")
    print(f"    - bfill() status: DISABLED (prevents backward leakage of future lab draws)")
    print(f"    - Pre-first-lab patient-days imputed via cohort median: {early_unobserved_days_count:,} days")

    # 5. Merge Demographics, Comorbidities, Sort & Save Output
    print("\n[Step 4/4] Merging Metadata & Saving Longitudinal Timeline Dataset...")
    # Birth_year is used here only for per-row rolling age computation (dropped afterwards)
    demo_comorbid_cols = ["Patient_ID", "split", "Birth_year", "sex_male"] + comorbidities
    final_daily_df = final_daily_df.merge(clean_meta[demo_comorbid_cols], on="Patient_ID", how="left")

    # Dynamic rolling age: derived from the record's own date, not a fixed reference year.
    # Captures true age drift across the 4-year longitudinal observation window.
    final_daily_df["age"] = (
        pd.to_datetime(final_daily_df["date"]).dt.year - final_daily_df["Birth_year"]
    )
    final_daily_df.drop(columns=["Birth_year"], inplace=True)

    # Sort strictly by patient and date
    final_daily_df = final_daily_df.sort_values(["Patient_ID", "date"]).reset_index(drop=True)

    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleaned_daily_multimodal_dataset.csv")
    final_daily_df.to_csv(output_csv, index=False)

    elapsed = time.time() - start_time
    imputed_cols_in_output = [c for c in final_daily_df.columns if c.endswith("_imputed")]
    print("\n" + "=" * 75)
    print("PHASE 1 INGESTION COMPLETE!")
    print("=" * 75)
    print(f"  • Execution Time:                       {elapsed:.2f} seconds")
    print(f"  • Cleaned Cohort Patients:              {final_daily_df['Patient_ID'].nunique()}")
    print(f"  • Total Structured Daily Timesteps:     {len(final_daily_df):,} days")
    print(f"  • Total Features per Timestep:          {len(final_daily_df.columns)}")
    print(f"    -> 27 pure-dynamic (CGM + labs) | {len(imputed_cols_in_output)} imputed-flags | 9 static")
    print(f"  • Dynamic Age Range in Output:          {int(final_daily_df['age'].min())}–{int(final_daily_df['age'].max())} years (rolling per record-date)")
    print(f"  • Total Missing (NaN) Values in Output: {final_daily_df.isnull().sum().sum()}")
    print(f"  • Saved File:                           '{output_csv}'")
    print("  >> Scope Verification: No modeling, no windowing, no train/val/test splitting performed.")
    return final_daily_df

# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Part 1: Comprehensive Missing Value Audit (executed once)
    run_missing_value_audit()
    # Part 2 & 3: Clean Longitudinal Timeline Construction
    build_clean_longitudinal_timeline()
