# Phase 1 · Data preparation

`phase1_ingestion.py` reads the four Granada source tables, audits missingness, selects the legacy cohort, and builds daily CGM/laboratory features. The original curation and exploration notebooks are kept in `notebooks/`.

Inputs belong in `raw_data/`; see [data access](../DATA_ACCESS.md). Patient tables and prepared CSVs are local-only.

The saved legacy dataset has 221,366 daily rows from 593 patients. Its laboratory imputation used whole-cohort medians, and the current source does not reproduce the saved schema exactly. Diagnosis onset and laboratory units are unresolved. These are [research audit findings](../docs/reports/RESEARCH_AUDIT_2026-10-19.md), not a leakage-free preprocessing claim.

The corrected CGM-only release prepares its data separately through `Phase_5_Application/train_corrected_release.py`. Keep the two data versions distinct. Historical preparation entrypoints can overwrite outputs; rebuild only in an isolated versioned copy.
