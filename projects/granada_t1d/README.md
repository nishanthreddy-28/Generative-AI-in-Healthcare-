> Public source distribution: patient tables, trained weights, FAISS binaries, patient-level evidence, generated reports and notebook outputs are excluded. Report-text workflows can run without private research artifacts. CGM/cohort simulation needs authorized local artifacts; see [DATA_ACCESS.md](DATA_ACCESS.md). The IEEE manuscript is an unvalidated historical draft; see [publication status](docs/IEEE_READINESS.md).

# ClearCare: Medical Report Assistant and Diabetes Research

A local Streamlit application for uploading **PDF, scanned PDF, image or text reports**, reviewing the extraction, and receiving explanations with medical-source citations. A separate experimental workspace runs the existing diabetes VAE, conditional generator and daily GRU.

```powershell
python -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. Use the current installed environment; no paid API is needed. See [RUN_AND_DEPLOY.md](docs/RUN_AND_DEPLOY.md) for dependency files, OCR portability, CLI commands, opt-in LLM assistance, security settings and deployment limitations.

## What the application does

- Reads up to five reports locally with bounded file/page limits.
- Uses PDF text extraction and installed local OCR for images/scans.
- Requires user review of values, units, ranges and dates before explanation.
- Extracts common numeric laboratory tests and explicit diagnosis/medication mentions, preserving negation, uncertainty, family history, recorded doses and source locations. Unknown labelled entries require review.
- Compares results with the ranges printed on the report; uncertain/incompatible inputs are flagged.
- Builds a unified laboratory/diagnosis/medication timeline, retaining conflicting and undated entries.
- Retrieves 16 sourced educational/guideline passages using BM25 and word/character TF-IDF, optionally combined with offline, revision-pinned BGE embeddings. The ADA extract includes a recommendation locator.
- Provides dated numeric comparisons when dates and units qualify, patient summaries, questions, evidence-based Q&A and Markdown/JSON downloads.
- Offers explicit paid structured LLM composition using GPT-6 Astra, GPT-6.1 Sol or a pinned GPT-4.1 snapshot. Generation is constrained to complete approved claims and citations; raw reports, names, exact dates and unknown entity text are withheld.
- Runs a newly trained CGM-pattern VAE/GRU and correlated residual simulations on measured user CGM, with coverage/unit checks, a baseline comparator and no cohort substitution. The optional legacy daily-feature route remains available separately.

**One report is enough for a snapshot.** Several dated measurements may support comparisons. Sparse reports are never padded into a model history or used to fabricate future forecasts.

## Research models

**My glucose simulation** defaults to measured `timestamp,glucose_mg_dl` CSV for the corrected release. It requires 30 consecutive days with at least 64 occupied 15-minute bins per day. This CGM-only model avoids inventing absent labs/diagnosis flags. Same-patient confirmation links its summary to reviewed reports. A single report supplies explanation but does not supply the missing sensor history. The serving route loads no cohort records.

An optional legacy route accepts the original 50-feature daily representation plus date. Its unresolved source-unit conventions and inherited scientific defects remain; it is not the corrected default.

The separate cohort research workspace requires explicit opt-in and authorized existing artifacts. Both inference routes compute the VAE latent live, align the final observed state and static profile, and run a separate conditional tabular generator and GRU. Models are hash-checked and loaded in restricted mode. Sensitivity spread is not displayed as calibrated clinical uncertainty; no treatment recommendations are generated.

The corrected CGM release was trained locally with patient-disjoint groups, train-only normalization, baseline comparison and marginal calibration. Its neural model narrowly won validation but did not improve test glucose error over the 30-day mean. Read [CGM_MODEL_CARD.md](docs/models/CGM_MODEL_CARD.md); this is internal research evaluation, not clinical validation. The unchanged legacy models retain the defects documented in [PROJECT_AUDIT_REPORT.md](docs/reports/PROJECT_AUDIT_REPORT.md).

## Project organization

| Phase | Responsibility | Main location |
|---|---|---|
| 1 | Raw data, curation, quality checks and daily features | `Phase_1_Data_Preparation` |
| 2 | Patient windows, normalization and latent VAE | `Phase_2_Latent_VAE` |
| 3 | Conditional generators, forecasts and benchmarks | `Phase_3_Generative_Forecasting` |
| 4 | Guideline retrieval and research report synthesis | `Phase_4_RAG_Support` |
| 5 | Integrated report application and corrected CGM release | `Phase_5_Application` |

```text
BTP/
├── Phase_1_Data_Preparation/
│   ├── raw_data/                   # Original clinical CSVs
│   ├── notebooks/
│   │   ├── curation/               # Original dataset construction
│   │   └── exploration/            # Clinical plots and statistics
│   ├── phase1_ingestion.py
│   └── cleaned_* / audit JSON      # Prepared legacy datasets
├── Phase_2_Latent_VAE/             # VAE code, tensors, scalers, latent vectors
├── Phase_3_Generative_Forecasting/ # GAN/forecast code, checkpoints, evaluations
├── Phase_4_RAG_Support/            # Retrieval code, guideline index, sample report
├── Phase_5_Application/
│   ├── app.py                     # Full Streamlit interface
│   ├── end_to_end_pipeline.py      # Full CLI implementation
│   ├── train_corrected_release.py
│   └── models/cgm/release_001/     # Corrected CGM research release
├── src/healthcare_app/             # Shared report and inference backend
├── tests/                         # Regression and integration checks
├── scripts/                       # Phase 1–4 launchers
├── requirements/                  # app.txt, dev.txt, research.txt
├── docs/
│   ├── guides/                    # Roadmap, walkthrough and IEEE drafts
│   ├── models/                    # CGM model card and retraining plan
│   ├── references/                # Healthcare requirements PDF
│   ├── reports/                   # Audit and verification reports
│   ├── PROJECT_STRUCTURE.md
│   ├── RUN_AND_DEPLOY.md
│   └── WORKSPACE_INVENTORY.md
├── evidence/                      # Dated audit and verification artifacts
├── outputs/legacy_sample/         # Existing generated sample reports
├── .streamlit/                    # App configuration
├── app.py                         # Root Streamlit launcher
├── end_to_end_pipeline.py          # Root CLI launcher
├── README.md
├── Dockerfile
└── pytest.ini
```

See [project structure](docs/PROJECT_STRUCTURE.md) for phase commands and ownership, and [the complete inventory](docs/WORKSPACE_INVENTORY.md) for every file's purpose.

## Verification and boundaries

```powershell
python -m pytest tests -q
python end_to_end_pipeline.py --report path/to/report.pdf --outdir new_report_output
python end_to_end_pipeline.py --patient LIB193266 --simulations 10 --outdir new_research_output
python end_to_end_pipeline.py --daily-history path/to/daily_features.csv --simulations 10 --outdir new_patient_output
python end_to_end_pipeline.py --cgm path/to/measured_cgm.csv --simulations 10 --outdir new_cgm_output
```

Tests use fictional report fixtures, mocked API calls and existing local research models. Existing datasets, weights and saved results were preserved. Read [WORKFLOW_COMPLETION_REPORT.md](docs/reports/WORKFLOW_COMPLETION_REPORT.md) for the current six-step status, actual results and remaining gaps. [IMPLEMENTATION_VERIFICATION.md](docs/reports/IMPLEMENTATION_VERIFICATION.md) preserves the earlier implementation checkpoint.

This is an educational research application, **not a clinically validated diagnostic/treatment system**. Clinical mentions are extracted within a bounded English vocabulary/labelled-section contract; arbitrary narrative, imaging and pathology reports are not medically interpreted. OCR and clinical extraction cannot be guaranteed error-free. Corpus coverage is limited and no similar-patient retrieval is provided. Neither the strongest available LLM nor output constraints repair forecasting validity.

Public mode requires an access token and disables research data by default. A production medical-data deployment additionally needs HTTPS, proper identity/authorization, rate limits, retention controls, document-worker isolation and operational validation. Controlled dataset/model artifacts must not be distributed without resolving their use rights. Unrelated root air-pollution CSVs were removed during workspace cleanup. Older manuscript outputs are not deployment assets.
