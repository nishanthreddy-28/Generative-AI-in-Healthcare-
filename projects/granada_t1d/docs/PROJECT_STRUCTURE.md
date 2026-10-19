# BTP project structure

The workspace is organized into five phases. Supporting source, tests, documentation, evidence and dependencies have their own locations. The original datasets and saved model weights were moved without regenerating them.

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

## Run from the project root

```powershell
python -m streamlit run app.py
python end_to_end_pipeline.py --help
python -m pytest tests -q
```

The root `app.py` and `end_to_end_pipeline.py` are small launchers. The implementations live in Phase 5; shared imports live under `src/healthcare_app`.

## Phase entry points

| Purpose | Command from BTP root |
|---|---|
| Phase 1 ingestion | `python scripts/run_phase_1.py` |
| Phase 2 VAE pipeline | `python scripts/run_phase_2.py` |
| Phase 3 tabular GAN pipeline | `python scripts/run_phase_3.py` |
| Phase 3 comparative models | `python Phase_3_Generative_Forecasting/train_comparative_models.py` |
| Phase 4 simulation and RAG | `python scripts/run_phase_4.py` |
| Corrected CGM retraining | `python Phase_5_Application/train_corrected_release.py --outdir outputs/generated/new_cgm_release` |

Training commands regenerate phase artifacts. They were not executed during folder organization. The corrected release remains separately versioned under Phase 5.

## Supporting folders

- `src/healthcare_app`: report extraction/OCR, laboratory findings, clinical history, evidence retrieval, summaries, model contracts and inference.
- `tests`: automated checks; disposable fixtures are placed in ignored `.runtime/test_temp`.
- `requirements`: `app.txt` for the core app, `dev.txt` for app/test dependencies, and `research.txt` for ML and optional API dependencies.
- `docs/guides`: original research roadmap, teaching guide, walkthrough and IEEE drafts.
- `docs/models`: corrected model card and retraining plan.
- `docs/reports`: recorded project audit and implementation checkpoints.
- `docs/references`: source requirements PDF.
- `evidence/audit_20261011` and `evidence/implementation_20261011`: historical audit/verification artifacts. Logs, original source backups and original manifest hashes record earlier versions.
- `evidence/organization_20261012`: verification of the folder reorganization.
- `outputs/legacy_sample`: existing generated research reports and chart.

## Notebook conventions

Open exploration notebooks with the notebook's containing folder as the working directory. They refer to `../../raw_data`. Curation notebooks preserve their original workflow and still require external `Originals` files, the diagnostic-code workbook and intermediate patient metadata that were not supplied in this workspace.

## Configuration and container

`.streamlit`, `.env.example`, `.gitignore`, `pytest.ini`, `Dockerfile` and `.dockerignore` stay at the root because their tools discover or build from that location. Docker copies the current `src` package and Phase 5 interface. Container execution remains unverified, as recorded in the deployment guide.

## Research documentation

Phase 1–4 READMEs describe the original research pipeline. Read historical performance and clinical claims alongside [the audit](reports/PROJECT_AUDIT_REPORT.md) and [the corrected model card](models/CGM_MODEL_CARD.md). Organization does not retrain or change model results.
