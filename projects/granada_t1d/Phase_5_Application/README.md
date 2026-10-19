# Phase 5: Integrated application

The full ClearCare Streamlit interface, report/simulation CLI and corrected CGM training entry point live here. Shared backend modules are in `../src/healthcare_app`.

| Item | Purpose |
|---|---|
| `app.py` | Report review, cited explanations, timeline, summaries and optional research simulations |
| `end_to_end_pipeline.py` | Report and research command-line workflows |
| `train_corrected_release.py` | CGM aggregate preparation, model fitting, baseline comparison and calibration |
| `models/cgm/release_001` | Versioned corrected weights, calibration, metadata and patient groups |
| `models/cgm/daily_cgm.csv`, `data_quality.json` | Earlier corrected preparation artifacts retained for provenance |

Run from the BTP root:

```powershell
python -m streamlit run app.py
python end_to_end_pipeline.py --help
```

Read [the model card](../docs/models/CGM_MODEL_CARD.md), [deployment instructions](../docs/RUN_AND_DEPLOY.md) and [project structure](../docs/PROJECT_STRUCTURE.md).
