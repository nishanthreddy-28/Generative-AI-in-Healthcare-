# Phase 5 · ClearCare application

The application brings report review and experimental glucose simulation into one local Streamlit interface.

- `app.py` implements the interface; launch it with the root `app.py`.
- `end_to_end_pipeline.py` provides report and simulation CLI commands.
- `train_corrected_release.py` prepares fresh CGM aggregates and trains the separate corrected release, including baseline selection and calibration.

Shared backend code is in `../src/healthcare_app`. The private corrected release lives at `models/cgm/release_001/` in a complete authorized local installation. Those assets are excluded from Git.

From the Granada project root:

```bash
python -m streamlit run app.py
python end_to_end_pipeline.py --help
```

See [setup](../README.md#setup), [model results](../docs/models/CGM_MODEL_CARD.md), and [data access](../DATA_ACCESS.md). Training requires a new empty output directory; it is separate from running the application.
