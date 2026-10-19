# Project structure

Run application commands from the Granada project root (`projects/granada_t1d` in this repository).

## Application

`app.py` launches `Phase_5_Application/app.py`. `end_to_end_pipeline.py` launches the CLI. Both use the shared backend in `src/healthcare_app/`.

The backend separates document extraction (`documents.py`), numeric/clinical parsing (`records.py`, `history.py`), analysis/timeline construction (`analysis.py`), source retrieval (`knowledge.py`), claim composition (`summary.py`, `llm.py`), and experimental model inference (`research.py`, `trajectory_v2.py`).

## Research paths

Phases 1–4 contain the original multimodal preparation, VAE, tabular generator, daily forecaster, comparisons, and research retrieval. Their launchers are in `scripts/run_phase_*.py`. These historical entrypoints can rebuild or overwrite artifacts; use an isolated copy and read the [audit](reports/RESEARCH_AUDIT_2026-10-19.md) first.

The corrected CGM release is trained by `Phase_5_Application/train_corrected_release.py` using the models in `src/healthcare_app/trajectory_v2.py` and aggregation in `cgm.py`. Its private artifacts belong in `Phase_5_Application/models/cgm/release_001/`.

## Tests and documentation

`tests/` holds application, CLI, input-validation, and model integration checks. Full model checks need authorized assets. `scripts/AUDITING.md` describes independent saved-data, raw-CGM, and artifact checks.

Start with the [project README](../README.md), [data access](../DATA_ACCESS.md), [run/deployment guide](RUN_AND_DEPLOY.md), [model card](models/CGM_MODEL_CARD.md), and [research audit](reports/RESEARCH_AUDIT_2026-10-19.md). The documents in `guides/` and older audit reports are dated project history; they do not override current results.
