# Generative AI in Healthcare

Research projects organized by dataset. Each project has its own input contract, environment, entrypoints and evaluation; their results are not interchangeable.

- [Pima diabetes](projects/pima_diabetes/README.md): eight-feature binary classification with a FastAPI backend, frontend, medical retrieval and optional LLM explanations. Run commands from `projects/pima_diabetes`.
- [Early-stage diabetes](projects/early_stage_diabetes/README.md): symptom-based classification notebook and 520-row dataset, formerly named `pre - diabetic`. This is not an independently established prediabetes class.
- [Granada type 1 diabetes](projects/granada_t1d/README.md): five-phase longitudinal research project, local report assistant and experimental CGM forecasting. Run commands from `projects/granada_t1d`.

See the [dataset catalog](docs/dataset_catalog.md), [migration details](docs/REORGANIZATION.md), and [Granada IEEE readiness](projects/granada_t1d/docs/IEEE_READINESS.md).

## Research status

These are educational research prototypes. Pima's historical model selection uses test scores. The early-stage notebook has repeated predictor profiles crossing its random split. Granada's corrected neural forecaster did not outperform the 30-day mean on held-out glucose MAE; its historical IEEE draft needs correction. There is no clinical validation or demonstrated zero-hallucination guarantee. Planned similarity-based reasoning and cost planning must not be treated as implemented evidence.

## Data and credentials

Pima and early-stage CSVs already present in this repository remain with their projects; their source attribution should be verified before reuse. Granada raw/derived patient data, checkpoints, indexes, notebook outputs and patient-level evidence are excluded from this source publication. Obtain authorized local assets as described in [Granada data access](projects/granada_t1d/DATA_ACCESS.md). Never commit environment secrets or uploaded reports. The code license does not supersede dataset/document terms.
