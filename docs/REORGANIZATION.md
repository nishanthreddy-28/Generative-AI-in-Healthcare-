# Dataset project reorganization — 19 October 2026

The former root application, notebook, data, frontend, backend, scripts, tests and dependency files now live together under `projects/pima_diabetes`. Their code/data bytes were preserved. Root architecture assets moved to `docs/architecture` and the Pima README links were updated. The former `pre - diabetic` notebook/data now live under `projects/early_stage_diabetes` with their filenames preserved.

The local Granada project was added under `projects/granada_t1d` using a source/documentation allowlist. Its phase layout, root launchers and src imports remain intact. Ten Granada notebooks retain their source but have cleared execution outputs and counts. Raw/cleaned patient data, tensors, checkpoints, binary retrieval indexes, generated patient reports, private Streamlit configuration, PDFs and patient-level audit evidence were not copied. Existing phase-level aggregate metric/scaler JSONs are retained; corrected-release aggregate metadata is included under `docs/models/results`.

No models were retrained and original datasets/checkpoints were not changed. Historical scientific defects are identified in project READMEs; repository organization does not repair experimental validity. Full Granada file-by-file auditing follows this initial publication. Source-data access and missing model assets are documented in Granada DATA_ACCESS.md.

From the repository root, change into the relevant project before running its documented commands. Do not combine all three dependency stacks into one environment. Full Granada integration tests require authorized local artifacts and therefore are not claimed to run from a bare public clone.

## Relocation checks

All 45 moved pre-existing code/data/notebook/asset files outside the edited README and ignore rules were compared byte for byte against the original Git HEAD. All 65 published Python files parsed. All ten newly published Granada notebooks have cleared outputs. The staging allowlist excludes private data/configuration, research binaries and local test assets.

Pima offline tests with real embedding-model tests deselected: 49 passed, 22 skipped, 4 deselected. Skips reflect absent generated artifacts/documents; this is not a full application validation. Limiting native numerical threads avoided the earlier local native crash in this subset. Missing-model embedding initialization remains unverified.

Granada's relocated full suite subsequently passed 130 tests in 32.71 seconds with authorized local assets kept outside Git. The completed [research audit](../projects/granada_t1d/docs/reports/RESEARCH_AUDIT_2026-10-19.md) independently checks the raw daily aggregation, saved split/scaler contracts and exact-checkpoint metrics. Front-facing READMEs have since been rewritten around the actual projects and current results; historical research drafts remain clearly identified.
