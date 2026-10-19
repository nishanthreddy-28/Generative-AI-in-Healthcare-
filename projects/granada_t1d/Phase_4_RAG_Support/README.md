# Phase 4 · Research retrieval and summaries

`rag_clinical_engine.py` connects research summaries to the shared local knowledge base. The legacy `ADAGuidelinesRAG` class name is retained for compatibility; its active corpus and vectors come from `src/healthcare_app/knowledge.py` and `knowledge.json`.

`phase4_trajectory_and_rag.py` is an older standalone rollout demo. It initializes the final observed target state with zeros and describes uncalibrated latent perturbations as confidence bands. Use the current application interfaces for guarded inference, and do not treat this demo's chart as clinical uncertainty.

The six-entry ADA JSON and old binary index belong to the earlier experiment; they are not the current complete guideline library. The binary index and patient-specific generated samples are excluded from the public checkout. See [audit findings](../docs/reports/RESEARCH_AUDIT_2026-10-19.md).
