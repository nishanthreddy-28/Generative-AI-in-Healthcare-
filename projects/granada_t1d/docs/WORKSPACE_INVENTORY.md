# BTP: complete workspace inventory

Snapshot: 12 October 2026, after cleanup and five-phase organization. 261 files and 36 folders, including this inventory and hidden configuration. Disposable runtime/cache directories are excluded and removed after verification.

See [project structure](PROJECT_STRUCTURE.md) for layout and commands. Main code purposes were checked against sources. Historical fixture descriptions inferred from filenames remain marked.

## Every folder

| Folder | Purpose |
|---|---|
| .streamlit | Streamlit server, theme and local credential configuration. |
| docs | Project documentation and complete inventory. |
| docs/guides | Roadmap, walkthrough, teaching guide and IEEE drafts. |
| docs/models | CGM model card and retraining plan. |
| docs/references | Healthcare project requirements/reference PDF. |
| docs/reports | Historical audit and implementation reports. |
| evidence | Dated audit and verification records. |
| evidence/audit_20261011 | Historical baseline audit scripts, logs, fixtures, results and extracts. |
| evidence/audit_20261011/pipeline_fixture_outputs | Historical pipeline samples organized by split. |
| evidence/audit_20261011/pipeline_fixture_outputs/test | Historical audit sample output for the test split. |
| evidence/audit_20261011/pipeline_fixture_outputs/train | Historical audit sample output for the train split. |
| evidence/audit_20261011/pipeline_fixture_outputs/val | Historical audit sample output for the val split. |
| evidence/implementation_20261011 | Historical implementation checks, screenshots, source backups and fixtures. |
| evidence/implementation_20261011/cli_report | Saved report CLI verification output. |
| evidence/implementation_20261011/cli_report_verified | Follow-up report CLI verification output. |
| evidence/implementation_20261011/cli_research | Legacy research CLI verification output. |
| evidence/organization_20261012 | Current five-phase organization verification. |
| outputs | Generated output examples and future conventions. |
| outputs/legacy_sample | Existing patient-specific note, guide and trajectory chart. |
| Phase_1_Data_Preparation | Data ingestion, preparation and prepared legacy outputs. |
| Phase_1_Data_Preparation/notebooks | Clinical dataset-curation and exploration notebooks. |
| Phase_1_Data_Preparation/notebooks/curation | Original dataset construction notebooks; external exports are not supplied. |
| Phase_1_Data_Preparation/notebooks/exploration | Clinical visualization/statistics notebooks. |
| Phase_1_Data_Preparation/raw_data | Original clinical CSV datasets and dataset documentation. |
| Phase_2_Latent_VAE | VAE code, tensors, scaling and learned representations. |
| Phase_3_Generative_Forecasting | Conditional generation, forecasting, saved models and evaluations. |
| Phase_4_RAG_Support | Retrieval/simulation code, guideline corpus/index and example report. |
| Phase_5_Application | Integrated app, CLI and corrected CGM training entry point. |
| Phase_5_Application/models | Integrated application model releases. |
| Phase_5_Application/models/cgm | Corrected CGM preparation artifacts and versioned release. |
| Phase_5_Application/models/cgm/release_001 | Corrected CGM weights, calibration, manifests, quality and patient groups. |
| requirements | Application, development and research dependencies. |
| scripts | Phase 1–4 launchers. |
| src | Shared Python source package. |
| src/healthcare_app | Report, evidence, summary and research-inference backend. |
| tests | Automated regression/integration tests. |

## Every file

### (workspace root)

| File | Purpose |
|---|---|
| [.dockerignore](<C:/Users/tejde/Downloads/BTP/.dockerignore>) | Excludes selected workspace files from Docker build context. |
| [.env.example](<C:/Users/tejde/Downloads/BTP/.env.example>) | Example app environment settings; the file is not automatically loaded. |
| [.gitignore](<C:/Users/tejde/Downloads/BTP/.gitignore>) | Defines files Git should ignore. |
| [app.py](<C:/Users/tejde/Downloads/BTP/app.py>) | Root launcher for the Phase 5 Streamlit interface. |
| [Dockerfile](<C:/Users/tejde/Downloads/BTP/Dockerfile>) | Container recipe adapted to src and Phase 5; container execution remains unverified. |
| [end_to_end_pipeline.py](<C:/Users/tejde/Downloads/BTP/end_to_end_pipeline.py>) | Root launcher/import adapter for the Phase 5 CLI. |
| [pytest.ini](<C:/Users/tejde/Downloads/BTP/pytest.ini>) | Pytest configuration. |
| [README.md](<C:/Users/tejde/Downloads/BTP/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |

### .streamlit

| File | Purpose |
|---|---|
| [config.toml](<C:/Users/tejde/Downloads/BTP/.streamlit/config.toml>) | Streamlit server, upload-limit, telemetry and theme configuration. |
| [credentials.toml](<C:/Users/tejde/Downloads/BTP/.streamlit/credentials.toml>) | Streamlit local onboarding/credential configuration; contents were not inspected. |

### docs

| File | Purpose |
|---|---|
| [PROJECT_STRUCTURE.md](<C:/Users/tejde/Downloads/BTP/docs/PROJECT_STRUCTURE.md>) | Five-phase layout, folder ownership, run commands and notebook conventions. |
| [RUN_AND_DEPLOY.md](<C:/Users/tejde/Downloads/BTP/docs/RUN_AND_DEPLOY.md>) | Local run commands, dependency setup, OCR behavior, optional LLM use and deployment requirements. |
| [WORKSPACE_INVENTORY.md](<C:/Users/tejde/Downloads/BTP/docs/WORKSPACE_INVENTORY.md>) | This complete current folder/file inventory. |

### docs/guides

| File | Purpose |
|---|---|
| [BTP_T1D_Master_Roadmap.md](<C:/Users/tejde/Downloads/BTP/docs/guides/BTP_T1D_Master_Roadmap.md>) | Type 1 diabetes BTP research roadmap. |
| [IEEE_Journal_Manuscript_Draft.md](<C:/Users/tejde/Downloads/BTP/docs/guides/IEEE_Journal_Manuscript_Draft.md>) | Earlier IEEE manuscript draft; verify claims against the audit and current model card. |
| [IEEE_Paper_Framework_and_Feature_Engineering.md](<C:/Users/tejde/Downloads/BTP/docs/guides/IEEE_Paper_Framework_and_Feature_Engineering.md>) | Paper framework and feature-engineering notes. |
| [implementation_plan.md](<C:/Users/tejde/Downloads/BTP/docs/guides/implementation_plan.md>) | Earlier project implementation plan. |
| [Phase_1_and_2_Complete_Class_Guide.pdf](<C:/Users/tejde/Downloads/BTP/docs/guides/Phase_1_and_2_Complete_Class_Guide.pdf>) | Teaching/reference guide for the original first two phases. |
| [walkthrough.md](<C:/Users/tejde/Downloads/BTP/docs/guides/walkthrough.md>) | Project implementation walkthrough. |

### docs/models

| File | Purpose |
|---|---|
| [CGM_MODEL_CARD.md](<C:/Users/tejde/Downloads/BTP/docs/models/CGM_MODEL_CARD.md>) | Corrected CGM release: input contract, training, baselines, measured results, calibration and limitations. |
| [RETRAINING_PLAN.md](<C:/Users/tejde/Downloads/BTP/docs/models/RETRAINING_PLAN.md>) | Plan for repairing scientific/data issues through isolated retraining. |

### docs/references

| File | Purpose |
|---|---|
| [GENERATIVE AI IN HEALTH CARE.pdf](<C:/Users/tejde/Downloads/BTP/docs/references/GENERATIVE AI IN HEALTH CARE.pdf>) | Project healthcare requirements/reference PDF; a text extraction exists in audit evidence. |

### docs/reports

| File | Purpose |
|---|---|
| [IMPLEMENTATION_VERIFICATION.md](<C:/Users/tejde/Downloads/BTP/docs/reports/IMPLEMENTATION_VERIFICATION.md>) | Earlier implementation checkpoint and recorded verification evidence. |
| [PROJECT_AUDIT_REPORT.md](<C:/Users/tejde/Downloads/BTP/docs/reports/PROJECT_AUDIT_REPORT.md>) | Baseline audit findings about implementation, data, models and scientific claims. |
| [WORKFLOW_COMPLETION_REPORT.md](<C:/Users/tejde/Downloads/BTP/docs/reports/WORKFLOW_COMPLETION_REPORT.md>) | Current workflow completion status, results and remaining gaps. |

### evidence

| File | Purpose |
|---|---|
| [README.md](<C:/Users/tejde/Downloads/BTP/evidence/README.md>) | Explains historical and current verification evidence. |

### evidence/audit_20261011

| File | Purpose |
|---|---|
| [app_audit.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/app_audit.py>) | Audit/verification helper for app audit (inferred from filename). |
| [app_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/app_console.log>) | Captured execution/test log for app console. |
| [app_fixture_runner.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/app_fixture_runner.py>) | Audit/verification helper for app fixture runner (inferred from filename). |
| [app_initial_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/app_initial_results.json>) | Saved audit/verification metadata or results for app initial results (inferred from filename). |
| [app_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/app_results.json>) | Saved audit/verification metadata or results for app results (inferred from filename). |
| [Biochemical_paramaters_curation_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Biochemical_paramaters_curation_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [cache_security_audit.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/cache_security_audit.py>) | Audit/verification helper for cache security audit (inferred from filename). |
| [cache_security_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/cache_security_console.log>) | Captured execution/test log for cache security console. |
| [cache_security_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/cache_security_results.json>) | Saved audit/verification metadata or results for cache security results (inferred from filename). |
| [cli_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/cli_console.log>) | Captured execution/test log for cli console. |
| [component_audit.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/component_audit.py>) | Audit/verification helper for component audit (inferred from filename). |
| [component_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/component_console.log>) | Captured execution/test log for component console. |
| [component_initial_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/component_initial_results.json>) | Saved audit/verification metadata or results for component initial results (inferred from filename). |
| [component_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/component_results.json>) | Saved audit/verification metadata or results for component results (inferred from filename). |
| [data_audit.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_audit.py>) | Audit/verification helper for data audit (inferred from filename). |
| [data_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_console.log>) | Captured execution/test log for data console. |
| [data_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_results.json>) | Saved audit/verification metadata or results for data results (inferred from filename). |
| [data_visual_Biochemical_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_visual_Biochemical_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [data_visual_diagnostics_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_visual_diagnostics_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [data_visual_glucose_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_visual_glucose_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [data_visual_patient_info_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/data_visual_patient_info_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [delivery_checks.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/delivery_checks.json>) | Saved audit/verification metadata or results for delivery checks (inferred from filename). |
| [delivery_checks.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/delivery_checks.py>) | Audit/verification helper for delivery checks (inferred from filename). |
| [Diagnostic_curation_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Diagnostic_curation_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [evaluation_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/evaluation_console.log>) | Captured execution/test log for evaluation console. |
| [final_checks.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/final_checks.json>) | Saved audit/verification metadata or results for final checks (inferred from filename). |
| [final_checks.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/final_checks.py>) | Audit/verification helper for final checks (inferred from filename). |
| [final_checks_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/final_checks_console.log>) | Captured execution/test log for final checks console. |
| [fixture_rag512.index](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/fixture_rag512.index>) | FAISS retrieval-index fixture retained for verification. |
| [GENERATIVE AI IN HEALTH CARE_extracted.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/GENERATIVE AI IN HEALTH CARE_extracted.txt>) | Extracted reference-PDF text retained for audit inspection. |
| [Get_patient_info_variables_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Get_patient_info_variables_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [Glucose_measurements_curation_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Glucose_measurements_curation_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [Graphs_and_stats_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Graphs_and_stats_notebook.txt>) | Text dump of notebook code/content retained for audit inspection. |
| [initial_runtime_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/initial_runtime_console.log>) | Captured execution/test log for initial runtime console. |
| [initial_runtime_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/initial_runtime_results.json>) | Saved audit/verification metadata or results for initial runtime results (inferred from filename). |
| [inventory.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/inventory.json>) | Saved audit/verification metadata or results for inventory (inferred from filename). |
| [inventory.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/inventory.py>) | Audit/verification helper for inventory (inferred from filename). |
| [original_manifest.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/original_manifest.json>) | Preserved pre-change source/document backup: manifest.json. |
| [Original_patient_info_curation_notebook.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Original_patient_info_curation_notebook.txt>) | Preserved pre-change source/document backup: patient_info_curation_notebook.txt. |
| [Phase_1_and_2_Complete_Class_Guide_extracted.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/Phase_1_and_2_Complete_Class_Guide_extracted.txt>) | Extracted reference-PDF text retained for audit inspection. |
| [pytest_collection.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pytest_collection.log>) | Captured execution/test log for pytest collection. |
| [recomputed_ctgan.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/recomputed_ctgan.png>) | Saved evaluation chart or verification image for recomputed ctgan (inferred from filename). |
| [recomputed_ctgan_metrics.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/recomputed_ctgan_metrics.json>) | Saved audit/verification metadata or results for recomputed ctgan metrics (inferred from filename). |
| [recomputed_forecast.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/recomputed_forecast.png>) | Saved evaluation chart or verification image for recomputed forecast (inferred from filename). |
| [recomputed_forecast_metrics.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/recomputed_forecast_metrics.json>) | Saved audit/verification metadata or results for recomputed forecast metrics (inferred from filename). |
| [requirements-1.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-1.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-2.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-2.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-3.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-3.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-4.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-4.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-5.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-5.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-6.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-6.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-7.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-7.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [requirements-8.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/requirements-8.png>) | Rendered page of the healthcare requirements PDF used in the audit. |
| [runtime_audit.py](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/runtime_audit.py>) | Audit/verification helper for runtime audit (inferred from filename). |
| [runtime_console.log](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/runtime_console.log>) | Captured execution/test log for runtime console. |
| [runtime_results.json](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/runtime_results.json>) | Saved audit/verification metadata or results for runtime results (inferred from filename). |
| [unsafe_chat_fixture.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/unsafe_chat_fixture.txt>) | Saved report, input fixture or verification text: unsafe chat fixture (inferred from filename). |

### evidence/audit_20261011/pipeline_fixture_outputs/test

| File | Purpose |
|---|---|
| [patient_guide_LIB193704.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/test/patient_guide_LIB193704.txt>) | Saved legacy patient-facing research output for the patient ID in the filename. |
| [physician_note_LIB193704.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/test/physician_note_LIB193704.txt>) | Saved legacy physician-facing research output for the patient ID in the filename. |
| [trajectory_forecast_LIB193704.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/test/trajectory_forecast_LIB193704.png>) | Saved legacy trajectory chart for the patient ID in the filename. |

### evidence/audit_20261011/pipeline_fixture_outputs/train

| File | Purpose |
|---|---|
| [patient_guide_LIB194068.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/train/patient_guide_LIB194068.txt>) | Saved legacy patient-facing research output for the patient ID in the filename. |
| [physician_note_LIB194068.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/train/physician_note_LIB194068.txt>) | Saved legacy physician-facing research output for the patient ID in the filename. |
| [trajectory_forecast_LIB194068.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/train/trajectory_forecast_LIB194068.png>) | Saved legacy trajectory chart for the patient ID in the filename. |

### evidence/audit_20261011/pipeline_fixture_outputs/val

| File | Purpose |
|---|---|
| [patient_guide_LIB193358.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/val/patient_guide_LIB193358.txt>) | Saved legacy patient-facing research output for the patient ID in the filename. |
| [physician_note_LIB193358.txt](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/val/physician_note_LIB193358.txt>) | Saved legacy physician-facing research output for the patient ID in the filename. |
| [trajectory_forecast_LIB193358.png](<C:/Users/tejde/Downloads/BTP/evidence/audit_20261011/pipeline_fixture_outputs/val/trajectory_forecast_LIB193358.png>) | Saved legacy trajectory chart for the patient ID in the filename. |

### evidence/implementation_20261011

| File | Purpose |
|---|---|
| [app_corrected_cgm.jpg](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_corrected_cgm.jpg>) | Application screenshot for app corrected cgm. |
| [app_corrected_cgm_charts.jpg](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_corrected_cgm_charts.jpg>) | Application screenshot for app corrected cgm charts. |
| [app_corrected_cgm_dom.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_corrected_cgm_dom.txt>) | Captured browser page/DOM text for UI verification. |
| [app_desktop.jpg](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_desktop.jpg>) | Application screenshot for app desktop. |
| [app_integrated_summary_dom.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_integrated_summary_dom.txt>) | Captured browser page/DOM text for UI verification. |
| [app_mobile.jpg](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_mobile.jpg>) | Application screenshot for app mobile. |
| [app_ready.jpg](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_ready.jpg>) | Application screenshot for app ready. |
| [app_tests_initial.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/app_tests_initial.xml>) | Machine-readable pytest/JUnit result report for app tests initial. |
| [browser_download.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/browser_download.json>) | Saved audit/verification metadata or results for browser download (inferred from filename). |
| [browser_timeline_snapshot.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/browser_timeline_snapshot.txt>) | Captured browser page/DOM text for UI verification. |
| [cli_existing_output.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_existing_output.log>) | Captured execution/test log for cli existing output. |
| [cli_missing_report.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_missing_report.log>) | Captured execution/test log for cli missing report. |
| [cli_report.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report.log>) | Captured execution/test log for cli report. |
| [cli_report_results.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report_results.json>) | Saved audit/verification metadata or results for cli report results (inferred from filename). |
| [cli_report_verified.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report_verified.log>) | Captured execution/test log for cli report verified. |
| [core_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/core_tests.xml>) | Machine-readable pytest/JUnit result report for core tests. |
| [corrected_release_tests.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/corrected_release_tests.log>) | Captured execution/test log for corrected release tests. |
| [corrected_release_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/corrected_release_tests.xml>) | Machine-readable pytest/JUnit result report for corrected release tests. |
| [extended_preservation.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/extended_preservation.log>) | Captured execution/test log for extended preservation. |
| [extended_tests.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/extended_tests.log>) | Captured execution/test log for extended tests. |
| [extended_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/extended_tests.xml>) | Machine-readable pytest/JUnit result report for extended tests. |
| [extended_verified_tests.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/extended_verified_tests.log>) | Captured execution/test log for extended verified tests. |
| [extended_verified_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/extended_verified_tests.xml>) | Machine-readable pytest/JUnit result report for extended verified tests. |
| [fictional_cgm.csv](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/fictional_cgm.csv>) | Saved research output or fictional verification input: fictional cgm (inferred from filename). |
| [final_checks.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_checks.json>) | Saved audit/verification metadata or results for final checks (inferred from filename). |
| [final_preservation_v2.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_preservation_v2.log>) | Captured execution/test log for final preservation v2. |
| [final_server_health.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_server_health.json>) | Saved audit/verification metadata or results for final server health (inferred from filename). |
| [final_tests.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_tests.log>) | Captured execution/test log for final tests. |
| [final_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_tests.xml>) | Machine-readable pytest/JUnit result report for final tests. |
| [final_wording_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/final_wording_tests.xml>) | Machine-readable pytest/JUnit result report for final wording tests. |
| [ocr_fixture.png](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/ocr_fixture.png>) | Saved evaluation chart or verification image for ocr fixture (inferred from filename). |
| [ocr_pdf_results.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/ocr_pdf_results.json>) | Saved audit/verification metadata or results for ocr pdf results (inferred from filename). |
| [ocr_result.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/ocr_result.json>) | Saved audit/verification metadata or results for ocr result (inferred from filename). |
| [original_app.py.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/original_app.py.txt>) | Preserved pre-change source/document backup: app.py.txt. |
| [original_end_to_end_pipeline.py.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/original_end_to_end_pipeline.py.txt>) | Preserved pre-change source/document backup: end_to_end_pipeline.py.txt. |
| [original_rag_clinical_engine.py.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/original_rag_clinical_engine.py.txt>) | Preserved pre-change source/document backup: rag_clinical_engine.py.txt. |
| [original_README.md.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/original_README.md.txt>) | Preserved pre-change source/document backup: README.md.txt. |
| [preservation.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/preservation.log>) | Captured execution/test log for preservation. |
| [preservation_results.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/preservation_results.json>) | Saved audit/verification metadata or results for preservation results (inferred from filename). |
| [prior_report.pdf](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/prior_report.pdf>) | Report-extraction/OCR verification PDF fixture: prior report. |
| [report_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/report_tests.xml>) | Machine-readable pytest/JUnit result report for report tests. |
| [retraining_v2.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/retraining_v2.log>) | Captured execution/test log for retraining v2. |
| [retraining_v2_release001.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/retraining_v2_release001.log>) | Captured execution/test log for retraining v2 release001. |
| [retrieval.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/retrieval.log>) | Captured execution/test log for retrieval. |
| [retrieval_results.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/retrieval_results.json>) | Saved audit/verification metadata or results for retrieval results (inferred from filename). |
| [scanned_report.pdf](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/scanned_report.pdf>) | Report-extraction/OCR verification PDF fixture: scanned report. |
| [semantic_app.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/semantic_app.log>) | Captured execution/test log for semantic app. |
| [semantic_app_results.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/semantic_app_results.json>) | Saved audit/verification metadata or results for semantic app results (inferred from filename). |
| [server_process.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/server_process.json>) | Saved audit/verification metadata or results for server process (inferred from filename). |
| [server_stderr.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/server_stderr.log>) | Captured execution/test log for server stderr. |
| [server_stdout.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/server_stdout.log>) | Captured execution/test log for server stdout. |
| [text_report.pdf](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/text_report.pdf>) | Report-extraction/OCR verification PDF fixture: text report. |
| [verified_tests.log](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verified_tests.log>) | Captured execution/test log for verified tests. |
| [verified_tests.xml](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verified_tests.xml>) | Machine-readable pytest/JUnit result report for verified tests. |
| [verify_ocr_and_pdf.py](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verify_ocr_and_pdf.py>) | Audit/verification helper for verify ocr and pdf (inferred from filename). |
| [verify_preservation.py](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verify_preservation.py>) | Audit/verification helper for verify preservation (inferred from filename). |
| [verify_retrieval.py](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verify_retrieval.py>) | Audit/verification helper for verify retrieval (inferred from filename). |
| [verify_semantic_app.py](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/verify_semantic_app.py>) | Audit/verification helper for verify semantic app (inferred from filename). |

### evidence/implementation_20261011/cli_report

| File | Purpose |
|---|---|
| [report_explanation.md](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report/report_explanation.md>) | Saved report, input fixture or verification text: report explanation (inferred from filename). |
| [report_findings.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report/report_findings.json>) | Saved audit/verification metadata or results for report findings (inferred from filename). |

### evidence/implementation_20261011/cli_report_verified

| File | Purpose |
|---|---|
| [report_explanation.md](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report_verified/report_explanation.md>) | Saved report, input fixture or verification text: report explanation (inferred from filename). |
| [report_findings.json](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_report_verified/report_findings.json>) | Saved audit/verification metadata or results for report findings (inferred from filename). |

### evidence/implementation_20261011/cli_research

| File | Purpose |
|---|---|
| [research_note.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_research/research_note.txt>) | Saved report, input fixture or verification text: research note (inferred from filename). |
| [research_patient_guide.txt](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_research/research_patient_guide.txt>) | Saved report, input fixture or verification text: research patient guide (inferred from filename). |
| [research_synthetic_states.csv](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_research/research_synthetic_states.csv>) | Saved research output or fictional verification input: research synthetic states (inferred from filename). |
| [research_trajectory.csv](<C:/Users/tejde/Downloads/BTP/evidence/implementation_20261011/cli_research/research_trajectory.csv>) | Saved research output or fictional verification input: research trajectory (inferred from filename). |

### evidence/organization_20261012

| File | Purpose |
|---|---|
| [organization_checks.json](<C:/Users/tejde/Downloads/BTP/evidence/organization_20261012/organization_checks.json>) | Syntax, manifests, artifact hashes, notebook paths and migration verification. |
| [README.md](<C:/Users/tejde/Downloads/BTP/evidence/organization_20261012/README.md>) | Organization test and artifact-preservation results. |
| [test_results.log](<C:/Users/tejde/Downloads/BTP/evidence/organization_20261012/test_results.log>) | Complete successful regression output: 130 tests passed. |
| [test_results.xml](<C:/Users/tejde/Downloads/BTP/evidence/organization_20261012/test_results.xml>) | JUnit results for the successful regression run. |

### outputs

| File | Purpose |
|---|---|
| [README.md](<C:/Users/tejde/Downloads/BTP/outputs/README.md>) | Sample output location and future generated-report conventions. |

### outputs/legacy_sample

| File | Purpose |
|---|---|
| [patient_guide_LIB193265.txt](<C:/Users/tejde/Downloads/BTP/outputs/legacy_sample/patient_guide_LIB193265.txt>) | Saved legacy patient-facing research output for the patient ID in the filename. |
| [physician_note_LIB193265.txt](<C:/Users/tejde/Downloads/BTP/outputs/legacy_sample/physician_note_LIB193265.txt>) | Saved legacy physician-facing research output for the patient ID in the filename. |
| [trajectory_forecast_LIB193265.png](<C:/Users/tejde/Downloads/BTP/outputs/legacy_sample/trajectory_forecast_LIB193265.png>) | Saved legacy trajectory chart for the patient ID in the filename. |

### Phase_1_Data_Preparation

| File | Purpose |
|---|---|
| [cleaned_cohort_metadata.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/cleaned_cohort_metadata.csv>) | Metadata for the selected legacy research cohort. |
| [cleaned_daily_multimodal_dataset.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv>) | Legacy cleaned daily glucose, laboratory, demographic and diagnostic feature table. |
| [missing_values_audit_report.json](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/missing_values_audit_report.json>) | Saved missingness/data-preparation audit. |
| [phase1_ingestion.py](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/phase1_ingestion.py>) | Phase 1 ingestion, missing-value audit, cohort preparation and daily feature engineering. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |

### Phase_1_Data_Preparation/notebooks

| File | Purpose |
|---|---|
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/README.md>) | Notebook locations, data paths and external curation prerequisites. |

### Phase_1_Data_Preparation/notebooks/curation

| File | Purpose |
|---|---|
| [Biochemical_paramaters_curation.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/curation/Biochemical_paramaters_curation.ipynb>) | Original exploration/curation notebook for Biochemical paramaters curation (purpose inferred from filename). |
| [Diagnostic_curation.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/curation/Diagnostic_curation.ipynb>) | Original exploration/curation notebook for Diagnostic curation (purpose inferred from filename). |
| [Get_patient_info_variables.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/curation/Get_patient_info_variables.ipynb>) | Original exploration/curation notebook for Get patient info variables (purpose inferred from filename). |
| [Glucose_measurements_curation.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/curation/Glucose_measurements_curation.ipynb>) | Original exploration/curation notebook for Glucose measurements curation (purpose inferred from filename). |
| [Original_patient_info_curation.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/curation/Original_patient_info_curation.ipynb>) | Original exploration/curation notebook for Original patient info curation (purpose inferred from filename). |

### Phase_1_Data_Preparation/notebooks/exploration

| File | Purpose |
|---|---|
| [data_visual_Biochemical.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/exploration/data_visual_Biochemical.ipynb>) | Original exploration/curation notebook for data visual Biochemical (purpose inferred from filename). |
| [data_visual_diagnostics.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/exploration/data_visual_diagnostics.ipynb>) | Original exploration/curation notebook for data visual diagnostics (purpose inferred from filename). |
| [data_visual_glucose.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/exploration/data_visual_glucose.ipynb>) | Original exploration/curation notebook for data visual glucose (purpose inferred from filename). |
| [data_visual_patient_info.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/exploration/data_visual_patient_info.ipynb>) | Original exploration/curation notebook for data visual patient info (purpose inferred from filename). |
| [Graphs_and_stats.ipynb](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/notebooks/exploration/Graphs_and_stats.ipynb>) | Exploratory plotting and statistical analysis notebook. |

### Phase_1_Data_Preparation/raw_data

| File | Purpose |
|---|---|
| [Biochemical_parameters.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/raw_data/Biochemical_parameters.csv>) | Raw laboratory/biochemical records. |
| [Diagnostics.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/raw_data/Diagnostics.csv>) | Raw diagnostic records. |
| [Glucose_measurements.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/raw_data/Glucose_measurements.csv>) | Raw continuous glucose monitoring observations. |
| [Patient_info.csv](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/raw_data/Patient_info.csv>) | Raw patient metadata/demographics. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_1_Data_Preparation/raw_data/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |

### Phase_2_Latent_VAE

| File | Purpose |
|---|---|
| [best_compact_vae.pt](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/best_compact_vae.pt>) | Saved legacy VAE checkpoint. |
| [latent_health_vectors.pt](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/latent_health_vectors.pt>) | Saved encoded latent representations. |
| [patient_latent_health_summary.csv](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/patient_latent_health_summary.csv>) | Patient-level latent representation summary. |
| [phase2_vae.py](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/phase2_vae.py>) | Legacy model-input preparation, two-branch VAE training and latent-vector extraction. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |
| [scaler_parameters.json](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/scaler_parameters.json>) | Legacy feature ordering and normalization parameters. |
| [timeseries_tensors.pt](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/timeseries_tensors.pt>) | Prepared legacy time-series tensors and associated model-input data. |
| [vae_training_history.json](<C:/Users/tejde/Downloads/BTP/Phase_2_Latent_VAE/vae_training_history.json>) | Legacy VAE training history. |

### Phase_3_Generative_Forecasting

| File | Purpose |
|---|---|
| [best_baseline_lstm.pt](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/best_baseline_lstm.pt>) | Saved baseline LSTM weights. |
| [best_ctgan_generator.pt](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/best_ctgan_generator.pt>) | Saved conditional tabular GAN generator weights. |
| [best_temporal_cgan_generator.pt](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/best_temporal_cgan_generator.pt>) | Saved temporal conditional GAN generator weights. |
| [best_trajectory_forecaster.pt](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/best_trajectory_forecaster.pt>) | Saved legacy trajectory forecaster weights. |
| [cgan_probabilistic_trajectory_simulation.png](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/cgan_probabilistic_trajectory_simulation.png>) | Saved evaluation chart or verification image for cgan probabilistic trajectory simulation (inferred from filename). |
| [ctgan_evaluation_report.json](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/ctgan_evaluation_report.json>) | Saved conditional tabular GAN fidelity metrics. |
| [ctgan_tabular_fidelity_comparison.png](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/ctgan_tabular_fidelity_comparison.png>) | Saved evaluation chart or verification image for ctgan tabular fidelity comparison (inferred from filename). |
| [evaluation_results.json](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/evaluation_results.json>) | Saved legacy trajectory evaluation metrics. |
| [model_benchmarking_results.json](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/model_benchmarking_results.json>) | Saved comparative model metrics. |
| [phase3_4_trajectory_forecasting.py](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/phase3_4_trajectory_forecasting.py>) | Legacy sequential trajectory-forecast training/evaluation. |
| [phase3_ctgan.py](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/phase3_ctgan.py>) | Conditional tabular GAN training, synthetic state generation and fidelity evaluation. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |
| [train_comparative_models.py](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/train_comparative_models.py>) | Trains/evaluates legacy baseline LSTM and temporal conditional GAN comparators. |
| [trajectory_simulation_evaluation.png](<C:/Users/tejde/Downloads/BTP/Phase_3_Generative_Forecasting/trajectory_simulation_evaluation.png>) | Saved evaluation chart or verification image for trajectory simulation evaluation (inferred from filename). |

### Phase_4_RAG_Support

| File | Purpose |
|---|---|
| [ada_diabetes_clinical_guidelines.json](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/ada_diabetes_clinical_guidelines.json>) | Legacy local ADA guideline passage collection. |
| [ada_guidelines_faiss.index](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/ada_guidelines_faiss.index>) | Persisted FAISS vector index for legacy ADA guideline retrieval. |
| [doctor_clinical_decision_report_sample.txt](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/doctor_clinical_decision_report_sample.txt>) | Saved sample legacy doctor-facing report. |
| [phase4_trajectory_and_rag.py](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/phase4_trajectory_and_rag.py>) | Legacy probabilistic trajectory simulation and RAG pipeline orchestration. |
| [probabilistic_trajectory_simulation.png](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/probabilistic_trajectory_simulation.png>) | Saved evaluation chart or verification image for probabilistic trajectory simulation (inferred from filename). |
| [rag_clinical_engine.py](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/rag_clinical_engine.py>) | Legacy ADA passage indexing/retrieval and clinical-report synthesis implementation. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_4_RAG_Support/README.md>) | Overview and usage documentation for its containing project or phase; older phase claims should be read alongside the current audit. |

### Phase_5_Application

| File | Purpose |
|---|---|
| [app.py](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/app.py>) | Full Streamlit report review, explanation, timeline and research simulation UI. |
| [end_to_end_pipeline.py](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/end_to_end_pipeline.py>) | Full report and cohort/patient-owned model CLI. |
| [README.md](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/README.md>) | Integrated application and corrected model-release guide. |
| [train_corrected_release.py](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/train_corrected_release.py>) | Builds fresh CGM daily aggregates and trains/evaluates the corrected VAE/GRU release with separate patient groups. |

### Phase_5_Application/models/cgm

| File | Purpose |
|---|---|
| [daily_cgm.csv](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/daily_cgm.csv>) | Corrected CGM daily aggregates; parent-folder version and release copy are separate files. |
| [data_quality.json](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/data_quality.json>) | Quality/coverage audit for corrected raw CGM aggregation. |

### Phase_5_Application/models/cgm/release_001

| File | Purpose |
|---|---|
| [calibration.npz](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/calibration.npz>) | Residual calibration arrays for uncertainty bounds and correlated simulation paths. |
| [daily_cgm.csv](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/daily_cgm.csv>) | Corrected CGM daily aggregates; parent-folder version and release copy are separate files. |
| [data_quality.json](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/data_quality.json>) | Quality/coverage audit for corrected raw CGM aggregation. |
| [manifest.json](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/manifest.json>) | Release artifact integrity/hash manifest; copies under test_temp are test fixtures. |
| [patient_splits.json](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/patient_splits.json>) | Corrected release patient memberships for training, validation, calibration and test. |
| [pattern_forecaster.pt](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/pattern_forecaster.pt>) | Corrected CGM forecaster weights. |
| [pattern_vae.pt](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/pattern_vae.pt>) | Corrected CGM-pattern VAE weights. |
| [release.json](<C:/Users/tejde/Downloads/BTP/Phase_5_Application/models/cgm/release_001/release.json>) | Corrected model-release metadata, settings and evaluation results; test_temp copies are fixtures. |

### requirements

| File | Purpose |
|---|---|
| [app.txt](<C:/Users/tejde/Downloads/BTP/requirements/app.txt>) | Pinned core Streamlit/report-processing dependencies. |
| [dev.txt](<C:/Users/tejde/Downloads/BTP/requirements/dev.txt>) | Includes app.txt and pytest for development checks. |
| [README.md](<C:/Users/tejde/Downloads/BTP/requirements/README.md>) | Dependency-file guide. |
| [research.txt](<C:/Users/tejde/Downloads/BTP/requirements/research.txt>) | Original/research ML and optional service dependencies. |

### scripts

| File | Purpose |
|---|---|
| [run_phase_1.py](<C:/Users/tejde/Downloads/BTP/scripts/run_phase_1.py>) | Phase 1 ingestion launcher; executes within its phase directory. |
| [run_phase_2.py](<C:/Users/tejde/Downloads/BTP/scripts/run_phase_2.py>) | Phase 2 VAE launcher; executes within its phase directory. |
| [run_phase_3.py](<C:/Users/tejde/Downloads/BTP/scripts/run_phase_3.py>) | Phase 3 GAN launcher; executes within its phase directory. |
| [run_phase_4.py](<C:/Users/tejde/Downloads/BTP/scripts/run_phase_4.py>) | Phase 4 simulation/RAG launcher; executes within its phase directory. |

### src/healthcare_app

| File | Purpose |
|---|---|
| [__init__.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/__init__.py>) | Marks healthcare_app as an importable Python package. |
| [analysis.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/analysis.py>) | Combines reviewed facts, provenance, dated comparisons, grounded explanations, Q&A and exports. |
| [artifact_manifest.json](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/artifact_manifest.json>) | Trusted hashes for legacy research artifacts. |
| [cgm.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/cgm.py>) | Validates measured CGM uploads and aggregates observed readings into daily metrics. |
| [documents.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/documents.py>) | Bounded local PDF/image/text extraction and OCR coordination. |
| [history.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/history.py>) | Extracts explicit diagnosis/medication events and builds the clinical timeline. |
| [knowledge.json](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/knowledge.json>) | Versioned sourced educational/guideline passage corpus for the current app. |
| [knowledge.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/knowledge.py>) | Local evidence retrieval using lexical/vector representations and optional offline semantic embeddings. |
| [llm.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/llm.py>) | Opt-in API adapter and validation of approved evidence/claim selection. |
| [model_inputs.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/model_inputs.py>) | Validates the legacy uploaded daily-feature input contract. |
| [records.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/records.py>) | Extracts numeric lab findings, units, dates and printed reference ranges. |
| [research.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/research.py>) | Runs trusted legacy research models with aligned patient windows and constrained outputs. |
| [summary.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/summary.py>) | Composes and renders patient summaries from approved claims and citations. |
| [trajectory_v2.py](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/trajectory_v2.py>) | Corrected CGM-pattern VAE, GRU forecaster, baselines, constraints and release loader/simulator. |
| [windows_ocr.ps1](<C:/Users/tejde/Downloads/BTP/src/healthcare_app/windows_ocr.ps1>) | Windows OCR/PDF-rendering helper invoked by document extraction. |

### tests

| File | Purpose |
|---|---|
| [conftest.py](<C:/Users/tejde/Downloads/BTP/tests/conftest.py>) | Shared pytest setup; blocks paid/external network calls in tests. |
| [test_app.py](<C:/Users/tejde/Downloads/BTP/tests/test_app.py>) | Streamlit interface, session isolation, authentication and simulation regression tests. |
| [test_cli.py](<C:/Users/tejde/Downloads/BTP/tests/test_cli.py>) | CLI output and invalid-input regression tests. |
| [test_clinical_workflow.py](<C:/Users/tejde/Downloads/BTP/tests/test_clinical_workflow.py>) | Clinical assertions, medications, timelines and patient-summary tests. |
| [test_corrected_cgm.py](<C:/Users/tejde/Downloads/BTP/tests/test_corrected_cgm.py>) | Measured-CGM validation, corrected inference, patient splits, integrity and summary-linking tests. |
| [test_model_uploads.py](<C:/Users/tejde/Downloads/BTP/tests/test_model_uploads.py>) | Legacy daily-feature upload schema and own-patient inference tests. |
| [test_reports.py](<C:/Users/tejde/Downloads/BTP/tests/test_reports.py>) | Document extraction, lab parsing, evidence retrieval, report exports and LLM constraint tests. |
| [test_research.py](<C:/Users/tejde/Downloads/BTP/tests/test_research.py>) | Legacy research alignment, reproducibility, constraints and RAG repair tests. |
