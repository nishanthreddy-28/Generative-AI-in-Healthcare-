# IEEE research readiness: not ready for submission

The source is published for research review, not as evidence of clinical validation. `guides/IEEE_Journal_Manuscript_Draft.md` is a historical draft with unsupported claims and mismatches against the current corrected release. It must be rewritten before submission.

The corrected release is CGM-only VAE/GRU forecasting with four daily input/target metrics, 30 observed days, 60 forecast days and 39 held-out test patients. Saved test glucose MAE is 23.97 mg/dL for the neural model versus 23.33 for the 30-day mean baseline. Nominal-80% marginal coverage is approximately 77–79%. No superiority, individual clinical reliability, privacy guarantee, causal treatment effect or zero hallucination follows from these results.

The historical draft instead describes multimodal temporal-cGAN forecasts, other cohort/window counts, clinical validation, nocturnal treatment decisions and guaranteed safety. These must not be represented as results of the corrected release. GMI is not measured HbA1c. The legacy multimodal artifacts retain unresolved data/preprocessing limitations.

Before submission: audit every file and data transformation, preserve explicit patient-disjoint split manifests, verify train-only transforms and time availability, reconcile code/checkpoint/result lineage, evaluate patient-level uncertainty and relevant baselines, report limitations and data-use terms, and choose the target IEEE venue's current author requirements. The complete file-by-file audit is pending following the repository reorganization.

The existing local regression suite passed 130 tests during the October 19 review. Passing software tests is separate from scientific or clinical validation.
