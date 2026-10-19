# IEEE manuscript status

The historical manuscript is not ready for submission. The [19 October research audit](reports/RESEARCH_AUDIT_2026-10-19.md) documents the confirmed leakage, reproduction failures, checkpoint/evaluation mismatches, and unsupported claims that need to be addressed.

The corrected CGM release has verified patient-disjoint groups and training-only normalization. Its checkpoint results are reproducible, but it did not outperform the 30-day mean baseline on held-out glucose MAE. See the [model card](models/CGM_MODEL_CARD.md).

Choose the exact experiment and target IEEE venue before rewriting the paper. Report the cohort flow, inputs, outcomes, split/selection procedure, baselines, uncertainty, data-access conditions, ethics statement, related work, and limitations. Venue-specific formatting and article requirements still need to be checked. Software test success is not clinical validation.
