# Granada research audit

19 October 2026 · Source added in commit `048bffb` · No retraining performed

## Verdict

The corrected CGM release has reproducible daily aggregates, patient-disjoint groups and training-only normalization. Its saved test metrics can be reproduced from the supplied checkpoint. The legacy multimodal artifacts contain confirmed imputation leakage and cannot be used as evidence of a leakage-free model. The historical manuscript does not describe the corrected release and is not ready for IEEE submission.

This is a scientific/code review, not a certification of clinical safety or an acceptance prediction for a particular IEEE venue.

## What was checked

Every one of the **262 original local files** was inventoried and SHA-256 checked. All remained unchanged. The inventory covers 49 Python files, ten notebooks, 12 CSVs, 41 JSON files, nine PyTorch artifacts, one NumPy archive, two FAISS indexes, five PDFs, 24 images, ten XML files and the remaining text/configuration files.

All Python sources parsed; notebook code cells parsed after removing notebook magic. CSV rows were read, JSON/TOML/XML parsed, all nine PyTorch artifacts loaded in restricted mode, NumPy arrays checked without pickle, and both index dimensions inspected. All PDF pages were text-extracted. Images were integrity-checked; this does not establish that every historical chart is scientifically valid or that every screenshot was exhaustively interpreted. Archived audit scripts/logs were inspected as evidence, not executed blindly.

Detailed scientific review concentrated on ingestion, temporal availability, splits, scaling, target pairing, checkpoint evaluation, the active CGM release, and manuscript claims. The local file ledger records the review method and limitations per file; it is retained locally because archived filenames and artifacts can contain patient-level information. A parser pass is not a finding of scientific correctness.

The relocated full application suite passed **130 tests in 32.71 seconds** with local authorized assets. An initial relocation check failed because test checkpoints were symlinked outside the allowed artifact root; replacing those test-only symlinks with local copies resolved it. Product code was not weakened to accept external paths. The original-folder suite also passed 130 tests in the earlier check.

## Confirmed blockers

### G01 — The saved legacy imputation uses held-out data

Files: `Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv`, `phase1_ingestion.py`, `Phase_2_Latent_VAE/scaler_parameters.json` and dependent legacy weights.

For all 14 laboratory columns, the saved imputed values match medians calculated across the eligible whole cohort. For eight biomarkers, those values differ from medians calculated using the saved training patients only. For example, creatinine fills use 0.79 versus a training-only median of 0.78; glucose fills use 144 versus 143. These are stored feature values/units from this dataset, not clinical interpretations.

The scaler itself matches the training-patient rows to floating-point tolerance. That does not repair features already imputed using held-out records. Current source code's train-only median logic also does not retroactively repair the saved CSV, tensors or weights.

**Required correction:** rebuild a new versioned multimodal dataset using a saved split manifest before fitting transforms, then retrain/evaluate dependent models. Preserve the existing release as historical. Do not keep old performance numbers after changing preprocessing.

### G02 — Phase 2 cannot rebuild from the delivered cleaned tables

File: `Phase_2_Latent_VAE/phase2_vae.py`, approximately lines 148–154.

Neither the saved cleaned daily table nor cohort metadata has the `split` column required by current preparation code. An isolated invocation reproduced `KeyError: 'split'` before saving new artifacts. Source and saved artifacts represent different preparation versions.

**Required correction:** define one versioned input schema and explicit split file. The preparation stage must validate compatibility before doing work and must not silently generate a new experimental split.

### G03 — Manuscript architecture, cohort and results do not match the artifacts

File: `docs/guides/IEEE_Journal_Manuscript_Draft.md`.

The draft describes a multimodal temporal cGAN, 39 features, 3,006 test windows from 76 patients, clinical validation, zero hallucination and safe treatment decisions. The current corrected release is CGM-only VAE/GRU, four modeled metrics, 30 observed days, 60 forecast days, and 39 test patients.

Even the available legacy inputs differ from the draft: there are 2,068 legacy VAE test windows from 90 test patients; the current 60-day target pairing produces **953 test trajectory windows from 42 patients**. Different stages legitimately retain different windows, but the paper must state which population and artifact each result belongs to. The draft's numbers cannot be substituted without provenance.

**Required correction:** choose the experiment reported by the paper, rewrite methods/results around its exact source/data/checkpoint, distinguish GMI from measured HbA1c, use percentage points for TIR/TBR/CV errors, and remove unsupported clinical/nocturnal-treatment/zero-hallucination claims. Sixty days is not a quarter.

### G04 — Legacy evaluation can assess a different epoch from the saved checkpoint

Files: `Phase_3_Generative_Forecasting/phase3_ctgan.py`, `phase3_4_trajectory_forecasting.py`, `train_comparative_models.py`.

The tabular GAN and GRU training wrappers save the best validation checkpoint but pass the final in-memory model into their evaluation functions. The comparative LSTM likewise saves a best checkpoint and evaluates its final weights; the temporal GAN's final weights are named `best_temporal_cgan_generator.pt` without equivalent best-validation selection.

**Required correction:** evaluate explicitly reloaded selected checkpoints and include their hashes in the result JSON. The corrected CGM training script does reload its selected state; this finding is about the legacy trainers.

### G05 — Legacy clinical feature provenance is insufficient

Files: `Phase_1_Data_Preparation/phase1_ingestion.py`, raw lab and diagnosis tables.

The raw lab table has no unit column; diagnosis records have no onset/recording date. The code maps full-history diagnosis information onto every daily row and treats absent matched diagnoses as zero. It is not possible to verify that a condition was known at a particular historical cutoff from these inputs. This is a temporal-availability risk, not proof that every diagnosis is future information. The raw patient table has 225 missing diagnosis counts; missing documentation does not establish a confirmed negative condition.

The lab join chooses the latest blood-draw row as a whole. It does not independently obtain the latest known valid value and age for each analyte. The raw labs also have 1,808 repeated patient/date/analyte keys; averaging requires an explicit duplicate/conflict policy rather than an assumption about repeated measurements.

**Required correction:** establish units and per-analyte timestamps/expiry, represent unknown diagnosis status explicitly, and remove undated information from prospective prediction unless its availability is justified. The corrected CGM release excludes these lab/diagnosis inputs and avoids this particular problem.

### G06 — Legacy preprocessing is not fully reproducible across processes

File: `Phase_1_Data_Preparation/phase1_ingestion.py`, approximately lines 317–330.

`list(selected_pats)` converts a set into an order before a seeded shuffle. A fixed NumPy seed does not fix hash-dependent set ordering. Saved legacy patient groups are disjoint (415 train, 88 validation, 90 test), but rerunning source is not guaranteed to recreate those groups.

**Required correction:** sort identifiers before seeded permutation and save the exact split manifest with the artifact/data hashes. The corrected trainer sorts IDs and saves its four patient groups.

### G07 — Legacy aggregation and output handling change the task

Files: `phase1_ingestion.py`, `phase3_ctgan.py`, `phase3_4_trajectory_forecasting.py`, `train_comparative_models.py`.

Legacy coverage counts rows rather than occupied sampling intervals; repeated scans can affect daily coverage and weighting. Its 67-of-96 threshold is about 69.8%, and is distinct from the corrected policy of 64 occupied bins (66.7%). Neither should be described imprecisely as an identical coverage rule.

The tabular GAN's substring-based glucose clipping also applies 40–500 to `glucose_std` and `glucose_cv`. Its 30-day nocturnal-low sum is normalized with daily count statistics, then clipped to 100 at evaluation. Those quantities need their own units, scaling and limits. These counts are readings below threshold, not independently defined hypoglycemia episodes.

The legacy trajectory evaluator clips reference targets as well as predictions. This can alter the outcome being evaluated. Its train/validation/test windows overlap within patients; this is not cross-split patient leakage, but uncertainty estimates must respect patient clustering rather than treat every overlapping daily prediction as independent.

**Required correction:** specify typed feature/target metadata, preserve unmodified reference targets for evaluation, use metrics at the patient level, and re-evaluate the exact served constraints. Inspect bounds and change counts transparently.

### G08 — The direct Phase 4 demo does not use the observed final state

File: `Phase_4_RAG_Support/phase4_trajectory_and_rag.py`, approximately lines 130–140.

The older demo sets `x_last_sample` to six zeros instead of the patient's final observed normalized target state. It can also fabricate age/sex defaults if the profile lookup fails. Latent perturbations are described as confidence bands without calibration. The newer `ResearchPipeline` aligns the last observed state and labels its spread as uncalibrated; it does not call this old demo for current serving.

**Required correction:** retire the old demo or route it through the guarded current interface, with explicit data provenance and no profile fallback. Do not cite its old chart as a calibrated clinical forecast.

### G09 — Original curation notebooks are not a reproducible current pipeline

Files: the ten notebooks under `Phase_1_Data_Preparation/notebooks`.

The glucose curation notebook has a patient-ID naming mismatch and applies `.date()` after converting its time column to `datetime.time`. The patient-information notebook has manual transition/counting and positional-assignment logic requiring correction with boundary fixtures and keyed joins. Original source tables/CMS reference material are missing from the delivered inputs; one notebook uses Windows-specific `ansi` encoding. The statistics notebook reports DataFrame element counts as bytes, calculates age for 2017 under a different caption and hardcodes cohort dimensions.

**Required correction:** document source provenance, replace positional joins/counts, test first/last/single-patient boundaries, use portable paths/encodings and regenerate outputs. Clearing outputs for publication prevents accidental disclosure; it does not fix notebook code.

## Corrected CGM release: independent results

The raw audit read **22,671,708 rows across 736 patients**. It independently discarded **22,026 conflicting timestamps**, combined **354,937 readings within 15-minute bins**, excluded **23,602 low-coverage days**, and reproduced all **234,180 retained daily rows**. No saved daily row was missing or extra, and no daily metric differed by more than `1e-8` (the maximum observed numerical difference was about `5.7e-14`).

The corrected daily table contains 735 patients; 312 have a qualifying 90-day segment. Train/validation/calibration/test groups contain **187/47/39/39 patients**, with zero pairwise overlap. Training-only scaler means match exactly and standard deviations differ by at most `3.6e-15`. Calibration/test use one window per patient.

Recomputed 60-day test glucose MAE:

- Last observed day: **32.16 mg/dL**.
- Past seven-day mean: **25.53 mg/dL**.
- Past 30-day mean: **23.33 mg/dL**.
- Validation-selected VAE/GRU: **23.97 mg/dL**.

Saved MAEs were reproduced within about `1.4e-7`. The neural model is approximately **0.64 mg/dL worse** than the past-30-day mean on this test set. Validation selection precedes test evaluation in the trainer; there is no basis to claim that this neural model beats the best tested baseline.

Recomputed nominal-80% marginal interval coverage: **77.44% glucose, 77.39% TIR, 77.22% TBR, 79.15% CV**. The conformal order statistic is 32 out of 39 calibration patients; the saved radius was reproduced within `9.2e-6`. These are marginal horizon/target results, not simultaneous trajectory coverage or a demonstrated probability for a particular person.

This check validates the supplied daily preparation, splits, scaler and inference results. It does not rule out all sources of leakage, establish broader population representativeness, verify model-training history independently, prove synthetic privacy, or provide external clinical validation. The selection of well-covered histories and one regional cohort must be described.

## Report assistant and repository quality

The report assistant has bounded parsers, required UI review, provenance, missing-input rejection, local retrieval and constrained claim composition. It does not have an independently annotated clinical extraction benchmark. The 16-entry knowledge corpus and authored smoke queries cannot establish general clinical retrieval/answering accuracy. Live provider execution, clean installation/container execution, document-worker isolation and concurrent-user performance remain unverified here.

The current retrieval implementation builds a new compatible index in memory. The old saved ADA index is 256-dimensional with six vectors; the archived fixture is 512-dimensional with six vectors. Those are historical files and should not be confused with the active knowledge base or taken as evidence that the current report route still has the old dimension failure.

Several guides/phase READMEs repeat stale claims, wrong feature/comorbidity names, machine-specific paths or local-only evidence links. Current documentation must identify the active release and distinguish local private assets from public code. The repository review also found that a complete dependency installation has not been reproduced in a clean environment; software passing in an existing environment is a narrower claim.

## What is necessary before an IEEE submission

**Scientific corrections:** resolve G01–G09 for any legacy experiment retained in the paper; otherwise restrict the paper to the corrected release and explicitly scope out legacy claims. Report the actual cohort flow, time rules, outcome definitions, baselines, checkpoint selection and failure cases. Add patient-level confidence intervals and useful ablations; evaluate external/subgroup performance before making broader clinical claims. External validation is a scientific need for broader generalization claims, not a universal IEEE formatting checkbox.

**Reporting and ethics:** identify the actual authors, institutions, funding/conflicts, data-access conditions and applicable secondary-data ethics/IRB approval or exemption determination. The source study's approval is not automatically the author's own institutional authorization. IEEE's [submission policies](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/) describe human-subject oversight statements and disclosure of AI-generated article content. The authors must review the applicable policy and describe actual assistance; a less templated README does not remove a disclosure obligation.

**Reproducibility:** document a tested environment, exact code/data/checkpoint versions, seeds, commands and access instructions. IEEE [encourages sharing research outputs](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/); controlled patient data must still follow its agreement. [TRIPOD+AI](https://www.bmj.com/content/385/bmj-2023-078378) is a useful clinical-prediction reporting checklist, not an IEEE acceptance certificate.

**Venue preparation:** the draft mentions JBHI/TBME but no venue has been selected. Confirm scope, novelty requirements, article category, length, template, references and supplementary-material rules against that venue's current instructions before final formatting. The current manuscript also needs a real related-work/reference section and experiment-specific discussion.

## Evidence and reproduction

Aggregate results: [raw-data comparison](audit_20261019/raw_results.json), [saved-data/split/scaler/metric checks](audit_20261019/science_results.json), [artifact and negative-input checks](audit_20261019/artifact_results.json), [index metadata](audit_20261019/index_results.json).

[Reproduction commands](../../scripts/AUDITING.md) run against authorized local assets and write separate audit outputs. Original datasets, weights and results were not overwritten. The complete 262-file ledger remains in the local review folder.
