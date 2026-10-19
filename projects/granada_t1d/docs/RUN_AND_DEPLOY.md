# Run and deploy ClearCare

## Local application

From the BTP project root, using the existing environment:

```powershell
python -m streamlit run app.py
```

Open `http://127.0.0.1:8501`. The server binds to the local computer by default. The implementation verification instance uses port 8503. No paid API is needed.

If creating a new environment, obtain approval for dependency installation first. The core dependencies are in `requirements/app.txt`; research models additionally need the original ML dependencies in `requirements/research.txt`. The observed runtime was Windows / Python 3.13.5. A fresh install has not been executed in this work. The core direct dependencies are pinned; transitive dependencies are not yet fully locked.

## Report workflow

1. Upload up to five PDF, PNG, JPEG or UTF-8 TXT reports, or paste text. Maximum 10 MB each / 20 MB total; PDFs have a 20-page limit.
2. Review the transcription against the original. Correct OCR errors, decimal points, dates, units and ranges. Use `YYYY-MM-DD` for unambiguous dates. A supported line is `Glucose 146 mg/dL 70-99`.
3. Confirm values, units, ranges, dates, diagnosis assertions and medication names/dose text against the originals. For multiple files, confirm they belong to the same patient.
4. Explain findings, inspect citations, ask questions and download Markdown/JSON.

The supported numeric parser covers HbA1c, glucose, creatinine/eGFR, common CBC measurements, lipids, TSH, common liver tests and selected electrolytes. It deliberately withholds range classification for unknown units, missing units, mismatched reference units and bound results. Unsupported narrative, imaging, histopathology and arbitrary report layouts are not medically interpreted. Extracted text remains editable so that a recognized result can be corrected without inventing information.

A single report produces a snapshot. Multiple unambiguous dates and matching units allow numeric comparisons; different labs/methods/fasting status can limit their meaning. Neither a snapshot nor a pair of laboratory reports supplies the consecutive daily inputs needed by the diabetes models.

## OCR portability

Windows uses installed Windows OCR and Windows PDF rendering locally. The packaged helper runs in a child Windows PowerShell process with an execution-policy override scoped to that process; it does not change system policy. English OCR language support must be installed by the system administrator. This host has en-GB/en-US language support, and genuine image/scanned-PDF fixtures were executed successfully outside the Codex execution sandbox.

Linux deployments need local `tesseract` and `pdftoppm` executables. The proposed Dockerfile includes both. This Linux path/container was authored but has not been built or executed. OCR can make errors: review is always required. Temporary files are deleted when extraction succeeds or fails; memory is not guaranteed to be securely erased. Set `TEMP`/`TMP` to an appropriate writable private location if the hosting environment restricts temporary-file access.

## Optional LLM assistance

Expand **Optional LLM-assisted organization**, enter a session-specific API key, explicitly consent to external processing/cost, and click **Organize with LLM**. No automatic rerun/API call occurs. The key is passed directly to a client, never copied to process environment variables. Clear it using the key-clearing control or clear the whole report session.

The API sees numeric findings, recognized clinical mentions and approved evidence/claims. Raw PDFs/images, filenames, source quotes, names, exact report dates, unknown entity text and recorded medication doses are withheld from patient-summary calls. Clinical facts remain sensitive; explicit consent is required. The SDK has a 45-second timeout, storage disabled in the request and no automatic retries. Strict provider schemas and independent server validation reject missing/unknown/duplicate/inapplicable claims and citations. The LLM composes a four-section explanation by selecting and ordering complete approved sentences, preserving assertions and uncertainties. It cannot freely invent medical prose. GPT-6 Astra is the quality-oriented default; GPT-6.1 Sol and pinned GPT-4.1 are selectable. Account access and live behavior are untested; no paid call ran.

## Diabetes research

### Patient-owned simulation and clinical history

The report review now includes explicit diagnosis/medication mentions, their assertion status, literal dose text and source locations. Unknown labelled entities require verification; broad clinical narrative extraction is not validated. Clinical events join lab results in the Timeline tab. Dates represent document dates rather than confirmed onset/start/adherence. Conflicting and repeated source entries remain visible.

Use **My glucose simulation**, or follow **Check my history for glucose simulation** from the report workspace. The corrected default takes a measured CGM CSV with exactly `timestamp,glucose_mg_dl`, ISO local timestamps and reviewed type 1 diabetes measurements. It averages actual observations in 15-minute bins, requires 64 occupied bins per day and 30 consecutive qualifying days, removes exact duplicate observations and rejects conflicting timestamps. Missing bins are never interpolated. It loads only release artifacts, not cohort records. When reports exist, confirm the same person to link a distinctly labelled simulation summary to the cited report explanation. Changed report/model claims invalidate prior LLM summary plans.

```powershell
python end_to_end_pipeline.py --cgm path/to/measured_cgm.csv --simulations 10 --outdir new_cgm_output
```

The corrected model release lives at `Phase_5_Application/models/cgm/release_001`. It was trained locally after authorization, with fresh raw aggregates, train-only normalization, patient-disjoint splits, validation selection and separate calibration/test groups. Read [CGM_MODEL_CARD.md](models/CGM_MODEL_CARD.md) for actual errors, baseline comparisons and interval coverage. Neural complexity did not produce a demonstrated test advantage over the historical-mean baseline. Research simulations remain distinct from clinically reliable forecasts. The UI shows measured input history, a neural research curve and a historical-mean comparator; marginal calibration bounds and sensitivity paths are under the provenance expander. Dates start after the selected observation window rather than after today's date, and older qualifying windows are explicit.

The **Legacy multimodal daily features** option still requires all 50 numeric features plus ISO date, known flags and verified original units. Missing features are not imputed. Lab unit compatibility needs a separately reviewed contract. This old route preserves the original experimental models and is separate from the corrected CGM-only release.

```powershell
python end_to_end_pipeline.py --daily-history path/to/daily_features.csv --simulations 10 --outdir new_patient_output
```

The legacy UI shows its experimental curve beside a last-observation comparator. Automatic construction of missing multimodal features and similar historical-patient retrieval are not implemented. The corrected CGM route does not need to invent those missing features.

### Optional offline semantic retrieval

The portable default combines BM25 with word/character TF-IDF. To use already available reviewed BGE weights, set these **process environment** variables before startup:

```powershell
$env:APP_SEMANTIC_MODEL = 'BAAI/bge-base-en-v1.5'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
python -m streamlit run app.py
```

BGE revision `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a` is pinned; loading is local-only with safe tensors and remote code disabled. This host already had the model and libraries, so nothing was downloaded or installed. Missing optional resources show a warning and a labelled lexical fallback. The verified optional versions are sentence-transformers 6.0.0, transformers 5.16.1, torch 2.14.0, FAISS 1.15.0 and OpenAI SDK 3.3.1. These are not added to the lightweight core container. Install/download approval and fresh-environment validation are required before reproducing that optional stack elsewhere. One measured cold BGE initialization took 39.111 seconds; budget for cold starts and memory rather than treating it as a free upgrade.

The active corpus has 16 passages: limited NLM/NIDDK educational content and a short ADA 2026 recommendation extract. It is not exhaustive clinical guidance. [WORKFLOW_COMPLETION_REPORT.md](reports/WORKFLOW_COMPLETION_REPORT.md) records current tests and limitations.

The local research workspace requires explicit opt-in and existing authorized cohort/model artifacts. It runs live VAE encoding, a separate conditional tabular generator and the GRU. The latest qualifying consecutive 30-day window drives every component. Older qualifying windows are explicitly dated; absent history is rejected rather than replaced with an average latent. Patient age is read from that window.

These unchanged weights retain inherited preprocessing leakage and poor calibration. Latent-noise perturbations are labeled sensitivity paths; clinical confidence bands are not displayed. Presentation constraints repair percentage sums/subsets and derive GMI consistently but do not repair model accuracy or validate forecasts. GMI is not a measured HbA1c. No dosing decisions are generated. Raw outputs are retained in the result object for research evaluation. Corrections/retraining and external validation require a separately approved, versioned exercise.

Trusted checkpoint hashes are in `src/healthcare_app/artifact_manifest.json`; unknown or modified weights fail closed before restricted loading. Do not register untrusted models. Model/data changes require a new validated artifact release and server restart. The app does not retrain or overwrite existing datasets/results.

## CLI and automated checks

```powershell
python end_to_end_pipeline.py --report path/to/report.pdf --outdir new_report_output
python end_to_end_pipeline.py --patient LIB193266 --simulations 10 --outdir new_research_output
python -m pytest tests -q
```

CLI explanations are prominently marked unreviewed; use the UI to verify extraction. Existing nonempty output directories are rejected. Tests block external network calls. Evidence and original source backups are under `evidence/implementation_20261011`. Windows sandbox native-handle restrictions caused failed initial harness runs; the final regression run used the normal local runtime.

## Deployment boundaries

For a private remotely reachable deployment, set `APP_PUBLIC_DEPLOYMENT=1`, provide a strong `APP_ACCESS_TOKEN`, keep `APP_ENABLE_RESEARCH=0`, and use a managed HTTPS reverse proxy. Public mode fails closed without a token. A shared token is only basic access gating: production needs proper identity, rate limiting, per-user authorization, retention policy, isolated document workers, dependency vulnerability scanning, operational monitoring and a security review. Do not expose this medical-data application anonymously.

The proposed Docker image includes the report application and public educational sources, not the controlled cohort or model artifacts. A possible command, after build/install approval, is:

```powershell
docker build -t clearcare .
docker run --rm -p 127.0.0.1:8501:8501 -e APP_ACCESS_TOKEN clearcare
```

Set `APP_ACCESS_TOKEN` in the host process environment before this command. Do not put secrets in source, shell history or image layers. The server health endpoint checks Streamlit liveness; it does not certify OCR/model validity. Clean installation, container build, Linux OCR, multi-user load and public hosting remain unverified. Dataset rights must be resolved before any research-artifact distribution. Clinical use remains out of scope.

## Current local verification process

The verification server was launched as a hidden background process bound only to `127.0.0.1:8503`, with cached semantic retrieval and offline model settings. Its recorded process ID and local logs are in `evidence/implementation_20261011/server_process.json`, `server_stdout.log` and `server_stderr.log`. To stop it, check that the recorded PID still belongs to this Python/Streamlit server before using `Stop-Process -Id 2116`; process IDs can be reused. Starting another server on the same port will fail while this one is running.
