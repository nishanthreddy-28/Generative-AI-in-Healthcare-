# Reproduce the research checks

These scripts read authorized local data and checkpoints. They do not fit models or rewrite input datasets. A bare public clone lacks the private inputs; see [data access](../DATA_ACCESS.md). Use the research dependencies and run from the Granada project directory.

```bash
python scripts/audit_saved_data.py --outdir /tmp/granada-saved-data-audit
python scripts/audit_raw_cgm.py --outdir /tmp/granada-raw-cgm-audit
python scripts/audit_artifacts.py --outdir /tmp/granada-artifact-audit
```

Choose a new, empty output directory for each run. `--project-root` can point to an authorized complete local project, so the public scripts can audit private inputs without copying those inputs into Git. Limit OpenMP/BLAS threads if your native numerical libraries are unstable.

The saved-data audit checks legacy splits/scalers/imputation and recomputes corrected-release metrics/calibration from its checkpoint. The raw audit independently bins the entire sensor CSV and compares every retained daily aggregate. The artifact audit uses restricted PyTorch loading and checks the legacy forecast pairing and Phase 2 input contract; the Phase 2 negative check runs in the output directory and fails before saving when the legacy CSV lacks `split`. If a different input version allows Phase 2 to complete, it can write derived files into that output directory.

Results from 19 October 2026 are in [the audit evidence folder](../docs/reports/audit_20261019). Reproducing a saved checkpoint is distinct from reproducing the original model training.
