# Phase 2 · Latent VAE

`phase2_vae.py` prepares consecutive 30-day windows and trains the original two-branch VAE. The saved representation has 41 dynamic inputs (27 clinical features and 14 imputation flags), nine static inputs, and a 16-dimensional latent vector. Reconstruction excludes the imputation flags.

Saved scalers and training history are included. Patient windows, latent exports, tensors, and weights are kept locally.

The current preparation code expects a `split` column absent from the delivered legacy cleaned CSV. The legacy inputs also retain upstream imputation leakage. See the [audit](../docs/reports/RESEARCH_AUDIT_2026-10-19.md) before rebuilding or using these results in a paper. The corrected CGM-only VAE is implemented separately in `src/healthcare_app/trajectory_v2.py`.
