# Phase 2: Time-Series Windowing & Multi-Modal Latent Health VAE

> Current location: `Phase_2_Latent_VAE`. From the BTP root, run `python scripts/run_phase_2.py`. These commands regenerate original research outputs. See [project structure](../docs/PROJECT_STRUCTURE.md) and [the audit](../docs/reports/PROJECT_AUDIT_REPORT.md).

## Overview
Phase 2 transforms the longitudinal multi-modal daily dataset produced in Phase 1 into 30-day time-series windowed tensors and trains a **Two-Branch Multi-Modal Variational Autoencoder (VAE)**. The model compresses complex 30-day temporal glycemic and clinical trajectories into a compact, continuous latent health representation:

$$\\mathbf{z} \\in \\mathbb{R}^{16}$$

---

## Architecture & Pipeline Design

The complete Phase 2 workflow is implemented in `phase2_vae.py`:

```
                 30-Day Window Input (30 x 50)
                               │
       ┌───────────────────────┴───────────────────────┐
       ▼                                               ▼
41 Dynamic Features (27 raw + 14 flags)     9 Static Demographics & Diagnoses
       │                                               │
Bi-LSTM + Attention Encoder                     2-Layer MLP Embedding
       │                                               │
  128-dim dynamic vector                         32-dim static vector
       └───────────────────────┬───────────────────────┘
                               ▼
                    Concatenation (160-dim)
                               │
                   fc_mu & fc_logvar Projections
                               │
                               ▼
               Latent Space: z in R^16 (Reparameterization)
                               │
                               ▼
                     Reconstruction Decoder
                               │
                               ▼
                 36 Continuous Clinical Targets
           (27 Pure Dynamic + 9 Static; flags excluded)
```

---

## Pipeline Components

### 1. Patient-Disjoint Partitioning & Normalization
* **Patient-Disjoint Splitting**: 70% Train (415 patients), 15% Validation (88 patients), 15% Test (90 patients).
  - Ensures zero patient overlap between splits to prevent data leakage and evaluate true clinical generalizability.
* **Strict Training-Set Scaling**:
  - Continuous clinical features normalized using z-score scalers fitted exclusively on the training cohort.
  - Normalization parameters exported to `scaler_parameters.json`.

### 2. Temporal Windowing
* Packages 30-day sliding windows with strict consecutive daily continuity checks.
* Preserves canonical feature ordering:
  `[pure_dynamic (27) | imputation_flags (14) | static (9)]`
* Tensors cached in `timeseries_tensors.pt`.

### 3. Two-Branch Multi-Modal VAE (`TwoBranchHealthVAE`)
* **Dynamic Branch**:
  - Bidirectional LSTM with hidden size 64 (128 bidirectional output).
  - Additive Attention pooling across the 30-day sequence length to capture transient glycemic events and volatility.
* **Static Branch**:
  - Multi-layer perceptron (Linear $\\to$ LayerNorm $\\to$ ReLU $\\to$ Dropout $\\to$ Linear) projecting 9 static features to a 32-dim embedding.
* **Latent Bottleneck**:
  - Concatenates dynamic (128) and static (32) representations into 160-dim vector.
  - Linear heads produce mean $\\boldsymbol{\\mu}$ and log-variance $\\log \\boldsymbol{\\sigma}^2$.
* **Reconstruction Decoder**:
  - Reconstructs the 36 continuous clinical variables (27 pure-dynamic across 30 days + 9 static).
  - Binary imputation indicator flags are excluded from MSE loss.

### 4. Training Dynamics & KL Annealing
* **Loss Function**:
  $$\\mathcal{L} = \\text{MSE}(\\hat{X}_{\\text{clinical}}, X_{\\text{clinical}}) + \\beta(e) \\cdot D_{\\text{KL}}(q(\\mathbf{z}|X) \\parallel p(\\mathbf{z}))$$
* **Linear KL Annealing**: $\\beta$ warm-up ramps from $0.0$ to $0.005$ across the first 15 epochs to prevent posterior collapse.
* **Early Stopping & Optimization**:
  - AdamW optimizer with weight decay.
  - Monitored on validation reconstruction loss (patience = 10, max epochs = 80).
  - Best checkpoint saved to `best_compact_vae.pt`.

---

## Execution

To execute Phase 2:
```bash
# Navigate to Phase 2 folder
cd Phase_2_Latent_VAE
python3 phase2_vae.py
```

### Outputs Produced
* `timeseries_tensors.pt`: Preprocessed PyTorch dataset splits (`train_dyn`, `train_stat`, etc.).
* `scaler_parameters.json`: Feature normalization means and standard deviations.
* `best_compact_vae.pt`: PyTorch model weights for `TwoBranchHealthVAE`.
* `latent_health_vectors.pt`: Inferred latent $\\boldsymbol{\\mu}$ and $\\mathbf{z}$ vectors for all windows.
* `patient_latent_health_summary.csv`: Patient-level mean latent health embeddings.
* `vae_training_history.json`: Epoch-by-epoch loss trajectories and convergence diagnostics.
