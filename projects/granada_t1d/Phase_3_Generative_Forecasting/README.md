# Phase 3: Comparative Benchmarking & Conditional Temporal GAN (cGAN)

> Current location: `Phase_3_Generative_Forecasting`. From the BTP root, run `python scripts/run_phase_3.py`. These commands regenerate original research outputs. See [project structure](../docs/PROJECT_STRUCTURE.md) and [the audit](../docs/reports/PROJECT_AUDIT_REPORT.md).

## Overview
Phase 3 trains comparative models (**Baseline Deterministic LSTM** vs **Multi-Modal Latent VAE** vs **Conditional Temporal GAN**) and generates **Monte Carlo probabilistic trajectory rollouts** (10th–90th percentile simulation cones) to capture metabolic risk boundaries.

---

## Scripts & Usage

### 1. `train_comparative_models.py`
* **Purpose**:
  - Trains Baseline LSTM (10 epochs)
  - Trains Conditional Temporal GAN Generator $G$ & Discriminator $D$ (10 epochs)
  - Evaluates all 3 models on 3,006 held-out test windows
  - Generates 50 Monte Carlo rollouts per patient to plot probabilistic uncertainty cones
* **Execution**: `python3 train_comparative_models.py`
* **Outputs**:
  - `best_baseline_lstm.pt`
  - `best_temporal_cgan_generator.pt`
  - `model_benchmarking_results.json`
  - `cgan_probabilistic_trajectory_simulation.png`

---

## IEEE Comparative Benchmarking Results

| Model Architecture | Glucose MAE | TIR MAE | TBR MAE | HbA1c MAE | Clarke Zone A+B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline Deterministic LSTM** | 23.58 mg/dL | 14.63% | 4.93% | 0.56% | 99.31% |
| **Multi-Modal Latent VAE** | 24.80 mg/dL | 15.05% | 4.85% | 0.59% | 99.07% |
| **Conditional Temporal GAN (cGAN)** | 24.85 mg/dL | 14.89% | 5.19% | 0.59% | 99.13% |

* **Probabilistic Trajectory Plot**: `cgan_probabilistic_trajectory_simulation.png` shows the 10th, 50th, and 90th percentile trajectory cones tracking true patient glucose.
