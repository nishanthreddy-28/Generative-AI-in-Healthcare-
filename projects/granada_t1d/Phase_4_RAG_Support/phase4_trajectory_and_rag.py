"""
================================================================================
PHASE 4: SEQUENTIAL TRAJECTORY SIMULATION & RAG DECISION SUPPORT
================================================================================
BTP Architectural Role:
  • Generates 60-day sequential trajectory rollouts using MultiModalTrajectoryForecaster.
  • Conducts Monte Carlo probabilistic rollouts with 10th, 50th, and 90th percentile confidence bands.
  • Connects trajectory prognosis directly to FAISS RAG over ADA Guidelines.
  • Outputs doctor consultation notes and patient guidance plans.
================================================================================
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    curr_dir = os.getcwd()
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

phase3_dir = os.path.abspath(os.path.join(curr_dir, "..", "Phase_3_Generative_Forecasting"))
if os.path.exists(phase3_dir) and phase3_dir not in sys.path:
    sys.path.insert(0, phase3_dir)

from rag_clinical_engine import ADAGuidelinesRAG, ClinicalDecisionSynthesizer, locate_file
from phase3_4_trajectory_forecasting import MultiModalTrajectoryForecaster, prepare_forecasting_dataset

def run_probabilistic_trajectory_simulation(
    z_vector,
    x_last,
    forecaster_ckpt_path=None,
    n_monte_carlo=100,
    noise_scale=0.03
):
    """
    Performs Monte Carlo probabilistic sequential trajectory rollout across 60 days.
    Returns:
      - p10, p50 (median), p90 trajectory curves (60, n_targets)
      - target_cols, target_mean, target_std
    """
    if forecaster_ckpt_path is None:
        forecaster_ckpt_path = locate_file("best_trajectory_forecaster.pt")

    ckpt = torch.load(forecaster_ckpt_path, map_location="cpu", weights_only=False)
    forecaster = MultiModalTrajectoryForecaster(
        target_dim=ckpt["target_dim"],
        latent_dim=ckpt["latent_dim"],
        hidden_dim=ckpt["hidden_dim"],
        out_len=ckpt["out_len"]
    )
    forecaster.load_state_dict(ckpt["model_state_dict"])
    forecaster.eval()

    target_mean = ckpt["target_mean"].numpy()
    target_std = ckpt["target_std"].numpy()
    target_cols = ckpt["target_cols"]

    if isinstance(z_vector, np.ndarray):
        z_t = torch.from_numpy(z_vector).float()
    else:
        z_t = z_vector.float()
    if z_t.dim() == 1:
        z_t = z_t.unsqueeze(0)

    if isinstance(x_last, np.ndarray):
        x_t = torch.from_numpy(x_last).float()
    else:
        x_t = x_last.float()
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)

    # Monte Carlo simulation with perturbation in latent health space
    simulations = []
    with torch.no_grad():
        for _ in range(n_monte_carlo):
            z_noisy = z_t + (torch.randn_like(z_t) * noise_scale)
            y_pred_norm = forecaster(z_noisy, x_t).numpy()[0]
            y_pred_denorm = (y_pred_norm * target_std) + target_mean

            # Clinical clipping
            y_pred_denorm[:, 0] = np.clip(y_pred_denorm[:, 0], 40.0, 500.0)
            for k in range(1, 5):
                y_pred_denorm[:, k] = np.clip(y_pred_denorm[:, k], 0.0, 100.0)
            y_pred_denorm[:, 5] = np.clip(y_pred_denorm[:, 5], 3.5, 20.0)

            simulations.append(y_pred_denorm)

    sims_arr = np.array(simulations)  # (n_monte_carlo, 60, target_dim)

    p10 = np.percentile(sims_arr, 10, axis=0)
    p50 = np.percentile(sims_arr, 50, axis=0)
    p90 = np.percentile(sims_arr, 90, axis=0)

    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "target_cols": target_cols,
        "simulations": sims_arr
    }

def run_phase4_end_to_end():
    print("=" * 75)
    print("PHASE 4: FULL PROBABILISTIC TRAJECTORY SIMULATION & RAG PIPELINE")
    print("=" * 75)

    rag = ADAGuidelinesRAG()
    synthesizer = ClinicalDecisionSynthesizer(rag)

    latent_path = locate_file("latent_health_vectors.pt")
    latents = torch.load(latent_path, weights_only=False)

    meta_test = latents["meta_test"]
    sample_idx = min(20, len(meta_test) - 1)
    sample_meta = meta_test[sample_idx]
    pid = sample_meta["patient_id"] if isinstance(sample_meta, dict) else sample_meta[0]

    print(f"  • Simulating Future Trajectory for Test Patient: {pid}")

    meta_path = locate_file("cleaned_cohort_metadata.csv")
    meta_df = pd.read_csv(meta_path)
    pat_rows = meta_df[meta_df["Patient_ID"] == pid]
    pat_profile = pat_rows.iloc[0].to_dict() if len(pat_rows) > 0 else {"Patient_ID": pid, "age": 45, "sex_male": 1}

    z_sample = latents["z_test"][sample_idx]
    x_last_sample = np.zeros(6, dtype=np.float32)

    sim_res = run_probabilistic_trajectory_simulation(z_sample, x_last_sample, n_monte_carlo=50)

    p50_traj = sim_res["p50"]
    doc_report, citations, engine_mode = synthesizer.generate_doctor_report(pat_profile, p50_traj, use_llm=False)
    pat_report, _, _ = synthesizer.generate_patient_guidance_report(pat_profile, p50_traj, use_llm=False)

    print(f"\n[Generated Physician Note]:\n{doc_report[:600]}...\n")
    print(f"[Generated Patient Guide]:\n{pat_report[:600]}...\n")

    # Plot 6-panel probabilistic trajectory fan chart
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    days = np.arange(1, 61)

    for i, col in enumerate(sim_res["target_cols"]):
        ax = axes[i]
        ax.plot(days, sim_res["p50"][:, i], color="#1f77b4", lw=2, label="Median Forecast (P50)")
        ax.fill_between(days, sim_res["p10"][:, i], sim_res["p90"][:, i], color="#1f77b4", alpha=0.25, label="10th-90th Percentile")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon (Days)", fontsize=9)
        unit = "mg/dL" if "glucose" in col else "%"
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    plt.suptitle(f"Phase 4: Probabilistic Trajectory Rollout (Patient {pid})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plot_file = os.path.join(curr_dir, "probabilistic_trajectory_simulation.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"  >> Saved probabilistic trajectory fan chart to '{plot_file}'")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_phase4_end_to_end()
