"""
================================================================================
PHASE 3 & PHASE 4: MULTI-MODAL TRAJECTORY FORECASTING & SIMULATION
================================================================================
Scope & Architectural Contract:
  â€¢ Consumes latent_health_vectors.pt produced by Phase 2 (CompactHealthVAE) as its
    primary input representation:
      - z_train, z_val, z_test (or mu_train, mu_val, mu_test) in R^16
      - Standardized metadata dictionaries: {"patient_id": ..., "start_date": ..., "end_date": ...}
  â€¢ Houses the autoregressive temporal trajectory forecaster migrated forward from Phase 2:
      - forecast_init: Maps [z, x_last] -> initial hidden state h0
      - forecast_gru: 2-layer autoregressive GRU cell
      - forecast_head: Projects hidden states to the 6 target clinical parameters
      - 60-day recursive rollout across future horizon
  â€¢ Serves as the official foundation and integration bridge for:
      - Phase 3: Conditional Tabular GAN (cGAN) for Synthetic Patient Trajectory Simulation
      - Phase 4: Sequential Probabilistic Sampling & Monte Carlo Rollouts (Multi-Trajectory Forecasting)
================================================================================
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Device Selection & Reproducibility
# -----------------------------------------------------------------------------
def get_device():
    """Select compute hardware with hierarchical priority: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_seed(seed=42):
    """Enforce reproducibility across all random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# -----------------------------------------------------------------------------
# 2. Path Resolution Utilities
# -----------------------------------------------------------------------------
def locate_file(filename, search_dirs=None):
    """Find model inputs relative to the project, independent of working directory."""
    phase_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(phase_dir)
    directories = [
        phase_dir,
        os.path.join(project_root, "Phase_2_Latent_VAE"),
        os.path.join(project_root, "Phase_1_Data_Preparation"),
        *(search_dirs or []),
        os.getcwd(),
    ]
    for directory in directories:
        candidate = os.path.join(directory, filename)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate project artifact: {filename}")


# -----------------------------------------------------------------------------
# 3. Standard Geometric Clarke Error Grid Analysis
# -----------------------------------------------------------------------------
def clarke_error_grid(ref, pred):
    """
    Standard Geometric Clarke Error Grid Algorithm (Clarke et al., Diabetes Care 1987).
    Calculates analytical zone distributions for reference vs. predicted glucose values (mg/dL).
    """
    assert len(ref) == len(pred), f"Length mismatch: {len(ref)} != {len(pred)}"
    total = len(ref)
    if total == 0:
        return {
            "zone_a_pct": 0.0, "zone_b_pct": 0.0, "zone_a_b_combined_pct": 0.0,
            "zone_c_pct": 0.0, "zone_d_pct": 0.0, "zone_e_pct": 0.0,
            "total_pairs_evaluated": 0
        }

    zone_a = 0
    zone_b = 0
    zone_c = 0
    zone_d = 0
    zone_e = 0

    for r, p in zip(ref, pred):
        # Zone A: Clinically accurate decisions (within 20% or both <= 70)
        if (r <= 70 and p <= 70) or (abs(p - r) <= 0.20 * r):
            zone_a += 1
            continue

        # Zone E: Confusion of hypo / hyper treatment
        if (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            zone_e += 1
            continue

        # Zone C: Overcorrection
        # Upper C: 70 <= r <= 290 and p > r + 110
        # Lower C: 130 <= r <= 180 and p < (7/5) * (r - 130)
        is_upper_c = (70 <= r <= 290) and (p > r + 110)
        is_lower_c = (130 <= r <= 180) and (p < (7.0 / 5.0) * (r - 130))
        if is_upper_c or is_lower_c:
            zone_c += 1
            continue

        # Zone D: Failure to detect dangerous hypo or hyper
        is_hypo_failure = (r <= 70) and (70 < p <= 180)
        is_hyper_failure = (r >= 240) and (70 <= p <= 180)
        if is_hypo_failure or is_hyper_failure:
            zone_d += 1
            continue

        # Zone B: Benign errors (everything else)
        zone_b += 1

    pct_a = (zone_a / total) * 100.0
    pct_b = (zone_b / total) * 100.0
    pct_c = (zone_c / total) * 100.0
    pct_d = (zone_d / total) * 100.0
    pct_e = (zone_e / total) * 100.0

    return {
        "zone_a_pct": round(pct_a, 2),
        "zone_b_pct": round(pct_b, 2),
        "zone_a_b_combined_pct": round(pct_a + pct_b, 2),
        "zone_c_pct": round(pct_c, 2),
        "zone_d_pct": round(pct_d, 2),
        "zone_e_pct": round(pct_e, 2),
        "total_pairs_evaluated": total
    }

# -----------------------------------------------------------------------------
# 4. Forecaster Input Pipeline: Wired to latent_health_vectors.pt
# -----------------------------------------------------------------------------
def prepare_forecasting_dataset(latent_vectors_path=None, cleaned_csv_path=None, out_len=60):
    """
    Ingests Phase 2 latent health vectors (mu/z in R^16) and pairs each window with:
      - x_last: Observed target features on the final day of the 30-day window (day 30)
      - Y: Future continuous 60-day target trajectories (days 31 to 90)
    Metadata format consumed: Standardized dict format {"patient_id": ..., "start_date": ..., "end_date": ...}.
    """
    start_time = time.time()
    print("=" * 75)
    print("PHASE 3/4: PREPARING FORECASTING INPUTS (WIRED TO Phase 2 LATENTS)")
    print("=" * 75)

    if latent_vectors_path is None:
        latent_vectors_path = locate_file("latent_health_vectors.pt")
    if cleaned_csv_path is None:
        cleaned_csv_path = locate_file("cleaned_daily_multimodal_dataset.csv")

    print(f"  â€¢ Loading Phase 2 Latent Representations: '{latent_vectors_path}'...")
    # weights_only=False is required: latent_health_vectors.pt serializes numpy types in metadata; safe local artifact
    latents = torch.load(latent_vectors_path, weights_only=False)

    print(f"  â€¢ Loading Phase 1 Cleaned Timeline:       '{cleaned_csv_path}'...")
    df = pd.read_csv(cleaned_csv_path)
    df["date_dt"] = pd.to_datetime(df["date"])

    target_cols = [
        "glucose_mean",      # Daily average glucose (mg/dL)
        "tir_70_180",        # Time in Range % (70-180 mg/dL)
        "tbr_below_70",      # Time Below Range % (<70 mg/dL)
        "tar_above_180",     # Time Above Range % (>180 mg/dL)
        "glucose_cv",        # Glycemic volatility %
        "gmi_estimated_a1c"  # Estimated HbA1c trajectory
    ]
    target_dim = len(target_cols)

    # Compute target normalization statistics strictly on training cohort patients
    train_pats = set(latents["train_pats"])
    train_df = df[df["Patient_ID"].isin(train_pats)]
    target_mean = train_df[target_cols].mean().values.astype(np.float32)
    target_std = train_df[target_cols].std().values.astype(np.float32)
    target_std[target_std < 1e-6] = 1.0

    print(f"  â€¢ Forecasting Target Dimensions: {target_dim} ({', '.join(target_cols)})")
    print(f"  â€¢ Target Normalization Fit on {len(train_pats)} Training Cohort Patients.")

    # Pre-index sorted patient timelines for O(1) temporal window alignment
    patient_timelines = {}
    for pid, group in df.groupby("Patient_ID"):
        group = group.sort_values("date_dt").reset_index(drop=True)
        dates = list(group["date"].values)
        date_map = {d: i for i, d in enumerate(dates)}
        norm_targets = (group[target_cols].values.astype(np.float32) - target_mean) / target_std
        date_dts = group["date_dt"].values
        patient_timelines[pid] = (date_map, norm_targets, date_dts)

    def extract_forecasting_windows(z_tensor, meta_list, split_name):
        z_out, x_last_out, y_out, meta_out = [], [], [], []

        for idx, item in enumerate(meta_list):
            # Consistently unpack standardized dict metadata format
            if isinstance(item, dict):
                pid = item["patient_id"]
                end_date = item["end_date"]
            else:
                pid, _, end_date = item[0], item[1], item[2]

            if pid not in patient_timelines:
                continue

            date_map, norm_targets, date_dts = patient_timelines[pid]
            if end_date not in date_map:
                continue

            end_idx = date_map[end_date]
            # Verify 60 future consecutive daily observations exist
            if end_idx + out_len < len(norm_targets):
                f_dates = date_dts[end_idx : end_idx + out_len + 1]
                diffs = (f_dates[1:] - f_dates[:-1]) / np.timedelta64(1, "D")
                if np.all(diffs == 1.0):
                    x_last = norm_targets[end_idx]
                    y_future = norm_targets[end_idx + 1 : end_idx + out_len + 1]

                    z_out.append(z_tensor[idx])
                    x_last_out.append(x_last)
                    y_out.append(y_future)
                    meta_out.append(item)

        if len(z_out) == 0:
            print(f"  â€¢ {split_name:10s}: 0 windows (no valid consecutive segments found)")
            target_dim_local = 6  # fallback; actual dim set by caller from dataset_bundle
            out_len_local = out_len
            return (
                torch.empty(0, 16),
                torch.empty(0, target_dim_local),
                torch.empty(0, out_len_local, target_dim_local),
                []
            )

        z_arr = torch.stack(z_out)
        x_last_arr = torch.from_numpy(np.array(x_last_out, dtype=np.float32))
        y_arr = torch.from_numpy(np.array(y_out, dtype=np.float32))

        print(f"  â€¢ {split_name:10s}: {len(y_arr):5d} windows | z: {z_arr.shape} | x_last: {x_last_arr.shape} | Y: {y_arr.shape}")
        return z_arr, x_last_arr, y_arr, meta_out

    z_tr, x_last_tr, y_tr, meta_tr = extract_forecasting_windows(latents["z_train"], latents["meta_train"], "Train Set")
    z_va, x_last_va, y_va, meta_va = extract_forecasting_windows(latents["z_val"], latents["meta_val"], "Val Set")
    z_te, x_last_te, y_te, meta_te = extract_forecasting_windows(latents["z_test"], latents["meta_test"], "Test Set")

    elapsed = time.time() - start_time
    print(f"  >> Extracted {len(y_tr) + len(y_va) + len(y_te):,} total forecasting windows in {elapsed:.2f}s.")

    dataset_bundle = {
        "train": (z_tr, x_last_tr, y_tr, meta_tr),
        "val": (z_va, x_last_va, y_va, meta_va),
        "test": (z_te, x_last_te, y_te, meta_te),
        "target_cols": target_cols,
        "target_mean": torch.from_numpy(target_mean),
        "target_std": torch.from_numpy(target_std),
        "latent_dim": latents.get("latent_dim", 16),
        "target_dim": target_dim,
        "out_len": out_len
    }
    return dataset_bundle

# -----------------------------------------------------------------------------
# 5. Multi-Modal Trajectory Forecaster Architecture
# -----------------------------------------------------------------------------
class MultiModalTrajectoryForecaster(nn.Module):
    """
    Autoregressive Multi-Step Trajectory Forecaster.
    Conditions on a compact latent health vector z in R^16 (from Phase 2) and initial condition x_last,
    performing recursive GRU rollout across future horizon (out_len = 60 days).
    """
    def __init__(self, target_dim=6, latent_dim=16, hidden_dim=64, out_len=60):
        super().__init__()
        self.target_dim = target_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.out_len = out_len

        # Forecast Initializer: maps [z, x_last] -> initial hidden state h0
        self.forecast_init = nn.Sequential(
            nn.Linear(latent_dim + target_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Autoregressive GRU cell
        self.forecast_gru = nn.GRU(
            input_size=target_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15
        )

        # Projection head to target clinical metrics
        self.forecast_head = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, x_last):
        """
        z: latent health vector (batch, latent_dim)
        x_last: initial observed state for targets (batch, target_dim)
        Returns: y_forecast (batch, out_len, target_dim)
        """
        batch_size = z.size(0)
        init_feat = torch.cat([z, x_last], dim=-1)
        h0 = self.forecast_init(init_feat).unsqueeze(0).repeat(2, 1, 1)  # 2 GRU layers

        curr_step = x_last.unsqueeze(1)  # (batch, 1, target_dim)
        predictions = []

        h = h0
        for _ in range(self.out_len):
            out, h = self.forecast_gru(curr_step, h)
            next_pred = self.forecast_head(out)
            predictions.append(next_pred)
            curr_step = next_pred

        y_forecast = torch.cat(predictions, dim=1)  # (batch, out_len, target_dim)
        return y_forecast

# -----------------------------------------------------------------------------
# 6. Training Routine for Trajectory Forecaster
# -----------------------------------------------------------------------------
def train_trajectory_forecaster(dataset_bundle, num_epochs=20, hidden_dim=64, checkpoint_path="best_trajectory_forecaster.pt"):
    """Trains the autoregressive GRU forecaster conditioned on latent health representations."""
    print("\n" + "=" * 75)
    print("PHASE 3/4: TRAINING AUTOREGRESSIVE TRAJECTORY FORECASTER")
    print("=" * 75)

    set_seed(42)
    device = get_device()
    print(f"  â€¢ Utilizing Device: {device} (CUDA -> MPS -> CPU hierarchy)")

    z_tr, x_last_tr, y_tr, _ = dataset_bundle["train"]
    z_va, x_last_va, y_va, _ = dataset_bundle["val"]

    train_loader = DataLoader(TensorDataset(z_tr, x_last_tr, y_tr), batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(z_va, x_last_va, y_va), batch_size=64, shuffle=False)

    target_dim = dataset_bundle["target_dim"]
    latent_dim = dataset_bundle["latent_dim"]
    out_len = dataset_bundle["out_len"]

    model = MultiModalTrajectoryForecaster(
        target_dim=target_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        out_len=out_len
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0.0

        for z_b, x_last_b, y_b in train_loader:
            z_b, x_last_b, y_b = z_b.to(device), x_last_b.to(device), y_b.to(device)
            optimizer.zero_grad()

            y_pred = model(z_b, x_last_b)
            loss = F.mse_loss(y_pred, y_b)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for z_v, x_last_v, y_v in val_loader:
                z_v, x_last_v, y_v = z_v.to(device), x_last_v.to(device), y_v.to(device)
                y_pred_v = model(z_v, x_last_v)
                loss_v = F.mse_loss(y_pred_v, y_v)
                total_val_loss += loss_v.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        scheduler.step(avg_val)

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "latent_dim": latent_dim,
                "target_dim": target_dim,
                "hidden_dim": hidden_dim,
                "out_len": out_len,
                "target_cols": dataset_bundle["target_cols"],
                "target_mean": dataset_bundle["target_mean"],
                "target_std": dataset_bundle["target_std"]
            }, checkpoint_path)
            flag = "(* BEST CHECKPOINT SAVED *)"
        else:
            flag = ""

        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f} {flag}")

    elapsed = time.time() - start_time
    print(f"\n  >> Training Complete in {elapsed:.2f}s | Best Val MSE: {best_val_loss:.4f}")
    return model, checkpoint_path

# -----------------------------------------------------------------------------
# 7. Comprehensive Clinical Trajectory Accuracy Evaluation
# -----------------------------------------------------------------------------
def evaluate_trajectory_forecaster(
    forecaster_model,
    dataset_bundle,
    sample_indices=None,
    output_image="trajectory_simulation_evaluation.png",
    output_json="evaluation_results.json"
):
    """
    Evaluates 60-day trajectory forecasts on unseen test patients:
      - Computes MAE, RMSE, Pearson r across all 6 targets.
      - Clarke Error Grid zone analysis.
      - Publication visualization.
    """
    print("\n" + "=" * 75)
    print("PHASE 3/4: COMPREHENSIVE CLINICAL ACCURACY EVALUATION (TEST COHORT)")
    print("=" * 75)

    device = get_device()
    forecaster_model.to(device)
    forecaster_model.eval()

    z_te, x_last_te, y_te, meta_te = dataset_bundle["test"]
    target_cols = dataset_bundle["target_cols"]
    target_mean = dataset_bundle["target_mean"].numpy()
    target_std = dataset_bundle["target_std"].numpy()

    test_loader = DataLoader(TensorDataset(z_te, x_last_te), batch_size=128, shuffle=False)
    preds = []
    with torch.no_grad():
        for z_b, x_last_b in test_loader:
            z_b, x_last_b = z_b.to(device), x_last_b.to(device)
            y_pred = forecaster_model(z_b, x_last_b)
            preds.append(y_pred.cpu())

    preds = torch.cat(preds, dim=0).numpy()
    ground_truth = y_te.numpy()

    # Denormalize with target-specific physiological bounds
    preds_denorm = (preds * target_std) + target_mean
    ground_truth_denorm = (ground_truth * target_std) + target_mean

    for t_idx, col_name in enumerate(target_cols):
        if col_name == "glucose_mean":
            preds_denorm[:, :, t_idx] = np.clip(preds_denorm[:, :, t_idx], 40.0, 500.0)
            ground_truth_denorm[:, :, t_idx] = np.clip(ground_truth_denorm[:, :, t_idx], 40.0, 500.0)
        elif col_name == "gmi_estimated_a1c":
            preds_denorm[:, :, t_idx] = np.clip(preds_denorm[:, :, t_idx], 3.5, 20.0)
            ground_truth_denorm[:, :, t_idx] = np.clip(ground_truth_denorm[:, :, t_idx], 3.5, 20.0)
        else:
            preds_denorm[:, :, t_idx] = np.clip(preds_denorm[:, :, t_idx], 0.0, 100.0)
            ground_truth_denorm[:, :, t_idx] = np.clip(ground_truth_denorm[:, :, t_idx], 0.0, 100.0)

    # Compute Metrics
    results = {}
    print("\nClinical Trajectory Forecast Metrics across 60 Days:")
    for t_idx, col_name in enumerate(target_cols):
        p_flat = preds_denorm[:, :, t_idx].flatten()
        g_flat = ground_truth_denorm[:, :, t_idx].flatten()

        mae = float(np.mean(np.abs(p_flat - g_flat)))
        rmse = float(np.sqrt(np.mean((p_flat - g_flat) ** 2)))
        r = float(np.corrcoef(p_flat, g_flat)[0, 1])
        unit = "mg/dL" if col_name == "glucose_mean" else "%"

        results[col_name] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "pearson_r": round(r, 3),
            "unit": unit
        }
        print(f"  â€¢ {col_name:18s}: MAE = {mae:6.2f} {unit:5s} | RMSE = {rmse:6.2f} {unit:5s} | Pearson r = {r:.3f}")

    # Clarke Error Grid on glucose_mean
    g_idx = target_cols.index("glucose_mean")
    clarke_results = clarke_error_grid(
        ground_truth_denorm[:, :, g_idx].flatten(),
        preds_denorm[:, :, g_idx].flatten()
    )
    results["clarke_error_grid"] = clarke_results

    print("\nStandard Geometric Clarke Error Grid Zone Breakdown:")
    print(f"  â€¢ Zone A (Clinically Accurate)         : {clarke_results['zone_a_pct']}%")
    print(f"  â€¢ Zone B (Benign Errors)               : {clarke_results['zone_b_pct']}%")
    print(f"  â€¢ Zone A + B Combined (Clinical Pass)  : {clarke_results['zone_a_b_combined_pct']}% (Target: > 95%)")
    print(f"  â€¢ Zone C (Overcorrection)              : {clarke_results['zone_c_pct']}%")
    print(f"  â€¢ Zone D (Failure to Detect Danger)    : {clarke_results['zone_d_pct']}%")
    print(f"  â€¢ Zone E (Erroneous Treatment)         : {clarke_results['zone_e_pct']}%")

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"  >> Saved evaluation metrics to '{output_json}'")

    # Plot trajectories
    n_test_samples = len(z_te)
    if sample_indices is None:
        sample_indices = [min(15, n_test_samples - 1)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    days = np.arange(1, 61)
    sample_idx = sample_indices[0]

    for t_idx, col_name in enumerate(target_cols):
        ax = axes[t_idx]
        ax.plot(days, ground_truth_denorm[sample_idx, :, t_idx], label="Ground Truth", color="black", lw=2)
        ax.plot(days, preds_denorm[sample_idx, :, t_idx], label="Forecast", color="crimson", ls="--", lw=2)
        unit = "mg/dL" if col_name == "glucose_mean" else "%"
        ax.set_title(f"{col_name} (MAE: {results[col_name]['mae']} {unit})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Horizon (Days)", fontsize=10)
        ax.set_ylabel(unit, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"  >> Saved trajectory forecast plots to '{output_image}'")
    return results

# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    bundle = prepare_forecasting_dataset()
    model, ckpt = train_trajectory_forecaster(bundle, num_epochs=15)
    evaluate_trajectory_forecaster(model, bundle)
