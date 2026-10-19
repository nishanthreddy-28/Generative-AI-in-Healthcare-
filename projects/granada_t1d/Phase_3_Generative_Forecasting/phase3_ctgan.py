"""
================================================================================
PHASE 3: CONDITIONAL TABULAR GAN (CTGAN) FOR 30-DAY CLINICAL STATE SIMULATION
================================================================================
BTP Architectural Role:
  â€¢ Consumes Phase 2 Latent Health Vectors (z in R^16) and Static Profiles (R^9).
  â€¢ Simulates the NEXT 30-DAY TABULAR CLINICAL HEALTH STATE:
      1. glucose_mean: 30-day mean glucose (mg/dL)
      2. glucose_std: 30-day glycemic standard deviation (mg/dL)
      3. glucose_cv: 30-day glycemic coefficient of variation (%)
      4. tir_70_180: 30-day Time in Range (%)
      5. tbr_below_70: 30-day Time Below Range / Hypoglycemia (%)
      6. tbr_severe_54: 30-day Severe Hypoglycemia (%)
      7. tar_above_180: 30-day Time Above Range / Hyperglycemia (%)
      8. tar_severe_250: 30-day Severe Hyperglycemia (%)
      9. nocturnal_mean: 30-day nocturnal glucose (00:00 - 06:00, mg/dL)
     10. nocturnal_hypo_count: 30-day nocturnal hypoglycemic events
     11. gmi_estimated_a1c: 30-day Glucose Management Indicator / HbA1c (%)
     12. lab_hba1c: projected laboratory HbA1c (%)
  â€¢ CTGAN Architecture:
      - Generator: Residual MLP mapping [Condition (z + static), Noise] -> Next 30-Day State
      - Critic / Discriminator: WGAN-GP conditioning on [Tabular State, Condition]
  â€¢ Produces:
      - best_ctgan_generator.pt
      - ctgan_evaluation_report.json
      - ctgan_tabular_fidelity_comparison.png
================================================================================
"""

import os
import sys
import json
import time
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
# 1. Device Hierarchy & Reproducibility
# -----------------------------------------------------------------------------
def get_device():
    """CUDA -> MPS -> CPU hierarchy."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

# -----------------------------------------------------------------------------
# 2. Path Resolution Utility
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
# 3. Tabular Dataset Extraction: Next 30-Day Clinical Target State
# -----------------------------------------------------------------------------
TARGET_TABULAR_COLS = [
    "glucose_mean",
    "glucose_std",
    "glucose_cv",
    "tir_70_180",
    "tbr_below_70",
    "tbr_severe_54",
    "tar_above_180",
    "tar_severe_250",
    "nocturnal_mean",
    "nocturnal_hypo_count",
    "gmi_estimated_a1c",
    "lab_hba1c"
]

STATIC_COLS = [
    "age",
    "sex_male",
    "hypothyroidism",
    "retinopathy",
    "hypertension",
    "dyslipidemia",
    "nephropathy",
    "neuropathy",
    "dka_history"
]

def prepare_ctgan_dataset(latent_vectors_path=None, cleaned_csv_path=None, horizon_days=30):
    """
    Extracts paired data:
      Condition c = [z (16-D), static (9-D)] in R^25
      Target Y = Next 30-day aggregate tabular features in R^12
    """
    start_time = time.time()
    print("=" * 75)
    print("PHASE 3: PREPARING NEXT 30-DAY TABULAR DATASET FOR CTGAN")
    print("=" * 75)

    if latent_vectors_path is None:
        latent_vectors_path = locate_file("latent_health_vectors.pt")
    if cleaned_csv_path is None:
        cleaned_csv_path = locate_file("cleaned_daily_multimodal_dataset.csv")

    print(f"  â€¢ Ingesting Phase 2 Latent Health Vectors: '{latent_vectors_path}'...")
    latents = torch.load(latent_vectors_path, weights_only=False)

    print(f"  â€¢ Ingesting Phase 1 Cleaned Multimodal CSV: '{cleaned_csv_path}'...")
    df = pd.read_csv(cleaned_csv_path)
    df["date_dt"] = pd.to_datetime(df["date"])

    train_pats = set(latents["train_pats"])
    train_df = df[df["Patient_ID"].isin(train_pats)]

    # Compute static normalization
    static_mean = train_df[STATIC_COLS].mean().values.astype(np.float32)
    static_std = train_df[STATIC_COLS].std().values.astype(np.float32)
    static_std[static_std < 1e-6] = 1.0

    # Compute target normalization
    target_mean = train_df[TARGET_TABULAR_COLS].mean().values.astype(np.float32)
    target_std = train_df[TARGET_TABULAR_COLS].std().values.astype(np.float32)
    target_std[target_std < 1e-6] = 1.0

    # Fast indexed cache of patient timelines
    print("  â€¢ Indexing patient longitudinal records...")
    patient_records = {}
    for pid, grp in df.groupby("Patient_ID"):
        grp = grp.sort_values("date_dt").reset_index(drop=True)
        dmap = {d: i for i, d in enumerate(grp["date"].values)}
        raw_targets = grp[TARGET_TABULAR_COLS].values.astype(np.float32)
        norm_stat = (grp[STATIC_COLS].iloc[0].values.astype(np.float32) - static_mean) / static_std
        dates = grp["date_dt"].values
        patient_records[pid] = (dmap, raw_targets, norm_stat, dates)

    def extract_split_tabular_windows(z_tensor, meta_list, split_name):
        c_list, y_list, meta_list_out = [], [], []

        for idx, item in enumerate(meta_list):
            if isinstance(item, dict):
                pid = item["patient_id"]
                end_date = item["end_date"]
            else:
                pid, _, end_date = item[0], item[1], item[2]

            if pid not in patient_records:
                continue

            dmap, raw_targets, norm_stat, dates = patient_records[pid]
            if end_date not in dmap:
                continue

            end_idx = dmap[end_date]
            # Check 30 consecutive calendar days ahead
            if end_idx + horizon_days < len(raw_targets):
                f_dates = dates[end_idx : end_idx + horizon_days + 1]
                diffs = (f_dates[1:] - f_dates[:-1]) / np.timedelta64(1, "D")
                if np.all(diffs == 1.0):
                    # Future 30-day tabular state: average daily values
                    f_window = raw_targets[end_idx + 1 : end_idx + horizon_days + 1]
                    f_summary = np.mean(f_window, axis=0)

                    # For count columns (nocturnal_hypo_count), use the 30-day sum
                    hypo_idx = TARGET_TABULAR_COLS.index("nocturnal_hypo_count")
                    f_summary[hypo_idx] = np.sum(f_window[:, hypo_idx])

                    z_vec = z_tensor[idx].numpy()
                    cond_vec = np.concatenate([z_vec, norm_stat])

                    c_list.append(cond_vec)
                    y_list.append(f_summary)
                    meta_list_out.append(item)

        c_arr = np.array(c_list, dtype=np.float32)
        y_arr = np.array(y_list, dtype=np.float32)

        # Normalize Y
        y_norm = (y_arr - target_mean) / target_std

        c_tensor = torch.from_numpy(c_arr)
        y_tensor = torch.from_numpy(y_norm)

        print(f"  â€¢ {split_name:10s}: {len(y_tensor):5d} windows | Condition c: {c_tensor.shape} | Next-30d Target: {y_tensor.shape}")
        return c_tensor, y_tensor, y_arr, meta_list_out

    c_tr, y_tr, y_tr_raw, meta_tr = extract_split_tabular_windows(latents["z_train"], latents["meta_train"], "Train Set")
    c_va, y_va, y_va_raw, meta_va = extract_split_tabular_windows(latents["z_val"], latents["meta_val"], "Val Set")
    c_te, y_te, y_te_raw, meta_te = extract_split_tabular_windows(latents["z_test"], latents["meta_test"], "Test Set")

    elapsed = time.time() - start_time
    print(f"  >> Tabular Extraction Completed in {elapsed:.2f}s ({len(c_tr) + len(c_va) + len(c_te):,} total windows).")

    return {
        "train": (c_tr, y_tr, y_tr_raw, meta_tr),
        "val": (c_va, y_va, y_va_raw, meta_va),
        "test": (c_te, y_te, y_te_raw, meta_te),
        "cond_dim": c_tr.shape[1],
        "target_dim": y_tr.shape[1],
        "latent_dim": latents.get("latent_dim", 16),
        "static_dim": len(STATIC_COLS),
        "target_cols": TARGET_TABULAR_COLS,
        "static_cols": STATIC_COLS,
        "target_mean": torch.from_numpy(target_mean),
        "target_std": torch.from_numpy(target_std),
        "static_mean": torch.from_numpy(static_mean),
        "static_std": torch.from_numpy(static_std)
    }

# -----------------------------------------------------------------------------
# 4. CTGAN Generator & Critic (Discriminator) Architecture
# -----------------------------------------------------------------------------
class ResidualDenseBlock(nn.Module):
    """Residual dense projection block with layer normalization and dropout."""
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(x + self.block(x))

class CTGANTabularGenerator(nn.Module):
    """
    Conditional Tabular GAN Generator G(c, epsilon).
    Maps [Condition c (z + static), Noise epsilon] -> Synthetic Next 30-Day Tabular State.
    """
    def __init__(self, cond_dim=25, noise_dim=16, target_dim=12, hidden_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.target_dim = target_dim

        self.input_layer = nn.Sequential(
            nn.Linear(cond_dim + noise_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.res1 = ResidualDenseBlock(hidden_dim)
        self.res2 = ResidualDenseBlock(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, target_dim)
        )

    def forward(self, cond, noise):
        # cond: (batch, cond_dim), noise: (batch, noise_dim)
        x = torch.cat([cond, noise], dim=-1)
        h = self.input_layer(x)
        h = self.res1(h)
        h = self.res2(h)
        out = self.head(h)
        return out

class CTGANTabularDiscriminator(nn.Module):
    """
    Conditional Tabular Critic D(Y, c).
    Evaluates realistic correlation and marginal consistency of tabular health state Y conditioned on c.
    """
    def __init__(self, target_dim=12, cond_dim=25, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(target_dim + cond_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, y, cond):
        x = torch.cat([y, cond], dim=-1)
        return self.net(x)

# -----------------------------------------------------------------------------
# 5. Gradient Penalty for WGAN-GP
# -----------------------------------------------------------------------------
def compute_gradient_penalty(discriminator, real_samples, fake_samples, cond, device):
    """Computes WGAN-GP gradient penalty to enforce 1-Lipschitz continuity."""
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, cond)

    fake_grad_outputs = torch.ones(d_interpolates.size(), device=device, requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
    return gradient_penalty

# -----------------------------------------------------------------------------
# 6. CTGAN Training Routine
# -----------------------------------------------------------------------------
def train_ctgan(dataset_bundle, num_epochs=20, batch_size=64, noise_dim=16, checkpoint_path="best_ctgan_generator.pt"):
    print("\n" + "=" * 75)
    print("PHASE 3: TRAINING CONDITIONAL TABULAR GAN (CTGAN / WGAN-GP)")
    print("=" * 75)

    set_seed(42)
    device = get_device()
    print(f"  â€¢ Utilizing Device: {device}")

    c_tr, y_tr, _, _ = dataset_bundle["train"]
    c_va, y_va, _, _ = dataset_bundle["val"]

    train_loader = DataLoader(TensorDataset(c_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(c_va, y_va), batch_size=batch_size, shuffle=False)

    cond_dim = dataset_bundle["cond_dim"]
    target_dim = dataset_bundle["target_dim"]

    generator = CTGANTabularGenerator(cond_dim=cond_dim, noise_dim=noise_dim, target_dim=target_dim).to(device)
    discriminator = CTGANTabularDiscriminator(target_dim=target_dim, cond_dim=cond_dim).to(device)

    opt_g = torch.optim.AdamW(generator.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    best_val_mae = float("inf")
    start_time = time.time()

    lambda_gp = 10.0
    lambda_rec = 5.0
    n_critic = 3

    for epoch in range(1, num_epochs + 1):
        generator.train()
        discriminator.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        n_batches = 0

        for c_batch, y_batch in train_loader:
            c_b = c_batch.to(device)
            y_b = y_batch.to(device)
            bs = c_b.size(0)

            # ---------------------------------------------------------
            # Train Discriminator / Critic (WGAN-GP)
            # ---------------------------------------------------------
            opt_d.zero_grad()
            noise = torch.randn(bs, noise_dim, device=device)
            y_fake = generator(c_b, noise).detach()

            d_real = discriminator(y_b, c_b)
            d_fake = discriminator(y_fake, c_b)

            gp = compute_gradient_penalty(discriminator, y_b, y_fake, c_b, device)
            d_loss = torch.mean(d_fake) - torch.mean(d_real) + (lambda_gp * gp)

            d_loss.backward()
            opt_d.step()
            total_d_loss += d_loss.item()

            # ---------------------------------------------------------
            # Train Generator (every n_critic iterations)
            # ---------------------------------------------------------
            if n_batches % n_critic == 0:
                opt_g.zero_grad()
                noise_g = torch.randn(bs, noise_dim, device=device)
                y_fake_g = generator(c_b, noise_g)
                d_fake_g = discriminator(y_fake_g, c_b)

                # Adversarial loss + Feature fidelity regularizer
                g_adv_loss = -torch.mean(d_fake_g)
                g_rec_loss = F.l1_loss(y_fake_g, y_b)
                g_loss = g_adv_loss + (lambda_rec * g_rec_loss)

                g_loss.backward()
                opt_g.step()
                total_g_loss += g_loss.item()

            n_batches += 1

        # Validation Evaluation
        generator.eval()
        val_l1_errors = []
        with torch.no_grad():
            for c_v, y_v in val_loader:
                c_v, y_v = c_v.to(device), y_v.to(device)
                bs_v = c_v.size(0)
                noise_v = torch.zeros(bs_v, noise_dim, device=device)  # deterministic zero noise for evaluation
                y_pred_v = generator(c_v, noise_v)
                val_l1_errors.append(torch.mean(torch.abs(y_pred_v - y_v)).item())

        avg_val_mae = float(np.mean(val_l1_errors))
        is_best = avg_val_mae < best_val_mae
        flag = ""
        if is_best:
            best_val_mae = avg_val_mae
            torch.save({
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "best_val_mae": best_val_mae,
                "cond_dim": cond_dim,
                "noise_dim": noise_dim,
                "target_dim": target_dim,
                "target_cols": dataset_bundle["target_cols"],
                "static_cols": dataset_bundle["static_cols"],
                "target_mean": dataset_bundle["target_mean"],
                "target_std": dataset_bundle["target_std"],
                "static_mean": dataset_bundle["static_mean"],
                "static_std": dataset_bundle["static_std"]
            }, checkpoint_path)
            flag = "(* BEST CHECKPOINT SAVED *)"

        avg_d = total_d_loss / n_batches
        avg_g = total_g_loss / (n_batches // n_critic + 1)
        print(f"Epoch {epoch:02d}/{num_epochs:02d} | Critic D Loss: {avg_d:6.3f} | Gen G Loss: {avg_g:6.3f} | Val Norm MAE: {avg_val_mae:6.3f} {flag}")

    elapsed = time.time() - start_time
    print(f"\n  >> CTGAN Training Complete in {elapsed:.2f}s | Best Val Norm MAE: {best_val_mae:.4f}")
    return generator, checkpoint_path

# -----------------------------------------------------------------------------
# 7. Comprehensive Tabular Fidelity Evaluation
# -----------------------------------------------------------------------------
def evaluate_ctgan_fidelity(
    generator,
    dataset_bundle,
    output_image="ctgan_tabular_fidelity_comparison.png",
    output_json="ctgan_evaluation_report.json"
):
    print("\n" + "=" * 75)
    print("PHASE 3: COMPREHENSIVE TABULAR FIDELITY EVALUATION (HELD-OUT TEST SET)")
    print("=" * 75)

    device = get_device()
    generator.to(device)
    generator.eval()

    c_te, y_te, y_te_raw, meta_te = dataset_bundle["test"]
    target_cols = dataset_bundle["target_cols"]
    target_mean = dataset_bundle["target_mean"].numpy()
    target_std = dataset_bundle["target_std"].numpy()

    # Generate synthetic tabular samples for each test condition
    n_test = len(c_te)
    test_loader = DataLoader(TensorDataset(c_te), batch_size=128, shuffle=False)

    preds_norm = []
    with torch.no_grad():
        for (c_b,) in test_loader:
            c_b = c_b.to(device)
            bs = c_b.size(0)
            noise = torch.randn(bs, 16, device=device)
            y_synth = generator(c_b, noise)
            preds_norm.append(y_synth.cpu())

    preds_norm = torch.cat(preds_norm, dim=0).numpy()
    preds_denorm = (preds_norm * target_std) + target_mean

    # Physiological clipping
    for idx, col in enumerate(target_cols):
        if "glucose" in col or "nocturnal_mean" in col:
            preds_denorm[:, idx] = np.clip(preds_denorm[:, idx], 40.0, 500.0)
        elif "a1c" in col:
            preds_denorm[:, idx] = np.clip(preds_denorm[:, idx], 3.5, 20.0)
        elif "count" in col:
            preds_denorm[:, idx] = np.clip(preds_denorm[:, idx], 0.0, 100.0)
        else:
            preds_denorm[:, idx] = np.clip(preds_denorm[:, idx], 0.0, 100.0)

    # Compute Statistical Metrics for Tabular Fidelity
    metrics = {}
    print("\nTabular Fidelity Metrics (Real vs. CTGAN Synthetic Next 30-Day States):")
    for idx, col in enumerate(target_cols):
        r_vals = y_te_raw[:, idx]
        s_vals = preds_denorm[:, idx]

        mae = float(np.mean(np.abs(r_vals - s_vals)))
        rmse = float(np.sqrt(np.mean((r_vals - s_vals) ** 2)))
        r_corr = float(np.corrcoef(r_vals, s_vals)[0, 1])

        unit = "mg/dL" if "glucose" in col or "nocturnal_mean" in col else ("%" if "count" not in col else "events")
        metrics[col] = {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "pearson_r": round(r_corr, 3),
            "real_mean": round(float(np.mean(r_vals)), 2),
            "synth_mean": round(float(np.mean(s_vals)), 2),
            "unit": unit
        }
        print(f"  â€¢ {col:20s}: MAE = {mae:6.2f} {unit:6s} | RMSE = {rmse:6.2f} {unit:6s} | Pearson r = {r_corr:.3f}")

    # Compute Tabular Correlation Preservation Matrix (Frobenius Norm Difference)
    corr_real = np.corrcoef(y_te_raw.T)
    corr_synth = np.corrcoef(preds_denorm.T)
    corr_frobenius_diff = float(np.linalg.norm(corr_real - corr_synth, ord="fro"))
    metrics["correlation_matrix_frobenius_distance"] = round(corr_frobenius_diff, 4)
    print(f"\n  â€¢ Correlation Matrix Preservation Frobenius Distance: {corr_frobenius_diff:.4f} (Lower = Better)")

    with open(output_json, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  >> Saved tabular fidelity report to '{output_json}'")

    # Plot Distribution Comparisons for 6 Key Metrics
    key_indices = [0, 2, 3, 4, 8, 10]  # glucose_mean, glucose_cv, tir_70_180, tbr_below_70, nocturnal_mean, gmi_estimated_a1c
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for p_idx, k_idx in enumerate(key_indices):
        ax = axes[p_idx]
        col_name = target_cols[k_idx]
        r_vals = y_te_raw[:, k_idx]
        s_vals = preds_denorm[:, k_idx]

        ax.hist(r_vals, bins=35, alpha=0.55, color="#1f77b4", label="Real Next 30-Day Cohort", density=True)
        ax.hist(s_vals, bins=35, alpha=0.55, color="#d62728", label="CTGAN Synthetic Next 30-Day", density=True)
        ax.set_title(f"{col_name} (r = {metrics[col_name]['pearson_r']})", fontsize=11, fontweight="bold")
        ax.set_xlabel(metrics[col_name]["unit"], fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Phase 3 CTGAN: Empirical Real vs. Synthetic Tabular Distribution Matching", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"  >> Saved distribution matching visualization to '{output_image}'")

    return metrics

# -----------------------------------------------------------------------------
# 8. Generation Helper for Downstream Pipeline & App
# -----------------------------------------------------------------------------
def generate_synthetic_30d_tabular_state(
    z_vector,
    static_profile,
    checkpoint_path=None,
    n_samples=50
):
    """
    Given a single patient's Phase 2 latent health vector z (16-D) and static profile (9-D),
    uses the trained CTGAN Generator to sample n_samples synthetic next 30-day tabular outcomes.
    Returns:
      - mean_projections: dict of expected 30-day tabular metrics
      - samples_df: DataFrame of all n_samples Monte Carlo synthetic outcomes
    """
    if checkpoint_path is None:
        checkpoint_path = locate_file("best_ctgan_generator.pt")

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    generator = CTGANTabularGenerator(
        cond_dim=ckpt["cond_dim"],
        noise_dim=ckpt["noise_dim"],
        target_dim=ckpt["target_dim"]
    )
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    target_mean = ckpt["target_mean"].numpy()
    target_std = ckpt["target_std"].numpy()
    static_mean = ckpt["static_mean"].numpy()
    static_std = ckpt["static_std"].numpy()
    target_cols = ckpt["target_cols"]

    # Normalize static profile
    norm_static = (static_profile - static_mean) / static_std
    cond_vec = np.concatenate([z_vector, norm_static]).astype(np.float32)
    cond_t = torch.from_numpy(cond_vec).unsqueeze(0).repeat(n_samples, 1)

    noise = torch.randn(n_samples, ckpt["noise_dim"])
    with torch.no_grad():
        synth_norm = generator(cond_t, noise).numpy()

    synth_denorm = (synth_norm * target_std) + target_mean

    # Physiological bounds
    for idx, col in enumerate(target_cols):
        if "glucose" in col or "nocturnal_mean" in col:
            synth_denorm[:, idx] = np.clip(synth_denorm[:, idx], 40.0, 500.0)
        elif "a1c" in col:
            synth_denorm[:, idx] = np.clip(synth_denorm[:, idx], 3.5, 20.0)
        elif "count" in col:
            synth_denorm[:, idx] = np.clip(synth_denorm[:, idx], 0.0, 100.0)
        else:
            synth_denorm[:, idx] = np.clip(synth_denorm[:, idx], 0.0, 100.0)

    samples_df = pd.DataFrame(synth_denorm, columns=target_cols)
    mean_projections = {col: float(np.mean(samples_df[col])) for col in target_cols}

    return mean_projections, samples_df

# -----------------------------------------------------------------------------
# Main Entrypoint
# -----------------------------------------------------------------------------
def run_phase3_ctgan_pipeline(num_epochs=15):
    bundle = prepare_ctgan_dataset()
    try:
        _phase3_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        _phase3_dir = "."
    ckpt_path = os.path.join(_phase3_dir, "best_ctgan_generator.pt")
    img_path = os.path.join(_phase3_dir, "ctgan_tabular_fidelity_comparison.png")
    json_path = os.path.join(_phase3_dir, "ctgan_evaluation_report.json")
    generator, ckpt = train_ctgan(bundle, num_epochs=num_epochs, checkpoint_path=ckpt_path)
    metrics = evaluate_ctgan_fidelity(generator, bundle, output_image=img_path, output_json=json_path)
    return generator, metrics

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_phase3_ctgan_pipeline(num_epochs=12)
