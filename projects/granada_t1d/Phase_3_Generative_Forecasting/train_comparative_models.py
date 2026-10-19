"""
================================================================================
COMPARATIVE TRAJECTORY MODELING FOR IEEE BENCHMARK
================================================================================
Benchmarked Architectures:
  1. Baseline Deterministic LSTM
  2. Autoregressive Multi-Modal Forecaster (Phase 4 GRU Rollout)
  3. Conditional Temporal GAN (Temporal Generator + Discriminator)
  4. Conditional Tabular GAN (Phase 3 CTGAN)
================================================================================
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure local imports
try:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    curr_dir = os.getcwd()
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from phase3_4_trajectory_forecasting import (
    MultiModalTrajectoryForecaster,
    prepare_forecasting_dataset,
    clarke_error_grid,
    get_device,
    set_seed
)

# -----------------------------------------------------------------------------
# 1. Baseline Deterministic LSTM Model
# -----------------------------------------------------------------------------
class BaselineLSTM(nn.Module):
    """Sequence-to-sequence baseline LSTM mapping [z, x_last] -> Y (60 days)."""
    def __init__(self, latent_dim=16, target_dim=6, hidden_dim=64, out_len=60):
        super().__init__()
        self.out_len = out_len
        self.proj = nn.Sequential(
            nn.Linear(latent_dim + target_dim, hidden_dim),
            nn.Tanh()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, x_last):
        cond = torch.cat([z, x_last], dim=-1)
        h0 = self.proj(cond).unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)

        # Static repeat input
        seq_in = cond.unsqueeze(1).repeat(1, self.out_len, 1)
        # Project seq_in to hidden_dim
        out, _ = self.lstm(h0[-1].unsqueeze(1).repeat(1, self.out_len, 1), (h0, c0))
        preds = self.head(out) + x_last.unsqueeze(1)
        return preds

# -----------------------------------------------------------------------------
# 2. Conditional Temporal GAN (Temporal Generator + Discriminator)
# -----------------------------------------------------------------------------
class TemporalGenerator(nn.Module):
    """Temporal Generator G(z, noise, x_last) producing 60-day future trajectories."""
    def __init__(self, latent_dim=16, noise_dim=16, target_dim=6, hidden_dim=64, out_len=60):
        super().__init__()
        self.out_len = out_len
        self.init_proj = nn.Sequential(
            nn.Linear(latent_dim + noise_dim + target_dim, hidden_dim),
            nn.Tanh()
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.15)
        self.head = nn.Linear(hidden_dim, target_dim)

    def forward(self, z, noise, x_last):
        cond = torch.cat([z, noise, x_last], dim=-1)
        # Fix: h0 must be (num_layers, batch, hidden_dim) for GRU
        h0_base = self.init_proj(cond)  # (batch, hidden_dim)
        h0 = h0_base.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)  # (num_layers, batch, hidden)

        seq_in = h0_base.unsqueeze(1).repeat(1, self.out_len, 1)
        out, _ = self.gru(seq_in, h0)   # pass h0 so GRU actually conditions on latent
        preds = self.head(out) + x_last.unsqueeze(1)
        return preds

class TemporalDiscriminator(nn.Module):
    """Temporal Discriminator D(Y, z) distinguishing real vs synthetic trajectories."""
    def __init__(self, target_dim=6, latent_dim=16, hidden_dim=64):
        super().__init__()
        self.z_proj = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.LeakyReLU(0.2))
        self.conv1 = nn.Conv1d(target_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, y, z):
        # y: (batch, 60, 6) -> transpose to (batch, 6, 60) for Conv1d
        y_trans = y.transpose(1, 2)
        f = F.leaky_relu(self.conv1(y_trans), 0.2)
        f = F.leaky_relu(self.conv2(f), 0.2)
        f = f.transpose(1, 2)

        lstm_out, _ = self.lstm(f)
        pooled = torch.mean(lstm_out, dim=1)

        z_feat = self.z_proj(z)
        combined = torch.cat([pooled, z_feat], dim=-1)
        validity = self.classifier(combined)
        return validity

# -----------------------------------------------------------------------------
# 3. Execution Pipeline: Training & Benchmarking
# -----------------------------------------------------------------------------
def run_comparison(num_epochs=10):
    print("=" * 75)
    print("IEEE MODEL COMPARISON: BASELINE LSTM vs AUTOREGRESSIVE GRU vs TEMPORAL cGAN")
    print("=" * 75)

    device = get_device()
    print(f"  • Utilizing Device: {device}")

    bundle = prepare_forecasting_dataset()
    z_tr, x_last_tr, y_tr, _ = bundle["train"]
    z_va, x_last_va, y_va, _ = bundle["val"]
    z_te, x_last_te, y_te, _ = bundle["test"]

    target_cols = bundle["target_cols"]
    target_mean = bundle["target_mean"].numpy()
    target_std = bundle["target_std"].numpy()

    train_loader = DataLoader(TensorDataset(z_tr, x_last_tr, y_tr), batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(z_va, x_last_va, y_va), batch_size=64, shuffle=False)

    # ---------------------------------------------------------
    # PART A: Train Baseline LSTM
    # ---------------------------------------------------------
    print("\n[Part 1/3] Training Baseline Deterministic LSTM...")
    lstm_model = BaselineLSTM(latent_dim=16, target_dim=6, hidden_dim=64, out_len=60).to(device)
    lstm_opt = torch.optim.AdamW(lstm_model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_lstm_val = float("inf")
    for epoch in range(1, num_epochs + 1):
        lstm_model.train()
        train_loss = 0.0
        for z_b, x_b, y_b in train_loader:
            z_b, x_b, y_b = z_b.to(device), x_b.to(device), y_b.to(device)
            lstm_opt.zero_grad()
            y_p = lstm_model(z_b, x_b)
            l = F.mse_loss(y_p, y_b)
            l.backward()
            torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), 1.0)
            lstm_opt.step()
            train_loss += l.item()

        lstm_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for z_v, x_v, y_v in val_loader:
                z_v, x_v, y_v = z_v.to(device), x_v.to(device), y_v.to(device)
                y_pv = lstm_model(z_v, x_v)
                val_loss += F.mse_loss(y_pv, y_v).item()

        val_loss /= len(val_loader)
        if val_loss < best_lstm_val:
            best_lstm_val = val_loss
            ckpt_path = os.path.join(curr_dir, "best_baseline_lstm.pt")
            torch.save(lstm_model.state_dict(), ckpt_path)
        print(f"  • LSTM Epoch {epoch:02d}/{num_epochs:02d} | Train MSE: {train_loss/len(train_loader):.4f} | Val MSE: {val_loss:.4f}")

    # ---------------------------------------------------------
    # PART B: Train Conditional Temporal GAN (cGAN)
    # ---------------------------------------------------------
    print("\n[Part 2/3] Training Conditional Temporal GAN (cGAN)...")
    netG = TemporalGenerator(latent_dim=16, noise_dim=16, target_dim=6, hidden_dim=64, out_len=60).to(device)
    netD = TemporalDiscriminator(target_dim=6, latent_dim=16, hidden_dim=64).to(device)

    optG = torch.optim.AdamW(netG.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    optD = torch.optim.AdamW(netD.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    for epoch in range(1, num_epochs + 1):
        netG.train()
        netD.train()
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0

        for z_b, x_b, y_b in train_loader:
            z_b, x_b, y_b = z_b.to(device), x_b.to(device), y_b.to(device)
            bs = z_b.size(0)

            # 1. Train Discriminator
            optD.zero_grad()
            noise = torch.randn(bs, 16, device=device)
            fake_y = netG(z_b, noise, x_b)

            d_real = netD(y_b, z_b)
            d_fake = netD(fake_y.detach(), z_b)

            loss_d_real = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real) * 0.9)
            loss_d_fake = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            optD.step()

            # 2. Train Generator
            optG.zero_grad()
            d_fake_g = netD(fake_y, z_b)
            loss_g_adv = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))
            loss_g_rec = F.mse_loss(fake_y, y_b)
            loss_g = loss_g_adv + (5.0 * loss_g_rec)
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
            optG.step()

            d_loss_epoch += loss_d.item()
            g_loss_epoch += loss_g.item()

        print(f"  • cGAN Epoch {epoch:02d}/{num_epochs:02d} | D Loss: {d_loss_epoch/len(train_loader):.4f} | G Loss: {g_loss_epoch/len(train_loader):.4f}")

    gen_ckpt_path = os.path.join(curr_dir, "best_temporal_cgan_generator.pt")
    torch.save(netG.state_dict(), gen_ckpt_path)

    # ---------------------------------------------------------
    # PART C: Comprehensive Comparative Evaluation on Test Set
    # ---------------------------------------------------------
    print("\n[Part 3/3] Benchmarking Models on Held-Out Test Set...")
    lstm_model.eval()
    netG.eval()

    def evaluate_model_preds(y_preds_norm, name):
        y_real = (y_preds_norm * target_std) + target_mean
        y_true_real = (y_te.numpy() * target_std) + target_mean

        y_real[:, :, 0] = np.clip(y_real[:, :, 0], 40, 500)
        for k in range(1, 6):
            y_real[:, :, k] = np.clip(y_real[:, :, k], 0, 100)

        mae_g = float(np.mean(np.abs(y_real[:, :, 0] - y_true_real[:, :, 0])))
        rmse_g = float(np.sqrt(np.mean((y_real[:, :, 0] - y_true_real[:, :, 0]) ** 2)))
        mae_tir = float(np.mean(np.abs(y_real[:, :, 1] - y_true_real[:, :, 1])))
        mae_tbr = float(np.mean(np.abs(y_real[:, :, 2] - y_true_real[:, :, 2])))
        mae_a1c = float(np.mean(np.abs(y_real[:, :, 5] - y_true_real[:, :, 5])))

        clarke = clarke_error_grid(y_true_real[:, :, 0].flatten(), y_real[:, :, 0].flatten())

        print(f"\nBenchmark Performance for {name}:")
        print(f"  • Glucose MAE : {mae_g:.2f} mg/dL | RMSE: {rmse_g:.2f} mg/dL")
        print(f"  • TIR MAE     : {mae_tir:.2f} %")
        print(f"  • TBR MAE     : {mae_tbr:.2f} %")
        print(f"  • HbA1c MAE   : {mae_a1c:.2f} %")
        print(f"  • Clarke Zone A+B: {clarke['zone_a_b_combined_pct']}% (Zone A: {clarke['zone_a_pct']}%, Zone B: {clarke['zone_b_pct']}%)")

        return {
            "glucose_mae": round(mae_g, 2),
            "glucose_rmse": round(rmse_g, 2),
            "tir_mae": round(mae_tir, 2),
            "tbr_mae": round(mae_tbr, 2),
            "hba1c_mae": round(mae_a1c, 2),
            "clarke_a_b": clarke["zone_a_b_combined_pct"]
        }

    # Evaluate LSTM
    test_loader = DataLoader(TensorDataset(z_te, x_last_te), batch_size=128, shuffle=False)
    lstm_preds = []
    with torch.no_grad():
        for z_b, x_b in test_loader:
            z_b, x_b = z_b.to(device), x_b.to(device)
            lstm_preds.append(lstm_model(z_b, x_b).cpu())
    lstm_res = evaluate_model_preds(torch.cat(lstm_preds, dim=0).numpy(), "Baseline Deterministic LSTM")

    # Evaluate Temporal cGAN
    cgan_preds = []
    with torch.no_grad():
        for z_b, x_b in test_loader:
            z_b, x_b = z_b.to(device), x_b.to(device)
            noise = torch.zeros(z_b.size(0), 16, device=device)
            cgan_preds.append(netG(z_b, noise, x_b).cpu())
    cgan_res = evaluate_model_preds(torch.cat(cgan_preds, dim=0).numpy(), "Conditional Temporal GAN (cGAN)")

    benchmark_dict = {
        "Baseline_LSTM": lstm_res,
        "Conditional_Temporal_GAN": cgan_res
    }
    out_json = os.path.join(curr_dir, "model_benchmarking_results.json")
    with open(out_json, "w") as f:
        json.dump(benchmark_dict, f, indent=4)
    print(f"\n  >> Saved comparative benchmark results to '{out_json}'")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_comparison(num_epochs=5)
