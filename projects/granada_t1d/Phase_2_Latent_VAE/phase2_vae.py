"""
================================================================================
PHASE 2: COMPACT REPRESENTATION OF PATIENT HEALTH (LATENT HEALTH VAE)
================================================================================
Scope:
  • Prepare Model Inputs:
      - Ingests cleaned longitudinal timeline from Phase 1.
      - Executes patient-disjoint train/val/test splitting (70% / 15% / 15%).
      - Normalizes features strictly on training cohort.
      - Enforces canonical feature ordering: [pure_dynamic | imputed_flags | static].
      - Packages 30-day sliding windows with strict daily continuity verification.
      - Consistently saves all three metadata splits (meta_train, meta_val, meta_test).
  • Two-Branch Latent VAE Architecture (TwoBranchHealthVAE):
      - Dynamic Branch: Bi-LSTM + Attention Encoder on 41 dynamic features
        (27 clinical + 14 imputation flags) → pooled representation (128-dim).
      - Static  Branch: 2-Layer MLP on 9 static features → embedding (32-dim).
      - Fusion: cat[(128), (32)] → 160-dim → fc_mu / fc_logvar → R^16.
      - Reparameterization trick (z = mu + eps * sigma).
      - Reconstruction Decoder: p(X_clinical | z) targeting 36 clinical dims
        (27 pure-dynamic + 9 static). Imputed flags are EXCLUDED from the MSE
        loss — binary indicators are not amenable to continuous reconstruction.
      - STRICTLY NO forecasting components.
  • Pure VAE Optimization with KL Annealing & Early Stopping:
      - Loss = MSE(recon_36, target_36) + beta(epoch) * KL Divergence.
      - Linear KL warm-up: beta ramps 0.0 → 0.005 over first 15 epochs.
      - Early Stopping tied to val reconstruction loss (patience = 10).
      - Max epochs = 80; ReduceLROnPlateau scheduler on val total loss.
  • Final Output:
      - Latent health vectors (mu / z in R^16) per window and per patient.
================================================================================
"""

import os
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# -----------------------------------------------------------------------------
# Device & Seed Utilities
# -----------------------------------------------------------------------------
def get_device():
    """Hierarchical device selection: CUDA -> MPS -> CPU."""
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
# 1. Prepare Model Inputs
# -----------------------------------------------------------------------------
def prepare_model_inputs():
    """
    Ingests the cleaned daily multimodal dataset from Phase 1, executes a
    patient-disjoint train/val/test split, fits a normalizer on the training
    cohort only, and packages strictly consecutive 30-day sliding windows.

    Enforces canonical feature ordering:
        [pure_dynamic_cols | imputed_flag_cols | static_cols]
    so that downstream tensor slicing (batch_dyn[:, :, :n_pure_dynamic]) is
    deterministic and architecture-agnostic.

    Saves meta_train, meta_val, and meta_test consistently alongside
    n_pure_dynamic, n_dyn, and n_stat for the two-branch model.
    """
    start_time = time.time()
    print("=" * 75)
    print("PHASE 2 - STEP 1: PREPARE MODEL INPUTS (WINDOWING & SPLITTING)")
    print("=" * 75)

    # Locate cleaned dataset from Phase 1
    csv_candidates = [
        "cleaned_daily_multimodal_dataset.csv",
        os.path.join("..", "Phase_1_Data_Preparation",
                     "cleaned_daily_multimodal_dataset.csv"),
        os.path.join("Phase_1_Data_Preparation",
                     "cleaned_daily_multimodal_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "..", "Phase_1_Data_Preparation",
                     "cleaned_daily_multimodal_dataset.csv"),
        os.path.join(os.path.dirname(__file__), "cleaned_daily_multimodal_dataset.csv")
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError(
            "Could not find 'cleaned_daily_multimodal_dataset.csv'. Run Phase 1 first!"
        )

    print(f"  • Loading cleaned timeline from '{csv_path}'...")
    df = pd.read_csv(csv_path)
    df["date_dt"] = pd.to_datetime(df["date"])

    # ------------------------------------------------------------------
    # Canonical feature ordering: [pure_dynamic | imputed_flags | static]
    # This guarantees:
    #   batch_dyn[:, :, :n_pure_dynamic]  = 27 clinical features only
    #   batch_dyn[:, :, n_pure_dynamic:]  = 14 imputation masks
    #   batch_full[:, :, n_dyn:]          = 9 static features
    # ------------------------------------------------------------------
    meta_cols = ["Patient_ID", "date", "date_dt", "readings_count", "split"]
    STATIC_COLS = [
        "age", "sex_male", "hypothyroidism", "retinopathy", "hypertension",
        "dyslipidemia", "nephropathy", "neuropathy", "dka_history"
    ]
    IMPUTED_SUFFIX = "_imputed"

    all_available = [c for c in df.columns if c not in meta_cols]
    pure_dyn_cols = [
        c for c in all_available
        if c not in STATIC_COLS and not c.endswith(IMPUTED_SUFFIX)
    ]
    imputed_cols = [c for c in all_available if c.endswith(IMPUTED_SUFFIX)]
    static_cols  = [c for c in all_available if c in STATIC_COLS]

    # Enforce canonical ordering
    input_cols     = pure_dyn_cols + imputed_cols + static_cols
    n_pure_dynamic = len(pure_dyn_cols)                     # 27
    n_dyn          = len(pure_dyn_cols) + len(imputed_cols) # 41
    n_stat         = len(static_cols)                       # 9
    input_dim      = len(input_cols)                        # 50

    print(f"  • Loaded {len(df):,} daily records across {df['Patient_ID'].nunique()} patients.")
    print(f"  • Feature split: {n_pure_dynamic} pure-dynamic | "
          f"{len(imputed_cols)} imputed-flags | {n_stat} static = {input_dim} total")

    # Patient-Disjoint Train/Val/Test Split (Generated Upstream in Phase 1)
    train_pats = set(df[df["split"] == "train"]["Patient_ID"])
    val_pats   = set(df[df["split"] == "val"]["Patient_ID"])
    test_pats  = set(df[df["split"] == "test"]["Patient_ID"])

    print(f"  • Upstream Split: {len(train_pats)} Train | "
          f"{len(val_pats)} Val | {len(test_pats)} Test")

    # Fit Normalizer Strictly on Training Cohort Only
    train_df = df[df["Patient_ID"].isin(train_pats)]
    feat_mean = train_df[input_cols].mean().values.astype(np.float32)
    feat_std  = train_df[input_cols].std().values.astype(np.float32)
    # Avoid divide-by-zero on invariant binary flags (std ≈ 0)
    feat_std[feat_std < 1e-6] = 1.0

    scaler_params = {
        "input_cols":     input_cols,
        "pure_dyn_cols":  pure_dyn_cols,
        "imputed_cols":   imputed_cols,
        "static_cols":    static_cols,
        "feat_mean":      feat_mean.tolist(),
        "feat_std":       feat_std.tolist(),
        "n_pure_dynamic": n_pure_dynamic,
        "n_dyn":          n_dyn,
        "n_stat":         n_stat
    }
    with open("scaler_parameters.json", "w") as f:
        json.dump(scaler_params, f, indent=4)
    print("  >> Saved 'scaler_parameters.json'")

    # Extract 30-Day Sliding Windows (Stride = 7, Strict Daily Continuity)
    T_IN   = 30
    STRIDE = 7

    def extract_windows(cohort_pats, name):
        X_list    = []
        meta_list = []
        sub_df    = df[df["Patient_ID"].isin(cohort_pats)]

        for pid, group in sub_df.groupby("Patient_ID"):
            group  = group.sort_values("date_dt").reset_index(drop=True)
            n_rows = len(group)
            if n_rows < T_IN:
                continue

            values      = group[input_cols].values.astype(np.float32)
            norm_values = (values - feat_mean) / feat_std
            dates       = group["date"].values

            for start_idx in range(0, n_rows - T_IN + 1, STRIDE):
                end_idx      = start_idx + T_IN
                window_dates = group.loc[start_idx:end_idx - 1, "date_dt"].values
                date_diffs   = (window_dates[1:] - window_dates[:-1]) / np.timedelta64(1, "D")

                # Strict daily consecutive observations — reject any window with gaps
                if not np.all(date_diffs == 1.0):
                    continue

                X_list.append(norm_values[start_idx:end_idx])
                meta_list.append({
                    "patient_id": pid,
                    "start_date": str(dates[start_idx]),
                    "end_date":   str(dates[end_idx - 1])
                })

        X_arr = np.array(X_list, dtype=np.float32)
        print(f"  • {name:10s}: {len(X_arr):5d} windows | Tensor shape: {X_arr.shape}")
        return torch.from_numpy(X_arr), meta_list

    X_train, meta_train = extract_windows(train_pats, "Train Set")
    X_val,   meta_val   = extract_windows(val_pats,   "Val Set")
    X_test,  meta_test  = extract_windows(test_pats,  "Test Set")

    tensor_path = "timeseries_tensors.pt"
    tensor_data = {
        "X_train":        X_train,
        "X_val":          X_val,
        "X_test":         X_test,
        "train_pats":     list(train_pats),
        "val_pats":       list(val_pats),
        "test_pats":      list(test_pats),
        "meta_train":     meta_train,       # Consistently saved!
        "meta_val":       meta_val,         # Consistently saved!
        "meta_test":      meta_test,        # Consistently saved!
        "input_cols":     input_cols,
        "scaler_params":  scaler_params,
        # Two-branch split dimensions — consumed by TwoBranchHealthVAE
        "n_pure_dynamic": n_pure_dynamic,   # 27: pure clinical dynamic features
        "n_dyn":          n_dyn,            # 41: dynamic + imputed flags
        "n_stat":         n_stat,           # 9:  static demographics + comorbidities
    }
    torch.save(tensor_data, tensor_path)
    elapsed = time.time() - start_time
    print(f"  >> Saved '{tensor_path}' "
          f"({os.path.getsize(tensor_path) / 1024 / 1024:.2f} MB in {elapsed:.2f}s)")
    return tensor_data


# -----------------------------------------------------------------------------
# 2. Two-Branch Latent Health VAE Architecture
# -----------------------------------------------------------------------------
class AttentionPooling(nn.Module):
    """Temporal attention pooling over the Bi-LSTM output sequence."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        scores  = self.attn(x)                     # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        pooled  = torch.sum(x * weights, dim=1)   # (B, hidden_dim)
        return pooled


class TwoBranchHealthVAE(nn.Module):
    """
    Two-Branch Multi-Modal Latent Health VAE.

    Separates time-varying dynamic features (CGM metrics + lab values +
    imputation flags) from static patient-level features (demographics +
    comorbidities) into dedicated encoder branches, fusing them before the
    latent projections.

    Encoder:
        Dynamic branch : (B, T, n_dyn)  -> Bi-LSTM(2L, hidden=64) -> AttentionPool -> (B, 128)
        Static  branch : (B, n_stat)    -> MLP(n_stat -> 64 -> 32) -> (B, 32)
        Fusion         : cat[(B,128),(B,32)] -> (B, 160) -> fc_mu / fc_logvar -> (B, latent_dim)

    Decoder  [p(X_clinical | z)]:
        z (B, latent_dim) -> rec_proj(16->64) -> repeat(T) -> LSTM(2L, 64) -> rec_out(64 -> 36)
        recon_dim = n_pure_dynamic + n_stat = 36.
        Imputed flags are excluded from reconstruction — binary indicators are not
        amenable to MSE optimization and would distort the loss landscape.

    Contains NO forecasting rollouts, NO trajectory heads, NO GRU extrapolators.
    """
    def __init__(self, n_dyn, n_stat, n_pure_dynamic,
                 latent_dim=16, hidden_dim=64, seq_len=30):
        super().__init__()
        self.n_dyn          = n_dyn
        self.n_stat         = n_stat
        self.n_pure_dynamic = n_pure_dynamic
        self.latent_dim     = latent_dim
        self.hidden_dim     = hidden_dim
        self.seq_len        = seq_len
        self.recon_dim      = n_pure_dynamic + n_stat   # 27 + 9 = 36

        # --- Dynamic Branch: Bi-LSTM Encoder ---
        self.encoder_lstm = nn.LSTM(
            input_size=n_dyn,        # 41 (27 clinical + 14 imputation flags)
            hidden_size=hidden_dim,  # 64
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.15
        )
        enc_lstm_out = hidden_dim * 2   # 128 (bidirectional concatenation)
        self.attn_pool = AttentionPooling(enc_lstm_out)

        # --- Static Branch: 2-Layer MLP ---
        static_out = hidden_dim // 2    # 32
        self.static_mlp = nn.Sequential(
            nn.Linear(n_stat, hidden_dim),  # 9 -> 64
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, static_out)  # 64 -> 32
        )

        # --- Fusion -> Latent Projections ---
        fusion_dim = enc_lstm_out + static_out   # 128 + 32 = 160
        self.fc_mu     = nn.Linear(fusion_dim, latent_dim)
        self.fc_logvar = nn.Linear(fusion_dim, latent_dim)

        # --- Reconstruction Decoder: p(X_clinical | z) -> 36 dims ---
        self.rec_proj = nn.Linear(latent_dim, hidden_dim)
        self.rec_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15
        )
        self.rec_out = nn.Linear(hidden_dim, self.recon_dim)   # 64 -> 36

    def encode(self, x_dyn, x_stat):
        """
        Args:
            x_dyn  : (B, T, n_dyn)  — normalized dynamic features (27 clinical + 14 flags)
            x_stat : (B, n_stat)    — normalized static features
        Returns:
            mu, logvar : each (B, latent_dim)
        """
        # Dynamic branch
        lstm_out, _ = self.encoder_lstm(x_dyn)    # (B, T, 128)
        dyn_emb     = self.attn_pool(lstm_out)    # (B, 128)
        # Static branch
        stat_emb    = self.static_mlp(x_stat)     # (B, 32)
        # Fusion
        fused       = torch.cat([dyn_emb, stat_emb], dim=-1)   # (B, 160)
        mu          = self.fc_mu(fused)
        logvar      = self.fc_logvar(fused)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        """
        Args:
            z : (B, latent_dim)
        Returns:
            x_recon : (B, T, recon_dim=36) — reconstructed clinical features only
        """
        z_proj      = self.rec_proj(z).unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, 64)
        rec_out, _  = self.rec_lstm(z_proj)                                      # (B, T, 64)
        return self.rec_out(rec_out)                                             # (B, T, 36)

    def forward(self, x_dyn, x_stat):
        """
        Args:
            x_dyn  : (B, T, n_dyn)  — dynamic branch input
            x_stat : (B, n_stat)    — static branch input
        Returns:
            x_recon : (B, T, 36)    — reconstructed clinical output (imputed flags excluded)
            mu      : (B, latent_dim)
            logvar  : (B, latent_dim)
        """
        mu, logvar = self.encode(x_dyn, x_stat)
        z          = self.reparameterize(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar


# -----------------------------------------------------------------------------
# 3. VAE Training Loop (Reconstruction + Annealed KL + Early Stopping)
# -----------------------------------------------------------------------------
def train_compact_vae(tensor_data=None, num_epochs=80, latent_dim=16, hidden_dim=64,
                      early_stop_patience=10, kl_anneal_epochs=15):
    """
    Trains TwoBranchHealthVAE with pure VAE loss (MSE reconstruction + annealed KL).

    Key training controls:
      • KL annealing   : beta rises linearly 0.0 → 0.005 over kl_anneal_epochs,
                         then held constant. Prevents early posterior collapse.
      • Early stopping : halts when val reconstruction loss fails to improve for
                         early_stop_patience consecutive epochs.
      • Checkpointing  : saves best model weights by total val loss (rec + KL term).
      • LR scheduler   : ReduceLROnPlateau on val total loss (factor=0.5, patience=3).
    """
    print("\n" + "=" * 75)
    print("PHASE 2 - STEP 2: TRAIN TWO-BRANCH LATENT HEALTH VAE")
    print("=" * 75)

    set_seed(42)
    device = get_device()
    print(f"  • Utilizing Device: {device} (CUDA -> MPS -> CPU hierarchy)")

    if tensor_data is None:
        tensor_path = "timeseries_tensors.pt"
        if not os.path.exists(tensor_path):
            tensor_path = os.path.join("Phase_2_Latent_VAE", "timeseries_tensors.pt")
        tensor_data = torch.load(tensor_path, map_location="cpu", weights_only=False)

    X_train    = tensor_data["X_train"]
    X_val      = tensor_data["X_val"]
    n_pure_dyn = tensor_data["n_pure_dynamic"]   # 27
    n_dyn      = tensor_data["n_dyn"]            # 41
    n_stat     = tensor_data["n_stat"]           # 9

    print(f"  • Feature split — pure-dynamic: {n_pure_dyn} | "
          f"with-imputed: {n_dyn} | static: {n_stat}")
    print(f"  • Decoder target: {n_pure_dyn + n_stat} dims "
          f"(clinical only; imputed flags excluded from MSE)")
    print(f"  • Training controls — Max epochs: {num_epochs} | "
          f"Early-stop patience: {early_stop_patience} | "
          f"KL anneal: {kl_anneal_epochs} epochs")

    train_dataset = TensorDataset(X_train)
    val_dataset   = TensorDataset(X_val)
    batch_size    = 64
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=True, drop_last=True)
    val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    model = TwoBranchHealthVAE(
        n_dyn=n_dyn,
        n_stat=n_stat,
        n_pure_dynamic=n_pure_dyn,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seq_len=30
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # KL annealing: linear ramp from beta_start → beta_end over kl_anneal_epochs
    beta_start = 0.0
    beta_end   = 0.005

    best_val_loss = float("inf")
    best_val_rec  = float("inf")
    es_counter    = 0
    checkpoint_path = "best_compact_vae.pt"
    history = []

    print("\nStarting Two-Branch VAE Training (Reconstruction + Annealed KL Loss):")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):

        # ---- KL Annealing: linear ramp then constant ----
        beta_kl = min(beta_end,
                      beta_start + (beta_end - beta_start) * (epoch / kl_anneal_epochs))

        # ==== Training Pass ====
        model.train()
        total_train_loss = 0.0
        total_rec_loss   = 0.0
        total_kl_loss    = 0.0

        for (x_b,) in train_loader:
            x_b = x_b.to(device)   # (B, 30, 50)

            # Two-branch split
            batch_dyn  = x_b[:, :, :n_dyn]    # (B, 30, 41) — dynamic + imputed flags
            batch_stat = x_b[:, 0, n_dyn:]    # (B, 9)      — static (invariant across T)

            # Target: pure clinical (27) + static broadcast (9) = 36 dims.
            # Imputed flags are sliced OUT of the reconstruction target here.
            target_dyn      = batch_dyn[:, :, :n_pure_dyn]             # (B, 30, 27)
            target_combined = torch.cat([
                target_dyn,
                batch_stat.unsqueeze(1).repeat(1, 30, 1)               # (B, 30, 9)
            ], dim=-1)                                                  # (B, 30, 36)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch_dyn, batch_stat)

            # 1. Reconstruction Loss (MSE on 36 clinical dims only)
            loss_rec = F.mse_loss(x_recon, target_combined)
            # 2. KL Divergence
            loss_kl  = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )
            # 3. Total VAE Loss with annealed beta
            loss = loss_rec + (beta_kl * loss_kl)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_rec_loss   += loss_rec.item()
            total_kl_loss    += loss_kl.item()

        # ==== Validation Pass ====
        model.eval()
        total_val_loss = 0.0
        total_val_rec  = 0.0
        total_val_kl   = 0.0

        with torch.no_grad():
            for (x_v,) in val_loader:
                x_v = x_v.to(device)

                val_dyn  = x_v[:, :, :n_dyn]
                val_stat = x_v[:, 0, n_dyn:]

                target_dyn_v      = val_dyn[:, :, :n_pure_dyn]         # (B, 30, 27)
                target_combined_v = torch.cat([
                    target_dyn_v,
                    val_stat.unsqueeze(1).repeat(1, 30, 1)             # (B, 30, 9)
                ], dim=-1)                                              # (B, 30, 36)

                x_recon_v, mu_v, logvar_v = model(val_dyn, val_stat)

                loss_rec_v = F.mse_loss(x_recon_v, target_combined_v)
                loss_kl_v  = -0.5 * torch.mean(
                    torch.sum(1 + logvar_v - mu_v.pow(2) - logvar_v.exp(), dim=1)
                )
                val_loss = loss_rec_v + (beta_kl * loss_kl_v)

                total_val_loss += val_loss.item()
                total_val_rec  += loss_rec_v.item()
                total_val_kl   += loss_kl_v.item()

        avg_train     = total_train_loss / len(train_loader)
        avg_train_rec = total_rec_loss   / len(train_loader)
        avg_val       = total_val_loss   / len(val_loader)
        avg_val_rec   = total_val_rec    / len(val_loader)

        scheduler.step(avg_val)

        # ---- Checkpoint: save on best total val loss ----
        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_val_loss,
                # Two-branch architecture dimensions
                "n_dyn":            n_dyn,
                "n_stat":           n_stat,
                "n_pure_dynamic":   n_pure_dyn,
                "latent_dim":       latent_dim,
                "hidden_dim":       hidden_dim,
                "seq_len":          30
            }, checkpoint_path)
            flag = "(* BEST *)"
        else:
            flag = ""

        print(f"Epoch {epoch:03d}/{num_epochs:03d} | "
              f"beta={beta_kl:.5f} | "
              f"Train: {avg_train:.4f} (Rec: {avg_train_rec:.4f}) | "
              f"Val: {avg_val:.4f} (Rec: {avg_val_rec:.4f}) {flag}")

        history.append({
            "epoch":        epoch,
            "beta_kl":      round(beta_kl, 6),
            "train_loss":   avg_train,
            "train_rec":    avg_train_rec,
            "val_loss":     avg_val,
            "val_rec_loss": avg_val_rec
        })

        # ---- Early Stopping: monitor val reconstruction loss ----
        if avg_val_rec < best_val_rec:
            best_val_rec = avg_val_rec
            es_counter   = 0
        else:
            es_counter += 1
            if es_counter >= early_stop_patience:
                print(f"\n  >> Early stopping triggered at epoch {epoch} "
                      f"(val_rec stagnant for {early_stop_patience} consecutive epochs).")
                break

    elapsed = time.time() - start_time
    print(f"\n  >> Training Complete in {elapsed:.2f}s | Best Val Loss: {best_val_loss:.4f}")
    with open("vae_training_history.json", "w") as f:
        json.dump(history, f, indent=4)

    return model, checkpoint_path


# -----------------------------------------------------------------------------
# 4. Latent Vector Extraction & Final Output
# -----------------------------------------------------------------------------
def extract_and_save_latent_vectors(checkpoint_path="best_compact_vae.pt"):
    """
    Extracts latent health vectors (mu and sampled z in R^16) for all patient
    windows across Train, Val, and Test sets using the trained TwoBranchHealthVAE
    encoder. In eval mode the encoder returns mu directly (no noise sampling).

    Saves:
      - latent_health_vectors.pt            : Window-level latent vectors + metadata.
      - patient_latent_health_summary.csv   : Per-patient mean latent representations.
    """
    print("\n" + "=" * 75)
    print("PHASE 2 - STEP 3: EXTRACT & SAVE LATENT HEALTH VECTORS")
    print("=" * 75)

    device = get_device()

    # Load tensor dataset
    tensor_path = "timeseries_tensors.pt"
    if not os.path.exists(tensor_path):
        tensor_path = os.path.join("Phase_2_Latent_VAE", "timeseries_tensors.pt")
    tensor_data = torch.load(tensor_path, map_location="cpu", weights_only=False)

    # Load checkpoint — weights_only=True is safe: contains only model weights,
    # tensors, and Python primitives (int, float, str).
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join("Phase_2_Latent_VAE", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    n_dyn      = ckpt["n_dyn"]
    n_stat     = ckpt["n_stat"]
    n_pure_dyn = ckpt["n_pure_dynamic"]
    latent_dim = ckpt["latent_dim"]

    model = TwoBranchHealthVAE(
        n_dyn=n_dyn,
        n_stat=n_stat,
        n_pure_dynamic=n_pure_dyn,
        latent_dim=latent_dim,
        hidden_dim=ckpt["hidden_dim"],
        seq_len=ckpt.get("seq_len", 30)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()   # reparameterize returns mu directly in eval mode

    def encode_split(X_tensor):
        mu_list, z_list = [], []
        with torch.no_grad():
            for i in range(0, len(X_tensor), 128):
                xb      = X_tensor[i:i + 128].to(device)
                xb_dyn  = xb[:, :, :n_dyn]     # (B, 30, 41)
                xb_stat = xb[:, 0, n_dyn:]     # (B, 9) — static, take from any timestep
                mu, logvar = model.encode(xb_dyn, xb_stat)
                z = model.reparameterize(mu, logvar)
                mu_list.append(mu.cpu())
                z_list.append(z.cpu())
        return torch.cat(mu_list, dim=0), torch.cat(z_list, dim=0)

    print("  • Encoding multi-modal windows into latent space R^16 "
          "via Two-Branch Encoder...")
    mu_train, z_train = encode_split(tensor_data["X_train"])
    mu_val,   z_val   = encode_split(tensor_data["X_val"])
    mu_test,  z_test  = encode_split(tensor_data["X_test"])

    # Save Window-Level Latent Vectors
    latent_export = {
        "mu_train":   mu_train,
        "z_train":    z_train,
        "meta_train": tensor_data["meta_train"],
        "mu_val":     mu_val,
        "z_val":      z_val,
        "meta_val":   tensor_data["meta_val"],
        "mu_test":    mu_test,
        "z_test":     z_test,
        "meta_test":  tensor_data["meta_test"],
        "latent_dim": latent_dim,
        "train_pats": tensor_data["train_pats"],
        "val_pats":   tensor_data["val_pats"],
        "test_pats":  tensor_data["test_pats"]
    }
    latent_pt_file = "latent_health_vectors.pt"
    torch.save(latent_export, latent_pt_file)
    print(f"  >> Saved window latent vectors: '{latent_pt_file}' "
          f"({os.path.getsize(latent_pt_file) / 1024 / 1024:.2f} MB)")

    # Aggregate Patient-Level Latent Health Vectors (mean over longitudinal windows)
    print("  • Aggregating patient-level mean latent health vectors...")
    all_rows = []
    splits   = [
        ("train", mu_train, tensor_data["meta_train"]),
        ("val",   mu_val,   tensor_data["meta_val"]),
        ("test",  mu_test,  tensor_data["meta_test"])
    ]
    for split_name, mu_mat, meta_list in splits:
        mu_np = mu_mat.numpy()
        for idx, item in enumerate(meta_list):
            row = {
                "patient_id": item["patient_id"],
                "split":      split_name,
                "start_date": item["start_date"],
                "end_date":   item["end_date"]
            }
            for d in range(latent_dim):
                row[f"latent_z_{d:02d}"] = round(float(mu_np[idx, d]), 5)
            all_rows.append(row)

    df_latents  = pd.DataFrame(all_rows)
    latent_cols = [f"latent_z_{d:02d}" for d in range(latent_dim)]
    agg_funcs   = {col: "mean" for col in latent_cols}
    agg_funcs["split"] = "first"
    patient_summary = df_latents.groupby("patient_id", as_index=False).agg(agg_funcs)
    patient_summary = patient_summary[["patient_id", "split"] + latent_cols]

    summary_csv = "patient_latent_health_summary.csv"
    patient_summary.to_csv(summary_csv, index=False)
    print(f"  >> Saved patient latent summary: '{summary_csv}' "
          f"({len(patient_summary)} patients)")

    print("\n" + "=" * 75)
    print("PHASE 2 COMPLETE: LATENT HEALTH VECTORS EXTRACTED & SAVED SUCCESSFULLY!")
    print("=" * 75)
    print(f"  • Total Windows Encoded:        "
          f"{len(mu_train) + len(mu_val) + len(mu_test):,}")
    print(f"  • Latent Space Dimensionality:  {latent_dim} features per health state")
    print(f"  • Latent Coordinate Mean/Std:   "
          f"mu={mu_train.mean():.4f}, std={mu_train.std():.4f}")
    print("  >> Scope Verification: Latent health representations produced "
          "without trajectory prediction.")
    return latent_export


# -----------------------------------------------------------------------------
# Main Execution Entrypoint
# -----------------------------------------------------------------------------
def run_phase2_pipeline():
    # 1. Prepare Model Inputs (canonical ordering, windowing, splitting)
    tensor_data = prepare_model_inputs()
    # 2. Train Two-Branch Compact VAE (KL annealing + early stopping)
    model, ckpt_path = train_compact_vae(tensor_data=tensor_data, num_epochs=80)
    # 3. Extract & Save Latent Vectors
    extract_and_save_latent_vectors(checkpoint_path=ckpt_path)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_phase2_pipeline()
