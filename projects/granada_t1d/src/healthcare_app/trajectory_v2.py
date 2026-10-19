"""Fresh CGM-pattern VAE and direct forecaster; optional torch dependency."""
import numpy as np
import torch
from torch import nn
from .cgm import CORE


class PatternVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(4, 32, batch_first=True)
        self.mu = nn.Linear(32, 16)
        self.logvar = nn.Linear(32, 16)
        self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 120))

    def encode(self, x):
        _, hidden = self.encoder(x)
        return self.mu(hidden[-1]), self.logvar(hidden[-1]).clamp(-10, 6)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z).reshape(-1, 30, 4), mu, logvar


class PatternForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.history_encoder = nn.GRU(4, 32, batch_first=True)
        self.head = nn.Sequential(nn.Linear(52, 96), nn.ReLU(), nn.Linear(96, 240))

    def forward(self, x, z):
        _, hidden = self.history_encoder(x)
        base = x[:, -7:].mean(dim=1)
        return base[:, None, :] + self.head(torch.cat([hidden[-1], z, base], dim=-1)).reshape(-1, 60, 4)


def constrain_core(values):
    result = np.array(values, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite values; output withheld.")
    result[..., 0] = np.clip(result[..., 0], 40, 500)
    result[..., 1:3] = np.clip(result[..., 1:3], 0, 100)
    total = result[..., 1:3].sum(axis=-1, keepdims=True)
    result[..., 1:3] *= np.minimum(1, 100 / np.maximum(total, 1e-8))
    result[..., 3] = np.clip(result[..., 3], 0, 150)
    return result


def expanded_metrics(core):
    core = constrain_core(core)
    return np.concatenate([core, (100 - core[..., 1] - core[..., 2])[..., None],
                           (3.31 + 0.02392 * core[..., 0])[..., None]], axis=-1)


def baseline(x, name):
    length = {"last_day": 1, "past_7_day_mean": 7, "past_30_day_mean": 30}[name]
    return np.repeat(x[:, -length:].mean(axis=1)[:, None, :], 60, axis=1)


class CorrectedSimulator:
    def __init__(self, directory=None):
        import hashlib
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        self.directory = Path(directory or root / "Phase_5_Application/models/cgm/release_001")
        manifest = json.loads((self.directory / "manifest.json").read_text())
        required = {"release.json", "pattern_vae.pt", "pattern_forecaster.pt", "calibration.npz"}
        if set(manifest) != required:
            raise ValueError("The corrected release manifest is incomplete.")
        for name in required:
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != manifest[name]:
                raise ValueError("A corrected model artifact changed; output withheld.")
        self.metadata = json.loads((self.directory / "release.json").read_text())
        if self.metadata["input_features"] != CORE or self.metadata["selected_model"] not in {"last_day", "past_7_day_mean", "past_30_day_mean", "vae_gru"}:
            raise ValueError("Unsupported corrected release representation.")
        self.mean = np.asarray(self.metadata["mean"])
        self.std = np.asarray(self.metadata["std"])
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid corrected training scaler.")
        self.vae, self.forecaster = PatternVAE(), PatternForecaster()
        self.vae.load_state_dict(torch.load(self.directory / "pattern_vae.pt", weights_only=True, map_location="cpu"))
        self.forecaster.load_state_dict(torch.load(self.directory / "pattern_forecaster.pt", weights_only=True, map_location="cpu"))
        self.vae.eval(); self.forecaster.eval()
        with np.load(self.directory / "calibration.npz", allow_pickle=False) as arrays:
            self.radius, self.factor, self.diagonal = [arrays[k].copy() for k in ("radius", "factor", "diagonal")]
        if self.radius.shape != (60, 4) or self.factor.shape[0] != 240 or self.diagonal.shape != (240,) or not all(np.isfinite(a).all() for a in (self.radius, self.factor, self.diagonal)):
            raise ValueError("Invalid calibration artifact.")
        self.fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()

    def run(self, daily, n_paths=20, seed=42):
        from .research import latest_contiguous_window, validate_count
        import pandas as pd
        validate_count(n_paths)
        window = latest_contiguous_window(daily)
        x = window[CORE].to_numpy(dtype=float)
        if x.shape != (30, 4) or not np.isfinite(x).all() or (x[:, 0] < 40).any() or (x[:, 0] > 500).any() or (x[:, 1:] < 0).any() or (x[:, 1:3].sum(axis=1) > 100.01).any():
            raise ValueError("Invalid observed CGM aggregates.")
        normalized = (x - self.mean) / self.std
        if (np.abs(normalized) > 8).any():
            raise ValueError("CGM history is far outside the corrected training representation; simulation withheld.")
        tensor = torch.tensor(normalized[None], dtype=torch.float32)
        with torch.inference_mode():
            z, _ = self.vae.encode(tensor)
            if not torch.isfinite(z).all():
                raise ValueError("Invalid latent representation; output withheld.")
            if self.metadata["selected_model"] == "vae_gru":
                point = self.forecaster(tensor, z).numpy()[0] * self.std + self.mean
            else:
                point = baseline(x[None], self.metadata["selected_model"])[0]
        point = constrain_core(point)
        rng = np.random.default_rng(seed)
        correlated = rng.normal(size=(n_paths, self.factor.shape[1])) @ self.factor.T
        independent = rng.normal(size=(n_paths, 240)) * self.diagonal
        core_paths = constrain_core(point[None] + (correlated + independent).reshape(n_paths, 60, 4))
        paths = expanded_metrics(core_paths)
        columns = CORE + ["tar_above_180", "gmi_estimated_a1c"]
        interval_low, interval_high = point - self.radius, point + self.radius
        interval_low[:, 0] = np.maximum(interval_low[:, 0], 40); interval_high[:, 0] = np.minimum(interval_high[:, 0], 500)
        interval_low[:, 1:] = np.maximum(interval_low[:, 1:], 0)
        interval_high[:, 1:3] = np.minimum(interval_high[:, 1:3], 100)
        interval_high[:, 3] = np.minimum(interval_high[:, 3], 150)
        from datetime import date, timedelta
        first_future_date = date.fromisoformat(str(window.date.iloc[-1])) + timedelta(days=1)
        dates = pd.date_range(str(first_future_date), periods=60, freq="D")
        from .knowledge import KnowledgeBase
        evidence = KnowledgeBase().retrieve("ADA CGM time in range glycemic assessment glucose", top_k=3)
        note = "CORRECTED CGM RESEARCH SIMULATION — not a treatment recommendation\n"
        note += f"Observed window: {window.date.iloc[0]} to {window.date.iloc[-1]}.\n"
        note += f"Latest qualifying uploaded daily observation: {daily.date.max()}. Simulation dates start after the selected window, not after today's date.\n"
        note += f"Observed 30-day average of daily mean glucose: {x[:, 0].mean():.2f} mg/dL.\n"
        note += f"Selected algorithm from validation comparison: {self.metadata['selected_model']}.\n"
        note += f"Simulated next-60-day central mean glucose: {point[:, 0].mean():.2f} mg/dL. This is simulated, not measured.\n"
        note += "Intervals are marginal research calibration estimates under patient-exchangeability assumptions. They are not individual clinical probabilities, simultaneous trajectory guarantees or external validation. Medication, diet, illness and treatment changes are not modeled. GMI is derived, not measured HbA1c.\n"
        note += "\n".join(f"[{e['id']}] {e['content']} Source: {e['url']}" for e in evidence)
        return {"window_dates": (window.date.iloc[0], window.date.iloc[-1]), "selected_model": self.metadata["selected_model"],
                "observed_window": pd.DataFrame(x, columns=CORE, index=pd.to_datetime(window.date).to_numpy()),
                "latest_available_daily_date": str(daily.date.max()),
                "point": pd.DataFrame(expanded_metrics(point), columns=columns, index=dates),
                "baseline_point": pd.DataFrame(expanded_metrics(baseline(x[None], "past_30_day_mean")[0]), columns=columns, index=dates),
                "core_interval_low": pd.DataFrame(interval_low, columns=CORE, index=dates),
                "core_interval_high": pd.DataFrame(interval_high, columns=CORE, index=dates),
                "simulations": paths, "columns": columns, "latent_z": z.numpy()[0], "note": note,
                "metadata": self.metadata, "artifact_fingerprint": self.fingerprint, "seed": seed, "evidence": evidence,
                "observed_mean_glucose": float(x[:, 0].mean())}
