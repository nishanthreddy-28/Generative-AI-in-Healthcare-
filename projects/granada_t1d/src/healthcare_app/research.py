"""Explicitly experimental cohort inference with aligned windows and trusted artifacts."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def trusted_checkpoint(path):
    import torch
    path = Path(path).resolve()
    manifest = json.loads((Path(__file__).with_name("artifact_manifest.json")).read_text())
    relative = path.relative_to(ROOT).as_posix()
    if relative not in manifest or hashlib.sha256(path.read_bytes()).hexdigest() != manifest[relative]:
        raise ValueError("Model artifact hash is unknown or changed. Validate it before enabling research inference.")
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.StrDType]):
        return torch.load(path, map_location="cpu", weights_only=True)


def latest_contiguous_window(frame, length=30):
    if frame.empty:
        raise ValueError("Patient was not found in the supplied timeline.")
    work = frame.copy()
    work["date_dt"] = pd.to_datetime(work["date"], format="%Y-%m-%d", errors="raise")
    if work["date_dt"].isna().any() or work["date_dt"].duplicated().any():
        raise ValueError("Timeline dates must be valid and unique.")
    work = work.sort_values("date_dt").reset_index(drop=True)
    groups = work["date_dt"].diff().dt.days.ne(1).cumsum()
    for _, segment in reversed(list(work.groupby(groups))):
        if len(segment) >= length:
            return segment.tail(length).reset_index(drop=True)
    raise ValueError(f"At least {length} consecutive daily observations are required. Sparse lab reports do not meet this requirement.")


def coherent_outputs(values, columns):
    """Presentation constraints, not a correction of model validity/calibration."""
    result = np.asarray(values, dtype=np.float32).copy()
    if result.shape[-1] != len(columns) or not np.isfinite(result).all():
        raise ValueError("Model outputs have invalid shape or non-finite values.")
    for i, col in enumerate(columns):
        if col in {"glucose_mean", "glucose_std", "glucose_min", "glucose_max", "nocturnal_mean"}:
            result[..., i] = np.clip(result[..., i], 0 if col == "glucose_std" else 40, 500)
        elif col == "nocturnal_hypo_count":
            result[..., i] = np.maximum(result[..., i], 0)  # 30-day sensor-low exposure, not episodes
        else:
            result[..., i] = np.clip(result[..., i], 0, 100)
    ranges = ["tir_70_180", "tbr_below_70", "tar_above_180"]
    if all(c in columns for c in ranges):
        indices = [columns.index(c) for c in ranges]
        sub = result[..., indices]
        total = sub.sum(axis=-1, keepdims=True)
        sub = np.where(total > 1e-6, sub / np.maximum(total, 1e-6) * 100, np.array([100, 0, 0]))
        result[..., indices] = sub
    for subset, parent in [("tbr_severe_54", "tbr_below_70"), ("tar_severe_250", "tar_above_180")]:
        if subset in columns and parent in columns:
            result[..., columns.index(subset)] = np.minimum(result[..., columns.index(subset)], result[..., columns.index(parent)])
    if "gmi_estimated_a1c" in columns and "glucose_mean" in columns:
        # Bergenstal et al., Diabetes Care 2018;41:2275. GMI is not measured HbA1c.
        result[..., columns.index("gmi_estimated_a1c")] = 3.31 + 0.02392 * result[..., columns.index("glucose_mean")]
    return result


def validate_count(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ValueError("Simulation count must be an integer between 1 and 100.")


class ResearchPipeline:
    def __init__(self, load_cohort=True):
        import torch
        torch.set_num_threads(2)
        self.device = torch.device("cpu")
        self.df = pd.DataFrame()
        self.meta_df = pd.DataFrame()
        if load_cohort:
            self.df = pd.read_csv(ROOT / "Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv")
            self.df["date_dt"] = pd.to_datetime(self.df["date"])
            self.meta_df = pd.read_csv(ROOT / "Phase_1_Data_Preparation/cleaned_cohort_metadata.csv")
        self.scaler = json.loads((ROOT / "Phase_2_Latent_VAE/scaler_parameters.json").read_text())
        self.input_cols = self.scaler["input_cols"]
        self.feat_mean = np.asarray(self.scaler["feat_mean"], dtype=np.float32)
        self.feat_std = np.asarray(self.scaler["feat_std"], dtype=np.float32)
        if not np.isfinite(self.feat_mean).all() or not np.isfinite(self.feat_std).all() or (self.feat_std <= 0).any():
            raise ValueError("Invalid model scaler.")
        vae_mod = source_module("health_vae_runtime", "Phase_2_Latent_VAE/phase2_vae.py")
        gan_mod = source_module("health_gan_runtime", "Phase_3_Generative_Forecasting/phase3_ctgan.py")
        forecast_mod = source_module("health_forecast_runtime", "Phase_3_Generative_Forecasting/phase3_4_trajectory_forecasting.py")
        self.vae_ckpt = trusted_checkpoint(ROOT / "Phase_2_Latent_VAE/best_compact_vae.pt")
        self.vae = vae_mod.TwoBranchHealthVAE(**{k: self.vae_ckpt[k] for k in ("n_dyn", "n_stat", "n_pure_dynamic", "latent_dim")})
        self.vae.load_state_dict(self.vae_ckpt["model_state_dict"])
        self.vae.eval()
        self.ctgan_ckpt_path = ROOT / "Phase_3_Generative_Forecasting/best_ctgan_generator.pt"
        self.ctgan_ckpt = trusted_checkpoint(self.ctgan_ckpt_path)
        self.generator = gan_mod.CTGANTabularGenerator(**{k: self.ctgan_ckpt[k] for k in ("cond_dim", "noise_dim", "target_dim")})
        self.generator.load_state_dict(self.ctgan_ckpt["generator_state_dict"])
        self.generator.eval()
        self.forecaster_path = ROOT / "Phase_3_Generative_Forecasting/best_trajectory_forecaster.pt"
        self.fc_ckpt = trusted_checkpoint(self.forecaster_path)
        self.forecaster = forecast_mod.MultiModalTrajectoryForecaster(**{k: self.fc_ckpt[k] for k in ("target_dim", "latent_dim", "hidden_dim", "out_len")})
        self.forecaster.load_state_dict(self.fc_ckpt["model_state_dict"])
        self.forecaster.eval()

    def run_inference_for_patient(self, patient_id=None, n_simulations=20, output_dir=None, seed=42):
        validate_count(n_simulations)
        if self.df.empty:
            raise ValueError("Cohort access was not enabled. Upload a complete patient-owned daily feature history.")
        if patient_id is None:
            patient_id = sorted(self.df.Patient_ID.unique())[0]
        if not isinstance(patient_id, str) or patient_id not in set(self.df.Patient_ID):
            raise ValueError("Choose a patient ID present in the supplied research cohort.")
        p_df = self.df[self.df.Patient_ID == patient_id]
        window = latest_contiguous_window(p_df)
        return self._run_window(window, patient_id, str(p_df.date.max()), n_simulations, output_dir, seed)

    def run_uploaded_history(self, frame, n_simulations=20, seed=42, output_dir=None):
        from .model_inputs import validate_daily_frame
        validate_count(n_simulations)
        work = validate_daily_frame(frame)
        window = latest_contiguous_window(work)
        normalized = (window[self.input_cols].to_numpy(dtype=float) - self.feat_mean) / self.feat_std
        if not np.isfinite(normalized).all() or (np.abs(normalized) > 12).any():
            raise ValueError("Input is far outside the supplied scaler representation. Verify units/profile; simulation withheld.")
        result = self._run_window(window, "Uploaded patient", str(work.date.max()), n_simulations, output_dir, seed)
        result["input_source"] = "Patient-owned uploaded daily feature CSV; no cohort identity or profile substituted"
        return result

    def _run_window(self, window, patient_id, latest_observation, n_simulations, output_dir, seed):
        import torch
        raw = window[self.input_cols].to_numpy(dtype=np.float32)
        if raw.shape != (30, len(self.input_cols)) or not np.isfinite(raw).all():
            raise ValueError("The selected model window contains invalid or missing input values.")
        norm = (raw - self.feat_mean) / self.feat_std
        profile = window.iloc[0][self.ctgan_ckpt["static_cols"]].to_dict()
        profile["Patient_ID"] = patient_id
        static = np.asarray([profile[col] for col in self.ctgan_ckpt["static_cols"]], dtype=np.float32)
        generator_rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.inference_mode():
            mu, _ = self.vae.encode(torch.from_numpy(norm[:, :self.vae_ckpt["n_dyn"]]).unsqueeze(0), torch.from_numpy(norm[0, self.vae_ckpt["n_dyn"]:]).unsqueeze(0))
            z = mu.numpy()[0]
            c = self.ctgan_ckpt
            norm_static = (static - c["static_mean"].numpy()) / c["static_std"].numpy()
            condition = torch.from_numpy(np.concatenate([z, norm_static])).unsqueeze(0).repeat(n_simulations, 1)
            generated = self.generator(condition, torch.randn(n_simulations, c["noise_dim"], generator=generator_rng)).numpy()
            raw_tabular = generated * c["target_std"].numpy() + c["target_mean"].numpy()
            samples = coherent_outputs(raw_tabular, c["target_cols"])
            f = self.fc_ckpt
            observed = window[f["target_cols"]].iloc[-1].to_numpy(dtype=np.float32)
            x = torch.from_numpy((observed - f["target_mean"].numpy()) / f["target_std"].numpy()).unsqueeze(0).repeat(n_simulations, 1)
            z_paths = mu.repeat(n_simulations, 1) + torch.randn(n_simulations, f["latent_dim"], generator=generator_rng) * 0.03
            raw_paths = self.forecaster(z_paths, x).numpy() * f["target_std"].numpy() + f["target_mean"].numpy()
        paths = coherent_outputs(raw_paths, f["target_cols"])
        trajectory = {
            "p10": np.percentile(paths, 10, axis=0), "p50": coherent_outputs(np.percentile(paths, 50, axis=0), f["target_cols"]),
            "p90": np.percentile(paths, 90, axis=0), "simulations": paths, "raw_simulations": raw_paths,
            "target_cols": f["target_cols"], "band_type": "Uncalibrated latent-perturbation sensitivity; not clinical confidence intervals",
        }
        rag_mod = source_module("health_rag_runtime", "Phase_4_RAG_Support/rag_clinical_engine.py")
        synth = rag_mod.ClinicalDecisionSynthesizer(rag_mod.ADAGuidelinesRAG())
        doctor, evidence, _ = synth.generate_doctor_report(profile, trajectory["p50"], use_llm=False)
        guide, _, _ = synth.generate_patient_guidance_report(profile, trajectory["p50"], use_llm=False)
        result = {
            "patient_id": patient_id, "window_dates": (str(window.date.iloc[0]), str(window.date.iloc[-1])), "pat_profile": profile,
            "latest_observation": latest_observation, "latent_z": z, "ctgan_projections": dict(zip(c["target_cols"], samples.mean(axis=0).astype(float))),
            "ctgan_samples": pd.DataFrame(samples, columns=c["target_cols"]), "raw_ctgan_samples": raw_tabular,
            "trajectory_simulation": trajectory, "retrieved_guidelines": evidence, "doctor_note": doctor, "patient_guide": guide,
            "seed": seed, "model_status": "Experimental: inherited preprocessing leakage and poor interval calibration remain; no model was retrained.",
        }
        if output_dir:
            output = Path(output_dir)
            if output.exists() and any(output.iterdir()):
                raise ValueError("Output directory must be new or empty; existing results will not be overwritten.")
            output.mkdir(parents=True, exist_ok=True)
            # Patient IDs are restricted to known cohort IDs; neutral filenames avoid PHI in paths.
            (output / "research_note.txt").write_text(doctor, encoding="utf-8")
            (output / "research_patient_guide.txt").write_text(guide, encoding="utf-8")
            result["ctgan_samples"].to_csv(output / "research_synthetic_states.csv", index=False)
            pd.DataFrame(trajectory["p50"], columns=f["target_cols"]).to_csv(output / "research_trajectory.csv", index=False)
        return result
