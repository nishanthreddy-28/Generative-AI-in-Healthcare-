"""Authorized isolated CGM-only release: fresh raw aggregates, patient splits,
train-fitted scaler, VAE/GRU candidates, validation selection and calibration.

No laboratory/diagnosis features are used: their original unit/onset conventions
are unresolved. Reports still supply the separate evidence-grounded history.
"""

import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from healthcare_app.cgm import aggregate_cgm, CORE
from healthcare_app.trajectory_v2 import PatternVAE, PatternForecaster, baseline, constrain_core

ROOT = Path(__file__).resolve().parents[1]


def raw_daily(source, output):
    closed, carry, pieces, audit = set(), pd.DataFrame(), [], {"raw_rows": 0, "exact_duplicates_removed": 0, "days_excluded_for_low_coverage": 0, "conflicting_timestamps_discarded": 0, "readings_combined_in_bins": 0, "patients_rejected": []}
    def process(patient, frame):
        converted = pd.DataFrame({"timestamp": frame.Measurement_date + " " + frame.Measurement_time, "glucose_mg_dl": frame.Measurement})
        try:
            daily, quality = aggregate_cgm(converted, conflict_policy="discard")
        except ValueError as exc:
            audit["patients_rejected"].append({"patient": patient, "reason": str(exc)})
            return
        daily["Patient_ID"] = patient
        pieces.append(daily)
        for key in ("exact_duplicates_removed", "days_excluded_for_low_coverage", "conflicting_timestamps_discarded", "readings_combined_in_bins"):
            audit[key] += quality[key]
    for chunk in pd.read_csv(source, chunksize=500000, dtype={"Patient_ID": str, "Measurement_date": str, "Measurement_time": str}):
        audit["raw_rows"] += len(chunk)
        combined = pd.concat([carry, chunk], ignore_index=True)
        last = combined.Patient_ID.iloc[-1]
        for patient, frame in combined[combined.Patient_ID != last].groupby("Patient_ID", sort=False):
            if patient in closed:
                raise ValueError("Raw file is not grouped by patient; streaming aggregation refuses an incomplete grouping.")
            process(patient, frame)
            closed.add(patient)
        carry = combined[combined.Patient_ID == last].copy()
        print("Raw rows processed:", audit["raw_rows"], flush=True)
    if not carry.empty:
        patient = carry.Patient_ID.iloc[0]
        if patient in closed:
            raise ValueError("Patient order is inconsistent.")
        process(patient, carry)
    if not pieces:
        raise ValueError("No qualifying daily records survived quality checks.")
    daily = pd.concat(pieces, ignore_index=True).sort_values(["Patient_ID", "date"])
    daily.to_csv(output / "daily_cgm.csv", index=False)
    (output / "data_quality.json").write_text(json.dumps(audit, indent=2))
    return daily


def windows(daily):
    entries = {}
    for patient, frame in daily.groupby("Patient_ID", sort=True):
        days = pd.to_datetime(frame.date)
        segments = days.diff().dt.days.ne(1).cumsum()
        rows = []
        for _, segment in frame.groupby(segments):
            values = segment[CORE].to_numpy(dtype=np.float32)
            for start in range(0, len(values) - 89, 15):
                rows.append((values[start:start + 30], values[start + 30:start + 90]))
        if rows:
            entries[patient] = rows
    return entries


def train(output, epochs=20):
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.use_deterministic_algorithms(True)
    daily = raw_daily(ROOT / "Phase_1_Data_Preparation/raw_data/Glucose_measurements.csv", output)
    entries = windows(daily)
    patients = np.array(sorted(entries))
    rng = np.random.default_rng(42)
    rng.shuffle(patients)
    if len(patients) < 80:
        raise ValueError("Too few qualifying patients for separate training/selection/calibration/test groups.")
    a, b, c = int(len(patients) * .60), int(len(patients) * .75), int(len(patients) * .875)
    splits = dict(train=patients[:a], validation=patients[a:b], calibration=patients[b:c], test=patients[c:])
    manifest = {k: v.tolist() for k, v in splits.items()}
    (output / "patient_splits.json").write_text(json.dumps(manifest, indent=2))
    train_values = daily[daily.Patient_ID.isin(splits["train"])][CORE].to_numpy(dtype=float)
    mean, std = train_values.mean(axis=0), train_values.std(axis=0)
    std = np.maximum(std, 1e-3)
    bundle = {}
    for name, ids in splits.items():
        rows, owners = [], []
        for patient in ids:
            candidates = entries[patient]
            # Training/validation can overlap internally; calibration and test
            # have only one window per patient, so interval scores have a
            # patient-level independent unit rather than thousands of repeats.
            if name in {"calibration", "test"}:
                candidates = candidates[-1:]
            elif len(candidates) > 40:
                candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, 40).astype(int)]
            rows.extend(candidates)
            owners.extend([patient] * len(candidates))
        x = np.stack([r[0] for r in rows]); y = np.stack([r[1] for r in rows])
        bundle[name] = (x, y, np.array(owners))
    x_train = torch.tensor((bundle["train"][0] - mean) / std, dtype=torch.float32)
    x_val = torch.tensor((bundle["validation"][0] - mean) / std, dtype=torch.float32)
    vae = PatternVAE()
    optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    best, state, history = float("inf"), None, []
    loader = DataLoader(TensorDataset(x_train), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(42))
    for epoch in range(epochs):
        vae.train()
        for (x,) in loader:
            optimizer.zero_grad()
            reconstruction, mu, logvar = vae(x)
            loss = (reconstruction - x).square().mean() + .001 * (-.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(vae.parameters(), 1); optimizer.step()
        vae.eval()
        with torch.inference_mode():
            reconstruction, _, _ = vae(x_val)
            value = float((reconstruction - x_val).square().mean())
        history.append({"epoch": epoch + 1, "validation_reconstruction_mse": value})
        if value < best:
            best, state = value, {k: v.detach().clone() for k, v in vae.state_dict().items()}
        print("VAE", epoch + 1, value, flush=True)
    vae.load_state_dict(state); vae.eval()
    torch.save(vae.state_dict(), output / "pattern_vae.pt")
    with torch.inference_mode():
        z_train, _ = vae.encode(x_train); z_val, _ = vae.encode(x_val)
    model = PatternForecaster()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    y_train = torch.tensor((bundle["train"][1] - mean) / std, dtype=torch.float32)
    y_val = torch.tensor((bundle["validation"][1] - mean) / std, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train, z_train.clone(), y_train), batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(43))
    best, state, forecast_history = float("inf"), None, []
    for epoch in range(epochs):
        model.train()
        for x, z, y in loader:
            optimizer.zero_grad()
            loss = (model(x, z) - y).square().mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        model.eval()
        with torch.inference_mode():
            value = float((model(x_val, z_val) - y_val).square().mean())
        forecast_history.append({"epoch": epoch + 1, "validation_mse": value})
        if value < best:
            best, state = value, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print("Forecaster", epoch + 1, value, flush=True)
    model.load_state_dict(state); model.eval()
    torch.save(model.state_dict(), output / "pattern_forecaster.pt")
    def predict(x, name):
        if name != "vae_gru":
            return constrain_core(baseline(x, name))
        tensor = torch.tensor((x - mean) / std, dtype=torch.float32)
        with torch.inference_mode():
            z, _ = vae.encode(tensor)
            result = model(tensor, z).numpy() * std + mean
        return constrain_core(result)
    candidates = ["last_day", "past_7_day_mean", "past_30_day_mean", "vae_gru"]
    validation_scores = {}
    for name in candidates:
        prediction = predict(bundle["validation"][0], name)
        error = np.abs(prediction - bundle["validation"][1]) / std
        validation_scores[name] = float(np.mean([error[bundle["validation"][2] == patient].mean() for patient in splits["validation"]]))
    selected = min(validation_scores, key=validation_scores.get)
    x_cal, y_cal, _ = bundle["calibration"]
    residual = y_cal - predict(x_cal, selected)
    # Split conformal per-horizon/target, one score per calibration patient.
    n = len(residual); order = min(n - 1, int(np.ceil((n + 1) * .8)) - 1)
    radius = np.sort(np.abs(residual), axis=0)[order]
    centered = residual.reshape(n, -1) - residual.reshape(n, -1).mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    rank = min(12, n - 1)
    factor = vectors[:rank].T * singular[:rank] / np.sqrt(n - 1)
    diagonal = np.sqrt(np.maximum(centered.var(axis=0, ddof=1) - (factor ** 2).sum(axis=1), 0))
    np.savez_compressed(output / "calibration.npz", radius=radius, factor=factor, diagonal=diagonal)
    metrics = {}
    for name in candidates:
        x, y, _ = bundle["test"]
        prediction = predict(x, name)
        error = prediction - y
        metrics[name] = {"mae": dict(zip(CORE, np.abs(error).mean(axis=(0, 1)).tolist())),
                         "rmse": dict(zip(CORE, np.sqrt((error ** 2).mean(axis=(0, 1))).tolist())),
                         "horizon_mae": {str(h): dict(zip(CORE, np.abs(error[:, h - 1]).mean(axis=0).tolist())) for h in (1, 7, 30, 60)}}
        if name == selected:
            metrics[name]["nominal_80_interval_coverage"] = dict(zip(CORE, (np.abs(error) <= radius).mean(axis=(0, 1)).tolist()))
    with torch.inference_mode():
        tensor = torch.tensor((bundle["test"][0] - mean) / std, dtype=torch.float32)
        reconstruction, mu, _ = vae(tensor)
        vae_test = {"normalized_reconstruction_mse": float((reconstruction - tensor).square().mean()), "latent_std_mean": float(mu.std(dim=0).mean())}
    metadata = {"version": 2, "input_features": CORE, "history_days": 30, "future_days": 60, "mean": mean.tolist(), "std": std.tolist(),
                "selected_model": selected, "validation_scores": validation_scores, "test_metrics": metrics, "vae_test": vae_test,
                "window_counts": {k: len(v[0]) for k, v in bundle.items()}, "patient_counts": {k: len(v) for k, v in splits.items()},
                "epochs": epochs, "seed": 42, "seconds": time.perf_counter() - started,
                "source": "Fresh raw T1DiabetesGranada CGM aggregates; no original cleaned labs/scalers used",
                "status": "Patient-held-out internal research evaluation; no external clinical validation or causal treatment modeling",
                "interval": "Nominal 80% per-horizon/per-target split conformal; patient exchangeability assumption, not simultaneous or clinical coverage",
                "generation": "Zero-mean low-rank Gaussian residual sensitivity model fitted on calibration histories; not a learned treatment-response generator",
                "training_history": {"vae": history, "forecaster": forecast_history}}
    (output / "release.json").write_text(json.dumps(metadata, indent=2))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.name in {"release.json", "pattern_vae.pt", "pattern_forecaster.pt", "calibration.npz"}}
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2))
    print("Selected:", selected, "Test metrics:", json.dumps(metrics[selected]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=ROOT / "Phase_5_Application/models/cgm")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.outdir.exists() and any(args.outdir.iterdir()):
        parser.error("Choose a new or empty directory; original or prior artifacts are never overwritten.")
    if not 1 <= args.epochs <= 100:
        parser.error("Epoch budget must be between 1 and 100.")
    args.outdir.mkdir(parents=True, exist_ok=True)
    train(args.outdir, args.epochs)
