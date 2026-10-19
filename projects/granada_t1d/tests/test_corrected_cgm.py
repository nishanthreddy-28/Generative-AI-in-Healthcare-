import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from healthcare_app.cgm import aggregate_cgm, read_cgm_upload, CORE
from healthcare_app.trajectory_v2 import CorrectedSimulator

ROOT = Path(__file__).resolve().parents[1]


def measured_frame(days=30):
    timestamps = pd.date_range("2026-09-01", periods=days * 96, freq="15min")
    values = 150 + 30 * np.sin(np.arange(len(timestamps)) / 8) + 5 * np.sin(np.arange(len(timestamps)) / 96)
    return pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"), "glucose_mg_dl": values})


def test_observed_binning_duplicates_and_sparse_coverage():
    frame = measured_frame()
    duplicate = pd.concat([frame, frame.iloc[:5]], ignore_index=True)
    daily, quality = aggregate_cgm(duplicate)
    assert quality["exact_duplicates_removed"] == 5 and quality["interpolated_readings"] == 0
    assert len(daily) == 30 and daily.readings_count.eq(96).all()
    dense = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=96, freq="1min").strftime("%Y-%m-%d %H:%M:%S"), "glucose_mg_dl": 140})
    aggregated, quality = aggregate_cgm(dense)
    assert aggregated.empty and quality["days_excluded_for_low_coverage"] == 1


def test_conflicting_timestamps_rejected_or_audited_discard():
    frame = measured_frame()
    conflict = frame.iloc[:1].copy(); conflict.glucose_mg_dl = 200
    combined = pd.concat([frame, conflict], ignore_index=True)
    with pytest.raises(ValueError, match="Conflicting"):
        aggregate_cgm(combined)
    daily, quality = aggregate_cgm(combined, conflict_policy="discard")
    assert quality["conflicting_timestamps_discarded"] == 1 and daily.readings_count.iloc[0] == 95


@pytest.mark.parametrize("data", [b"", b"timestamp,glucose_mg_dl\n2026-01-01 00:00:00,5\n", b"timestamp,glucose_mg_dl\n01/01/2026,150\n", b"name,timestamp,glucose_mg_dl\nPrivate,2026-01-01 00:00:00,150\n", b"timestamp,glucose_mg_dl\n2026-01-01 00:00:00,nan\n"])
def test_bad_or_sparse_cgm_rejected(data):
    with pytest.raises(ValueError):
        read_cgm_upload(data)


@pytest.fixture(scope="module")
def corrected():
    return CorrectedSimulator()


def test_corrected_inference_is_finite_reproducible_and_coherent(corrected):
    daily, _ = read_cgm_upload(measured_frame().to_csv(index=False).encode())
    a = corrected.run(daily, 10); b = corrected.run(daily, 10)
    assert a["point"].shape == (60, 6) and a["simulations"].shape == (10, 60, 6)
    assert a["latent_z"].shape == (16,) and np.isfinite(a["latent_z"]).all()
    np.testing.assert_array_equal(a["simulations"], b["simulations"])
    assert np.isfinite(a["simulations"]).all()
    assert a["point"].index[0] == pd.Timestamp("2026-10-01")
    assert (a["core_interval_low"] <= a["core_interval_high"]).all().all()
    np.testing.assert_allclose(a["simulations"][..., 1] + a["simulations"][..., 2] + a["simulations"][..., 4], 100)
    np.testing.assert_allclose(a["simulations"][..., 5], 3.31 + .02392 * a["simulations"][..., 0])
    assert "not measured" in a["note"] and "Source:" in a["note"]


def test_new_scaler_uses_training_patients_only_and_disjoint_splits():
    directory = ROOT / "Phase_5_Application/models/cgm/release_001"
    splits = json.loads((directory / "patient_splits.json").read_text())
    sets = [set(v) for v in splits.values()]
    assert sum(len(v) for v in sets) == len(set.union(*sets))
    daily = pd.read_csv(directory / "daily_cgm.csv")
    training = daily[daily.Patient_ID.isin(splits["train"])][CORE]
    metadata = json.loads((directory / "release.json").read_text())
    np.testing.assert_allclose(training.mean().to_numpy(), metadata["mean"])
    np.testing.assert_allclose(training.std(ddof=0).to_numpy(), metadata["std"])
    assert metadata["window_counts"]["calibration"] == metadata["patient_counts"]["calibration"]
    assert metadata["window_counts"]["test"] == metadata["patient_counts"]["test"]
    assert metadata["selected_model"] == min(metadata["validation_scores"], key=metadata["validation_scores"].get)


def test_bad_count_and_extreme_history_withheld(corrected):
    daily, _ = aggregate_cgm(measured_frame())
    with pytest.raises(ValueError): corrected.run(daily, 0)
    daily["glucose_cv"] = 1e8
    with pytest.raises(ValueError, match="outside"):
        corrected.run(daily, 10)


def test_older_qualifying_window_is_explicit_and_plot_uses_that_window(corrected):
    daily, _ = aggregate_cgm(measured_frame())
    recent = daily.tail(10).copy()
    recent["date"] = pd.date_range("2026-11-01", periods=10).strftime("%Y-%m-%d")
    combined = pd.concat([daily, recent], ignore_index=True)
    result = corrected.run(combined, 10)
    assert result["window_dates"][1] == "2026-09-30"
    assert result["latest_available_daily_date"] == "2026-11-10"
    assert result["observed_window"].index[-1] == pd.Timestamp("2026-09-30")
    assert result["point"].index[0] == pd.Timestamp("2026-10-01")


def test_manifest_tampering_fails_closed(tmp_path):
    import shutil
    source = ROOT / "Phase_5_Application/models/cgm/release_001"
    for name in ("release.json", "manifest.json", "pattern_vae.pt", "pattern_forecaster.pt", "calibration.npz"):
        shutil.copyfile(source / name, tmp_path / name)
    with (tmp_path / "pattern_vae.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        CorrectedSimulator(tmp_path)


def test_linked_simulation_has_distinct_provenance_and_summary_refresh(corrected):
    from healthcare_app.documents import extract_document
    from healthcare_app.analysis import build_analysis, attach_simulation, export_markdown
    from healthcare_app.summary import local_summary, render_summary
    analysis = build_analysis([extract_document(b"Glucose 146 mg/dL 70-99", "fictional.txt").to_dict()], user_reviewed=True)
    daily, _ = aggregate_cgm(measured_frame())
    result = corrected.run(daily, 10)
    attached = attach_simulation(analysis, result)
    assert attached["simulation_summary"]["kind"] == "experimental_simulation"
    assert "simulated, not measured" in export_markdown(attached)
    plan = local_summary(attached)
    assert "[S1]" in render_summary(attached, plan)
    assert "[S1]" not in render_summary(analysis, plan)  # stale claim plan cannot survive removal
