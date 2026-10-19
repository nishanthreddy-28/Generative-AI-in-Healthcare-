import numpy as np
import pandas as pd
import pytest
from healthcare_app.research import latest_contiguous_window, coherent_outputs, source_module, validate_count


def test_window_selects_latest_valid_segment_without_gaps():
    dates = list(pd.date_range("2026-01-01", periods=35)) + list(pd.date_range("2026-03-01", periods=10))
    frame = pd.DataFrame({"date": [d.strftime("%Y-%m-%d") for d in dates]})
    window = latest_contiguous_window(frame)
    assert window.date.iloc[-1] == "2026-02-04"
    assert window.date.iloc[0] == "2026-01-06"
    assert len(window) == 30


@pytest.mark.parametrize("dates", [[], ["2026-01-01"], ["2026-01-01"] * 30, ["invalid"] * 30])
def test_bad_timeline_rejected(dates):
    with pytest.raises(ValueError):
        latest_contiguous_window(pd.DataFrame({"date": dates}))


@pytest.mark.parametrize("n", [0, -1, True, 1.5, 101])
def test_invalid_simulation_count(n):
    with pytest.raises(ValueError):
        validate_count(n)


def test_constraints_units_and_subsets():
    columns = ["glucose_mean", "glucose_cv", "tir_70_180", "tbr_below_70", "tar_above_180", "tbr_severe_54", "gmi_estimated_a1c", "nocturnal_hypo_count"]
    raw = np.array([[140, 31, 85, 10, 20, 15, 9, 500]])
    result = coherent_outputs(raw, columns)[0]
    assert result[1] == 31  # CV is a percentage, never clipped as glucose
    assert result[2:5].sum() == pytest.approx(100)
    assert result[5] <= result[3]
    assert result[6] == pytest.approx(3.31 + 0.02392 * 140)
    assert result[7] == 500  # readings across 30 days, not capped at 100 episodes


def test_legacy_rag_repairs_retrieval_without_rewriting_index():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "Phase_4_RAG_Support/ada_guidelines_faiss.index"
    old = path.read_bytes()
    module = source_module("health_rag_test", "Phase_4_RAG_Support/rag_clinical_engine.py")
    rag = module.ADAGuidelinesRAG(index_path=str(path))
    assert rag.retrieve("glucose blood sugar")
    assert rag.retrieve("zxqv football") == []
    assert rag.index.d == len(rag.vectorizer.get_feature_names_out())
    assert old == path.read_bytes()
    with pytest.raises(FileNotFoundError):
        module.ADAGuidelinesRAG(guidelines_path=str(path.with_name("missing-source.json")))
    synthesizer = module.ClinicalDecisionSynthesizer(rag)
    answer, _, _ = synthesizer.chat_with_clinical_assistant({}, {"tbr_below_70": 0}, "How should I adjust insulin at night?")
    assert "10-15%" not in answer and "cannot" in answer


@pytest.fixture(scope="module")
def pipeline():
    from healthcare_app.research import ResearchPipeline
    return ResearchPipeline()


def test_real_model_pipeline_aligned_reproducible(pipeline):
    a = pipeline.run_inference_for_patient("LIB193266", 10)
    b = pipeline.run_inference_for_patient("LIB193266", 10)
    assert np.isfinite(a["latent_z"]).all() and a["latent_z"].shape == (16,)
    assert (pd.Timestamp(a["window_dates"][1]) - pd.Timestamp(a["window_dates"][0])).days == 29
    assert a["pat_profile"]["age"] != 45
    assert a["trajectory_simulation"]["p50"].shape == (60, 6)
    np.testing.assert_array_equal(a["trajectory_simulation"]["simulations"], b["trajectory_simulation"]["simulations"])
    assert a["retrieved_guidelines"] and "EXPERIMENTAL" in a["doctor_note"]
    assert "clinical confidence" in a["trajectory_simulation"]["band_type"]


def test_missing_patient_and_invalid_counts_fail(pipeline):
    with pytest.raises(ValueError):
        pipeline.run_inference_for_patient("NOT-A-PATIENT", 10)
    with pytest.raises(ValueError):
        pipeline.run_inference_for_patient("LIB193266", 0)
