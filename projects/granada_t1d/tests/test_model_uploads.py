import numpy as np
import pandas as pd
import pytest
from healthcare_app.model_inputs import model_schema, read_daily_upload, validate_daily_frame
from healthcare_app.research import ResearchPipeline, latest_contiguous_window


@pytest.fixture(scope="module")
def upload_pipeline():
    return ResearchPipeline(load_cohort=False)


@pytest.fixture(scope="module")
def compatible_frame():
    # Existing authorized cohort representation is used only as an integration
    # fixture; the application's uploaded route itself never reads the cohort.
    from pathlib import Path
    source = Path(__file__).resolve().parents[1] / "Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv"
    data = pd.read_csv(source)
    window = latest_contiguous_window(data[data.Patient_ID == "LIB193266"])
    return window[["date"] + model_schema()["input_cols"]].copy()


def test_uploaded_route_uses_own_frame_and_no_cohort(upload_pipeline, compatible_frame):
    assert upload_pipeline.df.empty and upload_pipeline.meta_df.empty
    frame = read_daily_upload(compatible_frame.to_csv(index=False).encode())
    result = upload_pipeline.run_uploaded_history(frame, 10)
    assert result["patient_id"] == "Uploaded patient" and "no cohort" in result["input_source"]
    assert result["window_dates"] == (frame.date.iloc[0], frame.date.iloc[-1])
    assert result["pat_profile"]["age"] == frame.age.iloc[0]
    assert np.isfinite(result["trajectory_simulation"]["p50"]).all()


@pytest.mark.parametrize("defect", ["missing", "identifier", "nan", "infinite", "negative", "duplicate_date", "short", "binary", "inconsistent_profile", "percentage", "subset", "glucose_unit", "gmi", "date_format"])
def test_invalid_uploaded_history_fails_closed(compatible_frame, defect):
    frame = compatible_frame.copy()
    if defect == "missing": frame = frame.drop(columns="lab_hba1c")
    elif defect == "identifier": frame["Patient_ID"] = "secret-id"
    elif defect == "nan": frame.loc[0, "lab_hba1c"] = np.nan
    elif defect == "infinite": frame.loc[0, "lab_hba1c"] = np.inf
    elif defect == "negative": frame.loc[0, "lab_hba1c"] = -2
    elif defect == "duplicate_date": frame.loc[1, "date"] = frame.loc[0, "date"]
    elif defect == "short": frame = frame.head(29)
    elif defect == "binary": frame.loc[0, "sex_male"] = 2
    elif defect == "inconsistent_profile": frame.loc[0, "sex_male"] = 1 - frame.loc[1, "sex_male"]
    elif defect == "percentage": frame.loc[0, "tir_70_180"] = 105
    elif defect == "subset": frame.loc[0, "tbr_severe_54"] = frame.loc[0, "tbr_below_70"] + 1
    elif defect == "glucose_unit": frame["glucose_mean"] = 7
    elif defect == "gmi": frame.loc[0, "gmi_estimated_a1c"] = 1
    elif defect == "date_format": frame.loc[0, "date"] = "01/01/2026"
    with pytest.raises(ValueError): validate_daily_frame(frame)


def test_sparse_and_duplicate_headers_rejected():
    with pytest.raises(ValueError, match="Missing model inputs"):
        read_daily_upload(b"date,glucose_mean\n2026-10-01,146\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_daily_upload(b"date,date\n2026-10-01,2026-10-01\n")


def test_out_of_distribution_withheld(upload_pipeline, compatible_frame):
    frame = compatible_frame.copy()
    frame["lab_hba1c"] = 1e8
    with pytest.raises(ValueError, match="outside"):
        upload_pipeline.run_uploaded_history(frame, 10)
