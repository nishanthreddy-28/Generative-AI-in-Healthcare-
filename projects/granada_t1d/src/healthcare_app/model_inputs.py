"""Strict user-owned daily feature input contract; never fill missing values."""
import csv
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd

SCALER = Path(__file__).resolve().parents[2] / "Phase_2_Latent_VAE/scaler_parameters.json"


def model_schema():
    return json.loads(SCALER.read_text())


def daily_template():
    return ",".join(["date"] + model_schema()["input_cols"]) + "\n"


def read_daily_upload(data):
    if not isinstance(data, bytes) or not data or len(data) > 5 * 1024 * 1024:
        raise ValueError("Upload a nonempty UTF-8 daily-feature CSV of at most 5 MB.")
    try:
        text = data.decode("utf-8-sig")
        if "\x00" in text:
            raise ValueError("Binary content is not accepted.")
        headers = next(csv.reader(io.StringIO(text)))
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate CSV headers are not accepted.")
        frame = pd.read_csv(io.StringIO(text), nrows=3661)
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError("The CSV must be valid UTF-8 with a header and consistent rows.") from exc
    if len(frame) > 3660:
        raise ValueError("Use at most 3,660 daily observations per upload.")
    return validate_daily_frame(frame)


def validate_daily_frame(frame):
    schema = model_schema()
    columns = schema["input_cols"]
    required = {"date", *columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing model inputs: " + ", ".join(missing) + ". Sparse reports or CGM alone cannot supply the full multimodal input.")
    extra = set(frame.columns) - required
    if extra:
        raise ValueError("Unexpected CSV columns: " + ", ".join(sorted(extra)) + ". Upload exactly the single-patient template; omit identifiers.")
    if frame.empty:
        raise ValueError("The daily input has no rows.")
    work = frame.copy()
    if not work.date.map(lambda value: isinstance(value, str) and __import__('re').fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None).all():
        raise ValueError("Every daily date must use YYYY-MM-DD.")
    dates = pd.to_datetime(work.date, format="%Y-%m-%d", errors="coerce")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be valid and unique for one patient.")
    try:
        work[columns] = work[columns].apply(pd.to_numeric, errors="raise")
        values = work[columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All model features must be numeric in the training representation.") from exc
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Features must be finite and nonnegative; missing values are never filled automatically.")
    binary = schema["imputed_cols"] + [col for col in schema["static_cols"] if col != "age"]
    if not work[binary].isin([0, 1]).all().all():
        raise ValueError("Diagnosis, sex and missingness indicators must be explicitly known binary 0/1 values in the training convention.")
    if not work[schema["static_cols"]].drop(columns="age").nunique().le(1).all():
        raise ValueError("Static model profile must be consistent across the uploaded history.")
    if (work.age > 120).any() or work.age.max() - work.age.min() > 10:
        raise ValueError("Age values are outside the supported representation.")
    percentages = ["tir_70_180", "tbr_below_70", "tbr_severe_54", "tar_above_180", "tar_severe_250"]
    if (work[percentages] > 100).any().any() or not np.allclose(work[["tir_70_180", "tbr_below_70", "tar_above_180"]].sum(axis=1), 100, atol=0.2):
        raise ValueError("Time-in/below/above-range percentages must be bounded and total 100.")
    if (work.tbr_severe_54 > work.tbr_below_70 + 0.01).any() or (work.tar_severe_250 > work.tar_above_180 + 0.01).any():
        raise ValueError("Severe glucose subsets cannot exceed their parent percentages.")
    if (work.glucose_min > work.glucose_mean).any() or (work.glucose_mean > work.glucose_max).any():
        raise ValueError("Daily glucose minimum, mean and maximum are inconsistent.")
    if (work[["glucose_min", "glucose_max", "glucose_mean", "nocturnal_mean"]] < 40).any().any() or (work[["glucose_min", "glucose_max", "glucose_mean", "nocturnal_mean"]] > 500).any().any():
        raise ValueError("Daily glucose inputs must use mg/dL within the supplied sensor/model representation (40–500).")
    if not np.allclose(work.gmi_estimated_a1c, 3.31 + 0.02392 * work.glucose_mean, atol=0.06):
        raise ValueError("GMI must be derived consistently from observed mean glucose; measured HbA1c is a separate feature.")
    from .research import latest_contiguous_window
    latest_contiguous_window(work)
    return work.sort_values("date").reset_index(drop=True)
