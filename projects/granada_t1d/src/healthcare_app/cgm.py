"""Measured CGM input contract and per-day aggregates; no lab imputation."""
import io
import numpy as np
import pandas as pd

CORE = ["glucose_mean", "tir_70_180", "tbr_below_70", "glucose_cv"]


def aggregate_cgm(frame, min_readings=64, conflict_policy="reject"):
    required = {"timestamp", "glucose_mg_dl"}
    if set(frame.columns) != required or frame.empty:
        raise ValueError("Use exactly timestamp,glucose_mg_dl for one patient's measured CGM data.")
    work = frame.copy()
    if not work.timestamp.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}").all():
        raise ValueError("Timestamps must be unambiguous local YYYY-MM-DD HH:MM:SS without mixed time zones.")
    work["timestamp"] = pd.to_datetime(work.timestamp, format="ISO8601", errors="coerce")
    work["glucose_mg_dl"] = pd.to_numeric(work.glucose_mg_dl, errors="coerce")
    if work.timestamp.isna().any() or not np.isfinite(work.glucose_mg_dl).all() or not work.glucose_mg_dl.between(40, 500).all():
        raise ValueError("Invalid timestamp or glucose. Values must be measured mg/dL within the Granada sensor representation (40–500).")
    conflicts = work.groupby("timestamp").glucose_mg_dl.nunique().gt(1)
    conflict_count = int(conflicts.sum())
    if conflict_count and conflict_policy == "reject":
        raise ValueError("Conflicting glucose values at the same timestamp must be resolved before simulation.")
    if conflict_policy not in {"reject", "discard"}:
        raise ValueError("Unsupported conflict policy.")
    if conflict_count:
        work = work[~work.timestamp.isin(conflicts[conflicts].index)]
    duplicate_count = int(work.duplicated("timestamp").sum())
    work = work.drop_duplicates("timestamp").sort_values("timestamp")
    # Count occupied 15-minute bins, not scans or bursts. Average observed
    # readings within a bin; never interpolate missing bins.
    observed_rows = len(work)
    work["timestamp"] = work.timestamp.dt.floor("15min")
    work = work.groupby("timestamp", as_index=False).glucose_mg_dl.mean()
    work["date"] = work.timestamp.dt.strftime("%Y-%m-%d")
    work["in_range"] = work.glucose_mg_dl.between(70, 180)
    work["below_range"] = work.glucose_mg_dl.lt(70)
    daily = work.groupby("date").agg(readings_count=("glucose_mg_dl", "size"), glucose_mean=("glucose_mg_dl", "mean"),
        glucose_std=("glucose_mg_dl", "std"), tir_70_180=("in_range", "mean"), tbr_below_70=("below_range", "mean"))
    daily[["tir_70_180", "tbr_below_70"]] *= 100
    daily["glucose_cv"] = daily.glucose_std / daily.glucose_mean * 100
    retained = daily[daily.readings_count >= min_readings].reset_index()
    return retained, {"exact_duplicates_removed": duplicate_count, "days_excluded_for_low_coverage": int(len(daily) - len(retained)),
                      "minimum_readings_per_day": min_readings, "unit": "mg/dL", "interpolated_readings": 0,
                      "conflicting_timestamps_discarded": conflict_count, "readings_combined_in_bins": observed_rows - len(work)}


def read_cgm_upload(data):
    if not data or len(data) > 10 * 1024 * 1024:
        raise ValueError("Provide a nonempty UTF-8 CGM CSV of at most 10 MB.")
    try:
        text = data.decode("utf-8-sig")
        import csv
        header = next(csv.reader(io.StringIO(text)))
        if header != ["timestamp", "glucose_mg_dl"]:
            raise ValueError("CSV header must be timestamp,glucose_mg_dl. Omit names and identifiers.")
        frame = pd.read_csv(io.StringIO(text), nrows=200001)
    except (UnicodeError, StopIteration, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("Invalid UTF-8 CGM CSV.") from exc
    if len(frame) > 200000:
        raise ValueError("Use at most 200,000 sensor readings per upload.")
    daily, quality = aggregate_cgm(frame)
    from .research import latest_contiguous_window
    latest_contiguous_window(daily)
    return daily, quality
