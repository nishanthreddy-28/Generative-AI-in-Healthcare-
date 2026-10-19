"""Exercise the public command-line interface, including safe failures."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    return subprocess.run([sys.executable, "-B", str(ROOT / "end_to_end_pipeline.py"), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=40)


def test_cli_export_retains_unreviewed_status_and_clinical_events(tmp_path):
    report = tmp_path / "fictional.txt"
    report.write_text("Report date: 2026-10-01\nGlucose 146 mg/dL 70-99\nDiagnosis: no nephropathy\nMedication: insulin glargine 12 units nightly")
    output = tmp_path / "new"
    result = run("--report", report, "--outdir", output)
    assert result.returncode == 0
    analysis = json.loads((output / "report_findings.json").read_text())
    assert analysis["user_reviewed"] is False and len(analysis["clinical_events"]) == 2
    assert "NOT user-reviewed" in (output / "report_explanation.md").read_text()
    original = (output / "report_findings.json").read_bytes()
    rejected = run("--report", report, "--outdir", output)
    assert rejected.returncode == 2 and (output / "report_findings.json").read_bytes() == original


def test_cli_missing_or_directory_input_reports_failure(tmp_path):
    for path in (tmp_path / "missing.txt", tmp_path):
        result = run("--report", path)
        assert result.returncode == 2 and "Traceback" not in result.stderr


def test_cli_sparse_daily_history_fails_before_model_inference(tmp_path):
    path = tmp_path / "sparse.csv"
    path.write_text("date,glucose_mean\n2026-10-01,146\n")
    result = run("--daily-history", path)
    assert result.returncode == 2 and "Missing model inputs" in result.stderr


def test_corrected_cgm_cli_generates_real_outputs(tmp_path):
    import numpy as np
    import pandas as pd
    timestamps = pd.date_range("2026-09-01", periods=30 * 96, freq="15min")
    frame = pd.DataFrame({"timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"), "glucose_mg_dl": 150 + 30 * np.sin(np.arange(len(timestamps)) / 8)})
    source = tmp_path / "fictional_cgm.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "new_corrected"
    result = run("--cgm", source, "--simulations", 10, "--outdir", output)
    assert result.returncode == 0
    assert len(pd.read_csv(output / "corrected_research_trajectory.csv")) == 60
    assert "INPUT NOT USER-REVIEWED" in (output / "corrected_research_note.txt").read_text()
