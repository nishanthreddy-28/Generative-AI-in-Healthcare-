"""Local report explanation CLI and experimental diabetes pipeline."""

import sys
from pathlib import Path as _ProjectPath
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import argparse
import json
from pathlib import Path
from healthcare_app.documents import extract_document
from healthcare_app.analysis import build_analysis, export_markdown


class EndToEndT1DPipeline:
    def __init__(self):
        from healthcare_app.research import ResearchPipeline
        self._pipeline = ResearchPipeline()

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def run_inference_for_patient(self, patient_id=None, n_simulations=20, output_dir=None):
        return self._pipeline.run_inference_for_patient(patient_id, n_simulations, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Explain lab reports or run experimental cohort inference")
    parser.add_argument("--report", type=Path, help="PDF, PNG, JPEG or UTF-8 TXT report")
    parser.add_argument("--patient", help="ID from the diabetes research cohort")
    parser.add_argument("--daily-history", type=Path, help="Single-patient complete daily model-feature CSV; never imputes missing features")
    parser.add_argument("--cgm", type=Path, help="Measured timestamp,glucose_mg_dl CSV for the corrected CGM-only release")
    parser.add_argument("--outdir", type=Path, help="New or empty output directory")
    parser.add_argument("--simulations", type=int, default=20)
    args = parser.parse_args()
    if sum(bool(value) for value in (args.report, args.patient, args.daily_history, args.cgm)) > 1:
        parser.error("Choose report explanation, uploaded daily history or cohort inference.")
    try:
        if args.outdir and args.outdir.exists() and any(args.outdir.iterdir()):
            raise ValueError("Output directory must be new or empty; existing results will not be overwritten.")
        if args.report:
            document = extract_document(args.report.read_bytes(), args.report.name).to_dict()
            analysis = build_analysis([document])
            text = "EXTRACTION NOT USER-REVIEWED: verify source values before relying on this output.\n\n" + export_markdown(analysis)
            if args.outdir:
                args.outdir.mkdir(parents=True, exist_ok=True)
                (args.outdir / "report_explanation.md").write_text(text, encoding="utf-8")
                (args.outdir / "report_findings.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
            print(text)
        elif args.cgm:
            from healthcare_app.cgm import read_cgm_upload
            from healthcare_app.trajectory_v2 import CorrectedSimulator
            daily, _ = read_cgm_upload(args.cgm.read_bytes())
            result = CorrectedSimulator().run(daily, args.simulations)
            if args.outdir:
                args.outdir.mkdir(parents=True, exist_ok=True)
                result["point"].to_csv(args.outdir / "corrected_research_trajectory.csv")
                (args.outdir / "corrected_research_note.txt").write_text("INPUT NOT USER-REVIEWED\n\n" + result["note"], encoding="utf-8")
            print("INPUT NOT USER-REVIEWED\n\n" + result["note"])
        elif args.daily_history:
            from healthcare_app.model_inputs import read_daily_upload
            from healthcare_app.research import ResearchPipeline
            frame = read_daily_upload(args.daily_history.read_bytes())
            result = ResearchPipeline(load_cohort=False).run_uploaded_history(frame, args.simulations, output_dir=args.outdir)
            print("Model window:", *result["window_dates"])
            print(result["doctor_note"])
        else:
            result = EndToEndT1DPipeline().run_inference_for_patient(args.patient, args.simulations, args.outdir)
            print("Model window:", *result["window_dates"])
            print(result["doctor_note"])
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Unable to complete analysis: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
