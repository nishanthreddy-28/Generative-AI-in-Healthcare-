"""Launch the Phase 5 CLI from the project root."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "btp_application_pipeline", ROOT / "Phase_5_Application" / "end_to_end_pipeline.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
EndToEndT1DPipeline = _module.EndToEndT1DPipeline
main = _module.main

if __name__ == "__main__":
    main()
