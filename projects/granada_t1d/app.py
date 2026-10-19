"""Launch the Phase 5 Streamlit application from the project root."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "Phase_5_Application" / "app.py"), run_name="__main__")
