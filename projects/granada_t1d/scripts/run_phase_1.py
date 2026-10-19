"""
================================================================================
PHASE 1: DATA COLLECTION AND PATIENT RECORD INGESTION (PHASE LAUNCHER)
================================================================================
Points directly to:
Phase_1_Data_Preparation/phase1_ingestion.py
================================================================================
"""
import os
import sys
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

_phase1_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Phase_1_Data_Preparation", "phase1_ingestion.py")
_spec = importlib.util.spec_from_file_location("phase1_ingestion_module", _phase1_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_missing_value_audit = _mod.run_missing_value_audit
build_clean_longitudinal_timeline = _mod.build_clean_longitudinal_timeline

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase_1_Data_Preparation"))
    run_missing_value_audit()
    build_clean_longitudinal_timeline()
