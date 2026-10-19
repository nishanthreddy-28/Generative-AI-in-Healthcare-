"""
================================================================================
PHASE 4: SEQUENTIAL TRAJECTORY SIMULATION & RAG (PHASE LAUNCHER)
================================================================================
Direct root-level entrypoint pointing to:
Phase_4_RAG_Support/phase4_trajectory_and_rag.py
================================================================================
"""

import os
import sys
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

_phase4_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Phase_4_RAG_Support", "phase4_trajectory_and_rag.py")
_spec = importlib.util.spec_from_file_location("phase4_trajectory_and_rag_module", _phase4_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

run_probabilistic_trajectory_simulation = _mod.run_probabilistic_trajectory_simulation
run_phase4_end_to_end = _mod.run_phase4_end_to_end
ADAGuidelinesRAG = _mod.ADAGuidelinesRAG
ClinicalDecisionSynthesizer = _mod.ClinicalDecisionSynthesizer

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase_4_RAG_Support"))
    run_phase4_end_to_end()
