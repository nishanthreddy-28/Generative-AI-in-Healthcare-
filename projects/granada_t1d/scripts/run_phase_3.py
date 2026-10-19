"""
================================================================================
PHASE 3: CONDITIONAL TABULAR GAN (PHASE LAUNCHER)
================================================================================
Direct root-level entrypoint pointing to:
Phase_3_Generative_Forecasting/phase3_ctgan.py
================================================================================
"""

import os
import sys
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

_phase3_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Phase_3_Generative_Forecasting", "phase3_ctgan.py")
_spec = importlib.util.spec_from_file_location("phase3_ctgan_module", _phase3_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CTGANTabularGenerator = _mod.CTGANTabularGenerator
CTGANTabularDiscriminator = _mod.CTGANTabularDiscriminator
prepare_ctgan_dataset = _mod.prepare_ctgan_dataset
train_ctgan = _mod.train_ctgan
evaluate_ctgan_fidelity = _mod.evaluate_ctgan_fidelity
generate_synthetic_30d_tabular_state = _mod.generate_synthetic_30d_tabular_state
run_phase3_ctgan_pipeline = _mod.run_phase3_ctgan_pipeline

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase_3_Generative_Forecasting"))
    run_phase3_ctgan_pipeline(num_epochs=12)
