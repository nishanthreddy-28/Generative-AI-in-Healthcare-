"""
================================================================================
PHASE 2: COMPACT REPRESENTATION OF PATIENT HEALTH (PHASE LAUNCHER)
================================================================================
Points directly to:
Phase_2_Latent_VAE/phase2_vae.py
================================================================================
"""
import os
import sys
import importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

_phase2_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Phase_2_Latent_VAE", "phase2_vae.py")
_spec = importlib.util.spec_from_file_location("phase2_vae_module", _phase2_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

TwoBranchHealthVAE = _mod.TwoBranchHealthVAE
prepare_model_inputs = _mod.prepare_model_inputs
train_compact_vae = _mod.train_compact_vae
extract_and_save_latent_vectors = _mod.extract_and_save_latent_vectors
run_phase2_pipeline = _mod.run_phase2_pipeline

if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase_2_Latent_VAE"))
    run_phase2_pipeline()
