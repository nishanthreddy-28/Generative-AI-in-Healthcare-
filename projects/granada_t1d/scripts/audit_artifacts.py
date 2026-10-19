"""Independent saved-data checks. Requires authorized local research artifacts."""
from pathlib import Path
import json,sys,ast,contextlib,io,importlib.util,os,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,torch
import argparse
_parser = argparse.ArgumentParser(description="Read-only research artifact audit; no fitting or dataset rewriting")
_parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
_parser.add_argument("--outdir", type=Path, required=True, help="New directory for aggregate audit output")
_args = _parser.parse_args()
ROOT = _args.project_root.resolve()
OUT = _args.outdir.resolve()
if OUT.exists() and any(OUT.iterdir()):
    _parser.error("Choose a new or empty output directory.")
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)
allow=[np._core.multiarray.scalar,np.dtype,np.dtypes.StrDType,np.dtypes.Float64DType,np.dtypes.Int64DType];res={}
def inspect_tensor(v,counts):
 if isinstance(v,torch.Tensor):counts['arrays']+=1;counts['elements']+=v.numel();counts['nonfinite']+=int((~torch.isfinite(v)).sum()) if v.is_floating_point() else 0
 elif isinstance(v,dict):
  for x in v.values():inspect_tensor(x,counts)
 elif isinstance(v,(tuple,list)):
  for x in v:inspect_tensor(x,counts)
for p in sorted(ROOT.glob('Phase_*/*.pt'))+sorted(ROOT.glob('Phase_5_Application/models/cgm/release_001/*.pt')):
 with torch.serialization.safe_globals(allow):d=torch.load(p,map_location='cpu',weights_only=True)
 counts={'arrays':0,'elements':0,'nonfinite':0};inspect_tensor(d,counts);res[str(p.relative_to(ROOT))]={'restricted_load':True,'top_level_keys':list(d),'tensor_checks':counts}
# Phase 2 preparation should fail before any saving because legacy clean table has no split.
spec=importlib.util.spec_from_file_location('audit_vae',ROOT/'Phase_2_Latent_VAE/phase2_vae.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
previous=os.getcwd();os.chdir(OUT)
try:
 with contextlib.redirect_stdout(io.StringIO()):mod.prepare_model_inputs()
 res['phase2_rebuild']={'unexpected_success':True}
except Exception as e:res['phase2_rebuild']={'exception':type(e).__name__,'message':str(e)}
finally:os.chdir(previous)
# Forecast input pairing without unrestricted pickle deserialization.
spec=importlib.util.spec_from_file_location('audit_forecast',ROOT/'Phase_3_Generative_Forecasting/phase3_4_trajectory_forecasting.py');fc=importlib.util.module_from_spec(spec);spec.loader.exec_module(fc)
original=torch.load
def safe_load(*args,**kwargs):
 kwargs['weights_only']=True;kwargs['map_location']='cpu'
 with torch.serialization.safe_globals(allow):return original(*args,**kwargs)
from unittest.mock import patch
with patch.object(torch,'load',safe_load),contextlib.redirect_stdout(io.StringIO()):bundle=fc.prepare_forecasting_dataset(str(ROOT/'Phase_2_Latent_VAE/latent_health_vectors.pt'),str(ROOT/'Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv'))
res['legacy_forecast_pairs']={}
for name in ['train','val','test']:
 z,x,y,meta=bundle[name];ids={m['patient_id'] if isinstance(m,dict) else m[0] for m in meta};res['legacy_forecast_pairs'][name]={'windows':len(y),'patients':len(ids),'target_shape':list(y.shape),'finite':bool(torch.isfinite(y).all())}
# Safe XML parse of every existing JUnit file, treated as historical evidence.
res['junit_archives']={}
for p in (ROOT/'evidence').rglob('*.xml'):
 node=ET.fromstring(p.read_bytes());res['junit_archives'][str(p.relative_to(ROOT))]={'root':node.tag,'suites':[dict(n.attrib) for n in node.iter('testsuite')]}
(OUT/'artifact_results.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:v for k,v in res.items() if k in ['phase2_rebuild','legacy_forecast_pairs']},indent=2))
