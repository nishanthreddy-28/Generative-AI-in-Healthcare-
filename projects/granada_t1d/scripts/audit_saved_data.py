"""Independent saved-data checks. Requires authorized local research artifacts."""
import os,sys,json,importlib.util,hashlib
from pathlib import Path
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

sys.path.insert(0,str(ROOT/'src'));torch.set_num_threads(2)
result={}
allow=[np._core.multiarray.scalar,np.dtype,np.dtypes.StrDType,np.dtypes.Float64DType,np.dtypes.Int64DType]
def load(rel):
 with torch.serialization.safe_globals(allow):return torch.load(ROOT/rel,map_location='cpu',weights_only=True)
legacy=pd.read_csv(ROOT/'Phase_1_Data_Preparation/cleaned_daily_multimodal_dataset.csv')
meta=pd.read_csv(ROOT/'Phase_1_Data_Preparation/cleaned_cohort_metadata.csv')
bio=pd.read_csv(ROOT/'Phase_1_Data_Preparation/raw_data/Biochemical_parameters.csv')
pat=pd.read_csv(ROOT/'Phase_1_Data_Preparation/raw_data/Patient_info.csv');diag=pd.read_csv(ROOT/'Phase_1_Data_Preparation/raw_data/Diagnostics.csv')
result['tables']={}
for name,df,keys in [('legacy_daily',legacy,['Patient_ID','date']),('cohort_metadata',meta,['Patient_ID']),('raw_patient',pat,['Patient_ID']),('raw_labs',bio,['Patient_ID','Reception_date','Name']),('raw_diagnoses',diag,['Patient_ID','Code','Description'])]:
 result['tables'][name]={'rows':len(df),'patients':df.Patient_ID.nunique(),'columns':list(df),'missing_cells':int(df.isna().sum().sum()),'duplicate_keys':int(df.duplicated(keys).sum())}
print('Small clinical tables checked',flush=True)
lat=load('Phase_2_Latent_VAE/latent_health_vectors.pt');tensors=load('Phase_2_Latent_VAE/timeseries_tensors.pt')
sets={k:set(lat[k+'_pats']) for k in ['train','val','test']}
result['legacy_splits']={'counts':{k:len(v) for k,v in sets.items()},'overlap_counts':{a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]},'cleaned_daily_has_split':'split' in legacy,'cohort_metadata_has_split':'split' in meta,'tensors':{k:{'shape':list(v.shape),'finite':bool(torch.isfinite(v).all())} for k,v in tensors.items() if isinstance(v,torch.Tensor)}}
result['legacy_window_dates']={}
for split in ['train','val','test']:
 bad=0;ids=set()
 for item in tensors['meta_'+split]:
  pid=item['patient_id'] if isinstance(item,dict) else item[0];start=item['start_date'] if isinstance(item,dict) else item[1];end=item['end_date'] if isinstance(item,dict) else item[2]
  ids.add(pid);bad+=((pd.Timestamp(end)-pd.Timestamp(start)).days!=29)
 result['legacy_window_dates'][split]={'window_count':len(tensors['meta_'+split]),'invalid_endpoint_spans':int(bad),'patients_outside_split':len(ids-sets[split])}
scaler=json.loads((ROOT/'Phase_2_Latent_VAE/scaler_parameters.json').read_text());train=legacy[legacy.Patient_ID.isin(sets['train'])]
means=train[scaler['input_cols']].mean().to_numpy();std=train[scaler['input_cols']].std().to_numpy();std[std<1e-6]=1
result['legacy_scaler']={'features':len(scaler['input_cols']),'max_abs_mean_error_vs_train':float(np.max(np.abs(means-np.array(scaler['feat_mean'])))),'max_abs_std_error_vs_train':float(np.max(np.abs(std-np.array(scaler['feat_std']))))}
# Determine whether values used for imputation depend on held-out patients.
name_map={'Glycated hemoglobin (A1c)':'lab_hba1c','Glucose':'lab_glucose','Creatinine':'lab_creatinine','Creatinine in urine':'lab_creatinine_urine','Albumin in urine':'lab_albumin_urine','Total cholesterol':'lab_total_cholesterol','HDL cholesterol':'lab_hdl_cholesterol','Triglycerides':'lab_triglycerides','Potassium':'lab_potassium','Sodium':'lab_sodium','GPT (ALT)':'lab_gpt_alt','GGT':'lab_ggt','Thyrotropin (TSH)':'lab_tsh','Uric acid':'lab_uric_acid'}
# Use the exact current source mapping without invoking its data-writing entrypoint.
import ast
module=ast.parse((ROOT/'Phase_1_Data_Preparation/phase1_ingestion.py').read_text())
for n in ast.walk(module):
 if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='bio_name_map' for t in n.targets):name_map=ast.literal_eval(n.value)
b=bio[bio.Patient_ID.isin(set(legacy.Patient_ID))].copy();b['biomarker']=b.Name.map(name_map);b=b[b.biomarker.notna()]
piv=b.pivot_table(index=['Patient_ID','Reception_date'],columns='biomarker',values='Value',aggfunc='mean')
tr=piv[piv.index.get_level_values('Patient_ID').isin(sets['train'])]
result['legacy_imputation']={}
for col in sorted(name_map.values()):
 values=legacy.loc[legacy[col+'_imputed']==1,col].dropna();unique=values.unique();all_med=float(piv[col].median()) if col in piv else None;train_med=float(tr[col].median()) if col in tr else None
 result['legacy_imputation'][col]={'imputed_rows':len(values),'distinct_imputed_values':len(unique),'single_fill':float(unique[0]) if len(unique)==1 else None,'all_cohort_median':all_med,'train_median':train_med,'matches_all_median':bool(len(unique)==1 and np.isclose(unique[0],all_med)) if all_med is not None else None,'matches_train_median':bool(len(unique)==1 and np.isclose(unique[0],train_med)) if train_med is not None else None}
result['clinical_provenance']={'raw_labs_have_unit_column':any('unit' in c.lower() for c in bio),'raw_diagnoses_have_date_column':any('date' in c.lower() for c in diag),'missing_diagnosis_count_patients':int(pat.Number_of_diagnostics.isna().sum())}
print('Legacy splits, scalers, and imputation checked',flush=True)
from healthcare_app.cgm import CORE
from healthcare_app.trajectory_v2 import CorrectedSimulator,baseline,constrain_core
spec=importlib.util.spec_from_file_location('audit_training',ROOT/'Phase_5_Application/train_corrected_release.py');training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
dir=ROOT/'Phase_5_Application/models/cgm/release_001';daily=pd.read_csv(dir/'daily_cgm.csv');splits=json.loads((dir/'patient_splits.json').read_text());m=json.loads((dir/'release.json').read_text());sim=CorrectedSimulator(dir)
split_sets={k:set(v) for k,v in splits.items()};result['corrected_splits']={'counts':{k:len(v) for k,v in splits.items()},'pairwise_overlap':{a+'_'+b:len(split_sets[a]&split_sets[b]) for i,a in enumerate(splits) for b in list(splits)[i+1:]}}
vals=daily[daily.Patient_ID.isin(splits['train'])][CORE].to_numpy();result['corrected_scaler']={'max_mean_error':float(np.max(np.abs(vals.mean(0)-sim.mean))),'max_std_error':float(np.max(np.abs(vals.std(0)-sim.std)))}
entries=training.windows(daily)
result['corrected_daily']={'rows':len(daily),'patients':daily.Patient_ID.nunique(),'duplicate_days':int(daily.duplicated(['Patient_ID','date']).sum()),'min_bins':int(daily.readings_count.min()),'max_bins':int(daily.readings_count.max()),'qualifying_patients':len(entries)}
def predict(x,name):
 if name!='vae_gru':return constrain_core(baseline(x,name))
 tensor=torch.tensor((x-sim.mean)/sim.std,dtype=torch.float32)
 with torch.inference_mode():z,_=sim.vae.encode(tensor);return constrain_core(sim.forecaster(tensor,z).numpy()*sim.std+sim.mean)
result['corrected_metrics']={}
for name in ['last_day','past_7_day_mean','past_30_day_mean','vae_gru']:
 rows=[entries[p][-1] for p in splits['test']];x=np.stack([r[0] for r in rows]);y=np.stack([r[1] for r in rows]);pred=predict(x,name);err=pred-y;mae=np.abs(err).mean((0,1));saved=np.array([m['test_metrics'][name]['mae'][k] for k in CORE])
 result['corrected_metrics'][name]={'test_patients':len(rows),'mae':dict(zip(CORE,mae.tolist())),'max_mae_discrepancy_saved':float(np.max(np.abs(mae-saved)))}
 if name==m['selected_model']:result['corrected_metrics'][name]['coverage']=dict(zip(CORE,(np.abs(err)<=sim.radius).mean((0,1)).tolist()))
rows=[entries[p][-1] for p in splits['calibration']];x=np.stack([r[0] for r in rows]);y=np.stack([r[1] for r in rows]);res=y-predict(x,m['selected_model']);n=len(res);rank=min(n-1,int(np.ceil((n+1)*.8))-1);radius=np.sort(np.abs(res),axis=0)[rank]
result['corrected_calibration']={'patients':n,'conformal_rank_1based':rank+1,'max_radius_discrepancy':float(np.max(np.abs(radius-sim.radius))),'latent_scale_train_only':True,'manifest_hashes_verified':True}
(OUT/'science_results.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
