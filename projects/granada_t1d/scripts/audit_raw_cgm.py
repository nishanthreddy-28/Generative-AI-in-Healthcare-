"""Independent saved-data checks. Requires authorized local research artifacts."""
from pathlib import Path
import pandas as pd,numpy as np,json
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

root=ROOT
out=OUT
saved=pd.read_csv(root/'Phase_5_Application/models/cgm/release_001/daily_cgm.csv');saved=saved.set_index(['Patient_ID','date'])
cols=['readings_count','glucose_mean','glucose_std','tir_70_180','tbr_below_70','glucose_cv'];stats=dict(raw_rows=0,patients_processed=0,missing_cells=0,conflicting_timestamps=0,exact_duplicates_removed_after_conflicts=0,readings_combined_in_bins=0,excluded_low_coverage_days=0,retained_days=0,missing_saved_days=0,extra_saved_days=0,metric_mismatched_days=0,max_absolute_error={c:0.0 for c in cols})
carry=pd.DataFrame();closed=set()
def process(pid,df):
 assert pid not in closed;closed.add(pid);stats['patients_processed']+=1
 stats['missing_cells']+=int(df.isna().sum().sum())
 ts=pd.to_datetime(df.Measurement_date+' '+df.Measurement_time,format='%Y-%m-%d %H:%M:%S',errors='coerce');v=pd.to_numeric(df.Measurement,errors='coerce');assert ts.notna().all() and v.between(40,500).all()
 d=pd.DataFrame({'t':ts,'g':v})
 extrema=d.groupby('t').g.agg(['min','max']);bad=extrema.index[extrema['min']!=extrema['max']];stats['conflicting_timestamps']+=len(bad)
 d=d[~d.t.isin(bad)];stats['exact_duplicates_removed_after_conflicts']+=int(d.duplicated('t').sum());d=d.drop_duplicates('t')
 bins=d.groupby(d.t.dt.floor('15min')).g.mean();stats['readings_combined_in_bins']+=len(d)-len(bins)
 grouped=bins.groupby(bins.index.strftime('%Y-%m-%d'));daily=grouped.agg(['size','mean','std']);daily.columns=['readings_count','glucose_mean','glucose_std'];daily['tir_70_180']=bins.between(70,180).groupby(bins.index.strftime('%Y-%m-%d')).mean()*100;daily['tbr_below_70']=bins.lt(70).groupby(bins.index.strftime('%Y-%m-%d')).mean()*100;daily['glucose_cv']=100*daily.glucose_std/daily.glucose_mean
 stats['excluded_low_coverage_days']+=int((daily.readings_count<64).sum());daily=daily[daily.readings_count>=64];stats['retained_days']+=len(daily)
 prior=saved.loc[pid] if pid in saved.index.get_level_values(0) else pd.DataFrame(columns=cols)
 common=daily.index.intersection(prior.index);stats['missing_saved_days']+=len(daily.index.difference(prior.index));stats['extra_saved_days']+=len(prior.index.difference(daily.index))
 if len(common):
  delta=np.abs(daily.loc[common,cols].to_numpy()-prior.loc[common,cols].to_numpy());stats['metric_mismatched_days']+=int((delta>1e-8).any(1).sum())
  for i,c in enumerate(cols):stats['max_absolute_error'][c]=max(stats['max_absolute_error'][c],float(delta[:,i].max()))
for chunk in pd.read_csv(root/'Phase_1_Data_Preparation/raw_data/Glucose_measurements.csv',chunksize=500000):
 stats['raw_rows']+=len(chunk);combined=pd.concat([carry,chunk],ignore_index=True);last=combined.Patient_ID.iloc[-1]
 for pid,frame in combined[combined.Patient_ID!=last].groupby('Patient_ID',sort=False):process(pid,frame)
 carry=combined[combined.Patient_ID==last].copy()
 if stats['raw_rows']%5000000==0:print('Audited raw rows',stats['raw_rows'],flush=True)
if len(carry):process(carry.Patient_ID.iloc[0],carry)
(out/'raw_results.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2),flush=True)
