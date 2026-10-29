from __future__ import annotations
from pathlib import Path
import pandas as pd, json, time
from run_metric_research import run_case
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/metric_research/library_probe'
SYSTEMS=['logistic','common_driver','sharp_transition','varying_frequency']
SEEDS=[401,431,461,491,521]
SIZES=[40,65,120]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];fits=[];fails=[]
    for system in SYSTEMS:
        for seed in SEEDS:
            for size in SIZES:
                tic=time.perf_counter()
                try:
                    x,f,_,_=run_case(system,seed,'library_probe',20.,library=size,maxiter=65)
                    for a in x+f:a['library_size']=size
                    rows.extend(x);fits.extend(f)
                except Exception as exc: fails.append({'system':system,'seed':seed,'library_size':size,'error':str(exc)})
                print('library',system,seed,size, 'done',round(time.perf_counter()-tic,2),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps({'systems':SYSTEMS,'seeds':SEEDS,'sizes':SIZES,
       'K':2,'lambda_jac':20.,'E':3,'tau':1,'maxiter':65,'n':600,
       'phase':'exploratory after confirmation; cannot form a fresh confirmatory claim','failures':fails},indent=2))
    d=pd.DataFrame(rows).query('split=="test"').pivot(index=['system','seed','direction','library_size'],columns='model',values='rmse')
    for size,g in d.groupby(level='library_size'):
        print('library_size',size,'pairs',len(g))
        for k in ['GP-Fixed','GP-Full','BAM-original','BAM-warm','BAM-Jac']:
            z=g['GP-ARD']-g[k]
            print(k,'mean_delta_vs_ard',round(z.mean(),5),'wins',sum(z>0))
if __name__=='__main__':main()
