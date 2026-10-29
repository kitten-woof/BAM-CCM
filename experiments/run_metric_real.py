import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from evaluate import _direction
from metric_refinement import AblationGP,local_start
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import bam_ccm as bam
R=Path(__file__).resolve().parents[1]
OUT=R/'experiments/results/metric_research/real'
DATASETS=['appliances','bike_day','bike_hour','qqq_volume','sp500_volume','qqq_sp500','qqq_vix']

def main():
    lam=float(json.loads((R/'experiments/results/metric_research/selection.json').read_text())['selected_lambda'])
    rows=[];fits=[]
    for dataset in DATASETS:
        df,_,_,src=bam.load_dataset(dataset,'logistic',42)
        df=df[['x','y']].dropna().iloc[-600:].reset_index(drop=True)
        dirs,t,tr,va,te,_,_=_direction(df)
        for direction,X,y in dirs:
            li=np.unique(np.linspace(0,len(tr)-1,65,dtype=int));xl=X[li];yl=y[li]
            C,r=bam._choose_centers(xl,2);trained={}
            with threadpool_limits(limits=1,user_api='blas'):
                for model,kind,lam_jac in [('GP-Fixed','fixed',0),('GP-ARD','diag',0),('GP-Full','full',0),
                                             ('BAM-original','local',0),('BAM-warm','local',0),('BAM-Jac','local',lam)]:
                    o=AblationGP(xl,yl,kind,C,r,lam_jac)
                    starts=[o.initial()]
                    if model in ('BAM-warm','BAM-Jac'):
                        starts += [local_start(trained[k][1],b,3,len(C)) for k,b in [('GP-ARD','diag'),('GP-Full','full')]]
                    tic=time.perf_counter();z,info=o.fit(65,starts);dt=time.perf_counter()-tic
                    trained[model]=(o,z)
                    fits.append({'dataset':dataset,'direction':direction,'model':model,'seconds':dt,'success':info['converged']})
                    for split,ix in [('validation',va),('test',te)]:
                        pred=o.predict(z,X[ix]);truth=y[ix]
                        rows.append({'dataset':dataset,'direction':direction,'model':model,'split':split,
                                     'rmse':float(np.sqrt(np.mean((truth-pred)**2))),
                                     'mae':float(np.mean(abs(truth-pred))),'n':len(df),
                                     'data_source':src,'n_library':len(li)})
        print('real:',dataset,flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT/'metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(OUT/'fits.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps({'datasets':DATASETS,'lambda_jac':lam,
                 'notes':'Last 600 complete observations (or shorter) from each original data source; 60/20/20 chronological split; single time window; descriptive only'},indent=2))
    t=pd.DataFrame(rows).query('split=="test"')
    print(t.pivot(index=['dataset','direction'],columns='model',values='rmse').round(4))
if __name__=='__main__':main()
