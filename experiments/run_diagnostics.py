from pathlib import Path
import sys
import pandas as pd,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_synthetic import generate
from evaluate import _direction
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import bam_ccm as bam
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/diagnostics';OUT.mkdir(parents=True,exist_ok=True)

def lag_curve(system):
    df,meta=generate(system,600,42,.1,.2);directions,t,tr,va,te,means,std=_direction(df)
    rows=[]
    for label,X,y in directions:
        li=np.unique(np.linspace(0,len(tr)-1,65).astype(int))
        for lag in range(-8,9):
            # Target y[t+lag] must stay inside the same train/test period.
            target_idx_train=t[li]+lag;target_idx_test=t[te]+lag
            lo_train=t[tr[0]];hi_train=t[tr[-1]];lo_test=t[te[0]];hi_test=t[te[-1]]
            ok_tr=(target_idx_train>=lo_train)&(target_idx_train<=hi_train)
            ok_te=(target_idx_test>=lo_test)&(target_idx_test<=hi_test)
            if ok_tr.sum()<20 or ok_te.sum()<40: continue
            # y vector index corresponds to original t, offset 2.
            ly=y[target_idx_train[ok_tr]-t[0]];truth=y[target_idx_test[ok_te]-t[0]]
            pred=bam.simplex_predict(X[li[ok_tr]],ly,t[li[ok_tr]],X[te[ok_te]],t[te[ok_te]],3,1)
            rows.append({'system':system,'direction':label,'lag_target_minus_embedding':lag,
                'n_library':int(ok_tr.sum()),'n_test':int(ok_te.sum()),
                'rho':bam.score(truth,{'mean':pred})['rho'],
                'rmse':bam.score(truth,{'mean':pred})['rmse']})
    return rows

def surrogate(system,n=99):
    df,meta=generate(system,600,42,.1,.2)
    rng=np.random.default_rng(2026);rows=[]
    for replicate in range(n+1):
        z=df.copy()
        if replicate>0:
            shift=int(rng.integers(30,len(z)-30))
            z['y']=np.roll(z['y'].values,shift)
        dirs,t,tr,va,te,means,std=_direction(z)
        for label,X,y in dirs:
            li=np.unique(np.linspace(0,len(tr)-1,65).astype(int))
            pred=bam.simplex_predict(X[li],y[li],t[li],X[te],t[te],3,1)
            rows.append({'system':system,'replicate':replicate,'direction':label,'surrogate':replicate>0,
                'rho':bam.score(y[te],{'mean':pred})['rho']})
    return rows

if __name__=='__main__':
    a=[];b=[]
    for system in ['logistic','independent','common_driver','periodic']:
        a+=lag_curve(system);b+=surrogate(system)
    pd.DataFrame(a).to_csv(OUT/'lag_curves.csv',index=False)
    pd.DataFrame(b).to_csv(OUT/'circular_surrogate.csv',index=False)
    print('Lag results',len(a),'surrogate runs',len(b),'(diagnostic only; no causal p-values)')
