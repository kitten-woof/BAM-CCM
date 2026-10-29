from __future__ import annotations
import json, sys, time, traceback, platform
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'original'))
import bam_ccm as bam

MODELS=['CCM','Metric-CCM','GP-ARD','GP-Full','BAM-MAP','BAM-Bayes']

def _direction(df,E=3,tau=1):
    xy=df[['x','y']].astype(float).to_numpy()
    if len(xy)<120 or not np.isfinite(xy).all():raise ValueError('Need >=120 finite paired observations')
    idx=np.arange((E-1)*tau,len(xy));a=int(.6*len(idx));b=int(.8*len(idx))
    cutoff=idx[a-1]+1
    mean=xy[:cutoff].mean(axis=0);std=xy[:cutoff].std(axis=0)
    if (std<1e-12).any():raise ValueError('zero train-only standard deviation')
    z=(xy-mean)/std
    return ([( 'X_from_Y',bam.delay_embed(z[:,1],E,tau),z[(E-1)*tau:,0]),
             ('Y_from_X',bam.delay_embed(z[:,0],E,tau),z[(E-1)*tau:,1])],
            idx,np.arange(a),np.arange(a,b),np.arange(b,len(idx)),mean,std)

def evaluate_case(name,df,params,output,metadata=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    E=params.get('E',3);tau=params.get('tau',1); nlib=params.get('library_size',60)
    maxiter=params.get('map_maxiter',65);bayes=bool(params.get('bayes',False))
    dirs,t,tr,va,te,mean,std=_direction(df,E,tau)
    all_rows=[];all_predictions=[];all_curves=[];all_fit=[];all_chain=[]
    for j,(direction,X,y) in enumerate(dirs):
        li=np.unique(np.linspace(0,len(tr)-1,min(nlib,len(tr)),dtype=int))
        xl=X[li];yl=y[li];tl=t[li]
        C,radius=bam._choose_centers(xl,params.get('centers',2))
        kinds={}; fit_status={}
        with threadpool_limits(limits=1,user_api='blas'):
            for kind in ('diag','full','local'):
                gp=bam.GPModel(xl,yl,E,kind,len(C),C,radius)
                tic=time.perf_counter()
                theta,fit=gp.fit_map(maxiter)
                elapsed=time.perf_counter()-tic
                kinds[kind]=(gp,theta,elapsed)
                fit_status[kind]=fit
                all_fit.append({'case':name,'direction':direction,'kind':kind,'fit_seconds':elapsed,**fit})
            if bayes:
                tic=time.perf_counter()
                draws,diag=bam.sample_elliptical_slice(kinds['local'][0],kinds['local'][1],
                    params.get('mcmc_warmup',35),params.get('mcmc_steps',60),
                    params.get('mcmc_thin',2),params.get('mcmc_chains',2),
                    params.get('seed',42)+1009*j)
                mcmc_seconds=time.perf_counter()-tic
                chain=diag.pop('chain_draws')
                np.savez_compressed(output/f'posterior_{direction}.npz',draws=draws,chains=chain)
                all_chain.append({'case':name,'direction':direction,'mcmc_seconds':mcmc_seconds,
                                  **{k:v for k,v in diag.items() if isinstance(v,(float,int,bool,str))}})
            for split,ix in [('validation',va),('test',te)]:
                xx=X[ix]; yy=y[ix];tt=t[ix]
                for model in MODELS:
                    if model=='BAM-Bayes' and not bayes:continue
                    tic=time.perf_counter()
                    if model=='CCM': p={'mean':bam.simplex_predict(xl,yl,tl,xx,tt,E,tau)};fit_time=0.
                    elif model=='Metric-CCM':
                        L=bam._unpack(kinds['full'][1],E,0,'full')[0]
                        p={'mean':bam.simplex_predict(xl,yl,tl,xx,tt,E,tau,L)}
                        fit_time=kinds['full'][2]
                    elif model=='BAM-Bayes':
                        gp=kinds['local'][0];p=bam.predictive_mixture(gp,draws,xx,yy)
                        fit_time=kinds['local'][2]+mcmc_seconds
                    else:
                        kind={'GP-ARD':'diag','GP-Full':'full','BAM-MAP':'local'}[model]
                        gp,theta,fit_time=kinds[kind]
                        p=bam.point_gp_result(gp,theta,xx,yy)
                    pred_seconds=time.perf_counter()-tic
                    m=bam.score(yy,p)
                    all_rows.append({'case':name,'direction':direction,'split':split,'model':model,
                                     'n_train':len(tr),'n_val':len(va),'n_test':len(te),
                                     'n_library':len(li),'fit_seconds':fit_time,'predict_seconds':pred_seconds,
                                     'map_converged':fit_status[{'GP-ARD':'diag','GP-Full':'full','BAM-MAP':'local','Metric-CCM':'full'}.get(model,'diag')]['map_success'] if model not in ('CCM','BAM-Bayes') else np.nan,
                                     **m})
                    if split=='test':
                        for tt_i,truth,forecast in zip(tt,yy,p['mean']):
                            all_predictions.append({'case':name,'direction':direction,'model':model,'time_index':int(tt_i),
                                 'y_true_scaled':float(truth),'y_pred_scaled':float(forecast)})
            for libsz in sorted(set([max(E+6,int(len(li)*f)) for f in (.3,.5,.7,1.0)])):
                if libsz>len(li):continue
                pred=bam.simplex_predict(X[li[:libsz]],y[li[:libsz]],t[li[:libsz]],X[te],t[te],E,tau)
                all_curves.append({'case':name,'direction':direction,'model':'CCM','library_size':libsz,'rho_test':bam.score(y[te],{'mean':pred})['rho']})
    metrics=pd.DataFrame(all_rows);pred=pd.DataFrame(all_predictions);curves=pd.DataFrame(all_curves);fits=pd.DataFrame(all_fit)
    metrics.to_csv(output/'metrics.csv',index=False);pred.to_csv(output/'predictions.csv',index=False)
    curves.to_csv(output/'convergence.csv',index=False);fits.to_csv(output/'fit_diagnostics.csv',index=False)
    pd.DataFrame(all_chain).to_csv(output/'mcmc_diagnostics.csv',index=False)
    (output/'metadata.json').write_text(json.dumps({'case':name,'params':params,'data':metadata or {},
            'n_total':len(df),'split':'chronological 60/20/20','scaling':'training mean and std only',
            'mean':mean.tolist(),'std':std.tolist(),
            'interpretation':'Contemporaneous reconstruction, not causal identification; cannot use RMSE as causal evidence',
            'software':{'python':platform.python_version(),'numpy':np.__version__}},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,' | ',metrics.query('split=="test"').pivot(index='model',columns='direction',values='rmse').round(3).to_string(),flush=True)
    return metrics
