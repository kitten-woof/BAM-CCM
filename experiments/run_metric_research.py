from __future__ import annotations
import argparse,json,time,platform,traceback
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import t as tdist, binomtest
from threadpoolctl import threadpool_limits
from generate_synthetic import generate
from evaluate import _direction
from metric_refinement import AblationGP, local_start
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import bam_ccm as bam

ROOT=Path(__file__).resolve().parents[1];DST=ROOT/'experiments/results/metric_research'
PILOT_SEEDS=[13,42,71]
CONFIRM_SEEDS=[101,131,161,191,221,251,281,311,331,351]
SYSTEMS=['logistic','nonlinear','common_driver','independent','sharp_transition','varying_frequency']


def local_process(system,n,seed):
    rng=np.random.default_rng(seed);a=np.zeros((n+450,2))
    for t in range(1,len(a)):
        xprev,yprev=a[t-1]
        x=.78*xprev+.42*rng.normal()
        if system=='sharp_transition':f=np.tanh(7*x)
        elif system=='varying_frequency':f=np.sin(4*x*x)/(1+.15*abs(x))
        else:raise ValueError(system)
        y=.18*yprev+.85*f+.11*rng.normal()
        a[t]=x,y
    obs=a[450:]
    obs=obs+.05*obs.std(axis=0)*rng.normal(size=obs.shape)
    return pd.DataFrame(obs,columns=['x','y']),{'truth':'x->y',
         'model':'x_t=.78 x_{t-1}+.42 eps_x; y_t=.18 y_{t-1}+.85 f(x_t)+.11 eps_y; measurement noise=.05 column SD',
         'f': 'tanh(7x)' if system=='sharp_transition' else 'sin(4x^2)/(1+.15|x|)',
         'burnin':450,'n':n,'seed':seed}


def get_data(system,seed,n=600):
    if system in ('sharp_transition','varying_frequency'):return local_process(system,n,seed)
    return generate(system,n,seed,.1,.2)


def run_case(system,seed,phase,lam,library=65,maxiter=65):
    df,meta=get_data(system,seed)
    directions,idx,tr,va,te,_,_=_direction(df)
    rows=[];fits=[];preds=[]
    for direction,X,y in directions:
        li=np.unique(np.linspace(0,len(tr)-1,min(library,len(tr)),dtype=int))
        xl,yl=X[li],y[li];C,r=bam._choose_centers(xl,2)
        kinds={}
        specs=[('GP-Fixed','fixed',0),('GP-ARD','diag',0),('GP-Full','full',0),
               ('BAM-original','local',0),('BAM-warm','local',0),('BAM-Jac','local',lam)]
        with threadpool_limits(limits=1,user_api='blas'):
            for model,kind,lam_jac in specs:
                o=AblationGP(xl,yl,kind,C,r,lam_jac)
                init=[o.initial()]
                if model in ('BAM-warm','BAM-Jac'):
                    init=[o.initial(),local_start(kinds['GP-ARD'][1],'diag',3,len(C)),
                          local_start(kinds['GP-Full'][1],'full',3,len(C))]
                t0=time.perf_counter()
                theta,info=o.fit(maxiter,init)
                elapsed=time.perf_counter()-t0
                kinds[model]=(o,theta)
                Wnorm=float(np.linalg.norm(bam._unpack(theta,3,len(C),'local')[1])) if kind=='local' else np.nan
                fits.append({'phase':phase,'system':system,'seed':seed,'direction':direction,'model':model,
                             'lambda_jac':lam_jac,'W_norm':Wnorm,'n_params':o.dim,'n_starts':len(init),
                             'fit_seconds':elapsed,**{k:v for k,v in info.items() if k!='all_runs'},
                             'all_runs':json.dumps(info['all_runs'])})
                for split,ix in [('validation',va),('test',te)]:
                    pred=o.predict(theta,X[ix]);truth=y[ix]
                    rmse=np.sqrt(np.mean((truth-pred)**2));mae=np.mean(np.abs(truth-pred))
                    rows.append({'phase':phase,'system':system,'seed':seed,'direction':direction,
                                 'model':model,'split':split,'rmse':rmse,'mae':mae,
                                 'lambda_jac':lam_jac,'n_library':len(li),
                                 'n_test':len(te),'n_val':len(va)})
                    if split=='test':
                        preds.extend([{'phase':phase,'system':system,'seed':seed,'direction':direction,
                                      'model':model,'time_index':int(t),'truth':float(gt),'pred':float(pr)}
                                      for t,gt,pr in zip(idx[ix],truth,pred)])
    return rows,fits,preds,meta


def execute(phase,lam=None):
    seeds=PILOT_SEEDS if phase=='pilot' else CONFIRM_SEEDS
    out=DST/phase;out.mkdir(parents=True,exist_ok=True)
    if phase=='pilot':lams=[0.,2.,20.]
    else:
        setting=json.loads((DST/'selection.json').read_text(encoding='utf-8'))
        lams=[float(setting['selected_lambda'])]
    rows=[];fits=[];preds=[];failures=[];datas=[]
    for system in SYSTEMS:
        for seed in seeds:
            for lv in lams:
                st=time.perf_counter()
                try:
                    x,f,p,meta=run_case(system,seed,phase,lv)
                    if len(lams)>1 and lv!=lams[0]:
                        x=[z for z in x if z['model']=='BAM-Jac']
                        f=[z for z in f if z['model']=='BAM-Jac']
                        p=[z for z in p if z['model']=='BAM-Jac']
                    for o in x+f+p:
                        if o['model']=='BAM-Jac':o['model']=f'BAM-Jac-{lv:g}'
                    rows+=x;fits+=f;preds+=p
                    datas.append({'phase':phase,'system':system,'seed':seed,'lam_jac':lv,'metadata':meta})
                    print(f'{phase}: {system} seed={seed} lambda={lv:g} ok {time.perf_counter()-st:.1f}s',flush=True)
                except Exception as e:
                    print('ERROR',phase,system,seed,lv,repr(e),flush=True)
                    failures.append({'phase':phase,'system':system,'seed':seed,'lambda':lv,'exception':traceback.format_exc()})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(fits).to_csv(out/'fits.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'predictions.csv',index=False)
    (out/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2))
    (out/'metadata.json').write_text(json.dumps({'params':{'E':3,'tau':1,'library':65,'maxiter':65,
      'n':600,'noise_standard':.1,'coupling':.2,'centers':2,'lambdas':lams},
      'data':datas,'seed_split':{'pilot':PILOT_SEEDS,'confirm':CONFIRM_SEEDS},
      'software':{'python':platform.python_version(),'numpy':np.__version__},
      'method':'60/20/20 chronological split; normalization fit on training only; training-only GP MAP; no test model selection',
      'failures':len(failures)},ensure_ascii=False,indent=2))
    print(f'{phase} finished, {len(rows)} metrics, {len(failures)} errors',flush=True)
    return pd.DataFrame(rows)


def select_lambda(df):
    v=df[df.split=='validation']
    pivot=v.pivot_table(index=['system','seed','direction'],columns='model',values='rmse')
    summaries=[]
    for lam in [0.,2.,20.]:
        key=f'BAM-Jac-{lam:g}'
        diff=(pivot[key]-pivot['GP-ARD'])
        summaries.append({'lambda':lam,'mean_validation_rmse_diff_v_ard':float(diff.mean()),
                          'validation_win_rate':float((diff<0).mean()),'pairs':len(diff)})
    best=min(summaries,key=lambda x:x['mean_validation_rmse_diff_v_ard'])
    res={'selection_rule':'Minimum mean pilot validation RMSE difference versus GP-ARD; confirm seeds excluded',
         'selected_lambda':best['lambda'],'candidates':summaries}
    (DST/'selection.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
    print('pilot selection',res,flush=True)
    return res


def summarize():
    df=pd.read_csv(DST/'confirm/metrics.csv').query('split=="test"')
    p=df.pivot(index=['system','seed','direction'],columns='model',values='rmse')
    chosen=next(c for c in p.columns if c.startswith('BAM-Jac-'))
    columns=['GP-Fixed','GP-ARD','GP-Full','BAM-original','BAM-warm',chosen]
    results=[]
    for system,g in p.groupby(level='system'):
        for direction,part in g.groupby(level='direction'):
            for model in columns:
                a=part[model].to_numpy();ref=part['GP-ARD'].to_numpy()
                delta=ref-a
                crit=tdist.ppf(.975,len(delta)-1)
                ci=[float(delta.mean()-crit*delta.std(ddof=1)/np.sqrt(len(delta))),
                    float(delta.mean()+crit*delta.std(ddof=1)/np.sqrt(len(delta)))]
                results.append({'system':system,'direction':direction,'model':model,
                     'n_pairs':len(a),'rmse_mean':float(a.mean()),'rmse_sd':float(a.std(ddof=1)),
                     'mean_rmse_gain_vs_ard':float(delta.mean()),
                     'gain_ci95_low':ci[0],'gain_ci95_high':ci[1],
                     'wins_vs_ard':int((delta>0).sum()),
                     'binom_p_two_sided_uncorrected':float(binomtest(int((delta>0).sum()),len(delta),.5).pvalue)})
    for model in columns:
        per_seed=(p['GP-ARD']-p[model]).groupby(level='seed').mean().to_numpy()
        crit=tdist.ppf(.975,len(per_seed)-1)
        ci=[float(per_seed.mean()-crit*per_seed.std(ddof=1)/np.sqrt(len(per_seed))),
            float(per_seed.mean()+crit*per_seed.std(ddof=1)/np.sqrt(len(per_seed)))]
        results.append({'system':'ALL_seed_aggregate','direction':'both_averaged',
                        'model':model,'n_pairs':len(per_seed),
                        'rmse_mean':float(p[model].mean()),'rmse_sd':float(p[model].std(ddof=1)),
                        'mean_rmse_gain_vs_ard':float(per_seed.mean()),
                        'gain_ci95_low':ci[0],'gain_ci95_high':ci[1],
                        'wins_vs_ard':int((per_seed>0).sum()),
                        'binom_p_two_sided_uncorrected':float(binomtest(int((per_seed>0).sum()),len(per_seed),.5).pvalue)})
    summary=pd.DataFrame(results);summary.to_csv(DST/'confirm/summary.csv',index=False)
    print(summary.query('model.str.contains("BAM-Jac")',engine='python').to_string(index=False),flush=True)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['pilot','confirm','all'],default='all')
    a=ap.parse_args();DST.mkdir(parents=True,exist_ok=True)
    if a.phase in ('pilot','all'):select_lambda(execute('pilot'))
    if a.phase in ('confirm','all'):execute('confirm');summarize()
