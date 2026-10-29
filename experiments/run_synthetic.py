import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_synthetic import SPEC,generate
from evaluate import evaluate_case
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',choices=['pilot','full'],default='pilot')
    p.add_argument('--systems',nargs='*',default=None,choices=list(SPEC))
    p.add_argument('--seeds',nargs='+',type=int,default=None)
    p.add_argument('--library-size',type=int,default=65);p.add_argument('--map-maxiter',type=int,default=65)
    p.add_argument('--bayes',action='store_true');p.add_argument('--only-generate',action='store_true')
    a=p.parse_args(); systems=a.systems or (list(SPEC) if a.profile=='full' else ['logistic','henon','independent','common_driver','coupled_ar','nonlinear','delayed','regime_switch','heavy_tailed'])
    seeds=a.seeds or ([13,42,71,101,131,161,191,221,251,281] if a.profile=='full' else [42])
    grid=([(.0,.1,450),(.2,.05,650),(.35,.2,900)] if a.profile=='full' else [(.2,.1,600)])
    rows=[];events=[]
    for system in systems:
        for seed in seeds:
            for coupling,noise,n in grid:
                case=f'{system}_s{seed}_b{coupling:g}_noise{noise:g}_n{n}'
                if a.bayes or a.library_size!=65 or a.map_maxiter!=65:
                    case+=f'_L{a.library_size}_iter{a.map_maxiter}'+('_bayes' if a.bayes else '')
                dst=ROOT/'experiments/results/synthetic'/case
                df,meta=generate(system,n,seed,noise,coupling)
                dst.mkdir(parents=True,exist_ok=True)
                df.to_csv(dst/'input.csv',index=False)
                (dst/'generator.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
                if a.only_generate:continue
                try:
                    rr=evaluate_case(case,df,{'E':3,'tau':1,'library_size':a.library_size,
                        'map_maxiter':a.map_maxiter,'centers':2,'bayes':a.bayes,'seed':seed},dst,meta)
                    rows.append(rr)
                    events.append({'case':case,'status':'success'})
                except Exception as ex:
                    print('FAILED',case,repr(ex),flush=True)
                    events.append({'case':case,'status':'failure','error':repr(ex)})
                pd.DataFrame(events).to_csv(ROOT/'experiments/results/synthetic/run_manifest.csv',index=False)
                if rows:pd.concat(rows,ignore_index=True).to_csv(ROOT/'experiments/results/synthetic/summary.csv',index=False)
    print('Synthetic completed:',sum(x['status']=='success' for x in events),'/',len(events))
if __name__=='__main__':main()
