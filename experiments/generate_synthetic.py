from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

SPEC={
 'logistic':('x->y',r'x+=3.78x(1-x); y+=(1-b)3.67y(1-y)+b*3.78x(1-x)', 'Unidirectional coupled logistic maps'),
 'henon':('x<->y',r'x+=1-1.4x^2+y; y+=0.3x', 'Bidirectional state dependence of the Hénon map'),
 'lorenz':('x<->y',r'dx=10(y-x); dy=x(28-z)-y; dz=xy-(8/3)z', 'Coordinate coupling, not an independently actuated causal experiment'),
 'rossler':('x<->y',r'dx=-y-z; dy=x+0.2y; dz=0.2+z(x-5.7)', 'Coordinate coupling, not an independently actuated causal experiment'),
 'coupled_ar':('x->y',r'x+=0.65x+eps_x; y+=0.45y+b*x+eps_y', 'Linear Gaussian AR(1), process errors'),
 'nonlinear':('x->y',r'x+=0.6x+eps_x; y+=0.35y+b*tanh(2x)+eps_y', 'Nonlinear asymmetric transfer'),
 'delayed':('x->y',r'x+=0.7x+eps_x; y+=0.4y+b*x[t-3]+eps_y', 'Three-step delayed transfer'),
 'independent':('none',r'x+=0.7x+eps_x; y+=0.6y+eps_y, independent eps', 'Negative control with independent AR processes'),
 'common_driver':('z->x,z->y',r'z+=0.7z+eps_z; x+=0.45x+0.5z+eps_x; y+=0.5y+0.6z+eps_y', 'Unobserved common driver: x,y not independent'),
 'synchronized':('x->y',r'Coupled logistic map with b=0.45; may approach synchronization', 'Direction can become unidentifiable'),
 'heteroskedastic':('x->y',r'x+=0.7x+eps_x; y+=0.4y+b*x+sigma_t*eps_y; sigma_t=0.3+0.7|x|', 'State-dependent process noise'),
 'regime_switch':('x->y then y->x',r'First half: x->y, second half: y->x, AR(1) with coefficient b', 'Time-varying structural direction'),
 'heavy_tailed':('x->y',r'Coupled AR with Student-t(3) innovations normalized to var~1', 'Outlier robustness'),
 'irregular':('x->y',r'Coupled AR with 15% random time indices removed, unequal observation gaps', 'Irregular index spacings are deliberately not corrected'),
 'periodic':('none',r'x=sin(2pi*t/30)+eps_x; y=sin(2pi*t/30+0.8)+eps_y, no causal edge', 'Common periodic phase is not a causal link')}

def generate(system:str,n:int=600,seed:int=42,noise:float=.1,coupling:float=.2):
    if system not in SPEC: raise ValueError(f'Unknown system {system}')
    rng=np.random.default_rng(seed); total=n+450
    if system in ('logistic','synchronized'):
        b=.45 if system=='synchronized' else coupling
        x,y=rng.uniform(.2,.8,2);a=np.zeros((total,2))
        for t in range(total):
            fx=3.78*x*(1-x);fy=3.67*y*(1-y)
            x,y=fx,(1-b)*fy+b*fx
            a[t]=[x,y]
        clean=a[450:];meta={'coupling_used':b,'process_error':'none'}
    elif system=='henon':
        x,y=.11,.21; a=np.empty((total,2))
        for t in range(total):
            x,y=1-1.4*x*x+y,0.3*x
            a[t]=x,y
        clean=a[450:];meta={'process_error':'none'}
    elif system in ('lorenz','rossler'):
        if system=='lorenz':
            f=lambda t,s:[10*(s[1]-s[0]),s[0]*(28-s[2])-s[1],s[0]*s[1]-(8/3)*s[2]]
            initial=[1.1,2.1,3.1];step=.03
        else:
            f=lambda t,s:[-s[1]-s[2],s[0]+.2*s[1],.2+s[2]*(s[0]-5.7)]
            initial=[1.1,.2,.2];step=.08
        ts=np.arange(total)*step
        sol=solve_ivp(f,[0,ts[-1]],initial,t_eval=ts,rtol=1e-9,atol=1e-11)
        if not sol.success:raise RuntimeError(sol.message)
        clean=sol.y[:2,450:].T;meta={'dt':step,'process_error':'none'}
    elif system=='periodic':
        t=np.arange(n);clean=np.column_stack([np.sin(2*np.pi*t/30),np.sin(2*np.pi*t/30+.8)])
        meta={'process_error':'independent measurement only'}
    else:
        arr=np.zeros((total,3));arr[0]=rng.normal(size=3)*.5
        for t in range(1,total):
            x,y,z=arr[t-1]
            if system=='heavy_tailed':
                innov=rng.standard_t(3,3)/np.sqrt(3)
            else:innov=rng.normal(size=3)
            if system=='common_driver':
                arr[t]=[.45*x+.5*z+.4*innov[0],.5*y+.6*z+.4*innov[1],.7*z+.4*innov[2]]
            elif system=='independent':
                arr[t]=[.7*x+.5*innov[0],.6*y+.5*innov[1],0]
            elif system in ('coupled_ar','heavy_tailed'):
                arr[t]=[.65*x+.5*innov[0],.45*y+coupling*x+.5*innov[1],0]
            elif system=='nonlinear':
                arr[t]=[.6*x+.5*innov[0],.35*y+coupling*np.tanh(2*x)+.5*innov[1],0]
            elif system=='delayed':
                d=arr[max(0,t-1-3),0]
                arr[t]=[.7*x+.5*innov[0],.4*y+coupling*d+.5*innov[1],0]
            elif system=='heteroskedastic':
                arr[t]=[.7*x+.5*innov[0],.4*y+coupling*x+.3*(.3+.7*abs(x))*innov[1],0]
            elif system=='regime_switch':
                if t<450+n//2:arr[t]=[.7*x+.4*innov[0],.4*y+coupling*x+.4*innov[1],0]
                else:arr[t]=[.7*x+coupling*y+.4*innov[0],.4*y+.4*innov[1],0]
            elif system=='irregular':
                arr[t]=[.7*x+.4*innov[0],.4*y+coupling*x+.4*innov[1],0]
        clean=arr[450:,:2];meta={'process_error':'in dynamics; specified independent innovations'}
    if not np.isfinite(clean).all():raise ValueError(f'{system} diverged')
    observed=clean+noise*np.std(clean,axis=0)*rng.standard_normal(size=clean.shape)
    times=np.arange(n)
    if system=='irregular':
        keep=rng.uniform(size=n)>.15
        observed=observed[keep];times=times[keep]
        meta['rows_dropped']=int(n-len(times))
    df=pd.DataFrame({'t':times,'x':observed[:,0],'y':observed[:,1]})
    meta.update({'system':system,'n_requested':n,'n_returned':len(df),'seed':seed,
                 'measurement_noise':noise,'nominal_coupling':coupling,
                 'truth':SPEC[system][0],'equations':SPEC[system][1],
                 'purpose':SPEC[system][2],
                 'measurement_noise_definition':'Independent Gaussian error scaled by clean-column SD'})
    return df,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--system',choices=['all',*SPEC],default='all');p.add_argument('--n',type=int,default=600);p.add_argument('--seed',type=int,default=42);p.add_argument('--noise',type=float,default=.1);p.add_argument('--coupling',type=float,default=.2);p.add_argument('--out',default='experiments/results/generated')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];dst=root/a.out;dst.mkdir(parents=True,exist_ok=True)
    for kind in (SPEC if a.system=='all' else [a.system]):
        df,meta=generate(kind,a.n,a.seed,a.noise,a.coupling)
        df.to_csv(dst/f'{kind}_seed{a.seed}.csv',index=False)
        (dst/f'{kind}_seed{a.seed}.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print(f'{kind:20} n={len(df)} truth={meta["truth"]}')
if __name__=='__main__':main()
