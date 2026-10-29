#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BAM CCM(v0.6)"""


from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import logsumexp, ndtr
from scipy.stats import norm
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

CONFIG = {
    'dataset': 'synthetic', # synthetic | appliances | bike_day | bike_hour |
                            # qqq_volume | sp500_volume | qqq_sp500 | qqq_vix
    'system': 'logistic',   # logistic | henon | lorenz | rossler | common_driver | independent
    'embedding_dim': 3,
    'tau': 1,
    'seed': 42,
    'quick': True, 
    'max_train': 110,
    'n_centers': 2,
    'map_maxiter': 35,
    'mcmc_warmup': 60,
    'mcmc_steps': 100,
    'mcmc_thin': 2,
    'mcmc_chains': 4,
    'metric_curves': True,
    'output_tag': '',
    'synthetic_n': 900,
    'synthetic_noise': 0.10,
    'coupling': 0.05,
}

# --------------------------- Загрузка данных -------------------------------
def _numeric(col: pd.Series) -> pd.Series:
    """Parse quoted Nasdaq prices ($1,234.56), comma-thousands, and '.' missing."""
    return pd.to_numeric(col.astype(str).str.replace(r"[$,\s]", "", regex=True),
                         errors="coerce")


def _dates(col: pd.Series) -> pd.Series:
    text = col.astype(str)
    if text.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$").mean() > 0.9:
        return pd.to_datetime(text, format="%m/%d/%Y", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден: {path}\n")
    return path


def read_market(filename: str) -> pd.DataFrame:
    path = _require_file(DATA / "raw" / "financial" / filename)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.columns = raw.columns.astype(str).str.strip()
    lookup = {c.lower().replace(" ", ""): c for c in raw.columns}
    date_col = lookup.get("date") or lookup.get("timestamp")
    price_col = next((lookup[c] for c in ("close/last", "close", "adjclose", "adjustedclose")
                      if c in lookup), None)
    if date_col is None or price_col is None:
        raise ValueError(f"{filename}: нет Date или Close/Last. Столбцы: {list(raw.columns)}")
    result = pd.DataFrame({"date": _dates(raw[date_col]), "price": _numeric(raw[price_col])})
    if "volume" in lookup:
        result["volume"] = _numeric(raw[lookup["volume"]])
    result = result.dropna(subset=["date", "price"]).sort_values("date")
    if result.date.duplicated().any():
        raise ValueError(f"{filename}: повторяющиеся даты — проверьте CSV")
    if (result.price <= 0).any():
        raise ValueError(f"{filename}: цена должна быть > 0")
    if len(result) < 120:
        raise ValueError(f"{filename}: недостаточно записей: {len(result)}")
    return result.reset_index(drop=True)


def read_vix() -> pd.DataFrame:
    raw = pd.read_csv(_require_file(DATA / "raw" / "financial" / "VIXCLS.csv"))
    raw.columns = raw.columns.astype(str).str.strip()
    date_col = next((c for c in ("observation_date", "DATE", "Date", "date") if c in raw), None)
    if date_col is None or "VIXCLS" not in raw:
        raise ValueError(f"{list(raw.columns)}")
    frame = pd.DataFrame({"date": _dates(raw[date_col]), "vix": _numeric(raw["VIXCLS"])})
    return frame.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _load_synthetic(system: str, seed: int) -> tuple[pd.DataFrame, str]:
    file = DATA / "synthetic" / f"{system}_seed{seed}.csv"
    if file.exists():
        frame = pd.read_csv(file)
        if not {"x", "y"}.issubset(frame.columns):
            raise ValueError(f"{file} не содержит x и y")
        return pd.DataFrame({"x": _numeric(frame["x"]), "y": _numeric(frame["y"])}), str(file)
    if system != "logistic":
        raise FileNotFoundError(f"Нет {file}")
    rng = np.random.default_rng(seed)
    x, y = 0.217, 0.613
    rows = []
    for t in range(CONFIG["synthetic_n"] + 500):
        fx, fy = 3.78 * x * (1 - x), 3.67 * y * (1 - y)
        x, y = fx, (1 - CONFIG["coupling"]) * fy + CONFIG["coupling"] * fx
        if t >= 500:
            rows.append((x, y))
    out = np.array(rows, dtype=float)
    out += CONFIG["synthetic_noise"] * out.std(axis=0) * rng.normal(size=out.shape)
    frame = pd.DataFrame(out, columns=["x", "y"])
    file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(file, index=False)
    return frame, str(file)


def load_dataset(dataset: str, system: str, seed: int) -> tuple[pd.DataFrame, str, str, str]:
    if dataset == "synthetic":
        frame, origin = _load_synthetic(system, seed)
        return frame, "x", "y", origin
    if dataset == "appliances":
        file = _require_file(DATA / "raw" / "appliances" / "energydata_complete.csv")
        raw = pd.read_csv(file)
        raw["date"] = _dates(raw["date"])
        frame = raw.sort_values("date")
        return frame[["T_out", "Appliances"]].rename(columns={"T_out": "x", "Appliances": "y"}), "T_out", "Appliances", str(file)
    if dataset.startswith("bike_"):
        hourly = dataset == "bike_hour"
        file = _require_file(DATA / "raw" / "bike_sharing" / ("hour.csv" if hourly else "day.csv"))
        raw = pd.read_csv(file)
        if hourly:
            raw["time"] = _dates(raw["dteday"]) + pd.to_timedelta(raw["hr"], unit="h")
        else:
            raw["time"] = _dates(raw["dteday"])
        frame = raw.sort_values("time")
        return frame[["temp", "cnt"]].rename(columns={"temp": "x", "cnt": "y"}), "temp", "cnt", str(file)
    qqq_file = "HistoricalData_1793127119371_QQQ.csv"
    sp_file = "HistoricalData_1793127127422_SP500.csv"
    if dataset in ("qqq_volume", "sp500_volume"):
        name = qqq_file if dataset == "qqq_volume" else sp_file
        raw = read_market(name)
        if "volume" not in raw.columns:
            raise ValueError(f"{name}: нет Volume. Этот файл может быть индексом, а не ETF")
        raw = raw[(raw.volume > 0) & raw.price.notna()].copy()
        raw["x"] = np.log(raw["volume"])
        raw["y"] = np.log(raw["price"]).diff().abs()
        return raw[["x", "y"]].dropna(), "log volume", "abs(log return)", name
    if dataset == "qqq_sp500":
        qqq = read_market(qqq_file)[["date", "price"]].rename(columns={"price": "qqq"})
        sp = read_market(sp_file)[["date", "price"]].rename(columns={"price": "sp500"})
        qqq["x"] = np.log(qqq.qqq).diff()
        sp["y"] = np.log(sp.sp500).diff()
        joined = qqq[["date", "x"]].merge(sp[["date", "y"]], on="date", how="inner")
        return joined[["x", "y"]].dropna(), "QQQ log return", "SP500 file log return", "QQQ+SP500"
    if dataset == "qqq_vix":
        qqq = read_market(qqq_file)[["date", "price"]]
        qqq["x"] = np.log(qqq.price).diff().abs()
        vix = read_vix()
        vix["y"] = vix.vix.diff()
        joined = qqq[["date", "x"]].merge(vix[["date", "y"]], on="date", how="inner")
        return joined[["x", "y"]].dropna(), "QQQ abs(log return)", "VIX daily change", "QQQ+VIXCLS"
    raise ValueError(f"Неизвестный dataset: {dataset}")


# --------------------- Embedding and baselines ---------------------
def delay_embed(series: np.ndarray, E: int, tau: int) -> np.ndarray:
    series = np.asarray(series, dtype=float).reshape(-1)
    if E < 2 or tau < 1 or len(series) - (E-1)*tau < E+1:
        raise ValueError('Expected E>=2, tau>=1 and >=20 embedded samples')
    s = (E-1)*tau
    return np.column_stack([series[s-k*tau:len(series)-k*tau] for k in range(E)])


def simplex_predict(Xlib, ylib, tlib, Xtest, ttest, E, tau, L=None, theiler_radius=None):
    if L is None:
        L = np.eye(E)
    D = cdist(Xtest @ L.T, Xlib @ L.T)
    radius = E*tau if theiler_radius is None else theiler_radius
    D[np.abs(ttest[:, None]-tlib[None,:]) <= radius] = np.inf
    k=E+1
    if len(Xlib) <= k or np.sum(np.isfinite(D),axis=1).min() < k:
        raise ValueError('Insufficient CCM neighbors after exclusion')
    idx=np.argpartition(D,kth=k-1,axis=1)[:,:k]
    dist=np.take_along_axis(D,idx,axis=1)
    nearzero = dist <= 1.e-12
    w=np.empty_like(dist)
    zero=nearzero.any(axis=1)
    w[zero]=nearzero[zero]/nearzero[zero].sum(axis=1,keepdims=True)
    if (~zero).any():
        d=dist[~zero]
        u=np.exp(-d / d.min(axis=1,keepdims=True))
        w[~zero]=u/u.sum(axis=1,keepdims=True)
    return np.sum(w*ylib[idx],axis=1)


def _choose_centers(X: np.ndarray, n_centers: int):
    centers = [int(np.argmin(np.sum((X-X.mean(axis=0))**2,axis=1)))]
    dist2=np.sum((X-X[centers[0]])**2,axis=1)
    for _ in range(1,n_centers):
        j=int(np.argmax(dist2))
        centers.append(j)
        dist2=np.minimum(dist2,np.sum((X-X[j])**2,axis=1))
    C=X[centers]
    radius=float(np.median(np.min(cdist(X,C),axis=1)))
    radius=max(0.6, radius, 1e-5)
    return C,radius


def _layout(E:int, K:int, kind:str):
    if kind not in ('diag','full','local'):
        raise ValueError(kind)
    nL=E if kind=='diag' else E*(E+1)//2
    nW=K*E if kind=='local' else 0
    return nL,nW,nL+nW+2


def _unpack(theta,E,K,kind):
    nL,nW,d=_layout(E,K,kind)
    assert len(theta)==d
    L=np.zeros((E,E))
    if kind=='diag':
        L[np.diag_indices(E)]=np.exp(theta[:E])
    else:
        tr=np.tril_indices(E)
        L[tr]=theta[:nL]
        ix=np.diag_indices(E)
        L[ix]=np.exp(L[ix])
    W=theta[nL:nL+nW].reshape(K,E) if nW else np.zeros((K,E))
    return L,W,float(np.exp(theta[-2])),float(np.exp(theta[-1]))


def _theta0(E,K,kind):
    nL,nW,d=_layout(E,K,kind)
    z=np.zeros(d)
    z[-2]=0.0
    z[-1]=-1.8
    return z


def warp(X, theta, E, K, kind, centers=None, radius=None):
    """Training-feature map, with PSD global/local metric via J_phi' J_phi.

    phi(x)=Lx + sum_r W_r exp(-||x-c_r||^2/(2h^2)).
    Non-stationary via input warp; RBF(phi(x),phi(x')) is PSD.
    """
    L,W,_,_=_unpack(theta,E,K,kind)
    transformed=X@L.T
    if kind=='local':
        R2=cdist(X,centers,'sqeuclidean')
        gates=np.exp(-0.5*R2/radius**2)
        transformed=transformed+gates@W
    return transformed


def local_metric(X,theta,E,K,centers,radius):
    """Evaluate positive semidefinite local metric J_phi(x)^T J_phi(x)."""
    L,W,_,_=_unpack(theta,E,K,'local')
    difference=X[:,None,:]-centers[None,:,:]         # (n,K,E)
    gates=np.exp(-0.5*np.sum(difference**2,axis=2)/radius**2)
    grad_g=-gates[:,:,None]*difference/(radius**2) # (n,K,E)
    J=L[None,:,:]+np.einsum('ko,nkj->noj',W,grad_g) # (n,E,E)
    return np.einsum('nki,nkj->nij',J,J)


def _prior_log(theta,E,K,kind):
    nL,nW,_=_layout(E,K,kind)
    if kind=='diag':
        p=-.5*np.sum((theta[:nL]/0.8)**2)
    else:
        tr=np.tril_indices(E)
        off=(tr[0]!=tr[1]); diag=~off
        v=theta[:nL]
        p=-.5*(np.sum((v[diag]/0.8)**2)+np.sum((v[off]/0.6)**2))
    if nW:
        p-=0.5*np.sum((theta[nL:nL+nW]/0.4)**2)
    p-=0.5*(theta[-2]/1.2)**2
    p-=0.5*((theta[-1]+1.8)/1.0)**2
    return float(p)


class GPModel:
    def __init__(self,X,y,E,kind,K,centers=None,radius=None):
        self.X=np.asarray(X,float)
        self.y=np.asarray(y,float)
        self.E=E;self.kind=kind
        self.K=K if kind=='local' else 0
        self.centers=centers;self.radius=radius

    def _kernel(self,A,B,signal):
        return signal**2*np.exp(-.5*cdist(A,B,'sqeuclidean'))

    def log_posterior(self,theta):
        try:
            with np.errstate(over='raise',invalid='raise',under='ignore'):
                F=warp(self.X,theta,self.E,self.K,self.kind,self.centers,self.radius)
                _,_,signal,noise=_unpack(theta,self.E,self.K,self.kind)
                cov=self._kernel(F,F,signal)
                cov.flat[::len(cov)+1]+=noise**2+1e-7
                chol,lower=cho_factor(cov,lower=True,check_finite=False)
                alpha=cho_solve((chol,lower),self.y,check_finite=False)
                logm=-.5*self.y@alpha-np.log(np.diag(chol)).sum()-.5*len(self.y)*np.log(2*np.pi)
                return float(logm+_prior_log(theta,self.E,self.K,self.kind))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return -1e25

    def fit_map(self,maxiter):
        theta=_theta0(self.E,self.K,self.kind)
        nL,nW,_=_layout(self.E,self.K,self.kind)
        bounds=[(-2.5,2.5)]*nL+[(-1.5,1.5)]*nW+[(-2.5,2.5),(-4.5,0.7)]
        res=minimize(lambda t:-self.log_posterior(t),theta,method='L-BFGS-B',
                     bounds=bounds,options={'maxiter':maxiter,'ftol':1e-7})
        if not np.isfinite(res.fun):
            raise RuntimeError('MAP failed')
        return np.asarray(res.x),{'map_success':bool(res.success),'map_message':str(res.message),
                                  'log_posterior_map':float(-res.fun),'map_iterations':int(res.nit)}

    def predict(self,theta,Xtest):
        tr=warp(self.X,theta,self.E,self.K,self.kind,self.centers,self.radius)
        te=warp(Xtest,theta,self.E,self.K,self.kind,self.centers,self.radius)
        _,_,signal,noise=_unpack(theta,self.E,self.K,self.kind)
        cov=self._kernel(tr,tr,signal)
        cov.flat[::len(cov)+1]+=noise**2+1e-7
        chol,lower=cho_factor(cov,lower=True,check_finite=False)
        alpha=cho_solve((chol,lower),self.y,check_finite=False)
        cross=self._kernel(tr,te,signal)
        mu=cross.T@alpha
        v=solve_triangular(chol,cross,lower=True,check_finite=False)
        var=np.maximum(signal**2+noise**2-np.sum(v**2,axis=0),1e-9)
        return mu,np.sqrt(var)


def _mcmc_diagnostics(chains_array):
    C,N,D=chains_array.shape
    if N<8:return {'max_split_rhat':None,'min_ess_approx':None}
    half=N//2
    x=np.concatenate([chains_array[:,:half,:],chains_array[:,-half:,:]],axis=0)
    n=x.shape[1]
    W=x.var(axis=1,ddof=1).mean(axis=0)
    B=n*x.mean(axis=1).var(axis=0,ddof=1)
    rhat=np.sqrt(np.maximum(((n-1)*W/n+B/n)/np.maximum(W,1e-12),0))
    ess=[]
    for j in range(D):
        z=chains_array[:,:,j]-chains_array[:,:,j].mean(axis=1,keepdims=True)
        den=float(np.mean(np.sum(z*z,axis=1)/N))
        if den<=1e-12:
            ess.append(float('nan'));continue
        tau=1.
        for lag in range(1,max(2,N//2)):
            ac=float(np.mean(np.sum(z[:,:-lag]*z[:,lag:],axis=1)/(N-lag))/den)
            if not np.isfinite(ac) or ac<=0:break
            tau+=2*ac
        ess.append(float(C*N/min(max(tau,1.),C*N)))
    rh=float(np.nanmax(rhat)) if np.any(np.isfinite(rhat)) else float('nan')
    es=float(np.nanmin(ess)) if np.any(np.isfinite(ess)) else float('nan')
    return {'max_split_rhat':rh,'min_ess_approx':es,
            'diagnostic_warning':bool(not np.isfinite(rh) or rh>1.05 or es<100),
            'diagnostic_note':'это только аппроксимация'}


def _prior_vectors(E,K,kind):
    nL,nW,d=_layout(E,K,kind)
    mu=np.zeros(d);sigma=np.ones(d)
    if kind=='diag':
        sigma[:nL]=.8
    else:
        t=np.tril_indices(E)
        sigma[:nL]=np.where(t[0]==t[1],.8,.6)
    sigma[nL:nL+nW]=.4
    sigma[-2]=1.2
    mu[-1]=-1.8;sigma[-1]=1.0
    return mu,sigma


def sample_elliptical_slice(gp:GPModel,map_theta,warmup,steps,thin,chains,seed):
    mean,sd=_prior_vectors(gp.E,gp.K,gp.kind)
    nL,nW,_=_layout(gp.E,gp.K,gp.kind)
    blocks=[np.arange(nL)]
    if nW: blocks.append(np.arange(nL,nL+nW))
    blocks.append(np.arange(nL+nW,nL+nW+2))
    chains_draws=[];proposal_counts=[]
    for chain_id in range(chains):
        rng=np.random.default_rng(seed+1009*(chain_id+1))
        theta=map_theta+rng.normal(size=len(map_theta))*sd*.05
        def log_likelihood(a):
            return gp.log_posterior(a)-_prior_log(a,gp.E,gp.K,gp.kind)
        ll=log_likelihood(theta)
        saved=[];n_try=0;n_updates=0
        for i in range(warmup+steps):
            for cols in blocks:
                nu=rng.normal(size=len(cols))*sd[cols]
                center=theta[cols]-mean[cols]
                log_slice=ll+np.log(rng.uniform())
                angle=rng.uniform(0,2*np.pi)
                lo=angle-2*np.pi;hi=angle
                for attempt in range(250):
                    trial=theta.copy()
                    trial[cols]=mean[cols]+center*np.cos(angle)+nu*np.sin(angle)
                    test_ll=log_likelihood(trial)
                    n_try+=1
                    if test_ll>log_slice:
                        theta=trial;ll=test_ll;n_updates+=1
                        break
                    if angle<0:lo=angle
                    else:hi=angle
                    angle=rng.uniform(lo,hi)
                else:
                    raise RuntimeError('')
            if i>=warmup and (i-warmup)%thin==0:
                saved.append(theta.copy())
        chains_draws.append(np.asarray(saved))
        proposal_counts.append(n_try/max(n_updates,1))
    stack=np.stack(chains_draws)
    flat=stack.reshape(-1,stack.shape[-1])
    return flat,{'sampler':'block elliptical slice sampling',
                 'chains':chains,'warmup_each':warmup,'proposals_each':steps,
                 'thinning':thin,'posterior_draws':len(flat),
                 'average_loglikelihood_evaluations_per_block_update':proposal_counts,
                 **_mcmc_diagnostics(stack),'chain_draws':stack}


def predictive_mixture(gp,draws,Xtest,ytest=None):
    mus=[]; sig=[]
    for theta in draws:
        mu,s=gp.predict(theta,Xtest)
        mus.append(mu);sig.append(s)
    mus=np.asarray(mus);sig=np.asarray(sig)
    mean=mus.mean(axis=0)
    var=(sig**2+mus**2).mean(axis=0)-mean**2
    sd=np.sqrt(np.maximum(var,1e-12))
    lo=np.min(mus-8*sig,axis=0)
    hi=np.max(mus+8*sig,axis=0)
    def mixture_quantile(q):
        left=lo.copy();right=hi.copy()
        for _ in range(35):
            mid=(left+right)/2
            cdf=ndtr((mid[None,:]-mus)/sig).mean(axis=0)
            left=np.where(cdf<q,mid,left)
            right=np.where(cdf>=q,mid,right)
        return (left+right)/2
    p05=mixture_quantile(.05);p95=mixture_quantile(.95)
    out={'mean':mean,'std':sd,'low90':p05,'high90':p95}
    if ytest is not None:
        z=(ytest[None,:]-mus)/sig
        logdens=-.5*z*z-np.log(sig)-.5*np.log(2*np.pi)
        out['pointwise_nlpd']=-(logsumexp(logdens,axis=0)-np.log(len(draws)))
        out['nlpd']=float(np.mean(out['pointwise_nlpd']))
    return out


def _mixture_nlpd_cached(method,full_pred,sl):
    return float(np.mean(full_pred["pointwise_nlpd"][sl]))


def point_gp_result(gp,theta,Xtest,ytest=None):
    mean,std=gp.predict(theta,Xtest)
    out={'mean':mean,'std':std,'low90':mean-1.64485362695*std,
         'high90':mean+1.64485362695*std}
    if ytest is not None:
        out['nlpd']=float(np.mean(-norm.logpdf(ytest,loc=mean,scale=std)))
    return out


def score(y,pred):
    y=np.asarray(y);mu=np.asarray(pred['mean'])
    rho=float(np.corrcoef(y,mu)[0,1]) if np.std(y)>1e-10 and np.std(mu)>1e-10 else float('nan')
    out={'rho':rho,'rmse':float(np.sqrt(np.mean((y-mu)**2))),
         'mae':float(np.mean(np.abs(y-mu))),'coverage90':None,'nlpd':None}
    if 'low90' in pred:
        out['coverage90']=float(np.mean((y>=pred['low90']) & (y<=pred['high90'])))
        out['nlpd']=pred.get('nlpd')
    return out


def prepare(dataset,system,seed,E,tau,quick):
    frame,xname,yname,origin=load_dataset(dataset,system,seed)
    frame=frame[['x','y']].copy().apply(pd.to_numeric,errors='coerce').dropna().reset_index(drop=True)
    if quick and len(frame)>900:frame=frame.iloc[-900:].reset_index(drop=True)
    if len(frame)<220:raise ValueError('')
    x=frame.x.to_numpy();y=frame.y.to_numpy()
    ix=np.arange((E-1)*tau,len(x))
    n=len(ix);a=int(n*.6);b=int(n*.8)
    end=int(ix[a-1])+1
    means=np.array([np.mean(x[:end]),np.mean(y[:end])])
    scales=np.array([np.std(x[:end]),np.std(y[:end])])
    if np.any(scales<1e-12):raise ValueError('')
    x=(x-means[0])/scales[0];y=(y-means[1])/scales[1]
    return {'directions':[('X_from_Y',delay_embed(y,E,tau),x[(E-1)*tau:],0),
                          ('Y_from_X',delay_embed(x,E,tau),y[(E-1)*tau:],1)],
            't':ix,'train':np.arange(a),'val':np.arange(a,b),'test':np.arange(b,n),
            'means':means,'scales':scales,'origin':origin,'names':(xname,yname),'input_count':len(frame)}


def fit_direction(label,X,y,time,train,val,test,cfg,rngseed,results_dir):
    E=cfg['embedding_dim'];tau=cfg['tau']
    n=min(cfg['max_train'],len(train))
    li=np.unique(np.linspace(0,len(train)-1,n).astype(int))
    Xlib=X[li];ylib=y[li];tlib=time[li]
    C,radius=_choose_centers(Xlib,cfg['n_centers'])
    models={};metadata={'direction':label,'library_rows':len(li),'centers':C.tolist(),'radius':radius}
    with threadpool_limits(limits=1,user_api='blas'):
        for kind in ('diag','full','local'):
            gp=GPModel(Xlib,ylib,E,kind,cfg['n_centers'],C,radius)
            theta,fit_info=gp.fit_map(cfg['map_maxiter'])
            models[kind]=(gp,theta)
            metadata[f'fit_{kind}']=fit_info
        post,diag=sample_elliptical_slice(models['local'][0],models['local'][1],
                  cfg['mcmc_warmup'],cfg['mcmc_steps'],cfg['mcmc_thin'],
                  cfg['mcmc_chains'],rngseed)
        chain_draws=diag.pop('chain_draws')
        metadata['posterior_diagnostics']=diag
        if diag.get('diagnostic_warning'):
            print(f'  WARNING MCMC {label}: Rhat={diag.get("max_split_rhat")}, ESS={diag.get("min_ess_approx")}; posterior results preliminary',flush=True)
        np.savez_compressed(results_dir/f'posterior_{label}.npz',theta_draws=post,chain_draws=chain_draws)
        eig=np.linalg.eigvalsh(local_metric(Xlib[:min(20,len(li))],models['local'][1],E,
                                        cfg['n_centers'],C,radius))
        metadata['min_local_metric_eigenvalue']=float(np.min(eig))
        Lfull=_unpack(models['full'][1],E,0,'full')[0]
        metadata['full_global_metric_M']= (Lfull.T @ Lfull).tolist()
        rows=[];pred_rows=[]
        both=np.concatenate([val,test])
        Xt=X[both];Yt=y[both];Tt=time[both]
        all_predictions={
            'CCM':{'mean':simplex_predict(Xlib,ylib,tlib,Xt,Tt,E,tau)},
            'Metric-CCM':{'mean':simplex_predict(Xlib,ylib,tlib,Xt,Tt,E,tau,
                                         _unpack(models['full'][1],E,0,'full')[0])},
            'GP-ARD':point_gp_result(*models['diag'],Xt,Yt),
            'GP-Full':point_gp_result(*models['full'],Xt,Yt),
            'BAM-MAP':point_gp_result(*models['local'],Xt,Yt),
            'BAM-Bayes':predictive_mixture(models['local'][0],post,Xt,Yt),
        }
        for split,index,sl in [('validation',val,slice(0,len(val))),
                                ('test',test,slice(len(val),len(both)))]:
            yt=y[index];tt=time[index]
            for method,full_pred in all_predictions.items():
                p={k:(v[sl] if isinstance(v,np.ndarray) else v)
                   for k,v in full_pred.items() if k not in ('nlpd','pointwise_nlpd')}
                if 'std' in p:
                    if method=='BAM-Bayes':
                        p['nlpd']=_mixture_nlpd_cached(method,full_pred,sl)
                    else:
                        p['nlpd']=float(np.mean(-norm.logpdf(yt,loc=p['mean'],scale=p['std'])))
                metrics=score(yt,p)
                rows.append({'direction':label,'split':split,'model':method,
                             'n_library':len(li),'n_queries':len(index),**metrics})
                if split=='test':
                    for j,t in enumerate(tt):
                        pred_rows.append({'direction':label,'model':method,'time_index':int(t),
                                         'y_true_scaled':float(yt[j]),'mean_scaled':float(p['mean'][j]),
                                         'low90_scaled':float(p['low90'][j]) if 'low90'in p else np.nan,
                                         'high90_scaled':float(p['high90'][j]) if 'high90'in p else np.nan})
        curves=[]
        if cfg['metric_curves']:
            for size in sorted(set([max(E+6,int(len(li)*f)) for f in (.35,.55,.75,1.0)])):
                sub=li[:size]
                for name,L in [('CCM',None),('Metric-CCM',_unpack(models['full'][1],E,0,'full')[0])]:
                    pred=simplex_predict(X[sub],y[sub],time[sub],X[test],time[test],E,tau,L)
                    curves.append({'direction':label,'method':name,'library_size':len(sub),
                                   'rho_test':score(y[test],{'mean':pred})['rho']})
            metadata['convergence_protocol']='held-out chronological test; nested subsampled training libraries; metric fixed from full train'
    return rows,pred_rows,curves,metadata


def evaluate(cfg):
    data=prepare(cfg['dataset'],cfg['system'],cfg['seed'],cfg['embedding_dim'],cfg['tau'],cfg['quick'])
    suffix=f"{cfg['dataset']}_{cfg['system']}" if cfg['dataset']=='synthetic' else cfg['dataset']
    out=RESULTS/(suffix + ('_' + str(cfg['output_tag']) if cfg.get('output_tag') else ''));out.mkdir(parents=True,exist_ok=True)
    info={'config':cfg,'data_file':data['origin'],'variables':data['names'],
          'input_rows_used':data['input_count'],
          'train_rows':len(data['train']),'validation_rows':len(data['val']),'test_rows':len(data['test']),
          'posterior_interpretation':'Posterior for GP parameters/metric, NOT P(causal arrow | data)',
          'causal_pvalues_computed':False,
          'reference':'Heldout E+1 simplex baseline, not identical to random-library pyEDM CCM',
          'warnings':['No calibrated causal significance test; cross-map reconstruction does not prove a causal edge',
                      'Short MCMC chains in quick mode require diagnostic checks',
                      'BAM-CCM is nonlinear GP cross mapping: separate convergence and null tests are needed']}
    metrics=[];pred=[];curves=[]
    for i,(name,X,y,target_axis) in enumerate(data['directions']):
        print(f"\n{name}: rows={len(X)}  train={len(data['train'])} val={len(data['val'])} test={len(data['test'])}",flush=True)
        rr,pp,cc,dd=fit_direction(name,X,y,data['t'],data['train'],data['val'],data['test'],cfg,cfg['seed']+i*100000,out)
        metrics.extend(rr);pred.extend(pp);curves.extend(cc);info[name]=dd
        for rec in rr:
            if rec['split']=='test':
                print(f"  {rec['model']:<11s} rho={rec['rho']:7.3f} RMSE={rec['rmse']:.3f} coverage90={rec['coverage90']}",flush=True)
    frame=pd.DataFrame(metrics)
    frame.to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(pred).to_csv(out/'test_predictions.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'convergence.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,5))
    for direction,sub in frame[frame.split=='test'].groupby('direction'):
        ax.plot(sub.model,sub.rho,marker='o',label=direction)
    ax.set_ylabel('Held-out correlation rho');ax.set_title('Exploratory cross map, NOT causality proof')
    ax.legend();ax.tick_params(axis='x',rotation=35);fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    if len(curves):
        fig,ax=plt.subplots(figsize=(9,5))
        for (direction,method),sub in pd.DataFrame(curves).groupby(['direction','method']):
            ax.plot(sub.library_size,sub.rho_test,marker='o',label=f'{direction}: {method}')
        ax.set(xlabel='Historical library size',ylabel='Held-out rho',
               title='Diagnostic convergence: fixed held-out prediction set')
        ax.legend();fig.tight_layout();fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    print(f'\nSaved {out}',flush=True)
    return frame


def reference_check_pyedm():
    try:
        import pyEDM
    except ImportError:
        print('скачать pyEDM')
        return False
    rng=np.random.default_rng(123)
    v=rng.normal(size=180)
    E=3
    Z=delay_embed(v,E,1)
    y=np.sin(v[2:])
    df=pd.DataFrame(Z,columns=['E0','E1','E2'])
    df['target']=y
    L=110
    res=pyEDM.Simplex(dataFrame=df,columns='E0 E1 E2',target='target',
                      lib=f'1 {L}',pred=f'{L+1} {len(df)}',embedded=True,
                      E=E,knn=E+1,Tp=0,exclusionRadius=0,noTime=True)
    actual=np.asarray(res['Predictions'],float)
    ours=simplex_predict(Z[:L],y[:L],np.arange(L),Z[L:],
                         np.arange(L,len(Z)),E,1,theiler_radius=0)
    if len(actual)!=len(ours):
        raise AssertionError(f'pyEDM comparison sizes differ: {len(actual)} vs {len(ours)}')
    ok=np.isfinite(actual)&np.isfinite(ours)
    if ok.sum()<30:raise AssertionError('')
    err=float(np.max(np.abs(actual[ok]-ours[ok])))
    print(f'n={ok.sum()}, max abs разница={err:.8g}')
    if err>1e-5:
        raise AssertionError('Mismatch with pyEDM')
    print('REFERENCE CHECK PASSED')
    return True


def self_test():
    arr=np.arange(8,dtype=float)
    np.testing.assert_allclose(delay_embed(arr,3,2),[[4,2,0],[5,3,1],[6,4,2],[7,5,3]])
    X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.],[4.,4.],[5.,5.]])
    p=simplex_predict(X,np.arange(6.),np.arange(6),np.array([[2.1,2.1]]),np.array([30]),2,1)
    assert np.isfinite(p).all()
    E=3;K=2;kind='local'
    theta=_theta0(E,K,kind)
    C=np.array([[-1.,-1.,-1.],[1.,1.,1.]])
    zz=np.random.default_rng(12).normal(size=(30,E))
    phi=warp(zz,theta,E,K,kind,C,1.)
    np.testing.assert_allclose(phi,zz)
    eigen=np.linalg.eigvalsh(local_metric(zz,theta,E,K,C,1.))
    assert eigen.min()>-1e-10
    cov=np.exp(-.5*cdist(phi,phi,'sqeuclidean'))
    assert np.linalg.eigvalsh(cov).min()>-1e-8
    for k in ('diag','full','local'):
        gp=GPModel(zz[:20],zz[:20,0],E,k,K,C,1.)
        th=_theta0(E,gp.K,k)
        assert np.isfinite(gp.log_posterior(th))
        mean,std=gp.predict(th,zz[20:])
        assert len(mean)==10 and np.all(std>0)
    fake=pd.Series(['$765.37','1,500.20'])
    np.testing.assert_allclose(_numeric(fake),[765.37,1500.20])
    print('SELF-TEST PASSED: embedding, simplex, symmetric PSD local kernel, GP, parser')




def simulate_logistic_independent(n, seed, coupling=0.05, noise=0.1, burnin=700):
    if not 0 <= coupling <= 0.5 or noise < 0:
        raise ValueError('Требуется 0<=coupling<=0.5 and noise>=0')
    rng=np.random.default_rng(seed)
    x,y=rng.uniform(.15,.85,size=2)
    arr=np.empty((n,2),float)
    for t in range(n+burnin):
        fx=3.78*x*(1-x); fy=3.67*y*(1-y)
        x,y=fx,(1-coupling)*fy+coupling*fx
        if t>=burnin:arr[t-burnin]=x,y
    if not np.isfinite(arr).all():raise ValueError('')
    noisy=arr+rng.normal(size=arr.shape)*noise*arr.std(axis=0)
    return pd.DataFrame(noisy,columns=['x','y'])


def benchmark_logistic(cfg,repeats,noise,coupling):
    if repeats<2:raise ValueError('--repeats must be at least 2')
    if cfg['mcmc_warmup']<100 or cfg['mcmc_steps']<200:
        print('WARNING: SHORT MCMC. Диагностический benchmark',flush=True)
    import hashlib
    tag=hashlib.sha256(f'v06|logistic|n=1000|burnin=700|coupling={coupling:.17g}|noise={noise:.17g}'.encode()).hexdigest()[:12]
    benchmark_system=f'benchmark_logistic_{tag}'
    out=RESULTS/f'benchmark_v06_logistic_{tag}_seed{cfg["seed"]}';out.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    for rep in range(repeats):
        seed=int(cfg['seed'])+rep*12347
        data=simulate_logistic_independent(1000,seed,coupling,noise)
        file=DATA/'synthetic'/f'{benchmark_system}_seed{seed}.csv'
        if file.exists():
            existing=pd.read_csv(file)
            if existing.shape!=data.shape or not np.allclose(existing[['x','y']].to_numpy(),data[['x','y']].to_numpy(),rtol=1e-9,atol=1e-9):
                raise FileExistsError(f'Existing seed file differs; refusing overwrite: {file}')
        else:
            file.parent.mkdir(parents=True,exist_ok=True)
            data.to_csv(file,index=False)
        rcfg={**cfg,'dataset':'synthetic','system':benchmark_system,'seed':seed,
              'quick':False,'metric_curves':False,'output_tag':f'bench_v06_{rep:03d}'}
        print(f'\nBENCHMARK {rep+1}/{repeats}, seed={seed}',flush=True)
        table=evaluate(rcfg)
        table=table[table['split']=='test'].copy()
        table['replicate']=rep;table['seed']=seed;table['coupling']=coupling;table['noise']=noise
        all_rows.append(table)
        pd.concat(all_rows,ignore_index=True).to_csv(out/'all_metrics.csv',index=False)
    all_df=pd.concat(all_rows,ignore_index=True)
    records=[]
    for direction,sub in all_df.groupby('direction'):
        wide=sub.pivot(index='replicate',columns='model',values='rmse')
        for comparator in ('CCM','GP-ARD','BAM-MAP'):
            delta=(wide[comparator]-wide['BAM-Bayes']).dropna().to_numpy()
            rng=np.random.default_rng(2026)
            boots=np.mean(rng.choice(delta,(4000,len(delta)),replace=True),axis=1)
            records.append({'direction':direction,'comparator':comparator,
                            'mean_RMSE_reduction':float(np.mean(delta)),
                            'ci95_low':float(np.quantile(boots,.025)),
                            'ci95_high':float(np.quantile(boots,.975)),
                            'replicates':len(delta),
                            'positive_replicate_fraction':float(np.mean(delta>0))})
    summary=pd.DataFrame(records);summary.to_csv(out/'paired_summary.csv',index=False)
    (out/'PROTOCOL.txt').write_text(
    'Независимые начальные условия логистических отображений и шум измерений.\n'
    'Парный ресемплинг по независимым реализациям траекторий. Доверительные интервалы носят предварительный характер.\n'
    'Причинные p-значения, TPR и FPR не рассчитываются. Точность GP-реконструкции сама по себе не доказывает причинную связь.\n'
    'Перед использованием байесовских результатов необходимо проверить диагностику MCMC для каждой реализации.\n'
    'Не следует подбирать гиперпараметры по результатам итоговой тестовой выборки.\n',
    encoding='utf-8'
)

    print('\nСВОДКА ПО НЕЗАВИСИМЫМ РЕАЛИЗАЦИЯМ\n',
      summary.to_string(index=False), flush=True)
    print('Результаты сохранены в:', out, flush=True)

    return summary 


def main():
    parser = argparse.ArgumentParser(
        description='BAM-CCM v0.6: байесовский подход с полной и локально адаптивной метрикой'
    )

    parser.add_argument('--dataset', default=CONFIG['dataset'])
    parser.add_argument('--system', default=CONFIG['system'])
    parser.add_argument('--E', type=int, default=CONFIG['embedding_dim'])
    parser.add_argument('--tau', type=int, default=CONFIG['tau'])
    parser.add_argument('--max-train', type=int, default=CONFIG['max_train'])
    parser.add_argument('--mcmc-steps', type=int, default=CONFIG['mcmc_steps'])
    parser.add_argument('--mcmc-chains', type=int, default=CONFIG['mcmc_chains'])
    parser.add_argument('--mcmc-thin', type=int, default=CONFIG['mcmc_thin'])
    parser.add_argument(
        '--output-tag',
        default='',
        help='Уникальный суффикс для предотвращения перезаписи предыдущих результатов'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Сравнительное тестирование на независимых реализациях синтетической логистической системы'
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=5,
        help='Количество независимых реализаций: от 3 для предварительной проверки, от 30 для исследования'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=.10,
        help='Относительный уровень шума измерений'
    )
    parser.add_argument(
        '--coupling',
        type=float,
        default=.05,
        help='Сила связи между переменными в тестовой системе'
    )
    parser.add_argument('--mcmc-warmup', type=int, default=CONFIG['mcmc_warmup'])
    parser.add_argument('--map-maxiter', type=int, default=CONFIG['map_maxiter'])
    parser.add_argument('--n-centers', type=int, default=CONFIG['n_centers'])
    parser.add_argument(
        '--full',
        action='store_true',
        help='Использовать весь реальный набор данных без ограничения последними 900 наблюдениями'
    )
    parser.add_argument('--no-curves', action='store_true')
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument(
        '--reference-check',
        action='store_true',
        help='Сравнить результаты simplex на отложенной выборке с официальной реализацией pyEDM'
    )
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    if args.reference_check:
        return reference_check_pyedm()

    cfg = CONFIG.copy()

    cfg.update({
        'dataset': args.dataset,
        'system': args.system,
        'embedding_dim': args.E,
        'tau': args.tau,
        'max_train': args.max_train,
        'mcmc_steps': args.mcmc_steps,
        'mcmc_warmup': args.mcmc_warmup,
        'mcmc_chains': args.mcmc_chains,
        'mcmc_thin': args.mcmc_thin,
        'output_tag': args.output_tag,
        'map_maxiter': args.map_maxiter,
        'n_centers': args.n_centers,
        'seed': args.seed,
        'quick': cfg['quick'] and not args.full,
        'metric_curves': cfg['metric_curves'] and not args.no_curves
    })

    if (
        cfg['max_train'] < 30
        or cfg['n_centers'] < 1
        or cfg['n_centers'] > 8
        or cfg['mcmc_steps'] < 10
        or cfg['mcmc_warmup'] < 0
        or cfg['mcmc_chains'] < 2
        or cfg['mcmc_thin'] < 1
    ):
        parser.error(
            'Требования к параметрам: max-train >= 30, '
            '1 <= n-centers <= 8, '
            'не менее 10 шагов MCMC и не менее 2 цепей'
        )

    if args.benchmark:
        return benchmark_logistic(
            cfg, args.repeats, args.noise, args.coupling
        )

    evaluate(cfg)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError) as e:
        print(f'ОШИБКА ВХОДНЫХ ДАННЫХ: {e}', file=sys.stderr)
        sys.exit(2)