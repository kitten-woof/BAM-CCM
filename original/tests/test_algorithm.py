import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from bam_ccm import (delay_embed, _theta0, _unpack, warp, local_metric,
                     GPModel, simplex_predict, predictive_mixture,
                     sample_elliptical_slice, read_market, score, _choose_centers)


def test_embedding_lag():
    x=np.arange(12.)
    np.testing.assert_array_equal(delay_embed(x,3,2)[0],[4.,2.,0.])


def test_knn_reference_hand_computation():
    X=np.array([[0.,0.],[1.,0.],[1.,1.],[3.,3.],[4.,4.],[5.,5.]])
    y=np.array([5.,8.,4.,9.,2.,7.])
    Z=np.array([[1.8,1.8]])
    d=cdist(Z,X)[0]
    order=np.argsort(d)[:3]
    w=np.exp(-d[order]/d[order].min());w/=w.sum()
    expected=np.sum(w*y[order])
    got=simplex_predict(X,y,np.arange(len(X)),Z,np.array([100]),E=2,tau=1)
    np.testing.assert_allclose(got,expected)


def test_identical_states_zero_distance():
    X=np.array([[0.,0.],[1.,1.],[1.,1.],[2.,2.],[3.,3.],[4.,4.]])
    y=np.array([0.,2.,4.,1.,6.,7.])
    out=simplex_predict(X,y,np.arange(6),np.array([[1.,1.]]),np.array([100]),2,1)
    assert abs(out[0]-3)<1e-12


def test_full_metric_psd_and_nondiagonal():
    th=_theta0(3,0,'full')
    th[1]=.7
    L,_,_,_=_unpack(th,3,0,'full')
    M=L.T@L
    np.testing.assert_allclose(M,M.T)
    assert np.linalg.eigvalsh(M).min()>0
    assert np.max(np.abs(M-np.diag(np.diag(M))))>0


def test_nonstationary_metric_finite_difference():
    E=3;K=2
    th=_theta0(E,K,'local')
    th[6:12]=np.array([.1,-.2,.3,.3,.1,-.4])
    C=np.array([[-1.,0.,1.],[1.,1.,-1.]])
    x=np.array([[.2,.3,-.5]])
    h=1e-5
    base=warp(x,th,E,K,'local',C,1.)[0]
    J=np.zeros((E,E))
    for j in range(E):
        z=x.copy();z[0,j]+=h
        Z=x.copy();Z[0,j]-=h
        J[:,j]=(warp(z,th,E,K,'local',C,1.)[0]-warp(Z,th,E,K,'local',C,1.)[0])/(2*h)
    np.testing.assert_allclose(local_metric(x,th,E,K,C,1.)[0],J.T@J,atol=1e-7)


def test_nonstationary_kernel_psd():
    rng=np.random.default_rng(1)
    E=3;K=2
    th=_theta0(E,K,'local');th[6:12]=rng.normal(size=6)*.3
    X=rng.normal(size=(35,E))
    C=X[:K]
    T=warp(X,th,E,K,'local',C,1.)
    gram=np.exp(-cdist(T,T,'sqeuclidean')/2)
    assert np.linalg.eigvalsh(gram).min()>-1e-8


def test_mcmc_posterior_reproducibility_and_noise_uncertainty():
    rng=np.random.default_rng(3)
    X=rng.normal(size=(30,2));y=np.sin(X[:,0])+.1*rng.normal(size=30)
    C,h=_choose_centers(X,1)
    gp=GPModel(X,y,2,'local',1,C,h)
    th=_theta0(2,1,'local')
    s1,meta1=sample_elliptical_slice(gp,th,10,20,2,2,47)
    s2,meta2=sample_elliptical_slice(gp,th,10,20,2,2,47)
    np.testing.assert_array_equal(s1,s2)
    assert len(s1)==20
    pred=predictive_mixture(gp,s1,X[:4],y[:4])
    assert all(pred['high90']>=pred['low90'])
    assert np.all(pred['std']>0)
    assert np.isfinite(pred['nlpd'])


def test_score_perfect():
    res=score(np.array([1.,2.,3.]),{'mean':np.array([1.,2.,3.])})
    assert res['rmse']==0
    np.testing.assert_allclose(res['rho'],1.)


def test_synthetic_parser():
    from bam_ccm import _numeric, _dates
    assert _numeric(pd.Series(['$765.37','1,000.42'])).tolist()==[765.37,1000.42]
    assert _dates(pd.Series(['10/26/2026'])).iloc[0].year==2026


def test_nasdaq_csv_loader_and_order(tmp_path, monkeypatch):
    import bam_ccm
    d=tmp_path/'raw'/'financial'
    d.mkdir(parents=True)
    dates=pd.bdate_range('2025-01-01',periods=130)
    source=pd.DataFrame({'Date':dates.strftime('%m/%d/%Y'),
                         'Close/Last':['$'+f'{100+j/10:.2f}' for j in range(130)],
                         'Volume':['1,234,567']*130})
    source.iloc[::-1].to_csv(d/'HistoricalData_1793127119371_QQQ.csv',index=False)
    monkeypatch.setattr(bam_ccm,'DATA',tmp_path)
    out=bam_ccm.read_market('HistoricalData_1793127119371_QQQ.csv')
    assert len(out)==130
    assert out['date'].is_monotonic_increasing
    assert out.price.iloc[0]==100.
    assert out.volume.iloc[0]==1234567


def test_chronological_split_training_scale_only(monkeypatch):
    import bam_ccm
    n=300
    x=np.arange(n,dtype=float)
    y=x.copy()
    x[250:]+=5000
    y[250:]+=5000
    monkeypatch.setattr(bam_ccm,'load_dataset',lambda a,b,c:(pd.DataFrame({'x':x,'y':y}),'x','y','fixture'))
    z=bam_ccm.prepare('fixture','fixture',1,3,1,False)
    assert max(z['train'])<min(z['test'])
    np.testing.assert_allclose(z['means'][0],np.mean(x[:z['t'][z['train'][-1]]+1]))
    assert z['means'][0]<200
