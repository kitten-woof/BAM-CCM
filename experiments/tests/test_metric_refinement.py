import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import approx_fprime
from metric_refinement import AblationGP,local_start

def test_gradient_all_models():
    rng=np.random.default_rng(7);X=rng.normal(size=(15,3));y=rng.normal(size=15)
    for kind in ('fixed','diag','full','local'):
        for lv in ([0.,2.] if kind=='local' else [0.]):
            o=AblationGP(X,y,kind,X[:2],1.1,lv)
            t=o.initial()+rng.normal(size=o.dim)*.07
            f,g=o.objective(t)
            numeric=approx_fprime(t,lambda a:o.objective(a)[0],1e-6)
            assert np.allclose(g,numeric,rtol=4e-4,atol=2e-4), (kind,lv,np.max(abs(g-numeric)))

def test_nested_initializations():
    X=np.arange(90.).reshape(30,3)/30;y=np.sin(X[:,0])
    for k in ('diag','full'):
        gp=AblationGP(X,y,k)
        a=gp.initial()+.1
        loc=AblationGP(X,y,'local',X[:2],1.)
        z=local_start(a,k,3,2)
        np.testing.assert_allclose(loc.features(X,z),gp.features(X,a),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(z[-2:],a[-2:])

def test_new_gp_matches_original_implementation():
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'original'))
    import bam_ccm as bam
    rng=np.random.default_rng(6);X=rng.normal(size=(20,3));y=rng.normal(size=20)
    C,r=bam._choose_centers(X,2)
    for kind in ('diag','full','local'):
        o=AblationGP(X,y,kind,C,r)
        z=o.initial()+rng.normal(size=o.dim)*.08
        assert abs(-o.objective(z)[0]-bam.GPModel(X,y,3,kind,2,C,r).log_posterior(z))<1e-7
