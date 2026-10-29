import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from modified.bam_ccm_v07 import RefinedBAM
import numpy as np
import pytest

def test_refined_public_api():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(45,3));y=x[:,0]**2+.2*rng.normal(size=45)
    model=RefinedBAM(K=2,maxiter=20,jacobian_weight=20).fit(x[:35],y[:35])
    pred=model.predict(x[35:]);assert pred.shape==(10,)
    assert np.isfinite(pred).all()
    assert np.isfinite(model.predict_ard(x[35:])).all()
    assert model.diagnostics()['n_library']==35

def test_api_validation():
    with pytest.raises(RuntimeError):RefinedBAM().predict(np.zeros((4,3)))
    with pytest.raises(ValueError):RefinedBAM(jacobian_weight=-1).fit(np.zeros((10,3)),np.arange(10))
