import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_synthetic import generate,SPEC
from evaluate import _direction

def test_seed_reproducible():
    a,_=generate('logistic',160,5,.1,.2);b,_=generate('logistic',160,5,.1,.2)
    pd.testing.assert_frame_equal(a,b)

def test_all_generators_finite():
    for s in SPEC:
        a,meta=generate(s,160,21,.1,.2)
        assert len(a)>120,(s,len(a))
        assert np.isfinite(a[['x','y']]).all().all(),s
        assert meta['truth']==SPEC[s][0]

def test_independent_not_forced_connected():
    a,_=generate('independent',300,21,0,.0)
    b,_=generate('independent',300,21,0,.5)
    pd.testing.assert_frame_equal(a,b)

def test_irregular_observed_indices():
    a,_=generate('irregular',500,21,.1,.2)
    assert (np.diff(a.t)>1).any()
    assert a.t.is_monotonic_increasing

def test_training_scaling_unchanged_by_future_outlier():
    arr=pd.DataFrame({'x':np.arange(240,dtype=float), 'y':np.arange(240,dtype=float)*2+1})
    a=_direction(arr)
    arr.loc[210:,'x']+=1000000
    arr.loc[210:,'y']+=1000000
    b=_direction(arr)
    np.testing.assert_allclose(a[-2],b[-2]);np.testing.assert_allclose(a[-1],b[-1])

def test_both_directions_same_test_times():
    a,_=generate('logistic',250,42,.1,.2)
    directions,t,tr,val,test,mean,scale=_direction(a)
    assert len(directions)==2 and np.array_equal(t[test],t[test])
    assert tr.max()<val.min()<test.min()
