"""BAM-Jac CCM(v0.7)"""

from __future__ import annotations
import sys
from pathlib import Path
from dataclasses import dataclass
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
sys.path.insert(0,str(ROOT/'original'))
from metric_refinement import AblationGP,local_start
import bam_ccm as original


@dataclass
class RefinedBAM:
    K:int=2
    jacobian_weight:float=20.0
    maxiter:int=65
    def fit(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y,float).reshape(-1)
        if X.ndim!=2 or X.shape[0]!=len(y) or len(y)<=X.shape[1]+4:
            raise ValueError('Expected aligned 2D embedding matrix and nontrivial 1D target')
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError('X and y must be finite')
        if self.K<1 or self.K>len(y):raise ValueError('K must be in [1, n_library]')
        if self.jacobian_weight<0:raise ValueError('Jacobian weight must be nonnegative')
        E=X.shape[1]
        C,r=original._choose_centers(X,self.K)
        self.params_={};self.models_={};self.fit_info_={}
        for kind in ('diag','full'):
            gp=AblationGP(X,y,kind,C,r)
            theta,info=gp.fit(self.maxiter)
            self.params_[kind]=theta;self.models_[kind]=gp;self.fit_info_[kind]=info
        gp=AblationGP(X,y,'local',C,r,self.jacobian_weight)
        starts=[gp.initial(),local_start(self.params_['diag'],'diag',E,self.K),
                             local_start(self.params_['full'],'full',E,self.K)]
        theta,info=gp.fit(self.maxiter,starts)
        self.params_['local']=theta;self.models_['local']=gp;self.fit_info_['local']=info
        self.E_=E;self.centers_=C;self.radius_=r;self.n_library_=len(y)
        return self

    def predict(self,X):
        if not hasattr(self,'params_'):raise RuntimeError('Call fit before predict')
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=self.E_:raise ValueError('Incorrect embedding dimension')
        return self.models_['local'].predict(self.params_['local'],X)

    def predict_ard(self,X):
        if not hasattr(self,'params_'):raise RuntimeError('Call fit before predict')
        return self.models_['diag'].predict(self.params_['diag'],X)

    def diagnostics(self):
        if not hasattr(self,'params_'):raise RuntimeError('Call fit before diagnostics')
        W=original._unpack(self.params_['local'],self.E_,self.K,'local')[1]
        return {'n_library':self.n_library_,'K':self.K,'jacobian_weight':self.jacobian_weight,
                'W_frobenius':float(np.linalg.norm(W)),
                'map':self.fit_info_,'radius':self.radius_}
