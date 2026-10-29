from __future__ import annotations
from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import bam_ccm as bam


class AblationGP:
    def __init__(self,X,y,kind,centers=None,radius=None,lam_jac=0.):
        self.X=np.asarray(X,float);self.y=np.asarray(y,float);self.kind=kind
        self.E=self.X.shape[1];self.centers=centers;self.radius=radius
        self.K=len(centers) if kind=='local' else 0
        self.lam_jac=float(lam_jac)
        if kind=='local':
            self.gates=np.exp(-.5*cdist(self.X,centers,'sqeuclidean')/(radius**2))
            self.dg=-self.gates[:,:,None]*(self.X[:,None,:]-centers[None,:,:])/(radius**2)
        else:self.gates=None;self.dg=None
        self.nL=0 if kind=='fixed' else bam._layout(self.E,self.K,kind)[0]
        self.nW=self.K*self.E
        self.dim=self.nL+self.nW+2

    def initial(self):
        if self.kind=='fixed':return np.array([0.,-1.8])
        return bam._theta0(self.E,self.K,self.kind)

    def features(self,X,theta):
        if self.kind=='fixed':return np.asarray(X,float)
        return bam.warp(np.asarray(X,float),theta,self.E,self.K,self.kind,self.centers,self.radius)

    def objective(self,theta):
        try:
            if self.kind=='fixed':
                F=self.X
                L=None;W=None
            else:
                F=self.features(self.X,theta)
                L,W,_,_=bam._unpack(theta,self.E,self.K,self.kind)
            signal,noise=np.exp(theta[-2:])
            M=signal**2*np.exp(-.5*cdist(F,F,'sqeuclidean'))
            K=M.copy();K.flat[::len(K)+1]+=noise**2+1e-7
            chol,lower=cho_factor(K,lower=True,check_finite=False)
            alpha=cho_solve((chol,lower),self.y,check_finite=False)
            logm=-.5*self.y@alpha-np.log(np.diag(chol)).sum()-.5*len(self.y)*np.log(2*np.pi)
            B=np.outer(alpha,alpha)-cho_solve((chol,lower),np.eye(len(K)),check_finite=False)
            BM=B*M
            grad=np.zeros(len(theta),float)
            if self.kind!='fixed':
                GF=BM@F-BM.sum(axis=1)[:,None]*F
                GL=GF.T@self.X
                if self.kind=='diag':
                    grad[:self.E]=np.diag(GL)*np.diag(L)
                else:
                    r,c=np.tril_indices(self.E)
                    grad[:self.nL]=GL[r,c]*np.where(r==c,np.diag(L)[r],1.)
                if self.kind=='local':
                    GW=self.gates.T@GF
                    grad[self.nL:self.nL+self.nW]=GW.ravel()
                if self.kind=='diag':grad[:self.nL]-=theta[:self.nL]/(.8**2)
                else:
                    r,c=np.tril_indices(self.E)
                    scale=np.where(r==c,.8,.6)
                    grad[:self.nL]-=theta[:self.nL]/scale**2
                if self.nW:grad[self.nL:self.nL+self.nW]-=theta[self.nL:self.nL+self.nW]/(.4**2)
                prior=bam._prior_log(theta,self.E,self.K,self.kind)
            else:
                prior=-.5*(theta[-2]/1.2)**2-.5*((theta[-1]+1.8)/1.)**2
            grad[-2]=np.sum(BM)-theta[-2]/(1.2**2)
            grad[-1]=noise**2*np.trace(B)-(theta[-1]+1.8)
            obj=-float(logm+prior); grad=-grad
            if self.kind=='local' and self.lam_jac>0:
                # здесь R_J=lambda/(2n) sum_i || sum_r W_r outer grad(g_r)(x_i) ||_F^2.
                displacement=np.einsum('ro,nre->noe',W,self.dg)
                obj+=self.lam_jac/(2*len(self.X))*np.sum(displacement**2)
                gw=(self.lam_jac/len(self.X))*np.einsum('noe,nre->ro',displacement,self.dg)
                grad[self.nL:self.nL+self.nW]+=gw.ravel()
            return obj,grad
        except (np.linalg.LinAlgError,FloatingPointError,ValueError,OverflowError):
            return 1e25,np.zeros_like(theta)

    def fit(self,maxiter=75,starts=None):
        starts=starts or [self.initial()]
        bounds=([(-2.5,2.5)]*self.nL+[(-1.5,1.5)]*self.nW+
                [(-2.5,2.5),(-4.5,.7)])
        best=None;allruns=[]
        for init in starts:
            r=minimize(self.objective,np.asarray(init,float),jac=True,method='L-BFGS-B',bounds=bounds,
                       options={'maxiter':maxiter,'ftol':1e-7})
            allruns.append((bool(r.success),float(r.fun),int(r.nit)))
            if best is None or r.fun<best.fun:best=r
        return np.asarray(best.x),{'converged':bool(best.success),'objective':float(best.fun),
                                    'iterations':int(best.nit),'starts':len(starts),
                                    'all_runs':allruns}

    def predict(self,theta,X):
        F=self.features(self.X,theta);T=self.features(X,theta)
        signal,noise=np.exp(theta[-2:])
        K=signal**2*np.exp(-.5*cdist(F,F,'sqeuclidean'));K.flat[::len(K)+1]+=noise**2+1e-7
        C=signal**2*np.exp(-.5*cdist(F,T,'sqeuclidean'))
        cf=cho_factor(K,lower=True,check_finite=False)
        return C.T@cho_solve(cf,self.y,check_finite=False)


def local_start(base_theta,base_kind,E,K):
    if base_kind=='diag':
        t=np.zeros(bam._layout(E,K,'local')[2])
        r,c=np.tril_indices(E);t[:len(r)]=np.where(r==c,base_theta[r],0.)
    elif base_kind=='full':
        t=np.zeros(bam._layout(E,K,'local')[2]);nL=E*(E+1)//2;t[:nL]=base_theta[:nL]
    else:raise ValueError(base_kind)
    t[-2:]=base_theta[-2:]
    return t
