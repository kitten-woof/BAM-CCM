import json,sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from generate_synthetic import generate
from evaluate import _direction
from metric_refinement import AblationGP
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'original'))
import bam_ccm as bam

def main():
    df,_=generate('logistic',600,131,.1,.2)
    dirs,t,tr,va,te,*_=_direction(df)
    out=[]
    for direction,X,y in dirs:
        li=np.unique(np.linspace(0,len(tr)-1,65,dtype=int));xl=X[li];yl=y[li]
        C,r=bam._choose_centers(xl,2)
        with threadpool_limits(limits=1,user_api='blas'):
            for kind in ['diag','full','local']:
                native=bam.GPModel(xl,yl,3,kind,2,C,r)
                tn,dn=native.fit_map(65)
                fresh=AblationGP(xl,yl,kind,C,r)
                tf,dfit=fresh.fit(65)
                a=float(-native.log_posterior(tn));b=float(fresh.objective(tf)[0])
                pa=native.predict(tn,X[te])[0];pb=fresh.predict(tf,X[te])
                maxpred=float(np.max(abs(pa-pb)))
                rec={'direction':direction,'kind':kind,'negative_log_posterior_native':a,
                     'negative_log_posterior_new':b,'objective_abs_diff':abs(a-b),
                     'prediction_max_abs_diff':maxpred,
                     'objective_match':bool(abs(a-b)<1e-3),
                     'prediction_match':bool(maxpred<1e-3)}
                out.append(rec)
                assert rec['objective_match'] and rec['prediction_match'],rec
    file=R/'experiments/results/metric_research/parity.json'
    file.write_text(json.dumps({'test':'logistic 600 obs, seed 131, noise .1, coupling .2, E3, train-only scaling, 65 library, MAP maxiter65',
          'records':out},indent=2))
    print('Original MAP parity:',len(out),'checks passed; output:',file)
if __name__=='__main__':main()
