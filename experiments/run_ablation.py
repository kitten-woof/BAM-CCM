from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_synthetic import generate
from evaluate import evaluate_case
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    parts=[]
    for seed in [13,42,71]:
        df,meta=generate('logistic',600,seed,.1,.2)
        for centers in [1,2,4]:
            case=f'logistic_s{seed}_centers{centers}'
            x=evaluate_case(case,df,{'E':3,'tau':1,'library_size':65,'map_maxiter':65,
                      'centers':centers,'bayes':False,'seed':seed},
                      ROOT/'experiments/results/ablation'/case,meta)
            x['centers']=centers;x['seed']=seed;parts.append(x)
    pd.concat(parts,ignore_index=True).to_csv(ROOT/'experiments/results/ablation/summary.csv',index=False)
