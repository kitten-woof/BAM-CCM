from pathlib import Path
import argparse,json,sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate import evaluate_case
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--map-maxiter',type=int,default=65);p.add_argument('--library-size',type=int,default=65);a=p.parse_args()
    dst=ROOT/'experiments/results/archive_synthetic';dst.mkdir(parents=True,exist_ok=True)
    rows=[];errors=[]
    for source in sorted((ROOT/'original/data/synthetic').glob('*_seed42.csv')):
        sys_name=source.stem.split('_seed')[0]; meta=source.with_suffix('.json')
        frame=pd.read_csv(source)
        details=json.loads(meta.read_text()) if meta.exists() else {}
        try:
            out=evaluate_case('archive_'+sys_name,frame,{'E':3,'tau':1,'library_size':a.library_size,
                   'map_maxiter':a.map_maxiter,'centers':2,'bayes':False},dst/sys_name,
                   {**details,'source_file':str(source.relative_to(ROOT))})
            rows.append(out)
        except Exception as ex:errors.append({'name':sys_name,'error':str(ex)})
    if rows:pd.concat(rows,ignore_index=True).to_csv(dst/'summary.csv',index=False)
    (dst/'errors.json').write_text(json.dumps(errors,indent=2))
    print('Archive finished:',len(rows),'errors:',errors)
if __name__=='__main__':main()
