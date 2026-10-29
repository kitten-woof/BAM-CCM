import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'original'))
import bam_ccm
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluate import evaluate_case
ROOT=Path(__file__).resolve().parents[1]
DATASETS=['appliances','bike_day','bike_hour','qqq_volume','sp500_volume','qqq_sp500','qqq_vix']
def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='*',choices=DATASETS,default=DATASETS)
    p.add_argument('--full-record',action='store_true',help='Use entire dataset, not only the last 900 observations')
    p.add_argument('--library-size',type=int,default=65);p.add_argument('--map-maxiter',type=int,default=65);p.add_argument('--bayes',action='store_true')
    a=p.parse_args();rows=[];events=[];folder=ROOT/'experiments/results/real';folder.mkdir(parents=True,exist_ok=True)
    for name in a.datasets:
        try:
            df,x,y,origin=bam_ccm.load_dataset(name,'',42)
            before=len(df);df=df.apply(pd.to_numeric,errors='coerce').dropna().reset_index(drop=True)
            if not a.full_record:df=df.tail(900).reset_index(drop=True)
            meta={'file_origin':origin,'variables':[x,y],'original_transformed_rows':before,
                  'rows_used':len(df),'tail_900':not a.full_record,'observational':'Causality labels unknown'}
            dst=folder/name
            m=evaluate_case(name,df,{'E':3,'tau':1,'library_size':a.library_size,'map_maxiter':a.map_maxiter,
                 'centers':2,'bayes':a.bayes,'seed':42},dst,meta)
            rows.append(m);events.append({'case':name,'status':'success'})
        except Exception as ex:
            print('FAILED',name,repr(ex),flush=True);events.append({'case':name,'status':'failure','error':repr(ex)})
        pd.DataFrame(events).to_csv(folder/'run_manifest.csv',index=False)
        if rows:pd.concat(rows,ignore_index=True).to_csv(folder/'summary.csv',index=False)
    print('Real completed:',sum(x['status']=='success' for x in events),'/',len(events))
if __name__=='__main__':main()
