from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
records=[]
for f in sorted((ROOT/'original/data').rglob('*.csv')):
    d=pd.read_csv(f)
    date_column=next((c for c in ['date','Date','dteday','observation_date'] if c in d),None)
    dates=pd.to_datetime(d[date_column],errors='coerce',format='mixed') if date_column else None
    records.append({'filename':str(f.relative_to(ROOT)),'rows':len(d),'cols':len(d.columns),
          'columns':', '.join(d.columns),'missing_cells':int(d.isna().sum().sum()),
          'date_min':str(dates.min()) if dates is not None else '',
          'date_max':str(dates.max()) if dates is not None else '',
          'source_verified':'original archive only; UCI matches by field names (no checksum)'})
frame=pd.DataFrame(records)
output=ROOT/'experiments/results/data_inventory.csv';output.parent.mkdir(parents=True,exist_ok=True)
frame.to_csv(output,index=False)
print(frame[['filename','rows','missing_cells','date_min','date_max']].to_string(index=False))
