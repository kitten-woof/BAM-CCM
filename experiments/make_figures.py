from __future__ import annotations
from pathlib import Path
import sys,json,re
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,Ellipse,Rectangle
sys.path.insert(0,str(Path(__file__).resolve().parent))
from generate_synthetic import generate,SPEC
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'report/figures';TAB=ROOT/'report/tables'
OUT.mkdir(parents=True,exist_ok=True);TAB.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,'pdf.fonttype':42,'figure.dpi':130,'axes.grid':True,'grid.alpha':.2})
ORDER=['CCM','Metric-CCM','GP-ARD','GP-Full','BAM-MAP','BAM-Bayes']

def savefig(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight');plt.close(fig)

def read_table(which,name):
    return pd.read_csv(ROOT/f'experiments/results/{which}/{name}')

def collect(kind):
    d=ROOT/f'experiments/results/{kind}'
    sets=[]
    for f in sorted(d.glob('*/metrics.csv')):
        a=pd.read_csv(f);a['folder']=f.parent.name;sets.append(a)
    if not sets:return pd.DataFrame()
    dfa=pd.concat(sets,ignore_index=True)
    dfa.to_csv(TAB/f'{kind}_all_metrics.csv',index=False)
    return dfa

def main():
    synth=collect('synthetic');real=collect('real');archive=collect('archive_synthetic')
    s=synth[synth.split=='test'].copy();r=real[real.split=='test'].copy();ar=archive[archive.split=='test'].copy()
    grid=s[s.folder.str.match(r'^[a-z_]+_s42_b.*_n(450|650|900)$')]
    grid.to_csv(TAB/'pilot_grid_test.csv',index=False)
    repeats=s[s.folder.str.match(r'^(logistic|independent|common_driver)_s(13|42|71|101|131|161|191|221|251|281)_b0.2_noise0.1_n600$')]
    repeats.to_csv(TAB/'ten_seed_test.csv',index=False)
    bayes=s[s.folder.str.contains('_L50_iter65_bayes$',regex=True)]
    bayes.to_csv(TAB/'bayes_ten_seed_test.csv',index=False)
    table=r.pivot_table(index=['case','direction'],columns='model',values='rmse').reindex(columns=ORDER)
    table.to_csv(TAB/'real_rmse.csv')
    table=ar.pivot_table(index=['case','direction'],columns='model',values='rmse').reindex(columns=ORDER)
    table.to_csv(TAB/'archive_rmse.csv')
    w=grid.pivot_table(index=['case','direction'],columns='model',values='rmse')
    if len(w):
        w['bam_vs_ccm_pct']=100*(1-w['BAM-MAP']/w['CCM'])
        w['bam_vs_ard_pct']=100*(1-w['BAM-MAP']/w['GP-ARD'])
        w.reset_index().to_csv(TAB/'grid_relative_improvement.csv',index=False)
        tmp=w.reset_index();tmp['system']=tmp['case'].str.extract(r'^(.*?)_s42_')[0]
        med=tmp.groupby('system')[['bam_vs_ccm_pct','bam_vs_ard_pct']].median().sort_values('bam_vs_ccm_pct')
        fig,ax=plt.subplots(figsize=(9,5.3)); yy=np.arange(len(med));h=.36
        ax.barh(yy-h/2,med.bam_vs_ccm_pct,h,label='BAM-MAP vs CCM')
        ax.barh(yy+h/2,med.bam_vs_ard_pct,h,label='BAM-MAP vs GP-ARD')
        ax.set(yticks=yy,yticklabels=med.index,xlabel='Relative reduction in held-out RMSE (%)',title='15 synthetic systems: median of 3 configurations (not independent CI)')
        ax.axvline(0,color='black',linewidth=.8);ax.legend(loc='lower right');savefig(fig,'synthetic_grid_improvement')
    if len(repeats):
        w=repeats.pivot_table(index=['case','direction'],columns='model',values='rmse')
        w['system']=[re.match(r'^(.*?)_s',x[0]).group(1) for x in w.index]
        records=[]
        rng=np.random.default_rng(20261028)
        for (system,direction),sub in w.groupby(['system',w.index.get_level_values('direction')]):
            for cmp in ('CCM','GP-ARD','GP-Full'):
                improvement=100*(1-sub['BAM-MAP'].to_numpy()/sub[cmp].to_numpy())
                boots=rng.choice(improvement,(4000,len(improvement)),replace=True).mean(axis=1)
                records.append({'system':system,'direction':direction,'baseline':cmp,'n_seeds':len(improvement),
                                'mean_improvement_pct':improvement.mean(),
                                'ci_low':np.quantile(boots,.025),'ci_high':np.quantile(boots,.975),
                                'wins':int(np.sum(improvement>0)),'median_improvement_pct':np.median(improvement)})
        pd.DataFrame(records).to_csv(TAB/'bootstrap_paired.csv',index=False)
        cmp=w.reset_index(); fig,ax=plt.subplots(figsize=(9,4.4))
        vals=[100*(1-v['BAM-MAP'].to_numpy()/v['GP-ARD'].to_numpy()) for _,v in cmp.groupby(['system','direction'])]
        labs=[f'{a}/{b}' for (a,b),v in cmp.groupby(['system','direction'])]
        ax.boxplot(vals,tick_labels=labs,showmeans=True,meanline=True)
        ax.axhline(0,color='black',lw=.8);ax.set(ylabel='BAM-MAP vs GP-ARD: RMSE reduction (%)',title='Paired differences across 10 independent trajectories')
        ax.tick_params(axis='x',rotation=24);savefig(fig,'replicate_gp_ard')
    diagnostics=[]
    for f in sorted((ROOT/'experiments/results/synthetic').glob('*_L50_iter65_bayes/mcmc_diagnostics.csv')):
        if f.stat().st_size>5:diagnostics.append(pd.read_csv(f))
    if diagnostics:
        pd.concat(diagnostics,ignore_index=True).to_csv(TAB/'bayes_diagnostics.csv',index=False)
        bw=bayes.pivot_table(index=['case','direction'],columns='model',values='rmse')
        bgrp=bw.groupby(level='direction').agg(['mean','std'])
        bgrp.to_csv(TAB/'bayes_aggregate.csv')
        fig,ax=plt.subplots(figsize=(8,4));
        means=bw.groupby(level='direction').mean().reindex(columns=['CCM','GP-ARD','GP-Full','BAM-MAP','BAM-Bayes'])
        for d,ss in means.iterrows():ax.plot(ss.index,ss.values,marker='o',label=d)
        ax.set(ylabel='Mean held-out RMSE across 10 seeds',title='BAM-Bayes: short MCMC, diagnostic only');ax.legend();ax.tick_params(axis='x',rotation=22)
        savefig(fig,'bayes_rmse')
        fig,ax=plt.subplots(figsize=(8,3))
        ds=pd.concat(diagnostics)
        for d,dd in ds.groupby('direction'):
            ax.plot(np.arange(len(dd)),dd.max_split_rhat,marker='.',label=d)
        ax.axhline(1.05,color='crimson',ls='--',lw=1,label='R-hat 1.05 threshold')
        ax.set(ylabel='Maximum split R-hat (approximate)',xlabel='Independent dataset seed (ordered)',title='Short-chain convergence diagnostic');ax.legend();savefig(fig,'mcmc_rhat')
    if len(r):
        g=r.groupby(['case','model']).rmse.mean().unstack().reindex(columns=ORDER[:5])
        g.to_csv(TAB/'real_mean_directions.csv')
        fig,ax=plt.subplots(figsize=(9,4.4));xx=np.arange(len(g));width=.15
        for i,c in enumerate(g):ax.bar(xx+(i-2)*width,g[c],width,label=c)
        ax.set(xticks=xx,xticklabels=g.index,ylabel='Mean RMSE (train-standardized units)',title='Real datasets: mean of two reconstruction directions')
        ax.tick_params(axis='x',rotation=25);ax.legend(ncol=3,fontsize=8);savefig(fig,'real_rmse')
    f=ROOT/'experiments/results/synthetic/logistic_s42_b0.2_noise0.1_n600/convergence.csv'
    if f.exists():
        v=pd.read_csv(f);fig,ax=plt.subplots(figsize=(7,3.6))
        for d,g in v.groupby('direction'):ax.plot(g.library_size,g.rho_test,marker='o',label=d)
        ax.set(xlabel='Training library observations',ylabel='Held-out cross map correlation',title='CCM diagnostic (fixed chronological test)');ax.legend();savefig(fig,'library_convergence')
    f=ROOT/'experiments/results/synthetic/logistic_s42_b0.2_noise0.1_n600/predictions.csv'
    if f.exists():
        p=pd.read_csv(f);p=p[p.direction=='X_from_Y']
        fig,ax=plt.subplots(figsize=(9,3.3));truth=p[p.model=='CCM'].sort_values('time_index');ax.plot(truth.time_index,truth.y_true_scaled,label='Observed target',color='black',lw=1.2)
        for model in ['CCM','GP-ARD','BAM-MAP']:
            d=p[p.model==model].sort_values('time_index');ax.plot(d.time_index,d.y_pred_scaled,label=model,alpha=.8,lw=.8)
        ax.set(xlabel='Original time index',ylabel='Standardized X',title='Held-out contemporaneous reconstruction X from Y');ax.legend(ncol=4);savefig(fig,'predictions_logistic')
    demo=['logistic','henon','lorenz','rossler','common_driver','independent','delayed','periodic']
    fig,axs=plt.subplots(4,2,figsize=(9,9));
    for ax,name in zip(axs.flat,demo):
        z,_=generate(name,n=350,seed=9,noise=.1,coupling=.2)
        ax.plot(z.t.iloc[:180],z.x.iloc[:180],label='X',lw=.9)
        ax.plot(z.t.iloc[:180],z.y.iloc[:180],label='Y',lw=.9,alpha=.8)
        ax.set_title(name);ax.set_xlabel('sample index')
    axs.flat[0].legend(ncol=2);fig.suptitle('Illustrative newly generated observations (not performance results)',y=1.005)
    savefig(fig,'synthetic_trajectories')
    fig,axs=plt.subplots(2,4,figsize=(10,5))
    for ax,name in zip(axs.flat,demo):
        z,_=generate(name,n=350,seed=9,noise=.1,coupling=.2)
        ax.plot(z.x,z.y,marker='.',markersize=.5,lw=.35)
        ax.set(xlabel='X',ylabel='Y',title=name)
    savefig(fig,'synthetic_phase')
    fig,axs=plt.subplots(1,2,figsize=(9,2.8))
    for ax in axs:ax.axis('off');ax.set(xlim=(0,10),ylim=(0,4))
    for ax,title,blocks in zip(axs,['Classical CCM','BAM-MAP / BAM-Bayes'],
            [['Y_t','Delay embedding','Simplex neighbours','Estimate X_t'],
             ['Y_t','Delay embedding','Warp Lx+ΣWg','GP posterior X_t']]):
        ax.set_title(title)
        for i,b in enumerate(blocks):
            x=.2+i*2.5;ax.add_patch(Rectangle((x,1.3),2.1,1.1,fc='#e4eef9',ec='#385e86'))
            ax.text(x+1.05,1.85,b,ha='center',va='center',fontsize=8)
            if i<3:ax.annotate('',xy=(x+2.45,1.85),xytext=(x+2.15,1.85),arrowprops={'arrowstyle':'->'})
    savefig(fig,'architecture')
    fig,ax=plt.subplots(figsize=(7,3.2));ax.axis('off');ax.set(xlim=(0,7),ylim=(0,3.5))
    for t,x,y in [('X',1,2.7),('Y',5,2.7),('Z',3,0.6)]:
        ax.add_patch(Ellipse((x,y),1,.7,fc='#e4eef9',ec='#385e86'));ax.text(x,y,t,ha='center',va='center')
    for st,en in [((3,1),(1.3,2.4)),((3,1),(4.7,2.4))]:ax.add_patch(FancyArrowPatch(st,en,arrowstyle='-|>',mutation_scale=16,lw=1.4))
    ax.set_title('Common-driver negative control: Z→X and Z→Y, no X↔Y edge')
    savefig(fig,'common_driver_graph')
    fit=[]
    for base in ('synthetic','real','archive_synthetic'):
        for f in (ROOT/f'experiments/results/{base}').glob('*/fit_diagnostics.csv'):
            x=pd.read_csv(f);x['dataset_group']=base;fit.append(x)
    if fit:
        data=pd.concat(fit,ignore_index=True);data.to_csv(TAB/'optimizer_diagnostics.csv',index=False)
        fig,ax=plt.subplots(figsize=(6,3.5));data.boxplot(column='fit_seconds',by='kind',ax=ax,grid=False)
        ax.set(xlabel='GP metric type',ylabel='Training seconds per direction',title='Measured optimizer wall-clock time');fig.suptitle('')
        savefig(fig,'runtime')
    diagdir=ROOT/'experiments/results/diagnostics'
    lf=diagdir/'lag_curves.csv'
    if lf.exists():
        lag=pd.read_csv(lf)
        lag.to_csv(TAB/'lag_curves.csv',index=False)
        fig,axs=plt.subplots(2,2,figsize=(9,6),sharex=True)
        for ax,(system,sub) in zip(axs.flat,lag.groupby('system')):
            for direction,values in sub.groupby('direction'):
                ax.plot(values.lag_target_minus_embedding,values.rho,label=direction,marker='.',lw=1)
            ax.set(title=system,ylabel='Held-out CCM correlation',xlabel='Target minus embedding lag');ax.axvline(0,color='gray',lw=.7)
        axs.flat[0].legend();savefig(fig,'lagged_ccm')
    sf=diagdir/'circular_surrogate.csv'
    if sf.exists():
        su=pd.read_csv(sf);su.to_csv(TAB/'circular_surrogate.csv',index=False)
        fig,axs=plt.subplots(2,2,figsize=(9,6))
        for ax,(system,sub) in zip(axs.flat,su.groupby('system')):
            vals=sub[sub.direction=='X_from_Y']
            ax.hist(vals[vals.surrogate].rho,bins=15,alpha=.7,label='Circular shift null')
            ax.axvline(vals[~vals.surrogate].rho.iloc[0],color='crimson',lw=1.5,label='Unshifted')
            ax.set(title=system,xlabel='Held-out correlation X from Y',ylabel='Number of surrogate shifts')
        axs.flat[0].legend();savefig(fig,'surrogate_controls')
    af=ROOT/'experiments/results/ablation/summary.csv'
    if af.exists():
        ab=pd.read_csv(af);ab.to_csv(TAB/'ablation_centers.csv',index=False)
        sub=ab[(ab.split=='test')&(ab.model=='BAM-MAP')]
        fig,ax=plt.subplots(figsize=(7,3.4))
        for direction,g in sub.groupby('direction'):
            z=g.groupby('centers').rmse.agg(['mean','std']);ax.errorbar(z.index,z['mean'],yerr=z['std'],label=direction,marker='o',capsize=3)
        ax.set(xlabel='Number of local radial centers K',ylabel='Mean RMSE ± SD (3 seeds)',title='Ablation: local metric complexity, small exploratory study')
        ax.legend();savefig(fig,'ablation_centers')
    print('Generated vector charts:',len(list(OUT.glob('*.pdf'))),'tabular CSV:',len(list(TAB.glob('*.csv'))))
if __name__=='__main__':main()
