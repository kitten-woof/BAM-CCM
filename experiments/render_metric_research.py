import json,math
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
M=P/'experiments/results/metric_research'; T=P/'report/tables';F=P/'report/figures'
F.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
D=pd.read_csv(M/'confirm/metrics.csv').query('split=="test"');S=pd.read_csv(M/'confirm/summary.csv');
A=pd.read_csv(M/'confirm/fits.csv');R=pd.read_csv(M/'real/metrics.csv').query('split=="test"')
choose='BAM-Jac-20';models=['GP-Fixed','GP-ARD','GP-Full','BAM-original','BAM-warm',choose]

def write(name,lines):
    (T/name).write_text('\n'.join(lines)+'\n',encoding='utf-8')

all=S.query('system=="ALL_seed_aggregate"').set_index('model').loc[models]
lines=[r'\begin{table}[htbp]\centering\small',r'\caption{Отложенный тест: средняя RMSE на 6 системах, 10 новых seed и двух направлениях; ДИ построен по 10 seed-агрегатам (разные направления одного seed не считаются независимыми).}',r'\label{tab:metric-main}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Модель & $p$ & RMSE & $\Delta$ к ARD & 95\% ДИ для $\Delta$ \\',r'\midrule']
for i,m in enumerate(models):
    x=all.loc[m];p={'GP-Fixed':2,'GP-ARD':5,'GP-Full':8,'BAM-original':14,'BAM-warm':14,choose:14}[m]
    lines.append(f'{m.replace("-","--")} & {p} & {x.rmse_mean:.4f} & {x.mean_rmse_gain_vs_ard:+.4f} & [{x.gain_ci95_low:+.4f}; {x.gain_ci95_high:+.4f}] '+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];write('metric_main.tex',lines)

b=S[(S.model==choose)&(S.system!='ALL_seed_aggregate')].copy()
b['system']=pd.Categorical(b.system, categories=['logistic','nonlinear','independent','common_driver','sharp_transition','varying_frequency'],ordered=True)
b=b.sort_values(['system','direction'])
lines=[r'\begin{longtable}{llrrrr}',r'\caption{BAM-Jac-20 относительно GP-ARD: положительная $\Delta$ означает меньшую RMSE у BAM; парные интервалы по 10 seed отдельно для каждой системы и направления.}\label{tab:metric-by-case}\\',r'\toprule',r'Система & Реконструкция & GP-ARD & BAM-Jac-20 & $\Delta$ & 95\% ДИ \\',r'\midrule',r'\endfirsthead',r'\toprule',r'Система & Реконструкция & GP-ARD & BAM-Jac-20 & $\Delta$ & 95\% ДИ \\',r'\midrule',r'\endhead']
for _,z in b.iterrows():
    key=D[(D.system==z.system)&(D.direction==z.direction)&(D.model=='GP-ARD')].rmse.mean()
    lines.append(f'{str(z.system).replace("_",r"\_")} & {str(z.direction).replace("_",r"\_")} & {key:.3f} & {z.rmse_mean:.3f} & {z.mean_rmse_gain_vs_ard:+.3f} & [{z.gain_ci95_low:+.3f}; {z.gain_ci95_high:+.3f}] '+r'\\')
lines += [r'\bottomrule',r'\end{longtable}'];write('metric_cases.tex',lines)

p=json.loads((M/'selection.json').read_text())
lines=[r'\begin{table}[htbp]\centering',r'\caption{Пилотная валидация трёх значений регуляризации по 3 seed; тестовые seed не использовались для выбора $\lambda_J$.}',r'\label{tab:metric-pilot}',r'\begin{tabular}{rrr}',r'\toprule',r'$\lambda_J$ & RMSE(BAM)$-$RMSE(ARD) & Доля побед BAM \\',r'\midrule']
for c in p['candidates']:
 lines.append(f'{c["lambda"]:g} & {c["mean_validation_rmse_diff_v_ard"]:+.5f} & {c["validation_win_rate"]:.1%} '+r'\\')
lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];write('metric_pilot.tex',[line.replace('%', r'\%') if line.startswith(('0 &','2 &','20 &')) else line for line in lines])

q=R.pivot(index=['dataset','direction'],columns='model',values='rmse').loc[:,['GP-ARD','GP-Full','BAM-original','BAM-Jac']]
lines=[r'\begin{longtable}{llrrrr}',r'\caption{Новые абляции на семи реальных постановках из исходного архива; одна последняя хронологическая выборка длиной до 600 наблюдений на постановку.}\label{tab:metric-real}\\',r'\toprule',r'Набор & Направление & GP-ARD & GP-Full & BAM-MAP & BAM-Jac \\',r'\midrule',r'\endfirsthead',r'\toprule',r'Набор & Направление & GP-ARD & GP-Full & BAM-MAP & BAM-Jac \\',r'\midrule',r'\endhead']
for (name,d),z in q.iterrows():
 lines.append(f'{name.replace("_",r"\_")} & {d.replace("_",r"\_")} & {z["GP-ARD"]:.3f} & {z["GP-Full"]:.3f} & {z["BAM-original"]:.3f} & {z["BAM-Jac"]:.3f} '+r'\\')
lines += [r'\bottomrule',r'\end{longtable}'];write('metric_real.tex',lines)

plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'figure.dpi':120})
fig,ax=plt.subplots(figsize=(9,4.3));
means=all['mean_rmse_gain_vs_ard'].values;lo=all['gain_ci95_low'].values;hi=all['gain_ci95_high'].values
xx=np.arange(len(models));ax.bar(xx,means,yerr=[means-lo,hi-means],capsize=3)
ax.axhline(0,linewidth=.8);ax.set_xticks(xx,models,rotation=14,ha='right');ax.set_ylabel('RMSE(GP-ARD) − RMSE(model)');ax.set_title('Confirmatory 10 seeds, six systems (positive = better than ARD)');fig.tight_layout();fig.savefig(F/'metric_ablation_main.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.7,5.2));pos=np.arange(len(b));ax.errorbar(b['mean_rmse_gain_vs_ard'],pos,
              xerr=np.vstack([b.mean_rmse_gain_vs_ard-b.gain_ci95_low,b.gain_ci95_high-b.mean_rmse_gain_vs_ard]),fmt='o',capsize=2)
ax.axvline(0,linewidth=.8);ax.set_yticks(pos,[f'{r.system} / {r.direction}' for r in b.itertuples()],fontsize=7);ax.invert_yaxis()
ax.set_xlabel('Test RMSE gain vs GP-ARD (10 seeds, 95% t CI)');fig.tight_layout();fig.savefig(F/'metric_directional.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,3.6))
local_models=['BAM-original','BAM-warm','BAM-Jac-20']
ax.boxplot([A.loc[A.model==m,'W_norm'].dropna().values for m in local_models],
           tick_labels=local_models,showfliers=False)
ax.set_ylabel(r'$||W||_F$');ax.set_title('Local warp coefficient norms over independent fits')
fig.tight_layout();fig.savefig(F/'metric_warp_norm.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(8.7,4.8));g=(q['GP-ARD']-q['BAM-Jac']).sort_values()
ax.barh(np.arange(len(g)),g.values);ax.set_yticks(np.arange(len(g)),[f'{d} / {t}' for d,t in g.index],fontsize=7)
ax.axvline(0,linewidth=.8);ax.set_xlabel('GP-ARD RMSE minus BAM-Jac RMSE');ax.set_title('Real data: 14 reconstructions, no independent replications');fig.tight_layout();fig.savefig(F/'metric_real_delta.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.7,3.2));x=np.linspace(-2.5,2.5,450)
ax.plot(x,np.tanh(7*x),label=r'$\tanh(7x)$');ax.plot(x,np.sin(4*x*x)/(1+.15*abs(x)),label=r'$\sin(4x^2)/(1+0.15|x|)$')
ax.set_xlabel('Latent driver x');ax.set_ylabel('Nonlinear response f(x)');ax.legend();ax.set_title('Predefined synthetic mechanisms, not estimated curves');fig.tight_layout();fig.savefig(F/'metric_new_generators.pdf');plt.close(fig)
print('Saved 4 tables and 5 vector figures')

LP=M/'library_probe/metrics.csv'
if LP.exists():
    from scipy.stats import t as _tdist
    a=pd.read_csv(LP).query('split=="test"').pivot(index=['system','seed','direction','library_size'],columns='model',values='rmse')
    lines=[r'\begin{table}[htbp]\centering\small',
           r'\caption{Чувствительность к размеру библиотеки на пяти новых seeds и четырёх семействах: разница RMSE(GP-ARD)$-$RMSE(model), положительное значение означает преимущество модели. Последующая диагностическая проверка, без права на подтверждающее утверждение.}',
           r'\label{tab:metric-library}',r'\begin{tabular}{rrrr}',r'\toprule',
           r'Библиотека & GP-Full & BAM-original & BAM-Jac \\',r'\midrule']
    stats=[]
    for size,g in a.groupby(level='library_size'):
        record={'library':int(size)}
        for mod in ['GP-Full','BAM-original','BAM-Jac']:
            d=(g['GP-ARD']-g[mod]).groupby(level='seed').mean().values
            c=_tdist.ppf(.975,len(d)-1)*d.std(ddof=1)/np.sqrt(len(d))
            record[mod]=(float(d.mean()),float(c))
        stats.append(record)
        lines.append(f'{int(size)} & {record["GP-Full"][0]:+.4f} & {record["BAM-original"][0]:+.4f} & {record["BAM-Jac"][0]:+.4f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    write('metric_library.tex',lines)
    fig,ax=plt.subplots(figsize=(7.2,4.1))
    for mod in ['GP-Full','BAM-original','BAM-Jac']:
        xs=[i['library'] for i in stats]
        ys=np.array([i[mod][0] for i in stats]);ci=np.array([i[mod][1] for i in stats])
        ax.errorbar(xs,ys,yerr=ci,marker='o',capsize=3,label=mod)
    ax.axhline(0,linewidth=.8);ax.set_xticks([40,65,120]);ax.set_xlabel('Number of training library points')
    ax.set_ylabel('RMSE(GP-ARD) - RMSE(model)')
    ax.set_title('Exploratory sensitivity: 5 new seeds, four systems')
    ax.legend();fig.tight_layout();fig.savefig(F/'metric_library.pdf');plt.close(fig)
