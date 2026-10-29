import argparse,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--latex',action='store_true')
    a=p.parse_args()
    if a.latex and not (ROOT/'report/main.tex').is_file():
        p.error('--latex unavailable: report/main.tex is absent; the archive contains only the final PDF')
    def run(file,*args):
        cmd=[sys.executable,str(ROOT/'experiments'/file),*args]
        print('RUN:',*cmd,flush=True)
        subprocess.run(cmd,cwd=ROOT,check=True)
    run('data_audit.py')
    if a.quick:
        run('run_synthetic.py')
    else:
        run('run_synthetic.py','--profile','full','--seeds','42')
        run('run_synthetic.py','--profile','pilot','--systems','logistic','independent','common_driver',
            '--seeds','13','42','71','101','131','161','191','221','251','281')
        run('run_synthetic.py','--profile','pilot','--systems','logistic',
            '--seeds','13','42','71','101','131','161','191','221','251','281',
            '--library-size','50','--bayes')
        run('run_ablation.py')
    run('run_archive.py');run('run_real.py');run('run_diagnostics.py')
    run('make_figures.py');run('make_latex_tables.py')
    if a.latex:
        for t in ['xelatex','biber']:
            if not shutil.which(t):raise RuntimeError(f'Required LaTeX command not available: {t}')
        wd=ROOT/'report'
        for cmd in [['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                    ['biber','main'],['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                    ['xelatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
            subprocess.run(cmd,cwd=wd,check=True,stdout=subprocess.DEVNULL)
    print('All requested steps completed. See report/tables and experiments/results.')
if __name__=='__main__':main()
