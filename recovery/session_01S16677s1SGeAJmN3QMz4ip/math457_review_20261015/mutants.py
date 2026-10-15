import sys,os,shutil,subprocess,json,re
BASE='/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/m457'
G="grep -Eq '^Ran [1-9][0-9]* tests? in [0-9]+([.][0-9]+)?s$'"
N='python -B -S -m unittest discover -s tests -v 2>&1 | tee "$log_dir/normal.log"'
O='python -B -O -S -m unittest discover -s tests -v 2>&1 | tee "$log_dir/optimized.log"'
I="          "
GN=f'{I}{G} "$log_dir/normal.log"\n'; GO=f'{I}{G} "$log_dir/optimized.log"\n'
M={
 'M01_drop_pipefail': ('set -euo pipefail\n          log_dir', 'set -eu\n          log_dir'),
 'M02_drop_normal_guard': (GN, ''),
 'M03_drop_optimized_guard': (GO, ''),
 'M04_weaken_regex_zero_ok': (GN, GN.replace('[1-9][0-9]*','[0-9]+')),
 'M05_opt_guard_reads_normal_log': (GO, GO.replace('optimized.log','normal.log')),
 'M06_normal_guard_or_true': (GN, GN[:-1]+' || true\n'),
 'M07_normal_drops_2to1': (N, N.replace(' 2>&1','')),
 'M08_optimized_loses_O': (O, O.replace('-B -O -S','-B -S')),
 'M09_normal_loses_v': (N, N.replace('tests -v','tests')),
 'M10_if_always': ("if: github.event_name == 'workflow_dispatch'\n        run: |\n          set -euo pipefail\n          log_dir", "if: always()\n        run: |\n          set -euo pipefail\n          log_dir"),
 'M11_guards_after_both_runs': (f'{I}{N}\n{GN}{I}{O}\n{GO}', f'{I}{N}\n{I}{O}\n{GN}{GO}'),
 'M12_drop_trap': ("          trap 'rm -rf \"$log_dir\"' EXIT\n", ''),
 'M13_redirect_not_tee': (N, N.replace('| tee "$log_dir/normal.log"','> "$log_dir/normal.log"')),
 'M14_guard_count_regex_unanchored': (GN, GN.replace("'^Ran","'Ran").replace("s$'","s'")),
}
def one(name, wf, py):
    old,new=M[name]
    d=f'{BASE}/mut/{name}__{wf}__{py}'
    shutil.rmtree(d,ignore_errors=True); shutil.copytree(f'{BASE}/ctl',d)
    p=f'{d}/.github/workflows/{wf}.yml'; s=open(p).read()
    if s.count(old)!=1: return {'name':name,'wf':wf,'py':py,'error':f'old count {s.count(old)}'}
    open(p,'w').write(s.replace(old,new))
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    r=subprocess.run([py,'-B','-S','-m','unittest','tests.test_full_suite_routing','tests.test_c6_factorial_job_budget'],cwd=d,env=env,capture_output=True,text=True,timeout=900)
    out=r.stderr
    fails=sorted(set(re.findall(r'^(?:FAIL|ERROR): (\w+)',out,re.M)))
    summ=[l for l in out.splitlines() if l.startswith('Ran ') or l.startswith('OK') or l.startswith('FAILED')]
    shutil.rmtree(d,ignore_errors=True)
    return {'name':name,'wf':wf,'py':py,'rc':r.returncode,'summary':summ,'failing':fails}
if __name__=='__main__':
    name,wf,py=sys.argv[1:4]
    res=one(name,wf,py)
    open(f'{BASE}/mut/res_{name}__{wf}__{py}.json','w').write(json.dumps(res))
    print(json.dumps(res))
