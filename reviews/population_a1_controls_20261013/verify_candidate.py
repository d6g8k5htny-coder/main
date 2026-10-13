import hashlib, json, os, subprocess, sys, time
from pathlib import Path
repo = Path.cwd()
out = Path(__file__).resolve().parent
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=repo).decode().split('\0')[:-1]
def manifest():
    return {p: {'bytes': (repo/p).stat().st_size, 'sha256': hashlib.sha256((repo/p).read_bytes()).hexdigest()} for p in tracked if (repo/p).is_file()}
before=manifest(); (out/'candidate_sources_before.json').write_text(json.dumps(before,indent=2,sort_keys=True)+'\n')
commands=[]
for label,folder in [('population','experiments/universality/two_parameter_cusp_population'),('universality','experiments/universality'),('repository','tests')]:
    for mode in ('normal','optimized'):
        commands.append((label+'_'+mode,[sys.executable,'-B','-E']+(['-O'] if mode=='optimized' else [])+['-S','-m','unittest','discover','-s',folder,'-p','test_*.py','-v']))
commands += [('pilot',[sys.executable,'-B','-E','-S','experiments/universality/run_pilot.py','--output',str(out/'exploratory_pilot')]),('pilot_verify',[sys.executable,'-B','-E','-S','experiments/universality/run_pilot.py','--verify',str(out/'exploratory_pilot'/'observations.json')]),('published_verify',[sys.executable,'-B','-E','-S','experiments/universality/verify_published.py','experiments/universality/results/exploratory50'])]
results=[]
for label,cmd in commands:
    start=time.time()
    with (out/(label+'.stdout.log')).open('w') as stdout,(out/(label+'.stderr.log')).open('w') as stderr:
        proc=subprocess.run(cmd,cwd=repo,stdout=stdout,stderr=stderr)
    result={'label':label,'command':cmd,'exit_code':proc.returncode,'seconds':time.time()-start}
    results.append(result); (out/'verification_results.json').write_text(json.dumps({'python':sys.version,'results':results},indent=2)+'\n')
    print(json.dumps(result),flush=True)
after=manifest(); (out/'candidate_sources_after.json').write_text(json.dumps(after,indent=2,sort_keys=True)+'\n')
print('SOURCE_DRIFT',json.dumps([p for p in before if before[p]!=after.get(p)]),flush=True)
