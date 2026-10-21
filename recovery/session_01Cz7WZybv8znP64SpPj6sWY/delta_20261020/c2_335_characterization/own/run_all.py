"""Run the unchanged C2 torus-transfer baseline, controls and five omission mutants in both pinned flag sets.
Each child runs from the packet directory exactly as the workflow does; stdout and stderr are kept separately."""
import concurrent.futures as cf, hashlib, json, os, subprocess, sys, time
SRC = sys.argv[1]; OUT = sys.argv[2]
PKT = os.path.join(SRC, 'frontiers/c2_torus_transfer_20261001')
FLAGS = {'normal': ['-B', '-S'], 'optimized': ['-B', '-O', '-S']}
CASES = [('check', ['transfer.py', '--check']), ('controls', ['controls.py'])] + \
        [('mutant-' + m, ['transfer.py', '--mutant', m]) for m in ('no-image', 'no-theta', 'drop-ratio', 'eta-half', 'fp-const')]
def run(job):
    mode, (name, argv) = job
    cmd = [sys.executable] + FLAGS[mode] + argv
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    t0 = time.time(); p = subprocess.run(cmd, cwd=PKT, capture_output=True, env=env, timeout=3600); dt = time.time() - t0
    base = f'{OUT}/{mode}__{name}'
    open(base + '.stdout', 'wb').write(p.stdout); open(base + '.stderr', 'wb').write(p.stderr)
    rec = {'mode': mode, 'case': name, 'argv': cmd[1:], 'rc': p.returncode, 'seconds': round(dt, 1),
           'stdout_bytes': len(p.stdout), 'stdout_sha256': hashlib.sha256(p.stdout).hexdigest(),
           'stderr_bytes': len(p.stderr), 'stderr_sha256': hashlib.sha256(p.stderr).hexdigest(),
           'stdout_tail': p.stdout.decode('utf-8', 'replace')[-600:], 'stderr_tail': p.stderr.decode('utf-8', 'replace')[-1200:]}
    print(json.dumps({k: rec[k] for k in ('mode', 'case', 'rc', 'seconds')}), flush=True)
    return rec
jobs = [(m, c) for c in CASES for m in FLAGS]
with cf.ThreadPoolExecutor(max_workers=4) as ex:
    recs = list(ex.map(run, jobs))
json.dump({'python': sys.version, 'executable': sys.executable, 'packet': PKT, 'results': recs}, open(OUT + '/results.json', 'w'), indent=1)
print('DONE', len(recs), flush=True)
