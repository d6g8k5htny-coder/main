"""Run main's documented default-home checks (docs/REPRODUCE.md) on each open PR composed onto main. Data only."""
import concurrent.futures as cf, json, os, re, subprocess, sys, threading
CLONE, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TARGETS = [int(x) for x in sys.argv[4:]]
LOCK = threading.Lock()
os.makedirs(OUT, exist_ok=True)
CMDS = [('landing', ['python3', '-B', '-S', 'tools/workspace_landing_check.py']),
        ('navigation', ['python3', '-B', '-S', 'tools/navigation_check.py']),
        ('unittest', ['python3', '-B', '-S', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py']),
        ('unittest -O', ['python3', '-B', '-O', '-S', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_*.py'])]
def git(*a, cwd=CLONE, check=True):
    r = subprocess.run(['git', *a], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(r.stderr)
    return r
def target(pr):
    res = {'pr': pr}
    if pr == 0:
        commit = git('rev-parse', BASE).stdout.strip(); res['head'] = commit
    else:
        head = git('rev-parse', 'pr/%d' % pr).stdout.strip(); res['head'] = head
        mt = git('merge-tree', '--write-tree', BASE, head, check=False)
        if mt.returncode:
            res['status'] = 'CONFLICT'; json.dump(res, open(os.path.join(OUT, 'mt-%d.json' % pr), 'w')); print(pr, 'CONFLICT', flush=True); return res
        with LOCK: commit = git('commit-tree', mt.stdout.splitlines()[0], '-p', BASE, '-p', head, '-m', 'compose %d' % pr).stdout.strip()
    res['commit'] = commit
    wt = os.path.join(OUT, 'wt-%d' % pr)
    with LOCK:
        git('worktree', 'remove', '--force', wt, check=False); git('worktree', 'add', '-q', '--detach', wt, commit)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1')
    res['checks'] = {}
    for name, cmd in CMDS:
        r = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, env=env, timeout=3600)
        out = r.stdout + r.stderr
        failed = sorted(set(m.group(1) + ' ' + m.group(2) for m in re.finditer(r'^(?:FAIL|ERROR): (\S+) \(([^)\s]+)\)', out, re.M)))
        ran = re.search(r'^Ran (\d+) tests?', out, re.M); summ = re.search(r'^(OK.*|FAILED.*)$', out, re.M)
        res['checks'][name] = {'exit': r.returncode, 'ran': int(ran.group(1)) if ran else None, 'summary': summ.group(1) if summ else None,
                               'failed_tests': failed[:200], 'tail': out[-1500:] if r.returncode else ''}
    with LOCK: git('worktree', 'remove', '--force', wt, check=False)
    res['status'] = 'FAIL' if any(c['exit'] for c in res['checks'].values()) else 'PASS'
    json.dump(res, open(os.path.join(OUT, 'mt-%d.json' % pr), 'w'), indent=1)
    print(pr, res['status'], {k: (v['exit'], v['ran']) for k, v in res['checks'].items()}, flush=True)
    return res
with cf.ThreadPoolExecutor(max_workers=int(os.environ.get('WORKERS', '3'))) as ex:
    list(ex.map(target, TARGETS))
print('ALL_DONE', flush=True)
