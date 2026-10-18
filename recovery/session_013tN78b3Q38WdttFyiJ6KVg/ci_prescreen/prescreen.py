"""Local pre-screen of Math- pull_request CI on composed trees. Data only: no push, no carry, no review.

Usage: python3 prescreen.py CLONE BASE OUTDIR PR[:HEAD] ...   (PR 0 means the base itself)
For each target: compose (merge-tree base+head), parse workflows from the composed tree, select those GitHub
would trigger for a pull_request into main given the PR's changed paths (merge-base..head), run their steps.
"""
import concurrent.futures as cf, fnmatch, itertools, json, os, re, shutil, subprocess, sys, tempfile, time
import threading
import yaml
LOCK = threading.Lock()

CLONE, BASE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TARGETS = sys.argv[4:]
os.makedirs(OUT, exist_ok=True)
NOOP_ACTIONS = ('actions/checkout', 'actions/setup-python', 'actions/upload-artifact', 'actions/cache',
                'actions/download-artifact')
STEP_TIMEOUT = 900

def git(*args, cwd=CLONE, check=True):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError('git %s: %s' % (' '.join(args), r.stderr.strip()))
    return r

def glob_re(pattern):
    # GitHub filter globs: '**/' zero or more directories, ** any chars incl '/', * and ? exclude '/'
    out, i = '', 0
    while i < len(pattern):
        if pattern.startswith('**/', i): out += '(?:.*/)?'; i += 3
        elif pattern.startswith('**', i): out += '.*'; i += 2
        elif pattern[i] == '*': out += '[^/]*'; i += 1
        elif pattern[i] == '?': out += '[^/]'; i += 1
        else: out += re.escape(pattern[i]); i += 1
    return re.compile(out + r'\Z')

def matches(path, patterns):
    """Ordered include/negate semantics: last matching pattern wins."""
    hit = False
    for p in patterns:
        neg = p.startswith('!')
        if glob_re(p[1:] if neg else p).match(path): hit = not neg
    return hit

def pr_trigger(doc):
    on = doc.get('on', doc.get(True))
    if on is None: return None
    if isinstance(on, str): on = {on: None}
    elif isinstance(on, list): on = {k: None for k in on}
    if 'pull_request' not in on: return None
    return on['pull_request'] or {}

def triggered(cfg, changed, base_branch='main'):
    if cfg is None: return False, 'no pull_request trigger'
    if 'branches' in cfg and not any(fnmatch.fnmatchcase(base_branch, b) for b in cfg['branches']):
        return False, 'branches filter excludes main'
    if 'branches-ignore' in cfg and any(fnmatch.fnmatchcase(base_branch, b) for b in cfg['branches-ignore']):
        return False, 'branches-ignore excludes main'
    types = cfg.get('types')
    if types is not None and not set(types) & {'opened', 'synchronize', 'reopened'}:
        return False, 'types exclude synchronize'
    if 'paths' in cfg:
        if not any(matches(f, cfg['paths']) for f in changed): return False, 'paths filter not matched'
    if 'paths-ignore' in cfg:
        if all(any(glob_re(p).match(f) for p in cfg['paths-ignore']) for f in changed): return False, 'all paths ignored'
    return True, 'triggered'

EXPR = re.compile(r'\$\{\{\s*(.*?)\s*\}\}')

def eval_expr(e, ctx):
    e = e.strip()
    if e == 'toJSON(needs)' and 'toJSON(needs)' in ctx: return ctx['toJSON(needs)']
    if '||' in e:
        for part in e.split('||'):
            v = eval_expr(part, ctx)
            if v not in (None, ''): return v
        return ''
    if len(e) >= 2 and e[0] == e[-1] == "'": return e[1:-1]
    if e in ctx: return ctx[e]
    if e.startswith(('inputs.', 'github.event.before', 'github.event.inputs.')): return ''
    return None

def substitute(text, ctx):
    unsupported = []
    def rep(m):
        v = eval_expr(m.group(1), ctx)
        if v is None: unsupported.append(m.group(1)); return m.group(0)
        return str(v)
    return EXPR.sub(rep, text), unsupported

def matrix_combos(job):
    m = (job.get('strategy') or {}).get('matrix')
    if not m: return [{}]
    if isinstance(m, str): return [None]
    keys = [k for k, v in m.items() if isinstance(v, list) and k not in ('include', 'exclude')]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[m[k] for k in keys])] if keys else [{}]
    for inc in m.get('include', []) or []:
        if isinstance(inc, dict): combos.append(dict(inc))
    excl = m.get('exclude', []) or []
    combos = [c for c in combos if not any(all(c.get(k) == v for k, v in e.items()) for e in excl)]
    return combos[:12]

def run_job(tree, wf_name, job_id, job, combo, ctx_base):
    rec = {'workflow': wf_name, 'job': job_id, 'matrix': combo, 'steps': []}
    if combo is None:
        rec.update(status='UNSUPPORTED', reason='dynamic matrix'); return rec
    cond = str(job.get('if', ''))
    if cond and ('push' in cond and 'pull_request' not in cond or 'schedule' in cond or 'workflow_dispatch' in cond):
        rec.update(status='SKIPPED', reason='job if: ' + cond[:120]); return rec
    tmp = tempfile.mkdtemp(prefix='runner-')
    envfile, outfile, summary = (os.path.join(tmp, n) for n in ('env', 'out', 'summary'))
    for f in (envfile, outfile, summary): open(f, 'w').close()
    ctx = dict(ctx_base); ctx['runner.temp'] = tmp
    for k, v in combo.items(): ctx['matrix.' + k] = v
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(CI='true', GITHUB_ACTIONS='true', GITHUB_EVENT_NAME='pull_request', GITHUB_WORKSPACE=tree,
               RUNNER_TEMP=tmp, GITHUB_ENV=envfile, GITHUB_OUTPUT=outfile, GITHUB_STEP_SUMMARY=summary,
               GITHUB_SHA=str(ctx['github.sha']), GITHUB_BASE_REF='main', GITHUB_HEAD_REF=str(ctx['github.head_ref']),
               GITHUB_REF='refs/pull/merge', PYTHONDONTWRITEBYTECODE='1', GITHUB_REPOSITORY='d6g8k5htny-coder/Math-',
               GITHUB_REPOSITORY_OWNER='d6g8k5htny-coder', GITHUB_SERVER_URL='https://github.com', GITHUB_RUN_ID='0',
               GITHUB_RUN_ATTEMPT='1', GITHUB_REF_NAME='merge', GITHUB_WORKFLOW=wf_name, GITHUB_JOB=job_id,
               RUNNER_OS='Linux', GITHUB_TOKEN='')
    if os.environ.get('SHIM_DIR'): env['PATH'] = os.environ['SHIM_DIR'] + os.pathsep + env['PATH']
    for scope in (job.get('env') or {},):
        for k, v in scope.items():
            val, uns = substitute(str(v), ctx)
            if uns: rec.update(status='UNSUPPORTED', reason='env expression ' + ','.join(uns)); return rec
            env[k] = val; ctx['env.' + k] = val
    defaults = ((job.get('defaults') or {}).get('run') or {})
    status = 'PASS'
    for i, step in enumerate(job.get('steps') or []):
        name = step.get('name') or step.get('uses') or ('step %d' % i)
        if 'uses' in step:
            if any(step['uses'].startswith(a) for a in NOOP_ACTIONS): continue
            rec.update(status='UNSUPPORTED', reason='action ' + step['uses']); shutil.rmtree(tmp, True); return rec
        if 'run' not in step: continue
        scond = str(step.get('if', ''))
        if scond and (('push' in scond and 'pull_request' not in scond) or ('failure()' in scond and 'always()' not in scond)):
            rec['steps'].append({'name': name, 'status': 'SKIPPED', 'if': scond[:80]}); continue
        senv = dict(env)
        for k, v in (step.get('env') or {}).items():
            val, uns = substitute(str(v), ctx)
            if uns: rec.update(status='UNSUPPORTED', reason='step env expression ' + ','.join(uns)); shutil.rmtree(tmp, True); return rec
            senv[k] = val
        for line in open(envfile).read().splitlines():
            if '=' in line and '<<' not in line:
                k, v = line.split('=', 1); senv[k] = v; env[k] = v
        script, uns = substitute(step['run'], ctx)
        if uns: rec.update(status='UNSUPPORTED', reason='run expression ' + ','.join(sorted(set(uns)))); shutil.rmtree(tmp, True); return rec
        shell = step.get('shell') or defaults.get('shell') or 'bash'
        wd = step.get('working-directory') or defaults.get('working-directory') or '.'
        cwd = os.path.join(tree, substitute(wd, ctx)[0])
        if shell == 'bash': cmd = ['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', script]
        elif shell in ('python', 'python3'): cmd = ['python3', '-c', script]
        elif shell == 'sh': cmd = ['sh', '-e', '-c', script]
        else: rec.update(status='UNSUPPORTED', reason='shell ' + shell); shutil.rmtree(tmp, True); return rec
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=cwd, env=senv, capture_output=True, text=True, timeout=step.get('timeout-minutes', 15) * 60)
            code, out = r.returncode, (r.stdout + r.stderr)
        except subprocess.TimeoutExpired as exc:
            code, out = 124, 'TIMEOUT ' + str(exc)[:200]
        except FileNotFoundError as exc:
            code, out = 127, str(exc)
        failed = sorted(set(m.group(1) + ' ' + m.group(2) for m in re.finditer(r'^(?:FAIL|ERROR): (\S+) \(([^)\s]+)\)', out, re.M)))
        srec = {'name': name, 'exit': code, 'seconds': round(time.time() - t0, 1), 'tail': out[-2500:] if code else out[-200:],
                'failed_tests': failed[:400], 'markers': sorted(set(re.findall(r'\b[A-Z][A-Z0-9_]{6,}(?:_FAIL|_FAILED|_ERROR|REFUSED)\b[^\n]{0,120}', out)))[:20]}
        rec['steps'].append(srec)
        if code:
            if step.get('continue-on-error'): srec['continue_on_error'] = True; continue
            status = 'FAIL'; break
    rec['status'] = status
    shutil.rmtree(tmp, True)
    return rec

def target(spec):
    pr, _, head = spec.partition(':')
    pr = int(pr)
    res = {'pr': pr}
    if pr == 0:
        commit, changed = git('rev-parse', BASE).stdout.strip(), None
        res['head'] = commit
    else:
        head = git('rev-parse', head or 'pr/%d' % pr).stdout.strip(); res['head'] = head
        mt = git('merge-tree', '--write-tree', BASE, head, check=False)
        if mt.returncode:
            res['status'] = 'CONFLICT'; res['conflicts'] = [l for l in mt.stdout.splitlines() if 'CONFLICT' in l][:10]; return res
        tree = mt.stdout.splitlines()[0]
        with LOCK: commit = git('commit-tree', tree, '-p', BASE, '-p', head, '-m', 'prescreen compose %d' % pr).stdout.strip()
        mb = git('merge-base', BASE, head).stdout.strip()
        changed = [l for l in git('diff', '--name-only', '--no-renames', mb, head).stdout.splitlines() if l]
        res['changed_files'] = len(changed)
    res['commit'] = commit
    wt = os.path.join(OUT, 'wt-%d' % pr)
    with LOCK:
        if os.path.exists(wt): git('worktree', 'remove', '--force', wt, check=False)
        git('worktree', 'add', '-q', '--detach', wt, commit)
    wfdir = os.path.join(wt, '.github', 'workflows')
    selections, runs = [], []
    wanted = set(json.load(open(os.path.join(OUT, 'base_union.json')))) if pr == 0 and os.path.exists(os.path.join(OUT, 'base_union.json')) else None
    for fn in sorted(os.listdir(wfdir)) if os.path.isdir(wfdir) else []:
        if not fn.endswith(('.yml', '.yaml')): continue
        try: doc = yaml.safe_load(open(os.path.join(wfdir, fn)))
        except Exception as exc:
            selections.append({'workflow': fn, 'selected': False, 'why': 'yaml error %s' % exc}); continue
        cfg = pr_trigger(doc or {})
        if pr == 0:
            ok, why = (fn in wanted, 'baseline union') if wanted is not None else triggered(cfg, ['__all__'])
        else:
            ok, why = triggered(cfg, changed)
        selections.append({'workflow': fn, 'selected': ok, 'why': why})
        if not ok: continue
        ctx = {'github.workspace': wt, 'github.sha': commit, 'github.event_name': 'pull_request',
               'github.head_ref': 'pr-%d' % pr, 'github.base_ref': 'main', 'github.repository': 'd6g8k5htny-coder/Math-',
               'github.event.pull_request.head.sha': res['head'], 'github.event.pull_request.base.sha': git('rev-parse', BASE).stdout.strip(),
               'github.run_id': '0', 'github.run_attempt': '1', 'runner.os': 'Linux', 'github.token': '',
               'secrets.GITHUB_TOKEN': '', 'github.ref': 'refs/pull/merge', 'github.event.number': str(pr),
               'github.event.pull_request.number': str(pr)}
        jobs = (doc or {}).get('jobs') or {}
        order, done = [], set()
        def visit(j, stack=()):
            if j in done or j in stack or j not in jobs: return
            needs = jobs[j].get('needs') or []
            for n in ([needs] if isinstance(needs, str) else needs): visit(n, stack + (j,))
            done.add(j); order.append(j)
        for j in jobs: visit(j)
        results = {}
        for job_id in order:
            job = jobs[job_id]
            needs = job.get('needs') or []
            needs = [needs] if isinstance(needs, str) else list(needs)
            jctx = dict(ctx)
            jctx['toJSON(needs)'] = json.dumps({n: {'result': results.get(n, 'skipped'), 'outputs': {}} for n in needs})
            if any(results.get(n) == 'unsupported' for n in needs):
                runs.append({'workflow': fn, 'job': job_id, 'matrix': {}, 'steps': [], 'status': 'UNSUPPORTED',
                             'reason': 'needs an unsupported job: ' + ','.join(n for n in needs if results.get(n) == 'unsupported')})
                results[job_id] = 'unsupported'; continue
            statuses = []
            for combo in matrix_combos(job):
                git('checkout', '-q', '--', '.', cwd=wt, check=False); git('clean', '-qfdx', cwd=wt, check=False)
                rec = run_job(wt, fn, job_id, job, combo, jctx); runs.append(rec); statuses.append(rec['status'])
            results[job_id] = ('failure' if 'FAIL' in statuses else 'unsupported' if 'UNSUPPORTED' in statuses
                               else 'skipped' if statuses and all(x == 'SKIPPED' for x in statuses) else 'success')
    with LOCK: git('worktree', 'remove', '--force', wt, check=False)
    res['selected'] = [s['workflow'] for s in selections if s['selected']]
    res['runs'] = runs
    res['status'] = 'FAIL' if any(r['status'] == 'FAIL' for r in runs) else 'PASS'
    with open(os.path.join(OUT, 'target-%d.json' % pr), 'w') as fh: json.dump(res, fh, indent=1)
    print(pr, res['status'], len(res['selected']), 'workflows', sum(r['status'] == 'FAIL' for r in runs), 'failing jobs', flush=True)
    return res

if __name__ == '__main__':
    with cf.ThreadPoolExecutor(max_workers=int(os.environ.get('WORKERS', '3'))) as ex:
        for _ in ex.map(target, TARGETS): pass
