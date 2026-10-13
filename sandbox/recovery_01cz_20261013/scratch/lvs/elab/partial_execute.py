"""Local replica of check.py --execute WITHOUT leanchecker (OOM-killed at the sandbox's 13.5 GB cgroup limit).
Runs kernel_audit (#check/#print axioms of every registered declaration), environment_audit and the negative
controls exactly as check.py does, writing logs under .lake/evidence_partial/. Produces NO receipt.json:
leanchecker was not run, so no `leanchecker: PASS` record may exist. Evidence of elaboration only."""
import importlib.util, json, sys, shutil
from pathlib import Path
P = Path(sys.argv[1]); REPO = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('check', P / 'check.py'); check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
info = check.source_check(P, REPO, True)
lake = check.lake_binary(); env = check.lean_env(lake)
out = P / '.lake' / 'evidence_partial'
if out.exists(): shutil.rmtree(out)
out.mkdir(parents=True)
version = check.check_lean_version(check.run([lake, 'env', 'lean', '--version'], 'version', out, P, env))
revs = check.dependency_state(P, info['dependency_revisions'])
log = check.run([lake, 'build'], 'build', out, P, env)
modules = check.MODULES
names = [d['name'] for m in modules for d in info['scans'][m] if d['kind']]
axioms, sigs = check.kernel_audit(lake, P, env, out, names)
expected = {'statement': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] == 'def_prop'],
            'structure': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] == 'structure'],
            'auxiliary': [d['name'] for m in modules for d in info['scans'][m] if d['kind'] is None]}
environment = check.environment_audit(lake, P, env, out, [m[:-5].replace('/', '.') for m in modules], expected)
reg = check.load_json(check.safe_path(P, 'REGISTER.json').read_text(encoding='utf-8'))
sources, _ = check.load_sources(P, None, reg)
outcomes = check.negative_controls(lake, P, env, out, reg, info['scans'], sources, modules)
after = check.source_check(P, REPO, True)
assert after['register_sha256'] == info['register_sha256'] and after['module_sha256'] == info['module_sha256']
summary = {'leanchecker': 'NOT_RUN (OOM-killed at 13.5 GB in this sandbox; hosted run is authoritative)', 'lean_version': version,
           'dependency_revisions': revs, 'declarations_audited': len(names), 'axiom_sets': sorted({tuple(v) for v in axioms.values()}),
           'environment_audit': environment, 'negative_controls': outcomes, 'register_sha256': info['register_sha256']}
(out / 'PARTIAL_SUMMARY.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
print(json.dumps(summary, indent=2, sort_keys=True))
