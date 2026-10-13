import hashlib, json, sys
from pathlib import Path
E = Path('<SCRATCH>/repos/math_c264ctl/.lake/torus-filtration-evidence')
raw = (E / 'receipt.json').read_bytes()
r = json.loads(raw)
print('receipt.json sha256', hashlib.sha256(raw).hexdigest(), 'bytes', len(raw))
for k in ('schema', 'scientific_effect', 'formalization_status', 'alignment_status', 'scientific_acceptance', 'checked_commit',
          'repository', 'run_id', 'run_attempt', 'lean_version', 'theorem_count', 'definition_count', 'leanchecker', 'census'):
    print(k, '=', json.dumps(r[k], ensure_ascii=False))
print('declarations', len(r['declarations']), 'axioms entries', len(r['axioms']))
print('axiom sets', sorted({tuple(v) for v in r['axioms'].values()}))
print('log_sha256 entries', len(r['log_sha256']))
print('negative_controls', json.dumps(r['negative_controls'], indent=1))
print('owner_modules in provenance:', json.dumps(r['provenance']['owner_modules']['OrdinaryH0Controls.lean']['sha256']))
print('dependency_revisions', len(r['dependency_revisions']))
members = sorted(p.name for p in E.iterdir())
print('evidence dir members', len(members), '(incl. receipt.json)')
for n in ('build-OrdinaryH0Controls', 'RejectMergerKillsPointClass', 'RejectSeparationWithoutNonzeroCoeff', 'RejectNonmergerKillsDifference', 'leanchecker'):
    for ext in ('.status', '.log'):
        f = E / (n + ext)
        print('--', f.name, 'sha256', hashlib.sha256(f.read_bytes()).hexdigest()[:16], repr(f.read_text()))
print('leanchecker.cmd', repr((E / 'leanchecker.cmd').read_text()))
for t in ('tests.log', 'tests-optimized.log'):
    txt = (E / t).read_text().splitlines()
    print(t, txt[0], '|', [l for l in txt if l.startswith('Ran ')], '|', txt[-1])
