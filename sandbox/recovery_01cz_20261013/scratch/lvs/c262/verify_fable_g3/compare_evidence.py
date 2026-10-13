"""Compare check.py EVIDENCE_FILES (HEAD copy) against: replay.sh write targets, the worktree's local
evidence dir (read-only listing), the local HEAD receipt's log_sha256 keys, and the 0dd0cb17 receipt's keys."""
import importlib.util, json, os, re, sys
from pathlib import Path

V = Path('<SCRATCH>/lvs/c262/verify_fable_g3/tree')
SIDE = V / 'companions/torus_filtration_20261011'
spec = importlib.util.spec_from_file_location('check_copy', SIDE / 'check.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
E = set(c.EVIDENCE_FILES)
print('sys.flags.optimize =', sys.flags.optimize)
print('len(EVIDENCE_FILES) =', len(E), '; receipt.json in set:', 'receipt.json' in E)
print('EMITTED_FILES =', len(c.EMITTED_FILES), sorted(c.EMITTED_FILES))

# Enumerate replay.sh writes into $OUT / $TORUS_OUT by text (my own derivation, not the author's).
replay = (SIDE / 'replay.sh').read_text()
targets = set()
for m in re.finditer(r'"\$(?:OUT|TORUS_OUT)/([^"]+)"', replay):
    targets.add(m.group(1))
print('raw replay targets:', sorted(targets))
derived = set()
for t in targets:
    if t.startswith('$label.'):
        for label in ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types', 'leanchecker'):
            derived.add(label + t[len('$label'):])
    elif t.startswith('$control.'):
        for ctrl in c.CONTROLS:
            derived.add(ctrl + t[len('$control'):])
    elif t.endswith('.lean') and t.startswith('$control'):
        pass
    else:
        derived.add(t)
# The .lean inputs lean reads (Positive.lean, Audit.lean, $control.lean) are emit() outputs, not shell writes.
derived -= {'Positive.lean', 'Audit.lean', 'Types.lean'}
derived = {d for d in derived if not d.endswith('.lean')}
derived |= set(c.EMITTED_FILES)   # written by `check.py emit "$OUT"` at replay.sh line 15
print('derived replay.sh pre-finish set size:', len(derived))
print('derived - EVIDENCE_FILES:', sorted(derived - E))
print('EVIDENCE_FILES - derived:', sorted(E - derived))

wt = Path('<SCRATCH>/repos/math_c254/.lake/torus-filtration-evidence')
names = {p.name for p in wt.iterdir()}
print('worktree evidence dir entries:', len(names), '; minus receipt == EVIDENCE_FILES:', (names - {'receipt.json'}) == E)
print('all non-receipt entries regular non-symlink files:', all((wt / n).is_file() and not (wt / n).is_symlink() for n in names - {'receipt.json'}))
r = json.loads((wt / 'receipt.json').read_text())
keys = set(r['log_sha256'])
print('HEAD local receipt keys == EVIDENCE_FILES:', keys == E, '; receipt.json in keys:', 'receipt.json' in keys, '; n =', len(keys))
print('HEAD local receipt checked_commit:', r['checked_commit'], 'leanchecker:', r['leanchecker'])
import hashlib
bad = [n for n, d in r['log_sha256'].items() if hashlib.sha256((wt / n).read_bytes()).hexdigest() != d]
print('HEAD local receipt digest mismatches:', bad)
old = Path('<SCRATCH>/lvs/c254/local_evidence_0dd0cb17/receipt.json')
o = json.loads(old.read_text())
ok = set(o['log_sha256'])
print('0dd0cb17 local receipt keys == EVIDENCE_FILES:', ok == E, '; n =', len(ok), '; diff:', sorted(ok ^ E))
print('0dd0cb17 receipt checked_commit:', o.get('checked_commit'))
