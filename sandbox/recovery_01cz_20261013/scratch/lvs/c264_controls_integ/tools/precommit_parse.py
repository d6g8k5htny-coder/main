"""Parse the pre-commit Lean logs with the companion's own check.py functions (loaded from the worktree, unmodified)."""
import importlib.util, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
C = Path('<SCRATCH>/repos/math_c264ctl/companions/torus_filtration_20261011/check.py')
L = Path('<SCRATCH>/lvs/c264_controls_integ/precommit/lean')
spec = importlib.util.spec_from_file_location('chk', C)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
meta = c.metadata()
for label in ('positive', 'axioms', 'types', 'census', 'leanchecker'):
    print(label, 'status', (L / (label + '.status')).read_text().strip())
print('positive log empty:', not (L / 'positive.log').read_text().strip())
print('leanchecker log empty:', not (L / 'leanchecker.log').read_text().strip())
axioms = c.audit_axioms((L / 'axioms.log').read_text(), c.targets(meta))
print('audit_axioms OK:', len(axioms), 'targets; axiom sets used:', sorted({tuple(v) for v in axioms.values()}))
controls_targets = ['TorusFiltration.' + n for kind in ('theorem',) for n in meta['theorems']['OrdinaryH0Controls']] + \
    ['TorusFiltration.' + n for k in ('abbrev', 'def', 'instance') for n in meta['definitions']['OrdinaryH0Controls'][k]]
print('controls-module targets audited:', len(controls_targets), all(t in axioms for t in controls_targets),
      'axiom sets:', sorted({tuple(axioms[t]) for t in controls_targets}))
print('types OK:', len(c.types((L / 'types.log').read_text(), c.theorem_targets(meta))))
print('census:', json.dumps(c.census((L / 'census.log').read_text(), c.targets(meta), meta['census_auxiliary'])))
for control in c.CONTROLS:
    status = int((L / (control + '.status')).read_text().strip())
    print(control, 'status', status, '->', c.negative((L / (control + '.log')).read_text(), control, status))
for control in ('RejectMergerKillsPointClass', 'RejectSeparationWithoutNonzeroCoeff', 'RejectNonmergerKillsDifference'):
    p = L / ('discrim-' + control + '.log')
    if not p.exists():
        print('discriminator', control, 'not run yet'); continue
    status = int((L / ('discrim-' + control + '.status')).read_text().strip())
    try:
        out = c.negative(p.read_text(), control, status)
        print('discriminator', control, 'ACCEPTED (unexpected):', out)
    except ValueError as refusal:
        print('discriminator', control, 'status', status, 'refused:', refusal)
