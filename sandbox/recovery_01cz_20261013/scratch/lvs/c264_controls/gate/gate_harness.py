"""Drive the companion gate's OWN functions (check.py loaded unmodified from the worktree; nothing is written there).

In memory only, FALSE_CONTROLS gains the three proposed C264 controls (CONTROLS is recomputed from it), exactly the
dictionary entries a registration would add. Modes:
  emit <out>                          c.emit(out): the control files the gate itself writes (header 'import TorusFiltration').
  discrim <out> <discrim.json>        same byte format as emit() for the true sibling statements (not gate files).
  negative <control>:<log>:<status>…  c.negative(log text, control, status); prints the outcome or the refusal.
  axioms <log> <targets>              c.audit_axioms(log text, targets) with the gate's ALLOWED set.
  scan <module.lean>                  c.scan_module with MODULES and PRELUDE['import'] extended in memory (conditional).
"""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
CHECK = '<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011/check.py'
CONTROLS_JSON = '<SCRATCH>/lvs/c264_controls/gate/controls.json'
spec = importlib.util.spec_from_file_location('torus_check', CHECK)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
proposed = {k: tuple(v) for k, v in json.load(open(CONTROLS_JSON, encoding='utf-8')).items()}
assert not set(proposed) & set(c.FALSE_CONTROLS)
c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **proposed)
c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
mode = sys.argv[1]
if mode == 'emit':
    out = Path(sys.argv[2])
    c.emit(out)
    for name in c.FALSE_CONTROLS:
        raw = (out / (name + '.lean')).read_bytes()
        print(name, len(raw), hashlib.sha256(raw).hexdigest())
elif mode == 'discrim':
    out = Path(sys.argv[2])
    for name, (statement, tactic) in json.load(open(sys.argv[3], encoding='utf-8')).items():
        (out / (name + '.lean')).write_text('import ' + c.ROOT_MODULE + '\nopen ' + c.NAMESPACE + '\nexample : '
                                            + statement + ' := by\n  ' + tactic + '\n')
        print(name, 'written')
elif mode == 'negative':
    for item in sys.argv[2:]:
        control, log, status = item.split(':')
        try:
            outcome = c.negative(open(log, encoding='utf-8').read(), control, int(open(status).read().strip()))
        except ValueError as refusal:
            outcome = 'REFUSED: ' + str(refusal)
        print(control, '<-', log.rsplit('/', 2)[-2] + '/' + log.rsplit('/', 1)[-1], ':', outcome)
elif mode == 'axioms':
    names = open(sys.argv[3]).read().split()
    print('audit_axioms:', c.audit_axioms(open(sys.argv[2], encoding='utf-8').read(), names) and 'OK', len(names), 'targets')
elif mode == 'scan':
    c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
    c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
    scan = c.scan_module(open(sys.argv[2], encoding='utf-8').read(), 'OrdinaryH0Controls')
    print('scan_module (MODULES extended in memory):', {k: len(v) for k, v in scan.items()})
