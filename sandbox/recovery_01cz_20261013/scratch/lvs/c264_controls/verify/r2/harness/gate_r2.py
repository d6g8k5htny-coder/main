"""Verifier harness: loads the worktree check.py UNMODIFIED (read-only; no bytecode written) and calls its own functions.
In-memory registration emulation is explicit per mode and printed."""
import hashlib, importlib.util, json, re, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import reported
CHECK = '<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011/check.py'
def load():
    spec = importlib.util.spec_from_file_location('torus_check_verify', CHECK)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c); return c
def register_controls(c):
    assert not set(reported.CONTROLS) & set(c.FALSE_CONTROLS)
    c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **{k: (v[0], v[1]) for k, v in reported.CONTROLS.items()})
    c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
mode = sys.argv[1]
c = load()
if mode == 'scan_unmodified':
    try:
        print('UNMODIFIED scan_module:', c.scan_module(open(sys.argv[2], encoding='utf-8').read(), 'OrdinaryH0Controls'))
    except ValueError as e:
        print('UNMODIFIED scan_module REFUSED:', e)
elif mode == 'scan_registered':
    c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
    c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
    scan = c.scan_module(open(sys.argv[2], encoding='utf-8').read(), 'OrdinaryH0Controls')
    print('REGISTERED-EMULATION scan_module counts:', {k: len(v) for k, v in scan.items()})
    print(json.dumps(scan))
elif mode == 'emit':
    register_controls(c)
    out = Path(sys.argv[2]); c.emit(out)
    for name, (stmt, tac, test_file, _) in reported.CONTROLS.items():
        raw = (out / (name + '.lean')).read_bytes()
        print(name, len(raw), hashlib.sha256(raw).hexdigest()[:16], 'BYTE-EQUAL-TO-REPORTED' if raw == test_file.encode() else 'DIFFERS')
elif mode == 'negative':
    register_controls(c)
    for item in sys.argv[2:]:
        control, log, status = item.split(':')
        try:
            outcome = c.negative(open(log, encoding='utf-8').read(), control, int(open(status).read().strip()))
        except ValueError as e:
            outcome = 'REFUSED: ' + str(e)
        print(control, outcome)
elif mode == 'axioms':
    names = open(sys.argv[3]).read().split()
    print('audit_axioms', len(c.audit_axioms(open(sys.argv[2], encoding='utf-8').read(), names)), 'targets OK')
elif mode == 'census':
    names = open(sys.argv[3]).read().split()
    print(c.census(open(sys.argv[2], encoding='utf-8').read(), names, []))
