"""Verifier harness: loads check.py UNMODIFIED from the worktree; in-memory only additions."""
import hashlib, importlib.util, json, re, sys, tempfile
from pathlib import Path
sys.dont_write_bytecode = True
CHECK = '<SCRATCH>/repos/math_c263reg/companions/torus_filtration_20261011/check.py'
V = Path('<SCRATCH>/lvs/c264_controls/verify/round2')
def load():
    spec = importlib.util.spec_from_file_location('torus_check', CHECK)
    c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c); return c
rep = json.load(open(V / 'report_controls.json', encoding='utf-8'))
mode = sys.argv[1]
if mode == 'write':
    # write the report's test files byte-exactly, and compare with what emit() writes once the 3 controls are registered
    c = load()
    c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **{r['name']: (r['proposition'], r['tactic']) for r in rep})
    c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
    with tempfile.TemporaryDirectory() as t:
        c.emit(Path(t))
        for r in rep:
            (V / 'controls' / (r['name'] + '.lean')).write_text(r['test_file'], encoding='utf-8')
            emitted = (Path(t) / (r['name'] + '.lean')).read_bytes()
            print(r['name'], 'report test_file == emit() bytes:', emitted == r['test_file'].encode('utf-8'),
                  hashlib.sha256(emitted).hexdigest()[:16])
elif mode == 'negative':
    c = load()
    c.FALSE_CONTROLS = dict(c.FALSE_CONTROLS, **{r['name']: (r['proposition'], r['tactic']) for r in rep})
    c.CONTROLS = tuple(c.FALSE_CONTROLS) + tuple(c.INJECTIONS)
    for item in sys.argv[2:]:
        name, d = item.split(':')
        log = (V / d / (name + '.log')).read_text(encoding='utf-8'); st = int((V / d / (name + '.status')).read_text())
        try: out = c.negative(log, name, st)
        except ValueError as e: out = 'REFUSED: ' + str(e)
        print(d + '/' + name, '->', out)
elif mode == 'axioms':
    c = load(); names = (V / 'targets.txt').read_text().split()
    recs = c.audit_axioms((V / 'logs/axioms.log').read_text(encoding='utf-8'), names)
    print('audit_axioms OK', len(recs), 'targets; axiom sets:', sorted({tuple(v) for v in recs.values()}))
elif mode == 'census':
    c = load(); names = (V / 'targets.txt').read_text().split()
    print('census:', c.census((V / 'logs/census.log').read_text(encoding='utf-8'), names, []))
elif mode == 'scan':
    c = load(); text = (V / 'OrdinaryH0Controls.lean').read_text(encoding='utf-8')
    try: c.scan_module(text, 'OrdinaryH0Controls'); print('unmodified scan: accepted')
    except ValueError as e: print('unmodified scan refuses:', e)
    c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
    c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
    s = c.scan_module(text, 'OrdinaryH0Controls')
    print('scan with MODULES extended in memory:', {k: len(v) for k, v in s.items()})
    names = sorted('TorusFiltration.' + n for k in s for n in s[k])
    print('scan inventory == targets.txt:', names == sorted((V / 'targets.txt').read_text().split()))
