#!/usr/bin/env python3
"""Comment-splice and digit-prefix probes through the source CLI; also prints what the scanner's
comment stripper leaves of each line, to show why the token is invisible to it."""
import importlib.util
import sys
import tempfile
from pathlib import Path

TREE = Path(__file__).resolve().parent / 'tree'
SIDE = TREE / 'companions/torus_filtration_20261011'
sys.path.insert(0, str(SIDE))
spec = importlib.util.spec_from_file_location('test_torus_filtration', SIDE / 'test_torus_filtration.py')
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
c = T.checker()
NS = 'namespace TorusFiltration\n'

PROBES = [
    ('indented theorem/- -/hidden after namespace', lambda o: o.replace(NS, NS + '  theorem/- -/hidden : True := trivial\n')),
    ('indented alias/- -/hidden after namespace', lambda o: o.replace(NS, NS + '  alias/- -/hidden := Nat.add_comm\n')),
    ('same-line theorem/- -/hidden on a declaration line', lambda o: o.replace('theorem superlevel_mono : True := trivial',
                                                                              'theorem superlevel_mono : True := trivial theorem/- -/b_hidden : True := trivial')),
    ('indented def/-\\n-/hidden (comment spans lines)', lambda o: o.replace(NS, NS + '  def/-\n-/hidden : Nat := 0\n')),
    ('exact/- -/sorry inside a proof (ADMISSION scan)', lambda o: o.replace('theorem superlevel_mono : True := trivial',
                                                                        'theorem superlevel_mono : True := by\n  exact/- -/sorry')),
    ('by/- -/native_decide (ADMISSION scan)', lambda o: o.replace('theorem superlevel_mono : True := trivial',
                                                               'theorem superlevel_mono : 2 + 2 = 4 := by/- -/native_decide')),
    ('def body := 0alias hidden := ... (digit prefix)', lambda o: o.replace('noncomputable def superlevel_set : Nat := 0',
                                                                          'noncomputable def superlevel_set : Nat := 0alias hidden := Nat.add_comm')),
    ('def body := 0theorem hidden ... (digit prefix)', lambda o: o.replace('noncomputable def superlevel_set : Nat := 0',
                                                                          'noncomputable def superlevel_set : Nat := 0theorem hidden : True := trivial')),
    ('control: alias /- -/ hidden with spaces (should be refused)', lambda o: o.replace(NS, NS + '  alias /- -/ hidden := Nat.add_comm\n')),
    ('control: plain indented theorem (should be refused)', lambda o: o.replace(NS, NS + '  theorem hidden : True := trivial\n')),
]

with tempfile.TemporaryDirectory() as tmp:
    f = T.Fixture(c, tmp)
    module = f.side / 'ClosedSuperlevel.lean'
    original = module.read_text()
    for label, mutate in PROBES:
        text = mutate(original)
        assert text != original, label
        module.write_text(text)
        f.restore()
        run = f.run('source')
        changed = [ln for ln in text.split('\n') if ln not in original.split('\n')]
        stripped = c.code_lines('\n'.join(changed))
        msg = next((ln.strip()[:120] for ln in run.stderr.splitlines() if 'TORUS_FILTRATION_FAIL' in ln), '')
        print('%-62s | %-8s | scanner sees %r | %s' % (label, 'ACCEPTED' if run.returncode == 0 else 'REFUSED', stripped, msg))
print('optimize:', sys.flags.optimize, 'child:', T.child_python()[1:])
