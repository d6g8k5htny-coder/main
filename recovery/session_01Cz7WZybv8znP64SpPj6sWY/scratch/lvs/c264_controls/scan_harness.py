"""Run the companion gate's own scan_module on OrdinaryH0Controls.lean, unmodified check.py, MODULES extended in memory."""
import importlib.util
import re
import sys

sys.dont_write_bytecode = True
CHECK = sys.argv[1]
TARGET = sys.argv[2]
spec = importlib.util.spec_from_file_location('torus_check', CHECK)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
text = open(TARGET, encoding='utf-8').read()
scan = c.scan_module(text, 'OrdinaryH0Controls')
names = [n for k in ('theorem', 'abbrev', 'def', 'instance') for n in scan[k]]
bad = [n for n in names if not n.startswith('ordinaryH0Control_')]
assert not bad, bad
# docstring rule (convention of OrdinaryH0.lean): each declaration directly preceded by a docstring with exactly one
# Proves:/Method:/Does not establish: line
lines = text.split('\n')
for i, line in enumerate(lines):
    m = c.DECLARATION.match(line)
    if not m:
        continue
    j = i - 1
    assert lines[j].rstrip().endswith('-/'), ('no docstring before', m.group(2))
    k = j
    while not lines[k].startswith('/--'):
        k -= 1
    doc = lines[k:j + 1]
    for key in ('Proves:', 'Method:', 'Does not establish:'):
        count = sum(1 for d in doc if d.lstrip('/-').lstrip().startswith(key))
        assert count == 1, (m.group(2), key, count)
print('SCAN_OK', {k: len(v) for k, v in scan.items()})
for k in ('theorem', 'abbrev', 'def', 'instance'):
    print(k, scan[k])
