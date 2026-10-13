"""check.py scan_module (file unmodified, no bytecode) on the controls module; MODULES extended in memory only."""
import importlib.util, re, sys
sys.dont_write_bytecode = True
check, target, declared = sys.argv[1:4]
spec = importlib.util.spec_from_file_location('tc_scan', check)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
c.MODULES = c.MODULES + ('OrdinaryH0Controls',)
c.PRELUDE['import'] = re.compile(r'import (?:Mathlib(?:\.[A-Za-z0-9_]+)*|' + '|'.join(c.MODULES) + ')')
text = open(target, encoding='utf-8').read()
scan = c.scan_module(text, 'OrdinaryH0Controls')
names = [n for k in ('theorem', 'abbrev', 'def', 'instance') for n in scan[k]]
print('scan_module OK:', {k: len(v) for k, v in scan.items()}, 'total', len(names))
want = {l.split('.', 1)[1] for l in open(declared).read().split()}
print('scan inventory == reported declarations:', set(names) == want, '| dupes:', len(names) != len(set(names)))
print('all prefixed ordinaryH0Control_:', all(n.startswith('ordinaryH0Control_') for n in names))
# docstring rule: the line before every declaration closes a `/--` docstring holding exactly one of each key
lines = text.split('\n'); bad = []
for i, line in enumerate(lines):
    m = c.DECLARATION.match(line)
    if not m: continue
    j = i - 1
    if not lines[j].rstrip().endswith('-/'): bad.append((m.group(2), 'no docstring')); continue
    k = j
    while k >= 0 and '/--' not in lines[k] and '/-!' not in lines[k]: k -= 1
    if k < 0 or not lines[k].startswith('/--'): bad.append((m.group(2), 'not a /-- docstring')); continue
    doc = '\n'.join(lines[k:j + 1])
    for key in ('Proves:', 'Method:', 'Does not establish:'):
        n = len(re.findall(r'(?m)^(?:/--\s*)?' + re.escape(key), doc))
        n_any = doc.count(key)
        if n != 1 or n_any != 1: bad.append((m.group(2), key, n, n_any))
print('docstring rule violations:', bad)
# admission / escape tokens anywhere in code, independent of the scanner
code = c.strip_comments(text)
refused = ['sorry', 'admit', 'axiom', 'native_decide', 'set_option', 'opaque', 'unsafe', 'implemented_by', 'extern',
           'maxHeartbeats', 'decide', 'ofReduceBool', 'Lean.', 'elab', 'macro', 'attribute', '@[', '#']
print('raw substrings found in code:', [t for t in refused if t in code])
