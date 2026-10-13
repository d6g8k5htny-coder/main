import importlib.util, sys, re, subprocess
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
new = load(sys.argv[1], 'check_new'); old = load(sys.argv[2], 'check_old')
added = [c for c in new.COMMANDS if c not in old.COMMANDS]
print('old', len(old.COMMANDS), 'new', len(new.COMMANDS), 'added', len(added), 'removed', [c for c in old.COMMANDS if c not in new.COMMANDS])
print('added:', added)
readme = open(sys.argv[3], encoding='utf-8').read()
m = re.search(r'Batteries, Aesop and Mathlib at the pin \((.*?)\)\.', readme, re.S)
rl = re.findall(r'`([^`]+)`', m.group(1))
print('README list == additions (ordered):', rl == added, '| as sets:', set(rl) == set(added))
print('EVIDENCE_FILES', len(new.EVIDENCE_FILES), sorted(new.EVIDENCE_FILES))
print('EMITTED_FILES', len(new.EMITTED_FILES), 'BUILDS', new.BUILDS, 'CONTROLS', new.CONTROLS)
print('ADMISSION unchanged:', new.ADMISSION == old.ADMISSION, '| DECLARATION unchanged:', new.DECLARATION.pattern == old.DECLARATION.pattern, '| PRELUDE unchanged:', {k:v.pattern for k,v in new.PRELUDE.items()} == {k:v.pattern for k,v in old.PRELUDE.items()})
print('LEANCHECKER unchanged:', new.LEANCHECKER == old.LEANCHECKER, '| FALSE_CONTROLS unchanged:', new.FALSE_CONTROLS == old.FALSE_CONTROLS, '| INJECTIONS unchanged:', new.INJECTIONS == old.INJECTIONS, '| POSITIVE unchanged:', new.POSITIVE == old.POSITIVE)
# every module-level constant compared
for k in sorted(set(dir(old)) & set(dir(new))):
    if k.startswith('_'): continue
    a, b = getattr(old, k), getattr(new, k)
    if callable(a) or hasattr(a, 'pattern'): continue
    if a != b: print('CHANGED CONSTANT', k)
print('constants only in new:', sorted(set(dir(new)) - set(dir(old))))
print('names only in old:', sorted(set(dir(old)) - set(dir(new))))
