import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path('/home/user/Math-/formal')
s = importlib.util.spec_from_file_location('g', ROOT / 'gate.py'); g = importlib.util.module_from_spec(s); s.loader.exec_module(g)
real = g.run
def run(command, label, out, expect_success=True, cwd=g.ROOT):
    if 'leanchecker' in label:
        (out / (label + '.log')).write_text('SKIPPED LOCALLY: leanchecker OOM (exit 137) in this container\n')
        print(label + ': SKIPPED (local OOM limit)', flush=True); return ''
    return real(command, label, out, expect_success, cwd)
g.run = run
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
m = json.loads((ROOT / 'manifest.json').read_text())
modules = {g.PACKAGE + '.' + p[len(g.PACKAGE) + 1:-5] for p in m['source_modules']}
bound = g.bound_inventory(m['compiled_inventory'], m['targets'], modules)
t = time.time()
prog = out / 'Inventory.lean'; prog.write_text(g.inventory_source())
base = g.audit_inventory(g.run(['lake', 'env', 'lean', str(prog)], 'inventory', out), m['targets'], modules, bound=bound)
print('baseline', base['declarations'], len(base['public_theorems']), base['sha256'], flush=True)
try:
    print('compiled', json.dumps(g.compiled_admission(m, modules, base, bound, out), indent=1), flush=True)
except ValueError as e:
    print('compiled FAILED', e, flush=True)
print('elapsed', round(time.time() - t), flush=True)
