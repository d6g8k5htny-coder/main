import importlib.util, json, sys, time
from pathlib import Path
ROOT = Path('/home/user/Math-/formal')
s = importlib.util.spec_from_file_location('g', ROOT / 'gate.py'); g = importlib.util.module_from_spec(s); s.loader.exec_module(g)
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
m = json.loads((ROOT / 'manifest.json').read_text())
modules = {g.PACKAGE + '.' + p[len(g.PACKAGE) + 1:-5] for p in m['source_modules']}
t = time.time()
prog = out / 'Inventory.lean'; prog.write_text(g.inventory_source())
base = g.audit_inventory(g.run(['lake', 'env', 'lean', str(prog)], 'inventory', out), m['targets'], modules)
print('baseline', {k: (len(v) if isinstance(v, list) and k == 'public_theorems' else v) for k, v in base.items()}, flush=True)
P = (g.PACKAGE,)
adm = {
 'hidden_sorry': ('namespace ' + g.PACKAGE + '\n  private theorem hiddenAdmission : (1 : Nat) = 1 := by sorry\nend ' + g.PACKAGE + '\n', ('theorem','user',P+('hiddenAdmission',),True,g.LOCAL_MODULE), {('forbidden_axiom',('sorryAx',),False),('extra_theorem',P+('hiddenAdmission',),True)}),
 'hidden_theorem': ('namespace ' + g.PACKAGE + '\n  @[simp] theorem indentedAdmission : (1 : Nat) = 1 := rfl\nend ' + g.PACKAGE + '\n', ('theorem','user',P+('indentedAdmission',),False,g.LOCAL_MODULE), {('extra_theorem',P+('indentedAdmission',),False)}),
 'hidden_def': ('namespace ' + g.PACKAGE + '\n  noncomputable def unusedAdmission : Nat := by sorry\nend ' + g.PACKAGE + '\n', ('def','user',P+('unusedAdmission',),False,g.LOCAL_MODULE), {('forbidden_axiom',('sorryAx',),False)}),
 'outside_namespace': ('theorem outsideNamespaceAdmission : (1 : Nat) = 1 := rfl\n', ('theorem','user',('outsideNamespaceAdmission',),False,g.LOCAL_MODULE), {('extra_theorem',('outsideNamespaceAdmission',),False)})}
for label, (decl, inj, exp) in adm.items():
    path = out / (label + '.lean')
    path.write_text(g.inventory_source(include_local=True).replace('import Lean\n', 'import Lean\n' + decl, 1))
    log = g.run(['lake', 'env', 'lean', str(path)], label, out)
    try:
        print(label, json.dumps(g.check_admission(log, label, inj, exp, m['targets'], modules, True)), flush=True)
    except ValueError as e:
        print(label, 'FAILED', type(e).__name__, e, flush=True)
try:
    print('compiled', json.dumps(g.compiled_admission(m, modules, base, out)), flush=True)
except ValueError as e:
    print('compiled FAILED', e, flush=True)
print('elapsed', round(time.time() - t), flush=True)
