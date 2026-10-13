import importlib.util, sys
from pathlib import Path
P = Path(sys.argv[1]); M = Path(sys.argv[2]); SR = Path(sys.argv[3]); OUT = Path(sys.argv[4])
spec = importlib.util.spec_from_file_location('pcheck', P / 'check.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
registered, _, _ = mod.load_register(SR)
text = M.read_text(encoding='utf-8')
lines = text.split('\n')
imports = [l for l in lines if l.startswith('import ')]
body = '\n'.join(l for l in lines if not l.startswith('import '))
props = '#[' + ', '.join('`' + r for r in sorted(registered)) + ']'
src = ('import Lean\n' + '\n'.join(imports) + '\n' + body + '\n\nopen Lean Meta Elab Command\n\n'
       'def auditModules : Array Name := #[]\n'
       'def auditPrefix : Name := `UniversalLaw.Proofs.Side24Covariance\n'
       f'def registeredProps : Array Name := {props}\n' + mod.AUDIT_PROGRAM)
OUT.write_text(src, encoding='utf-8')
print('registered props:', len(registered))
