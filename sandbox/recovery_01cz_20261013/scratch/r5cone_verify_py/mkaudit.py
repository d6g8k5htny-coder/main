import importlib.util, sys
from pathlib import Path
P = Path(sys.argv[1]); F = Path(sys.argv[2]); OUT = Path(sys.argv[3])
spec = importlib.util.spec_from_file_location('pp_check', P / 'check.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
registered, structures, rsha = m.load_register((P / m.STATEMENT_DIR).resolve())
props = '#[' + ', '.join('`' + r for r in sorted(registered)) + ']'
text = F.read_text(encoding='utf-8')
tail = ('\nopen Lean Meta Elab Command\n\n'
        'def auditModules : Array Name := #[]\n'
        'def auditPrefix : Name := `UniversalLaw.Proofs.Side24ConeMoment\n'
        f'def registeredProps : Array Name := {props}\n' + m.AUDIT_PROGRAM)
OUT.write_text(text + tail, encoding='utf-8')
print('registered props:', len(registered))
