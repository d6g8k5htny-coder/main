import importlib.util, re
from pathlib import Path
S = Path('<SCRATCH>')
pkt = S/'repos/math/specifications/lean_statement_proofs_20261011'
spec = importlib.util.spec_from_file_location('check', pkt/'check.py')
check = importlib.util.module_from_spec(spec); spec.loader.exec_module(check)
src = (S/'lvs/round4/work/DerivativePairingBound.lean').read_text(encoding='utf-8')
names = re.findall(r'^theorem\s+(\S+)', src, re.M)
ns = 'UniversalLaw.Proofs.Side24Analysis'
fq = [ns + ('.Aux.' if n.startswith('dp_') else '.') + n for n in names]
tail = ['', '-- VERIFIER SCRATCH (not the deliverable)', 'open Lean Meta Elab Command', '',
        '#check @UniversalLaw.Proofs.Side24Analysis.derivativePairingBound',
        '#print UniversalLaw.Spec.Side24.DerivativePairingBound',
        '#print UniversalLaw.Proofs.Side24Analysis.derivativePairingBound',
        '#print UniversalLaw.Spec.Side24.Kinf', '#print UniversalLaw.Spec.Side24.pairingSum', '#print UniversalLaw.Spec.Side24.IsUnitTuple', '']
tail += [f'#print axioms {n}' for n in fq]
tail += ['', 'def auditModules : Array Name := #[]',
         'def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis',
         'def registeredProps : Array Name := #[`UniversalLaw.Spec.Side24.DerivativePairingBound]',
         check.AUDIT_PROGRAM]
out = 'import Lean\n' + src + '\n'.join(tail) + '\n'
(S/'lvs/round4/verify_dp/_dp_audit.lean').write_text(out, encoding='utf-8')
print(len(names), 'theorems:', names)
