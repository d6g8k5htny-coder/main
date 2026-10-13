import importlib.util, sys, re
from pathlib import Path
check_path, statement_root, deliverable, outdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
spec = importlib.util.spec_from_file_location('packet_check', check_path)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
registered = mod.load_register(statement_root)
if isinstance(registered, tuple): registered = registered[0]
text = deliverable.read_text()
assert text.startswith('import Specifications.Side24\n')
body = text[len('import Specifications.Side24\n'):]
names = re.findall(r'^theorem\s+(\S+)', mod.lex(text)[0], re.M)
print('theorems found:', len(names))
NS = 'UniversalLaw.Proofs.Side24Analysis'
fq = []
in_aux = False
for ln in mod.lex(text)[0].split('\n'):
    if ln.startswith('namespace Aux'): in_aux = True
    if ln.startswith('end Aux'): in_aux = False
    m = re.match(r'^theorem\s+(\S+)', ln)
    if m: fq.append(NS + ('.Aux.' if in_aux else '.') + m.group(1))
props = '#[' + ', '.join('`' + r for r in sorted(registered)) + ']'
tail = '\n\n-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====\n'
tail += ''.join(f'#print axioms {n}\n' for n in fq)
tail += f'#check @{NS}.derivativeBoundAtZero\n#check @{NS}.Aux.d0_derivativeBoundAtZero_of_derivativePairingBound\n'
tail += '#print UniversalLaw.Spec.Side24.DerivativeBoundAtZero\n#print UniversalLaw.Spec.Side24.DerivativePairingBound\n'
tail += '#print UniversalLaw.Spec.Side24.pairingSum\n#print UniversalLaw.Spec.Side24.Kinf\n#print UniversalLaw.Spec.Side24.IsUnitTuple\n'
tail += 'open Lean Meta Elab Command\n\n'
tail += 'def auditModules : Array Name := #[]\n'
tail += f'def auditPrefix : Name := `{NS}\n'
tail += f'def registeredProps : Array Name := {props}\n'
tail += '''
run_cmd do
  let env ← getEnv
  let some ci := env.find? `UniversalLaw.Proofs.Side24Analysis.derivativeBoundAtZero | throwError "inventory theorem missing"
  let expected : Expr := .const `UniversalLaw.Spec.Side24.DerivativeBoundAtZero []
  logInfo m!"EXACT_TYPE_INVENTORY {ci.type == expected} raw={ci.type}"
  let isThm := match ci with | .thmInfo _ => true | _ => false
  logInfo m!"INVENTORY_IS_THEOREM {isThm}"
  let some hci := env.find? `UniversalLaw.Proofs.Side24Analysis.Aux.d0_derivativeBoundAtZero_of_derivativePairingBound | throwError "helper missing"
  let ok := match hci.type with
    | .forallE _ (.const `UniversalLaw.Spec.Side24.DerivativePairingBound []) (.const `UniversalLaw.Spec.Side24.DerivativeBoundAtZero []) _ => true
    | _ => false
  logInfo m!"EXACT_TYPE_HELPER {ok} raw={hci.type}"
  let some pci := env.find? `UniversalLaw.Spec.Side24.DerivativeBoundAtZero | throwError "Prop missing"
  let modIdx := env.getModuleIdxFor? `UniversalLaw.Spec.Side24.DerivativeBoundAtZero
  let modName := match modIdx with | some i => toString env.header.moduleNames[i.toNat]! | none => "CURRENT_FILE"
  let pkind := match pci with | .defnInfo _ => "def" | .thmInfo _ => "thm" | .axiomInfo _ => "AXIOM" | .opaqueInfo _ => "OPAQUE" | _ => "other"
  logInfo m!"PROP_ORIGIN module={modName} kind={pkind} type={pci.type}"
  let modIdx2 := env.getModuleIdxFor? `UniversalLaw.Spec.Side24.DerivativePairingBound
  let modName2 := match modIdx2 with | some i => toString env.header.moduleNames[i.toNat]! | none => "CURRENT_FILE"
  logInfo m!"UPSTREAM_PROP_ORIGIN module={modName2}"
'''
tail += mod.AUDIT_PROGRAM
(outdir / 'D0_verify.lean').write_text('import Lean\nimport Specifications.Side24\n' + body + tail)
# Lean negative control: tighten 15 -> 14 in the helper statement only (must fail at q = 6)
neg = text.replace('theorem d0_pairingSum_zero_le (q : ℕ) (hq : q ≤ 6) : pairingSum q 0 ≤ 15 := by',
                   'theorem d0_pairingSum_zero_le (q : ℕ) (hq : q ≤ 6) : pairingSum q 0 ≤ 14 := by')
assert neg != text
(outdir / 'D0_negctrl_14.lean').write_text(neg)
# Python scanner negative controls
def expect_refusal(label, t):
    try:
        mod.scan_module(t, 'Side24Analysis', registered); print('NEGCTRL', label, 'NOT REFUSED  <-- PROBLEM')
    except ValueError as e:
        print('NEGCTRL', label, 'refused:', str(e)[:160])
expect_refusal('sorry', text.replace('Aux.d0_derivativeBoundAtZero_of_derivativePairingBound Aux.d0_derivativePairingBound', 'sorry'))
expect_refusal('smuggled hypothesis', text.replace('theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :=',
                                                   'theorem derivativeBoundAtZero (h : UniversalLaw.Spec.Side24.DerivativePairingBound) : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :='))
expect_refusal('implication type', text.replace('theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativeBoundAtZero :=',
                                                'theorem derivativeBoundAtZero : UniversalLaw.Spec.Side24.DerivativePairingBound → UniversalLaw.Spec.Side24.DerivativeBoundAtZero :='))
expect_refusal('hash command', text.replace('end UniversalLaw.Proofs.Side24Analysis', '#check 1\nend UniversalLaw.Proofs.Side24Analysis'))
expect_refusal('indented theorem', text.replace('\ntheorem d0_Kinf_pos', '\n  theorem d0_Kinf_pos'))
expect_refusal('Proves label missing', text.replace('/-- Proves: `UniversalLaw.Spec.Side24.DerivativeBoundAtZero`', '/-- Proves `UniversalLaw.Spec.Side24.DerivativeBoundAtZero`'))
print('fq names:'); print('\n'.join(fq))
