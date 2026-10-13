import sys
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(PKT))
import check
registered, structures, _ = check.load_register(STMT)
names = sorted(registered)
assert 'UniversalLaw.Spec.Side24.DensityComparison' in names
print("REGISTERED_PROPS", len(names))
text = FILE.read_text(encoding='utf-8')
assert text.startswith('import Specifications.Side24\n')
thms = ['UniversalLaw.Proofs.Side24Analysis.Aux.' + n for n in [
 'dc_epsilon_pos','dc_epsilon_lt_one','dc_one_sub_epsilon_pos','dc_one_add_epsilon_pos','dc_expand_identity',
 'dc_conj_identity','dc_inv_sub_inv_posSemidef','dc_quad_le','dc_one_le_det','dc_det_le_det','dc_det_smul',
 'dc_scale_identity','dc_assemble_lower','dc_assemble_upper',
 'dc_expand_identity._abel_1_1','dc_expand_identity._abel_1_2']] + ['UniversalLaw.Proofs.Side24Analysis.densityComparison']
tail = '\n-- ===== VERIFIER SCRATCH TAIL (never part of the deliverable) =====\nopen Lean Meta Elab Command\n\n'
tail += 'def auditModules : Array Name := #[]\n'
tail += 'def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis\n'
tail += 'def registeredProps : Array Name := #[' + ', '.join('`' + n for n in names) + ']\n'
tail += check.AUDIT_PROGRAM
tail += '\n#check @UniversalLaw.Proofs.Side24Analysis.densityComparison\n'
tail += '#print UniversalLaw.Spec.Side24.DensityComparison\n'
tail += ''.join(f'#print axioms {n}\n' for n in thms)
tail += '''
run_cmd do
  let env ← getEnv
  let some ci := env.find? `UniversalLaw.Proofs.Side24Analysis.densityComparison | throwError "VERIFY: theorem missing"
  logInfo m!"VERIFY levelParams={ci.levelParams} type={ci.type}"
  unless ci.type.isConstOf `UniversalLaw.Spec.Side24.DensityComparison do throwError "VERIFY: type is not the bare registered constant"
  unless ci.type.getAppNumArgs == 0 do throwError "VERIFY: type has arguments"
  match ci with
  | .thmInfo _ => logInfo "VERIFY kind=theorem"
  | _ => throwError "VERIFY: not a theorem"
  logInfo "VERIFY_TYPE_EXACT_CONST OK"
'''
OUT.write_text('import Lean\n' + text + tail, encoding='utf-8')
print("WROTE", OUT, len(('import Lean\n' + text + tail).splitlines()), "lines")
