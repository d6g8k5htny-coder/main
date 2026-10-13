import sys
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(PKT))
import check
registered, structures, _ = check.load_register(STMT)
props = '#[' + ', '.join('`' + r for r in sorted(registered)) + ']'
text = FILE.read_text(encoding='utf-8')
NS = 'UniversalLaw.Proofs.Side24Interfaces'
names = [NS + '.coneMomentD1Identity'] + [NS + '.Aux.' + n for n in (
    'c1_aemeasurable_of_map_eq_gaussian', 'c1_measurable_indicator_sq', 'c1_second_moment_gaussianReal',
    'c1_integrable_sq_gaussianReal', 'c1_setIntegral_sq_Iio_eq_Ioi', 'c1_setIntegral_sq_Iio', 'c1_integral_indicator_eq')]
tail = '\n-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====\n'
tail += '#check @' + names[0] + '\n'
tail += '#print UniversalLaw.Spec.Side24.ConeMomentD1Identity\n'
tail += '#print UniversalLaw.Spec.Side24.coneMomentD1\n'
tail += ''.join(f'#print axioms {n}\n' for n in names)
# packet audit program, auditModules empty => this file's constants under auditPrefix
tail += '\nopen Lean Meta Elab Command\n\n'
tail += 'def auditModules : Array Name := #[]\n'
tail += f'def auditPrefix : Name := `{NS}\n'
tail += f'def registeredProps : Array Name := {props}\n'
tail += check.AUDIT_PROGRAM
# full stage-2 listing (every constant this file adds, any namespace)
tail += '''
#eval show CommandElabM Unit from do
  let env ← getEnv
  let mut rows : Array String := #[]
  for (n, ci) in env.constants.map₂.toList do
    let kind := match ci with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM"
      | .opaqueInfo _ => "OPAQUE" | .quotInfo _ => "QUOT" | .inductInfo _ => "ind"
      | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (Meta.isProp ci.type)
    rows := rows.push s!"STAGE2 {kind} prop={isP} unsafe={ci.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo s!"STAGE2 TOTAL {rows.size}"
'''
OUT.write_text(text + tail, encoding='utf-8')
print('registered props:', len(registered)); print('wrote', OUT)
