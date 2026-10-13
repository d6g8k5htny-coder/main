import sys, json
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
print('REGISTER_TYPE', type(registered).__name__, 'N', len(registered))
names = sorted(registered.keys() if isinstance(registered, dict) else registered)
print('HAS_TARGET', 'UniversalLaw.Spec.Side24.LatticeImageSumBound' in names)
props = '#[' + ', '.join('`' + r for r in names) + ']'
body = FILE.read_text(encoding='utf-8')
thms = ['UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound'] + [
 'UniversalLaw.Proofs.Side24Analysis.Aux.' + t for t in
 ['li_geom_tail','li_pow6_succ_le','li_exp_succ_le','li_summable_int_tsum_le','li_gauss_int','li_pow_gauss_int',
  'li_sum_prod_le','li_prod_ite_eq','li_prod_exp','li_sum_cube_le','li_imageTerm_nonneg','li_imageTerm_le','li_partial_sum_le']]
tail = '\n\n-- VERIFIER SCRATCH ONLY (not the deliverable).\n'
tail += ''.join(f'#print axioms {n}\n' for n in thms)
tail += '#check @UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound\n'
tail += '#print UniversalLaw.Spec.Side24.LatticeImageSumBound\n'
tail += '#print UniversalLaw.Spec.Side24.imageTerm\n#print UniversalLaw.Spec.Side24.intNormSq\n'
tail += 'example : UniversalLaw.Spec.Side24.LatticeImageSumBound := UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound\n'
tail += '''
open Lean Meta Elab Command in
run_cmd do
  let env ← getEnv
  -- syntactic identity of the inventory theorem's type with the registered constant
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.latticeImageSumBound | throwError "missing theorem"
  let target : Expr := mkConst ``UniversalLaw.Spec.Side24.LatticeImageSumBound
  logInfo m!"TYPE_SYNTACTIC_EQ {ci.type == target} TYPE={ci.type} LEVELPARAMS={ci.levelParams}"
  let isThm := match ci with | .thmInfo _ => true | _ => false
  logInfo m!"IS_THEOREM {isThm} UNSAFE {ci.isUnsafe}"
  -- the registered Prop must be a def of type Prop in the imported statement module, not declared here
  let some pci := env.find? ``UniversalLaw.Spec.Side24.LatticeImageSumBound | throwError "missing Prop"
  let pmod := env.getModuleIdxFor? ``UniversalLaw.Spec.Side24.LatticeImageSumBound
  let pmodName := match pmod with | some i => toString env.header.moduleNames[i.toNat]! | none => "THIS_FILE"
  logInfo m!"PROP_MODULE {pmodName} PROP_TYPE {pci.type}"
  -- every constant this file adds (env.constants.map₂), classified like check.py's audit
  let allowed : Array Name := #[``propext, ``Classical.choice, ``Quot.sound]
  let mut rows : Array String := #[]
  let mut refused := 0
  for (n, ci) in env.constants.map₂.toList do
    let axioms ← collectAxioms n
    let bad := axioms.filter (!allowed.contains ·)
    let range ← findDeclarationRanges? n
    let user := (privateToUserName? n).getD n
    let isAux := isPrivateName n || user.components.contains `Aux
    let (head, exact) ← liftTermElabM do
      forallTelescope ci.type fun xs body => do
        let args := body.getAppArgs
        let exact := args.size == xs.size && args.all (fun a => xs.contains a) && xs.all (fun x => args.contains x)
        pure (body.getAppFn.constName?, exact)
    let headStr : String := match head with | some c => toString c | none => "nonconstant"
    let registeredHead := match head with | some c => registeredProps.contains c | none => false
    let isP ← liftTermElabM (isProp ci.type)
    let kind : String :=
      if ci.isUnsafe then "REFUSED:unsafe" else
      match ci with
        | .axiomInfo _ => "REFUSED:axiom"
        | .opaqueInfo _ => "REFUSED:opaque"
        | .quotInfo _ => "REFUSED:quot"
        | .inductInfo _ => "auxiliary"
        | .ctorInfo _ => "auxiliary"
        | .recInfo _ => "auxiliary"
        | .defnInfo _ => if isP then "REFUSED:proof_as_def" else "auxiliary"
        | .thmInfo _ =>
          if isAux then "aux_theorem"
          else if range.isNone then (if registeredHead then s!"REFUSED:generated_proof:{headStr}" else "generated")
          else if registeredHead then (if exact then "theorem" else s!"REFUSED:premise:{headStr}")
          else s!"REFUSED:type:{headStr}"
    let verdict : String :=
      if kind.startsWith "REFUSED" || bad.isEmpty then kind
      else "REFUSED:axioms:" ++ ",".intercalate (bad.map toString).toList
    if verdict.startsWith "REFUSED" then refused := refused + 1
    rows := rows.push s!"AUDIT {verdict} {n} head={headStr} prop={isP} axioms={axioms}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo m!"AUDIT END total={rows.size} refused={refused}"
'''
decl = f'\n/-- registered Props of the statement register (verifier scratch) -/\ndef registeredProps : Array Lean.Name := {props}\n'
OUT.write_text(body + decl + tail, encoding='utf-8')
print('WROTE', OUT, 'LEN', len(body + decl + tail))
