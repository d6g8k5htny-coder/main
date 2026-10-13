import sys, re
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(PKT))
import check
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
props = sorted(registered.keys())
assert 'UniversalLaw.Spec.Side24.ReferencePinDensityProduct' in props
assert 'UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks' in props
text = FILE.read_text(encoding='utf-8')
lines = text.split('\n')
assert lines[0] == 'import Specifications.Side24', lines[0]
body = text
names = re.findall(r'^theorem\s+(\S+)', text, flags=re.M)
assert len(names) == 20, names
fq = []
for n in names:
    fq.append(('UniversalLaw.Proofs.Side24Analysis.Aux.' if n.startswith('rp_') else 'UniversalLaw.Proofs.Side24Analysis.') + n)
tail = []
tail.append('\n\n-- VERIFIER SCRATCH TAIL (not part of the deliverable) ------------------------------')
tail.append('#check @UniversalLaw.Proofs.Side24Analysis.referencePinDensityProduct')
tail.append('#print UniversalLaw.Spec.Side24.ReferencePinDensityProduct')
tail.append('#check @UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks')
for n in fq:
    tail.append(f'#print axioms {n}')
tail.append('open Lean Meta Elab Command')
tail.append('def auditModules : Array Name := #[]')
tail.append('def auditPrefix : Name := `UniversalLaw.Proofs.Side24Analysis')
tail.append('def registeredProps : Array Name := #[' + ', '.join('`' + p for p in props) + ']')
# syntactic exactness check of the inventory theorem and the hypothesis-form helper
tail.append('''
run_cmd do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.referencePinDensityProduct
    | throwError "EXACT REFUSED: inventory theorem missing"
  let target : Expr := .const ``UniversalLaw.Spec.Side24.ReferencePinDensityProduct []
  unless ci.type == target do
    throwError "EXACT REFUSED: inventory type is not syntactically the Prop constant: {ci.type}"
  unless ci matches .thmInfo _ do throwError "EXACT REFUSED: inventory is not a theorem"
  logInfo m!"EXACT OK inventory type == Expr.const ReferencePinDensityProduct [] ; kind thm"
  let some hi := env.find? ``UniversalLaw.Proofs.Side24Analysis.Aux.rp_referencePinDensityProduct_of_referenceOddAndPinBlocks
    | throwError "EXACT REFUSED: hypothesis-form helper missing"
  match hi.type with
  | .forallE _ dom body _ =>
    unless dom == (.const ``UniversalLaw.Spec.Side24.ReferenceOddAndPinBlocks []) do
      throwError "EXACT REFUSED: hypothesis is not the upstream Prop: {dom}"
    unless body == target do
      throwError "EXACT REFUSED: hypothesis-form conclusion is not the Prop: {body}"
    unless !body.hasLooseBVars do throwError "EXACT REFUSED: dependent conclusion"
    logInfo m!"EXACT OK hypothesis-form helper : ReferenceOddAndPinBlocks → ReferencePinDensityProduct (single hypothesis, exact constants)"
  | t => throwError "EXACT REFUSED: hypothesis-form helper has unexpected shape: {t}"
''')
tail.append(check.AUDIT_PROGRAM)
OUT.write_text('import Lean\n' + body + '\n'.join(tail) + '\n', encoding='utf-8')
print('WROTE', OUT, 'theorems', len(fq), 'registered', len(props))
for n in fq: print('  ', n)
