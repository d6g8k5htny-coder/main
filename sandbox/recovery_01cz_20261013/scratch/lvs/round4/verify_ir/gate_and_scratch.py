import sys, json, re, unicodedata
from pathlib import Path
PKT = Path(sys.argv[1]); STMT = Path(sys.argv[2]); FILE = Path(sys.argv[3]); OUT = Path(sys.argv[4])
sys.path.insert(0, str(PKT))
import check
text = FILE.read_text(encoding='utf-8')

# --- 1. the real gate: load_register + scan_module ---
reg = check.load_register(STMT)
registered = reg[0] if isinstance(reg, tuple) else reg
if isinstance(registered, dict) and 'registered' in registered:
    registered = registered['registered']
print('REGISTER type', type(registered).__name__, 'len', len(registered))
names = list(registered.keys()) if isinstance(registered, dict) else list(registered)
print('REGISTER sample', names[:3])
TARGET = 'UniversalLaw.Spec.Side24.IntegrandRatioBounds'
print('TARGET_REGISTERED', TARGET in names)
if isinstance(registered, dict):
    print('TARGET_ENTRY', json.dumps(registered[TARGET], ensure_ascii=False)[:600])
try:
    info = check.scan_module(text, 'Side24Analysis', registered)
    print('SCAN imports', info['imports']); print('SCAN namespaces', info['namespaces'])
    print('SCAN aux_theorems', info['aux_theorems'], 'auxiliary_definitions', info['auxiliary_definitions'], 'set_options', info['set_options'])
    for t in info['theorems']:
        print('SCAN THEOREM', json.dumps(t, ensure_ascii=False))
    print('SCAN_OK')
except Exception as e:
    print('SCAN_FAILED', repr(e))

# --- 2. my own lexical checks on comment-stripped code ---
code, docs = check.lex(text)
(OUT / 'stripped_code.lean').write_text(code, encoding='utf-8')
words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
task_tokens = ['sorry','admit','axiom','native_decide','unsafe','opaque','partial','implemented_by','extern','macro','syntax','elab','notation','initialize','export','alias','include','_root_','set_option','sorryAx','native','ofReduceBool','trustCompiler','skipKernelTC','byAsSorry','macro_rules','elab_rules','infix','infixl','infixr','prefix','postfix','run_cmd','run_tac','run_elab','builtin_initialize','declare_syntax_cat','irreducible_def']
print('MY_REFUSED_HITS', sorted(set(task_tokens) & words))
print('MY_HASH_CMDS', [ (i+1, l) for i,l in enumerate(code.split('\n')) if re.search(r'#[A-Za-z_]', l)])
print('MY_INDENTED_DECL_KEYWORDS', [ (i+1, l) for i,l in enumerate(code.split('\n')) if re.match(r'\s+(theorem|lemma|def|abbrev|instance|structure|inductive|example|namespace|section|end|variable|universe|private|protected|noncomputable)\b', l)])
print('MY_FUN_ANGLE_OR_MATCH', [ (i+1, l) for i,l in enumerate(code.split('\n')) if re.search(r'fun\s*(⟨|\|)|\bmatch\b|\bfun\b.*=>', l)])
print('MY_OPEN_IN', [ (i+1, l) for i,l in enumerate(code.split('\n')) if re.search(r'\bopen\b.*\bin\b', l)])
print('MY_COL0_TOKENS', sorted(set(l.split()[0] for l in code.split('\n') if l and not l[0].isspace())))
print('MY_PROBE_MENTION', [ (i+1,l) for i,l in enumerate(text.split('\n')) if re.search(r'probe', l, re.I)])
# non-ASCII inventory, code vs whole file
def inv(s):
    d = {}
    for ch in s:
        if ord(ch) > 127:
            d[ch] = d.get(ch, 0) + 1
    return {f'{ch} U+{ord(ch):04X} {unicodedata.name(ch, "?")}': n for ch, n in sorted(d.items())}
print('NONASCII_CODE', json.dumps(inv(code), ensure_ascii=False))
print('NONASCII_FILE', json.dumps(inv(text), ensure_ascii=False))
# identifiers in code containing non-ASCII letters (homoglyph hunt)
print('NONASCII_IDENTS', sorted(set(w for w in re.findall(r"[^\W\d][\w'!?]*(?:\.[^\W\d][\w'!?]*)*", code) if any(ord(c) > 127 and unicodedata.category(c)[0] == 'L' and c not in 'ℝℕε' for c in w))))

# --- 3. the scratch Lean file: deliverable verbatim + tail ---
thms = ['UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds'] + [
  'UniversalLaw.Proofs.Side24Analysis.Aux.' + h for h in ['ir_epsilon_pos','ir_epsilon_lt','ir_one_sub_epsilon_pos','ir_one_le_one_add_epsilon','ir_rpow_le_pow_five','ir_bernoulli','ir_lower_rational','ir_upper_rational','ir_lower_bound','ir_upper_bound','ir_ratio_bounds','ir_exponentA_two','ir_exponentB_two','ir_exponentA_three','ir_exponentB_three']]
# every theorem name declared in the file, from the stripped code, to make sure the list above is complete
declared = re.findall(r'^theorem\s+(\S+)', code, re.M)
print('DECLARED_THEOREMS', len(declared), declared)
props = '#[' + ', '.join('`' + r for r in names) + ']'
tail = ['', '-- ===== VERIFIER SCRATCH TAIL (not part of the deliverable) =====']
tail.append('#check @' + thms[0])
tail.append('#print ' + TARGET)
tail.append('#print ' + thms[0])
tail += ['#print axioms ' + t for t in thms]
tail.append('''
open Lean Meta Elab Command in
run_cmd do
  let env ← getEnv
  let some ci := env.find? ``UniversalLaw.Proofs.Side24Analysis.integrandRatioBounds | throwError "missing theorem"
  let target := mkConst ``UniversalLaw.Spec.Side24.IntegrandRatioBounds
  logInfo m!"TYPE_PP {ci.type}"
  logInfo m!"TYPE_EXACT_EXPR_EQ {ci.type == target}"
  logInfo m!"TYPE_IS_THEOREM {match ci with | .thmInfo _ => true | _ => false}"
  let some pi := env.find? ``UniversalLaw.Spec.Side24.IntegrandRatioBounds | throwError "missing Prop"
  logInfo m!"PROP_VALUE {pi.value?.getD (mkConst `none)}"
  -- every constant this file adds (stage-2 map), with kind
  let mut rows : Array String := #[]
  for (n, c) in env.constants.map₂.toList do
    let kind := match c with
      | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "AXIOM" | .opaqueInfo _ => "OPAQUE"
      | .quotInfo _ => "QUOT" | .inductInfo _ => "ind" | .ctorInfo _ => "ctor" | .recInfo _ => "rec"
    let isP ← liftTermElabM (isProp c.type)
    rows := rows.push s!"STAGE2 {kind} prop={isP} unsafe={c.isUnsafe} {n}"
  for r in rows.qsort (· < ·) do logInfo r
  logInfo m!"STAGE2_TOTAL {rows.size}"
''')
# the gate's own audit program, prefix widened to `UniversalLaw so the generated eq lemmas are classified too
tail.append('open Lean Meta Elab Command')
tail.append('def auditModules : Array Name := #[]')
tail.append('def auditPrefix : Name := `UniversalLaw')
tail.append('def registeredProps : Array Name := ' + props)
tail.append(check.AUDIT_PROGRAM)
scratch = 'import Lean\n' + text + '\n'.join(tail) + '\n'
(OUT / 'scratch.lean').write_text(scratch, encoding='utf-8')
print('SCRATCH_WRITTEN', len(scratch), 'bytes; deliverable bytes', len(text))
