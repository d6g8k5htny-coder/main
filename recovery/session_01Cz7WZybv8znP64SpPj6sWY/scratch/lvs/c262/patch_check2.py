"""Second-round repairs to check.py after the pre-push adversarial verification of the G1-G3 commit."""
from pathlib import Path
p = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011/check.py')
s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:100])
    s = s.replace(old, new)

# --- G2: bounded wording for COMMANDS and the additional refused command keywords
rep("""# COMMANDS is an explicit enumeration, not a Lean grammar: a declaration-producing command absent from it would
# pass an indented or same-line position uncounted (C262 gate review G2, Math- #428 comment 6112740935, found
# `alias` absent at head a898bb44; Batteries/Tactic/Alias.lean at the pin elaborates `alias n := c` into a new
# declaration). The enumeration therefore lists every declaration-producing command of Lean core, Batteries,
# Aesop and Mathlib known at the pin, and the README bounds the census claim to this list.
""",
"""# COMMANDS is an explicit list of refused keywords, not a Lean grammar and not a complete census: a
# declaration-producing command or form absent from it passes an indented or same-line position uncounted (C262
# gate review G2, Math- #428 comment 6112740935, found `alias` absent at head a898bb44; the pre-push adversarial
# verification of the repair then enumerated further commands at the pin, listed below, and the `where` clause of
# a declaration, which no keyword list can refuse). The scanner is therefore only a first filter. The declaration
# census that the receipt stands on is Census.lean (CENSUS below): Lean itself lists every non-internal constant
# the companion modules add to the environment, and `finish` requires that list to equal the inventory plus the
# explicitly named compiler-generated auxiliaries (`census_auxiliary` in SOURCE_FILES.json), nothing else.
""")
rep("""            # declaration-producing commands of Batteries, Aesop and Mathlib at the pin (C262 G2)
            'alias', 'irreducible_def', 'simproc', 'simproc_decl', 'dsimproc', 'dsimproc_decl', 'builtin_simproc',
            'builtin_dsimproc', 'notation3', 'unif_hint', 'recall', 'binder_predicate', 'add_decl_doc', 'library_note',
            'assert_not_exists', 'register_simp_attr', 'declare_aesop_rule_sets', 'compile_inductive', 'compile_def',
            'nonrec')
""",
"""            # declaration-producing commands of Lean core, Batteries, Aesop and Mathlib found at the pin (C262 G2 and
            # the verification of its repair); refused wherever they appear, column 0 included
            'alias', 'irreducible_def', 'simproc', 'simproc_decl', 'dsimproc', 'dsimproc_decl', 'builtin_simproc',
            'builtin_simproc_decl', 'builtin_dsimproc', 'builtin_dsimproc_decl', 'cbv_simproc', 'cbv_simproc_decl',
            'builtin_cbv_simproc', 'builtin_cbv_simproc_decl', 'notation3', 'unif_hint', 'recall', 'binder_predicate',
            'add_decl_doc', 'library_note', 'assert_not_exists', 'register_simp_attr', 'register_builtin_option',
            'register_linter_set', 'register_label_attr', 'declare_aesop_rule_sets', 'add_aesop_rules',
            'erase_aesop_rules', 'compile_inductive', 'compile_def', 'nonrec', 'mk_iff_of_inductive_prop', 'run_meta',
            'grind_propagator', 'builtin_grind_propagator', 'declare_config_elab', 'declare_config_elab_legacy',
            'declare_command_config_elab_legacy', 'gen_injective_theorems', 'declare_uint_theorems',
            'declare_int_theorems', 'declare_bitwise_uint_theorems', 'declare_bitwise_int_theorems', 'elab_stx_quot',
            'declare_eval_bin', 'declare_eval_bin_bitwise', 'declare_eval_bin_bool_pred', 'name_poly_vars',
            'name_power_vars', 'with_weak_namespace', 'meta', 'public')
""")
rep("""COMMAND_TOKEN = re.compile(r'(?<![A-Za-z0-9_.])(?:' + '|'.join(COMMANDS) + r')(?![A-Za-z0-9_])')
""",
"""# Token scan for keyword detection. ASCII identifiers (dotted components, primes, `!`, `?`) and numerals are whole
# tokens, as Lean lexes them: `0alias` is the numeral `0` followed by the keyword `alias` (refused), `foo2alias` is
# one identifier (accepted). A component after a dot is a name component, not a keyword. Any non-ASCII character
# separates tokens, a refusal in the safe direction when a keyword abuts a unicode letter.
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_'!?]*(?:\\.[A-Za-z_][A-Za-z0-9_'!?]*)*|[0-9]+")
COMMAND_SET = frozenset(COMMANDS)


def command_token(text):
    \"\"\"The first command keyword that appears as a whole token in comment-free `text`, or None.\"\"\"
    for found in TOKEN.finditer(text):
        if found.group() in COMMAND_SET:
            return found.group()
    return None


def admission_tokens(text):
    \"\"\"Every ADMISSION word that appears as a whole token or as a dotted component in comment-free `text`.\"\"\"
    present = set()
    for found in TOKEN.finditer(text):
        present.update(found.group().split('.'))
    return [token for token in ADMISSION if token in present]
""")
rep("""    code_text = '\\n'.join(lines)
    for token in ADMISSION:
        require(re.search(r'(?<![A-Za-z0-9_])' + token + r'(?![A-Za-z0-9_])', code_text) is None,
                'admission token in ' + module + ': ' + token)
""",
"""    code_text = '\\n'.join(lines)
    for token in admission_tokens(code_text):
        require(False, 'admission token in ' + module + ': ' + token)
""")
rep("""        found = COMMAND_TOKEN.search(rest)
        require(found is None, 'command token outside a column-0 declaration in ' + where + ': ' + (found.group() if found else ''))
""",
"""        found = command_token(rest)
        require(found is None, 'command token outside a column-0 declaration in ' + where + ': ' + (found or ''))
""")
# --- comment lexer: a closed block comment leaves one space so tokens on either side cannot splice
rep("""    Handles `--` to end of line and nested `/- … -/` blocks (`/--` and `/-!` included). String
    literals are not modelled, so scan_module refuses any `"` left in code; a `-/` outside a comment
    or an unterminated block is refused rather than guessed at.
    \"\"\"
""",
"""    Handles `--` to end of line and nested `/- … -/` blocks (`/--` and `/-!` included); a closed
    block comment leaves one space in the code, as Lean treats it as whitespace between tokens, so
    `theorem/- -/hidden` is scanned as `theorem hidden`, not `theoremhidden`. String literals are
    not modelled, so scan_module refuses any `"` left in code; a `-/` outside a comment or an
    unterminated block is refused rather than guessed at.
    \"\"\"
""")
rep("""                elif two == '-/':
                    depth -= 1
                    i += 2
                else:
                    i += 1
""",
"""                elif two == '-/':
                    depth -= 1
                    i += 2
                    if depth == 0:
                        code.append(' ')
                else:
                    i += 1
""")
# --- census: emitted Lean file, evidence members, parser
rep("""BUILDS = ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types', 'leanchecker')
""",
"""BUILDS = ('build-TorusTopology', 'build-ClosedSuperlevel', 'build-TorusFiltration', 'positive', 'axioms', 'types', 'census',
          'leanchecker')
""")
rep("""EMITTED_FILES = ('Audit.lean', 'Types.lean', 'Positive.lean') + tuple(c + '.lean' for c in CONTROLS)
""",
"""EMITTED_FILES = ('Audit.lean', 'Types.lean', 'Positive.lean', 'Census.lean') + tuple(c + '.lean' for c in CONTROLS)
# The elaborated declaration census (C262 G2 closure): Lean lists every constant that the companion modules add to
# the environment and that is not an internal detail (`Name.isInternalDetail`: `_`-prefixed, `eq_n`, `match_n`,
# `proof_n`, `omega_n`, numeric components). `finish` requires the list to equal the inventory plus the named
# compiler-generated auxiliaries, so a declaration the scanner missed (an unlisted command, a `where` clause) is
# refused by the receipt, whatever its spelling.
CENSUS = '''import TorusFiltration
open Lean in
#eval show Lean.Elab.Command.CommandElabM Unit from do
  let env ← Lean.getEnv
  let modules : Array Lean.Name := #[`TorusTopology, `ClosedSuperlevel, `TorusFiltration]
  let idxs := modules.filterMap env.getModuleIdx?
  if idxs.size != modules.size then throwError "census: a companion module is not imported"
  let names := env.constants.fold (init := (#[] : Array Lean.Name)) fun acc n _ =>
    match env.getModuleIdxFor? n with
    | some i => if idxs.contains i && !n.isInternalDetail then acc.push n else acc
    | none => acc
  for n in names.qsort (fun a b => a.toString < b.toString) do
    IO.println n
  IO.println s!"CENSUS_END {names.size}"
'''
""")
rep("""if len(EVIDENCE_FILES) != 41:   # an explicit check, not an `assert`: the gate must hold under -O too
    raise RuntimeError('EVIDENCE_FILES must enumerate exactly 41 members')
""",
"""if len(EVIDENCE_FILES) != 44:   # an explicit check, not an `assert`: the gate must hold under -O too
    raise RuntimeError('EVIDENCE_FILES must enumerate exactly 44 members')
""")
rep("""    require(set(meta) == {'schema', 'scientific_effect', 'alignment_status', 'namespace', 'root_module', 'modules',
                          'definitions', 'theorems', 'files', 'provenance'}, 'SOURCE_FILES schema keys')
""",
"""    require(set(meta) == {'schema', 'scientific_effect', 'alignment_status', 'namespace', 'root_module', 'modules',
                          'definitions', 'theorems', 'files', 'provenance', 'census_auxiliary'}, 'SOURCE_FILES schema keys')
    aux = meta['census_auxiliary']
    require(isinstance(aux, list) and all(isinstance(n, str) and re.fullmatch(NAME, n) and n.startswith(NAMESPACE + '.')
                                          and n.count('.') >= 2 for n in aux) and len(aux) == len(set(aux)),
            'census_auxiliary must list dotted auxiliary names under the companion namespace')
""")
rep("""    (out / 'Positive.lean').write_text(POSITIVE)
    for name, (statement, tactic) in FALSE_CONTROLS.items():
""",
"""    (out / 'Positive.lean').write_text(POSITIVE)
    (out / 'Census.lean').write_text(CENSUS)
    for name, (statement, tactic) in FALSE_CONTROLS.items():
""")
rep("""def leanchecker_mode(text):
""",
"""def census(text, names, auxiliary):
    \"\"\"Parse census.log: one constant per line, then `CENSUS_END <count>`; the set must equal names ∪ auxiliary.\"\"\"
    lines = text.strip('\\n').split('\\n') if text.strip() else []
    require(lines and re.fullmatch(r'CENSUS_END [0-9]+', lines[-1]), 'census terminator absent')
    listed = lines[:-1]
    require(all(re.fullmatch(NAME, n) for n in listed), 'unrecognized census output')
    require(int(lines[-1].split()[1]) == len(listed) and len(set(listed)) == len(listed), 'census count mismatch')
    expected = set(names) | set(auxiliary)
    extra, missing = sorted(set(listed) - expected), sorted(expected - set(listed))
    require(not extra and not missing, 'declaration census differs: extra %s, missing %s' % (extra, missing))
    return {'constants': len(listed), 'auxiliary': sorted(auxiliary)}


def leanchecker_mode(text):
""")
rep("""    require(out.is_dir() and not out.is_symlink(), 'evidence directory absent: ' + str(out))
""",
"""    require(out.is_dir() and not out.is_symlink(), 'evidence directory absent, not a directory, or a symlink: ' + str(out))
""")
rep("""    types((out / 'types.log').read_text(), theorem_targets(meta))
""",
"""    types((out / 'types.log').read_text(), theorem_targets(meta))
    counted = census((out / 'census.log').read_text(), targets(meta), meta['census_auxiliary'])
""")
rep("""        'leanchecker': {'command': (out / 'leanchecker.cmd').read_text().strip(), 'mode': mode},
""",
"""        'leanchecker': {'command': (out / 'leanchecker.cmd').read_text().strip(), 'mode': mode},
        'census': counted,
""")
rep("""    require(not receipt_path.exists() and not receipt_path.is_symlink(), 'receipt.json appeared during finalization')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
""",
"""    # Exclusive creation (O_EXCL, O_NOFOLLOW): a receipt that appeared meanwhile, or a symlink planted at the path,
    # fails the open instead of being replaced or followed.
    descriptor = os.open(receipt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o644)
    with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
        handle.write(json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
""")
# --- G1: every CLI invocation reports the mode it runs in (observed by the test fixture for every child)
rep("""    args = parser.parse_args()
    try:
        if args.command == 'source':
""",
"""    args = parser.parse_args()
    # The interpreter mode this invocation actually runs in, for the test fixture (which requires it to equal the
    # outer mode for every child) and for the hosted job log (C262 G1).
    print('TORUS_FILTRATION_PYTHON: optimize=%d' % sys.flags.optimize, file=sys.stderr)
    try:
        if args.command == 'source':
""")
p.write_text(s, encoding='utf-8')
print('check.py second-round patch applied')
