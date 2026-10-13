"""C262 G2 + G3 edits to check.py (exact-string replacements; every anchor must match once)."""
from pathlib import Path
p = Path('<SCRATCH>/repos/math_c254/companions/torus_filtration_20261011/check.py')
s = p.read_text(encoding='utf-8')
def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)

# G2: bounded enumeration of refused command keywords, now including the declaration-producing commands
# available at the pin that the C262 gate review found absent (`alias`, Batteries/Tactic/Alias.lean).
rep("""# Module scanner (scan_module): a whitelist of what may start a column-0 code line. Lean itself accepts a
# command anywhere (indented, after another command on the same line, behind `open … in`), so every other
# command keyword is refused wherever it appears outside comments; the scanner refuses what it cannot place.
COMMANDS = ('theorem', 'lemma', 'def', 'abbrev', 'instance', 'axiom', 'axioms', 'example', 'structure', 'inductive',
            'class', 'opaque', 'macro', 'macro_rules', 'syntax', 'elab', 'elab_rules', 'notation', 'infix', 'infixl',
            'infixr', 'prefix', 'postfix', 'set_option', 'attribute', 'namespace', 'section', 'end', 'universe',
            'variable', 'import', 'private', 'protected', 'noncomputable', 'unsafe', 'partial', 'scoped', 'local',
            'mutual', 'deriving', 'initialize', 'builtin_initialize', 'declare_syntax_cat', 'register_option',
            'open', 'export', 'include', 'omit', 'run_cmd', 'run_tac', 'run_elab', 'in')
""",
"""# Module scanner (scan_module): a whitelist of what may start a column-0 code line. Lean itself accepts a
# command anywhere (indented, after another command on the same line, behind `open … in`), so every command
# keyword in COMMANDS is refused wherever it appears outside comments; the scanner refuses what it cannot place.
# COMMANDS is an explicit enumeration, not a Lean grammar: a declaration-producing command absent from it would
# pass an indented or same-line position uncounted (C262 gate review G2, Math- #428 comment 6112740935, found
# `alias` absent at head a898bb44; Batteries/Tactic/Alias.lean at the pin elaborates `alias n := c` into a new
# declaration). The enumeration therefore lists every declaration-producing command of Lean core, Batteries,
# Aesop and Mathlib known at the pin, and the README bounds the census claim to this list.
COMMANDS = ('theorem', 'lemma', 'def', 'abbrev', 'instance', 'axiom', 'axioms', 'example', 'structure', 'inductive',
            'class', 'opaque', 'macro', 'macro_rules', 'syntax', 'elab', 'elab_rules', 'notation', 'infix', 'infixl',
            'infixr', 'prefix', 'postfix', 'set_option', 'attribute', 'namespace', 'section', 'end', 'universe',
            'variable', 'import', 'private', 'protected', 'noncomputable', 'unsafe', 'partial', 'scoped', 'local',
            'mutual', 'deriving', 'initialize', 'builtin_initialize', 'declare_syntax_cat', 'register_option',
            'open', 'export', 'include', 'omit', 'run_cmd', 'run_tac', 'run_elab', 'in',
            # declaration-producing commands of Batteries, Aesop and Mathlib at the pin (C262 G2)
            'alias', 'irreducible_def', 'simproc', 'simproc_decl', 'dsimproc', 'dsimproc_decl', 'builtin_simproc',
            'builtin_dsimproc', 'notation3', 'unif_hint', 'recall', 'binder_predicate', 'add_decl_doc', 'library_note',
            'assert_not_exists', 'register_simp_attr', 'declare_aesop_rule_sets', 'compile_inductive', 'compile_def',
            'nonrec')
""")

# G3: the exact evidence membership finish() requires before it writes a receipt.
rep("""CONTROLS = tuple(FALSE_CONTROLS) + tuple(INJECTIONS)
""",
"""CONTROLS = tuple(FALSE_CONTROLS) + tuple(INJECTIONS)
# The exact membership of the evidence directory when `finish` runs (C262 gate review G3): the nine files emit()
# writes, a .log/.status pair per Lean process, the two protocol-test transcripts and the four environment records.
# finish() refuses a directory with any member missing or unexpected, any member that is not a regular file, and
# any directory already holding receipt.json: a receipt is written once and never replaced.
EMITTED_FILES = ('Audit.lean', 'Types.lean', 'Positive.lean') + tuple(c + '.lean' for c in CONTROLS)
EVIDENCE_FILES = frozenset(EMITTED_FILES) | {label + ext for label in BUILDS for ext in ('.log', '.status')} \\
    | {control + ext for control in CONTROLS for ext in ('.log', '.status')} \\
    | {'leanchecker.cmd', 'packages.log', 'toolchain.log', 'source.json', 'tests.log', 'tests-optimized.log'}
assert len(EVIDENCE_FILES) == 41
""")

rep("""def finish(out):
    initial = strict((out / 'source.json').read_bytes())
""",
"""def finish(out):
    require(out.is_dir() and not out.is_symlink(), 'evidence directory absent: ' + str(out))
    receipt_path = out / 'receipt.json'
    require(not receipt_path.exists() and not receipt_path.is_symlink(),
            'receipt.json already present: finish never replaces a receipt; a new attempt needs a fresh evidence directory')
    present = {p.name for p in out.iterdir()}
    missing, unexpected = sorted(EVIDENCE_FILES - present), sorted(present - EVIDENCE_FILES)
    require(not missing and not unexpected, 'evidence membership differs: missing %s, unexpected %s' % (missing, unexpected))
    for name in sorted(EVIDENCE_FILES):
        member = out / name
        require(member.is_file() and not member.is_symlink(), 'evidence member is not a regular file: ' + name)
    initial = strict((out / 'source.json').read_bytes())
""")

rep("""        'log_sha256': {p.name: sha(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
""",
"""        'log_sha256': {name: sha((out / name).read_bytes()) for name in sorted(EVIDENCE_FILES)}}
    require(not receipt_path.exists() and not receipt_path.is_symlink(), 'receipt.json appeared during finalization')
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\\n')
    return receipt
""")
p.write_text(s, encoding='utf-8')
print('check.py patched')
