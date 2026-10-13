Companion torus_filtration_20261011: apply the C262 engineering-gate AMEND (G1 child mode observed, G2 refused keywords and elaborated census, G3 immutable receipt and exact evidence membership)

One forward commit after the posted verdict (Math- #428 comments 6112740935 gate
review, 6112743879 disposition AMEND_ENGINEERING_GATE on head a898bb44).

G1 test_torus_filtration.py: check.py children are spawned with the outer
interpreter's optimization level (-O under -O) and without PYTHONOPTIMIZE,
ambient or injected. Every check.py invocation prints the mode it runs in
(TORUS_FILTRATION_PYTHON: optimize=N) to stderr; the fixture requires exactly
one such line per child and refuses a child whose mode differs from the outer
mode, on every call; a control rebinds the spawner to drop -O and shows the
guard refusing. The suite prints its own mode into tests.log and
tests-optimized.log; replay.sh unsets PYTHONOPTIMIZE. The -O transcripts of
hosted runs 38165155316 and 38165763430 keep their narrower attribution in the
README (CLI children ran unoptimized then).

G2 check.py, two layers. The keyword scanner now matches whole tokens, with
identifiers and numerals lexed as Lean does (Lean/Parser/Basic.lean numeral
grammar: decimal with `_`, fraction, exponent, 0x/0b/0o), and the comment
lexer leaves a space where a block comment closes, so the splices
`theorem/- -/hidden`, `0alias`, `0x1Ftheorem`, `1e5axiom`, `1_0def`,
`0b1axiom` are refused; COMMANDS lists the declaration-producing commands of
Lean core, Batteries, Aesop and Mathlib found at the pin (alias,
irreducible_def, the simproc families, notation3, unif_hint, recall,
binder_predicate, add_decl_doc, library_note, assert_not_exists,
register_simp_attr, register_builtin_option, register_linter_set,
register_label_attr, declare_aesop_rule_sets, add/erase_aesop_rules,
compile_inductive, compile_def, nonrec, mk_iff_of_inductive_prop, run_meta,
grind_propagator(s), declare_config_elab*, gen_injective_theorems,
declare_*_theorems, elab_stx_quot, declare_eval_bin*, name_poly_vars,
name_power_vars, with_weak_namespace, meta, public, aux_def, lrat_proof,
reprove, declare_aesop_exception, declare_uint/sint_simprocs, seal, unseal,
unset_option, suppress_compilation, grind_pattern, grind_annotated,
extend_docs, init_quot, initialize_simps_projections). The list is a first
filter, not the guarantee: the guarantee is Census.lean, an elaborated
declaration census that Lean prints for the three modules, each constant
tagged internal/user (Name.isInternalDetail) and range/generated (whether Lean
recorded a declaration range). finish requires the user-named ranged constants
to equal the inventory exactly, refuses any ranged constant with an
internal-style name (a user-written `proof_1`), and requires the user-named
generated constants to equal SOURCE_FILES.json census_auxiliary exactly (today
TorusFiltration.torus_periodBasis.congr_simp). A `where` clause, which no
keyword list refuses, is refused by that census.

G3 check.py finish: refuses a directory already holding receipt.json (a receipt
is written once, never replaced), requires exactly the 44 evidence members
(EVIDENCE_FILES) as regular files and nothing else, hashes exactly those 44,
and creates the receipt through write_new_file (O_EXCL|O_NOFOLLOW); controls
for a repeat finish, an existing receipt beside a newly failing input, every
member removed in turn, an extra file, a stray directory, a directory as a
required member, a symlinked member, a symlinked evidence directory, a
symlinked receipt path and write_new_file itself. The module-level 44-member
check is an explicit raise, not an assert.

Before the push, this lane's own adversarial verification of the first form of
this commit (same account, credit 0) refuted G1 and G2 as first repaired; the
repairs above are the folded-in result, so the AMEND stays one forward commit.
Rounds of that verification are recorded in the README Amendments section.

Protocol tests: 28 in both Python modes. SOURCE_FILES.json regenerated (schema
key census_auxiliary added). replay.sh changed by `unset PYTHONOPTIMIZE` and the
census invocation; no Lean byte, declaration, control statement or workflow line
changed. Scientific effect NONE.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
