# C254 compile-fix brief (two Lean modules proposed by the OpenAI/Codex lane; this lane supplies the pinned toolchain)

Scientific effect: NONE. You are making proposed Lean source elaborate at the pinned toolchain without changing what
it states. You decide no mathematical status and review nothing scientific. Organizational-independence credit: 0.

## Where things are (S=<SCRATCH>)

- Worktree of Math- at origin/main 2bbce4b6 with the C254 overlay applied: $S/repos/math_c254 (branch
  claude/c254-torus-filtration-20261011). The Lean package is $S/repos/math_c254/formal (lakefile.toml: lib
  ResearchFormalCoreR1; Mathlib d13f23b7 via .lake/packages symlink; Lean v4.34.1).
- The two files to fix, IN PLACE: $S/repos/math_c254/formal/ResearchFormalCoreR1/TorusTopology.lean and
  $S/repos/math_c254/formal/ResearchFormalCoreR1/ClosedSuperlevel.lean. Byte-identical originals (never edit):
  $S/lvs/c254/overlay/ResearchFormalCoreR1/{TorusTopology,ClosedSuperlevel}.lean. First local error logs:
  $S/lvs/c254/{TorusTopology,ClosedSuperlevel}.out.
- Pinned Mathlib source for grepping: $S/mathf/.lake/packages/mathlib/Mathlib (e.g. LinearAlgebra/Basis/Defs.lean:
  `namespace Module` line 76, `structure Basis` line 90; LinearAlgebra/Basis/SMul.lean: `namespace Module.Basis`,
  `unitsSMul` 103, `unitsSMul_apply` 107; Topology/Connected/PathConnected.lean: `ZerothHomotopy` 117 is
  `Quotient (pathSetoid X)`; Topology/Covering/{Basic,Quotient,Deck,AddCircle}.lean; Algebra/Module/ZLattice/*).
- Compile (the only command you need; at most ONE `lake env lean` running at a time; expect 15-120 s):
      export PATH=$HOME/.elan/bin:$PATH
      cd $S/repos/math_c254/formal && lake env lean -DwarningAsError=false ResearchFormalCoreR1/<File>.lean
  Do NOT run `lake build`, `lake update`, `lake exe cache`; do not touch .lake except through that command; do not
  edit any other file of the worktree (manifest.json, tests, README, the other module, the root import).

## Hard constraints (the package gate, formal/gate.py, enforces most of them)

1. Every `theorem`/`lemma` name must stay exactly as in the original, in the same order, and no `theorem`/`lemma`
   may be added or removed: the manifest's `targets` list is compared to the source declarations. Helper facts you
   need must be inlined into proofs (`have`), or be `def`/`abbrev`/`instance` (not theorems). If you truly cannot
   avoid a new theorem, STOP and report instead.
2. Every theorem STATEMENT must keep its meaning. Allowed: renaming an identifier to its name at this Mathlib
   revision (`Basis` -> `Module.Basis`), adding `open`/`open scoped`, adjusting implicit/instance arguments so the
   same proposition elaborates, fixing a mis-typed binder so the intended proposition is stated. Not allowed:
   dropping a conjunct, adding a hypothesis, changing a quantifier, weakening a conclusion, changing a definition's
   mathematical content (definitions may change only where needed to typecheck, and then say so).
3. No `sorry`, `admit`, `axiom`, `native_decide`, `unsafe`, `opaque`, `partial`, `implemented_by`, `extern`,
   `set_option ... skipKernelTC`, `autoImplicit`; axioms of every theorem within {propext, Classical.choice,
   Quot.sound} (check with `#print axioms` in a scratch copy, never in the deliverable).
4. Imports: `import Mathlib` is acceptable; replacing it by precise imports is welcome only if the file still
   elaborates and you list the imports used. Keep the module docstring; add one line to it recording that the
   compile fixes were made at the pin (what, by name), not altering any statement.
5. Keep the authors' structure and comments; make the minimal diff that elaborates. Record every change.

## Deliverable (StructuredOutput)

elaborates (bool), command (string), exit_code, errors_remaining (list of "line:col message"), changes
([{location, original, fixed, reason}]), statements_unchanged (bool, with your justification), axioms_checked
(list of "name: [axioms]" for every theorem of the file), imports_final (list), notes.
