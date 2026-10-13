# verifier_docs2 — adversarial check of the round-4 doc diffs (2026-10-11)

Scope: worktree diffs of Math- specifications/lean_statements_20261010/{README,SCOPE}.md,
specifications/lean_statement_proofs_20261011/{README,SCOPE}.md (clone HEAD 80e80243) and
lvs_main/docs/LEAN_VERIFICATION_SPECIFICATION.md (lvs_main HEAD dc97752f), against REGISTER.json
(worktree), the proof-packet inventory and the Lean modules at HEAD. Same-account verifier: credit 0.
No label moves anywhere; no file edited by this verifier; no git state changed.

## Refuting finding (one)
F1 main doc §1 bullet 4: "Each such theorem is L5 evidence for that Prop" now ranges over all 73
theorems including the 13 round-4 theorems that the same sentence group says have no hosted
receipt; the doc's own ladder (lines 29-31) makes "a target with a trusted kernel-checked receipt"
the L5 unit. By the doc's own standard the 13 are presented with receipted-grade evidence.
Origin: render_spec7.py bullet-4 edit keeps the sentence verbatim while changing its antecedent;
its negative control only forbids "kernel-checked by" near the new modules.

## Non-blocking
F2 main doc §5.1 triage paragraph enumerates 16 Dirac + Λ:=0 + two-point + 3 (p≡0/1) + 2 Side24 = 23
of 24 BLOCKED_INTERFACE Props; RN.PalmRouteSecondFactorialTheta (Dirac refutation in its `route`,
not its `finding`) is the uncounted one (render_spec7.py counts 'Dirac' in `finding` only).
F3 register OA-C6-RARE-CLUSTER-20260929-v1 next_step (rendered into main §4 and stmt SCOPE):
"the 1 theorem(s) of this row" — template plural.

## Verified (details in the logs here)
counts 73/72/13/69, 352/155/83, 197, 138/33, per-module 17,13,3,4,5,8,5,9,3,5,1; triage 10/5/7/24/28;
run-id bindings (merge sha + PR head in same paragraph/cell) in all five docs; tables well-formed
(7+8+2+1+13); no proved Prop listed as without-a-theorem; vocabulary clean; main diff = 12 hunks, all
declared edits; Lean identifiers and constants quoted by the docs present in the three modules and in
the Prop definitions; commit 80e80243 message records the local build/axiom audit the README cites.

## Check / test outputs
stmt_check.log, proof_check.log, proof_tests_mode1.log, proof_tests_mode2.log (all EXIT=0; 70 tests OK x2).
