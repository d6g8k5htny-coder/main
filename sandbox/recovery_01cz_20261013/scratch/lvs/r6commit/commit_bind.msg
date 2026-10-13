Proof packet round 6: bind the three theorems to their first hosted receipt (run 38277458954)

One forward commit on 9e305ea5 (draft PR #441). It changes REGISTER.json and the four
packet documents only: no .lean, lakefile, lake-manifest, lean-toolchain, check.py,
test or workflow byte changes.

The first hosted receipt of the round-6 theorems is workflow lean-specification-proofs.yml
run 38277458954, attempt 1, job 114906310832 (proof-evidence, success), event pull_request
on PR #441 head 9e305ea5141a1d1503560efd0e3deaf25441749e. Its checked_commit is the
synthetic merge 1376175ce5536a6ac74cb03706d18e65fdb695fa, parents a0323b94 (Math- main at
the merge of PR #440) and 9e305ea5. Unlike rounds 4 and 5, the tested merge's tree is not
the head's tree. It differs only outside both packets: 13 files under
companions/torus_filtration_20261011/ and .github/workflows/torus-filtration.yml. Both
packets are byte-identical between head and merge, and every binding sentence says so.
The receipt records formalization_status kernel-checked, leanchecker PASS,
theorem_count 78, 77 covered Props, controls as intended and worktree CLEAN. The
artifact is lean-specification-proofs-38277458954-1 (id 11759872846, 47,041 bytes),
GitHub digest sha256 1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79.

REGISTER.json (blob 84d0dd07 -> 4ba43ed9) changes 23 string leaves across 11 rows, in
round 5's shapes:
- In side24-coefficient, the pending tier becomes the receipted tier: run, PR head,
  tested merge with base, and the tree clause.
- All ten proof-packet kernel_evidence rows gain one corroboration clause: all 78
  theorems of PR head 9e305ea5 receipted by that run, with the tested-merge caveat.
- The round-6 triage items and the README-reasons sentence drop "no hosted receipt yet".
- In next_step, round 5's "every theorem of this row" wording is narrowed to "each of
  those 22 theorems" and "every proof-packet theorem of this row". The row also holds 27
  main-package theorems.
Every statements array, every formal_progress (138 none, 33 specified) and every triage
verdict is unchanged.

The statement packet README/SCOPE are re-rendered with no dagger left. The proof packet
README/SCOPE bind the run.

Reads, both nonauthor, same account, organizational-independence credit 0:
- Source/type review 5493661142: bounded PASS on 9e305ea5. This commit changes none of
  the consumed proof, Prop, dependency or pin identities it names.
- Hosted original-byte audit 6128623199: bounded PASS. It records that the statement
  workflow file is lean-specifications.yml, and that the hosted source-only suites skip
  the 3 LeanHelpers tests in each mode.

Local checks (records of computations on this lane's worktree, not receipts):
- the generators update_register14 / render_docs14 / edit_proof_docs14 (outside the
  tree): fail-closed, reproduce-before-render, exact-string edits with asserted counts;
- both packets' check.py --allow-dirty: SOURCE_CHECK_PASS;
- test_check.py, normal and -O modes: 97 and 70 tests OK.
Two adversarial verification rounds found missing tree caveats and wording collisions.
All of them are folded in.

Not established: anything about an informal claim. A receipt is a record of a
computation about the registered Props for the formal objects. Alignment remains
PENDING_INDEPENDENT_REVIEW. Scientific effect NONE.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
