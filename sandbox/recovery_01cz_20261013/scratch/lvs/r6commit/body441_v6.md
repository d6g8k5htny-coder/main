**Scientific effect: NONE.** This PR moves no claim-level label: every claim stays `specified` or `none`. It changes no `lemma_closed`, prize, premise, LANDING_CLAIMS disposition, PROOF_INDEX entry or theorem status. A theorem here is evidence about its registered Prop for the formal objects only, never about an informal claim. Alignment of every statement with its anchor stays `PENDING_INDEPENDENT_REVIEW`. Author: Anthropic / Claude, Claude Code cloud session `session_01Cz7WZybv8znP64SpPj6sWY`. This is the same project account as every lane, so organizational-independence credit is 0; there is no human review. Draft; this session does not merge.

## Current disposition

- **Head:** `4a3fbadf56c807f7f538fb42ab304ab789d77878`, the receipt-binding successor: one forward commit on the reviewed head `9e305ea5141a1d1503560efd0e3deaf25441749e`, which is one commit on Math- main `4afcca6a`. It changes `REGISTER.json` and the four packet documents only. No Lean, lake, toolchain, checker or workflow byte changes. The unchanged-source attestation and the successor checks are in the handoff [6129598488](https://github.com/d6g8k5htny-coder/Math-/pull/441#issuecomment-6129598488). Hosted checks on `4a3fbadf`: all five check runs passed. The proof run 38285814238 (job 114933796783) received `kernel-checked`, 78 theorems; its actual tested merge is `0017d0ceeabed4dddcca69a9fb44c3f29a59280c` with base `a0323b94`. The statement run is 38285814239 and the aggregate run 38285815086. Main is at `a0323b94`, after #428 and #440, which changed only the torus_filtration companion and its workflow. Both packets are byte-identical between each head and its tested merge.
- **Reads (both complete and released, on head `9e305ea5`):** the nonauthor source/type review by OpenAI dot / Codex `/root/review_side24_bound_proofs`, **bounded source/type PASS**, no blocking finding ([review 5493661142](https://github.com/d6g8k5htny-coder/Math-/pull/441#pullrequestreview-5493661142)). It names a carry-forward condition for the receipt-binding commit: an explicit unchanged-source attestation of the consumed proof, Prop, dependency and pin identities. The second read is the hosted execution / original-byte audit by OpenAI dot / Codex `/root/verify_side24_hosted_evidence`, **bounded PASS** ([6128623199](https://github.com/d6g8k5htny-coder/Math-/pull/441#issuecomment-6128623199)). It acquired the four original artifacts once each and checked their bytes, members and CRCs. It confirmed the proof receipt `receipt.json` (sha256 `49c134b2…`), and that every checkout and receipt binds tested merge `1376175c…`. Both reads were done on the same project account, so they earn organizational-independence credit 0. Neither is an acceptance, an alignment verdict or an integration decision.
- **Unresolved finding IDs:** none.
- **Hosted receipt (first, for the three new theorems):** `lean-specification-proofs.yml` run [38277458954](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38277458954) (attempt 1, job proof-evidence 114906310832, event pull_request) on PR head `9e305ea5141a1d1503560efd0e3deaf25441749e`. Its actual tested checkout, the receipt's `checked_commit`, is the synthetic merge `1376175ce5536a6ac74cb03706d18e65fdb695fa` of that head into base `a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc`; both packets are byte-identical between that merge and the head. The receipt records `formalization_status: kernel-checked`, `leanchecker` PASS, `theorem_count` 78, 77 covered Props, every negative control refused and the honest controls accepted, and worktree CLEAN. Artifact `lean-specification-proofs-38277458954-1` (id 11759872846, 47,041 bytes, 29 files), GitHub digest sha256 `1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79`. Lean 4.34.1 (`5045d005`). These facts were read from the job log; the artifact was not downloaded here.
  - Companion runs on the same head and tested merge: the statement packet's `lean-specifications.yml` run 38277458488 (the file name was given wrongly as `lean-specification-statements.yml` until 16:47Z; corrected per [6128623199](https://github.com/d6g8k5htny-coder/Math-/pull/441#issuecomment-6128623199)) (job statement-evidence 114906311074; receipt `specified`, 150 registered declarations, register sha256 `67f7fb1e…`; artifact `lean-specifications-38277458488-1`, digest `6d8cf532…`) and the aggregate run 38277459272 (downstream-replay, formal / formal-evidence, math-downstream-gates: success).
  - A receipt is a record of a computation about the registered Props for the formal objects, not a review or an acceptance. The register and docs bind this receipt at the successor head `4a3fbadf`, the receipt-binding forward commit (handoff 6129598488). At `9e305ea5` they still carried the pending tier. Until 18:31Z this sentence said so in the present tense.
- **Local validation** (records of computations on this lane's worktree, not receipts): 
  - both packets' `check.py --allow-dirty`: SOURCE_CHECK_PASS;
  - both `test_check.py` suites in normal and `-O` modes: 97 and 70 tests OK;
  - `lake env lean -DwarningAsError=true` on the module: exit 0;
  - `#check` of each theorem against its registered Prop;
  - `#print axioms` within `propext`, `Classical.choice`, `Quot.sound`;
  - `lake build` of the proof packet from the committed tree: success;
  - `python3 -B -S specifications/lean_statement_proofs_20261011/check.py --execute` on `9e305ea`, 2026-10-12T16:21:04Z to 16:26:07Z: exit 0, receipt `formalization_status: kernel-checked`, `leanchecker: PASS`, `theorem_count: 78`, 77 covered Props, every control classified as intended (negative controls refused, positive controls accepted), `worktree: CLEAN`, `passed: true`; local receipt sha256 `10f76c77…` (`run_id` local);
  - `git diff --exit-code` afterwards: clean.
- **Next action:**
  1. ~~Bind the hosted runs on this head~~ (done above, body only).
  2. ~~The nonauthor source/type review~~ (PASS, see Reads).
  3. ~~A receipt-binding commit after the first hosted receipt~~ (done: `4a3fbadf`, handoff 6129598488).
  4. Integration by a nonauthor lane, at the owner's decision.

## What this adds

One proof module. Each of its three inventory theorems has exactly the type of one registered Prop of `specifications/lean_statements_20261010/Specifications/Side24.lean`. These are the three Props that the round-4 triage rated BLOCKED_MISSING_MATHLIB (weeks). Its verdict stays in the register verbatim as the historical record.

**`SpecificationProofs/Side24Images.lean`** (862 lines, 49,482 bytes, sha256 `ca643c39144400074658c39bbdd0f04ac47d29bd0c59fc2327d5a90d4f5ae392`; 3 inventory theorems, 38 `Aux` helpers, one auxiliary definition):

- `imageBound : UniversalLaw.Spec.Side24.ImageBound` (display (2)): `|D^q K_24(0) u − D^q K_∞(0) u| < imageConstE` for `d ∈ {2,3}`, `q ≤ 6` and every unit tuple.
  - Mathlib's `iteratedFDeriv_tsum` needs bounds uniform over the whole space, which lattice translates lack. The module therefore proves a local version on an open preconnected set, the helper `Aux.ib_iteratedFDeriv_tsum_of_isOpen`. It uses the global lemma's induction, from the first-order `hasFDerivAt_tsum_of_isPreconnected`.
  - On the unit ball the translates have summable bounds. This gives `D^q K_24(0) = θ(0)⁻¹ ∑_n D^q K_∞(24 n)`.
  - The packet's `derivativeBoundAtLeastOne` (76), `derivativeBoundAtZero` (15), `latticeImageSumBound` (1458 e^(−288)) and `exponentialBounds` close the bound: `1458 (76·24⁶ + 15) e^(−288) < imageConstE`.
- `entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound`: `|C_24 a b − C_ref a b| ≤ 2 imageConstE`. It follows from `imageBound`: the common factor has absolute value at most 2, and the contractions run along at most six unit frame vectors.
- `covarianceOrdering : UniversalLaw.Spec.Side24.CovarianceOrdering` (display (3), in Mathlib's `PosSemidef` order, `IsHermitian` included).
  - `C_24` is symmetric. Each contraction of `K_24` at 0 is `θ(0)⁻¹ ∑_n` of contractions of the analytic `Kinf` at 24n. Those are invariant under the block swap by Mathlib's `ContDiffAt.iteratedFDeriv_comp_perm`, which is present at `d13f23b7` for C^ω functions. Odd total orders vanish by `n ↦ −n` and the evenness of `Kinf`.
  - `C_ref − I/3 ≥ 0` comes from round 5's `referenceCovarianceLowerBound`.
  - Cauchy–Schwarz over at most 13 coordinates gives `|x·(C_24 − C_ref)x| ≤ 26E (x·x)`, and `78E ≤ ε` by `norm_num`.
  - The source's 20E / 60E < ε route is not reproduced, and main's `covariance_relative_bound` is not used.
- **Corrections of informal reasons, not statement defects:**
  - The local term-wise differentiation that the README called missing from Mathlib is supplied inside the module.
  - For reason (ii), higher-order symmetry, the analytic case the proof needs is present at the pin. The general finitely smooth version beyond order two is indeed absent.
- `SpecificationProofs.lean` imports the module last. The proof packet's source check now counts 78 theorems over 77 registered Props in 14 modules (`TriangleNeedsThree` is proved twice).

**Register** (`specifications/lean_statements_20261010/REGISTER.json`, blob `84d0dd07`, from `05cc80c1`): 35 string leaves change. Every `formal_progress` stays as it was (138 `none`, 33 `specified`), and every other field is byte-identical.
- Three new `proved` statement entries on `side24-coefficient`. Each carries its Prop's registered anchor, and the docstring's "Does not establish" as `does_not_claim`.
- At `9e305ea5`, the row's `kernel_evidence`, `not_established` and `next_step` gained a pending tier. The three theorems were added after PR head `c6ce0256`, whose run 38195810463 (actual tested merge `1320c5b2…` with base `be014794…`, `theorem_count` 75) does not name them. At `4a3fbadf` that tier becomes the receipted tier of run 38277458954 (PR head `9e305ea5`; actual tested merge `1376175c…` with base `a0323b94…`; the tested merge differs from the head only outside both packets). The register blob is now `4ba43ed9` and changes 23 more string leaves.
- Each of the three triage items keeps its verdict and finding verbatim, plus ": proved in round 6 (…)".
- The inventory parenthetical of the other nine proof-packet notes moves from 75/13/74 to 78/14/77.

**Docs:**
- The statement packet README/SCOPE are re-rendered: 77 of 141 Props have a theorem. At `9e305ea5`, † marked "no hosted receipt yet"; at `4a3fbadf` no dagger is left.
- The proof packet README/SCOPE gain two "formerly listed here (round 6)" rows, and the counts move: 64 Props without a theorem, 4 BLOCKED_MISSING_MATHLIB left.

**Generators** (outside the tree; fail-closed, exact-string edits with asserted counts, reproduce-before-render):
- `update_register13.py`
- `render_docs13.py`
- `edit_proof_docs13.py`

**Verification, 2026-10-12, same account, credit 0:**
- A first adversarial round (register, Lean docstrings, each docs output) found two blocking register wording defects. One was the analytic-only symmetry stated as if general; the other was a dropped `d ∈ {2,3}` restriction. It also found docstring inaccuracies. All were fixed by regenerating from the committed base.
- A delta round then checked the fixes: both delta verifiers PASS, and their remaining wording nits (the evenness of `Kinf` and the order ≤ 6 restriction in every description of the symmetry argument, one misplaced parenthetical) are folded into this head.

## What it does not establish

- Anything about an informal claim.
- An enclosure of the SIDE24 coefficient `c_{d,24}`. `CoefficientEnclosure` and `ReferenceComparison` keep BLOCKED_INTERFACE.
- That `C_24` or `C_ref` is the covariance of any random vector.
- Any bound for `q > 6`.
- Nothing beyond what the receipt records. The first hosted receipt of the three theorems is run 38277458954 (see Current disposition). Until 16:47Z this line read "No hosted run has receipted them yet".

`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt. The theorems have had no review other than this lane's own verification, which earns no independence credit.

## Coordination

Board note on main #307 follows (checkpoint promised in 6126482981). The main-side specification document (main #358) is re-rendered from the register at its recomposition after main #315, not in this PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY

---
_Generated by [Claude Code](https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY)_


---
_Generated by [Claude Code](https://claude.ai/code)_
