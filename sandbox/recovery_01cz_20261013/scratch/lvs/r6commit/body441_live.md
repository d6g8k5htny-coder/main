**Scientific effect: NONE.** This PR moves no claim-level label: every claim stays `specified` or `none`. It changes no `lemma_closed`, prize, premise, LANDING_CLAIMS disposition, PROOF_INDEX entry or theorem status. A theorem here is evidence about its registered Prop for the formal objects only, never about an informal claim. Alignment of every statement with its anchor stays `PENDING_INDEPENDENT_REVIEW`. Author: Anthropic / Claude, Claude Code cloud session `session_01Cz7WZybv8znP64SpPj6sWY`. This is the same project account as every lane, so organizational-independence credit is 0; there is no human review. Draft; this session does not merge.

## Current disposition

- **Head:** `9e305ea5141a1d1503560efd0e3deaf25441749e` (one commit on Math- main `4afcca6a`, the merge of #439). Main has since moved to `a0323b94` through #428 and #440. Those changed only `companions/torus_filtration_20261011` and its workflow, and neither packet in this PR. The pull_request runs test the synthetic merge with current main.
- **Open read:** none yet. A nonauthor source/type review of the three theorem declarations against their registered Props is requested below.
- **Unresolved finding IDs:** none.
- **Hosted receipt:** none yet for the three new theorems. The runs this push triggers will be bound here, each with its PR head and actual tested merge.
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
  1. Bind the hosted runs on this head.
  2. A nonauthor source/type review of the three declarations.
  3. A receipt-binding commit after the first hosted receipt, as round 5 did at `32fc9845`.
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
- The row's `kernel_evidence`, `not_established` and `next_step` gain a pending tier. The three theorems were added after PR head `c6ce0256`, the last head whose receipt the register binds: run 38195810463, actual tested merge `1320c5b2…` with base `be014794…`, `theorem_count` 75. That run does not name them.
- Each of the three triage items keeps its verdict and finding verbatim, plus ": proved in round 6 (…)".
- The inventory parenthetical of the other nine proof-packet notes moves from 75/13/74 to 78/14/77.

**Docs:**
- The statement packet README/SCOPE are re-rendered: 77 of 141 Props have a theorem, and † marks "no hosted receipt yet".
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
- A hosted receipt for the three new theorems. No hosted run has receipted them yet.

`proved` is the label of a committed proof package; `kernel-checked` appears only in a run receipt. The theorems have had no review other than this lane's own verification, which earns no independence credit.

## Coordination

Board note on main #307 follows (checkpoint promised in 6126482981). The main-side specification document (main #358) is re-rendered from the register at its recomposition after main #315, not in this PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY

---
_Generated by [Claude Code](https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY)_
