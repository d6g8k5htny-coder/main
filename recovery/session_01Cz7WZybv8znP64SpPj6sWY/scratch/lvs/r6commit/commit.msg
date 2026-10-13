Proof packet round 6: ImageBound, EntrywiseCovarianceBound and CovarianceOrdering (Side24Images)

One new proof module, specifications/lean_statement_proofs_20261011/
SpecificationProofs/Side24Images.lean, imported last by SpecificationProofs.lean.
Its three inventory theorems each have exactly the type of one registered Prop
of Specifications/Side24.lean:

  imageBound               : UniversalLaw.Spec.Side24.ImageBound
  entrywiseCovarianceBound : UniversalLaw.Spec.Side24.EntrywiseCovarianceBound
  covarianceOrdering       : UniversalLaw.Spec.Side24.CovarianceOrdering

The round-4 triage rated all three BLOCKED_MISSING_MATHLIB (weeks). That verdict
is kept verbatim in the register as the historical record.

- ImageBound: a local, open-preconnected-set version of iteratedFDeriv_tsum is
  proved inside the module (Aux.ib_iteratedFDeriv_tsum_of_isOpen, by the global
  lemma's induction from hasFDerivAt_tsum_of_isPreconnected). It gives
  D^q K24(0) = theta(0)^-1 sum_n D^q Kinf(24 n) for q <= 6. The packet's
  derivative, lattice-image and exponential theorems then bound the difference
  by 1458 (76 * 24^6 + 15) e^-288 < imageConstE.
- EntrywiseCovarianceBound follows from imageBound: the common factor has
  absolute value at most 2, and the contractions run along at most six unit
  frame vectors.
- CovarianceOrdering:
  - C24 is symmetric. Contractions of the analytic Kinf are block-swap
    invariant by Mathlib's ContDiffAt.iteratedFDeriv_comp_perm, present at
    d13f23b7 for C^omega functions. Odd orders vanish by n -> -n and the
    evenness of Kinf.
  - Cref - I/3 >= 0 comes from the round-5 theorem.
  - Cauchy-Schwarz over at most 13 coordinates gives
    |x.(C24 - Cref)x| <= 26E (x.x), and 78E <= epsilon.
  - The source's 20E and 60E < epsilon are not reproduced.

Register (REGISTER.json, input blob 05cc80c1 at 4afcca6a):
- 35 string leaves change. Every formal_progress is unchanged (138 none,
  33 specified).
- Three `proved` entries are added on side24-coefficient (354 -> 357
  statements, 157 -> 160 proved), with a pending tier: no hosted receipt yet.
  The last receipt the register binds is run 38195810463 on PR head c6ce0256
  (actual tested merge 1320c5b2... with base be014794..., theorem_count 75).
  That run does not name these theorems.
- Each triage item gains ': proved in round 6 (...)'. For CovarianceOrdering,
  only the analytic case of the higher-order symmetry, which is the case the
  proof needs, is present at the pin. The general finitely smooth version is
  still absent.

Docs:
- Statement packet README/SCOPE are re-rendered (77 of 141 Props have a
  theorem; a dagger marks no hosted receipt yet).
- Proof packet README/SCOPE: two 'formerly listed here (round 6)' rows. The
  previous informal reasons are superseded as reasons, not as statement
  defects.
- Generators (outside the tree): update_register13.py, render_docs13.py and
  edit_proof_docs13.py. Each is fail-closed with exact-string edits and
  asserted counts.

Local checks (records of computations on this lane's worktree, not receipts):
- lake build exit 0.
- lake env lean -DwarningAsError=true exit 0.
- #check of each theorem against its registered Prop.
- #print axioms within {propext, Classical.choice, Quot.sound}.
- Both packets' check.py --allow-dirty: SOURCE_CHECK_PASS (78 theorems over
  77 Props in 14 modules).
- test_check.py, normal and -O modes: 97 and 70 tests OK.
- Two adversarial verification rounds (same account, organizational-
  independence credit 0). The first round's blocking findings were fixed,
  the delta round passed, and its wording nits are folded in.

Not established: anything about an informal claim, any enclosure of c_{d,24},
or a hosted receipt for the three theorems. Alignment remains
PENDING_INDEPENDENT_REVIEW. Scientific effect NONE.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Cz7WZybv8znP64SpPj6sWY
