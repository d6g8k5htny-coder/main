# A field-valued regression kernel in Lean, specified at every pin

**Object:** WHOLE-FIELD-REGRESSION-KERNEL-01CZ-20261101. Credit 0 · scientific effect NONE · OBL OPEN.

**Why.** [`docs/ELDER_EVENT_TETHER_20261031.md`](../../docs/ELDER_EVENT_TETHER_20261031.md)
found that tethering the pair-law `p` to the actual elder event needs condition T2: `Q` must be
the field's own regression law given the pins, defined at the prescribed pin value itself. The
repositories had this object in Lean only for finite-dimensional residuals (Math-
`GaussianRegressionKernel.lean`, blob `face5de8`). That was the first item of the tether's §7.

**What is proved.** In `WholeFieldRegression/Basic.lean`, compiled against Lean 4.34.1 and
Mathlib `d13f23b7` (the Math- pins). There are no `sorry`s, and only the axioms `propext`,
`Classical.choice` and `Quot.sound` are used.

| Declaration | Statement |
|---|---|
| `reconstructionKernel`, `reconstructionKernel_apply` | For a measurable reconstruction `g`, the kernel is `x ↦ (law R).map (g (x, ·))`. It has this literal value at **every** `x`, including probability-zero pins. |
| `independent_reconstruction_hasCondDistrib` | If the observation `X` is independent of the residual `R`, then `g (X, R)` has that kernel as a conditional law given `X`. |
| `fieldRegressionKernel`, `fieldRegressionKernel_apply` | Specialization to a residual **field** `R : Ω → T → ℝ` on an arbitrary index set `T`, with mean field `mY` and a measurable correction `c`: the kernel at `x` is the law of `t ↦ R t + mY t + c (x − mX) t`. |
| `fieldRegression_hasCondDistrib` | Display (12) of `PERSISTENCE_BRIDGE.md` §4.1, in shape: given pin/residual independence, `R + mY + c (X − mX)` has the field regression kernel. |
| `indepFun_pi_of_finite` | Independence of `X` from the whole field (product σ-algebra) follows from independence from every finite restriction `R|_S`. The proof goes through the measurable-cylinder π-system. |
| `fieldRegression_hasCondDistrib_of_finite` | The two combined: finite-restriction independence gives the field regression kernel. |

The two finite-dimensional lemmas it reuses (`regression_compProd_id_const`,
`regression_pair_hasCondDistrib`) are inlined unchanged from the Math- file, apart from the
namespace, so that this package depends on Mathlib only.

## Checks

`./run_checks.sh` replays all of them. Outputs from this session are in `evidence/`.

- **Build:** `lake build` succeeds.
- **Kernel replay:** `leanchecker WholeFieldRegression.Basic` exits 0. The same check on a
  nonexistent module exits 1.
- **Axiom audit** (`controls/Audit.lean`): 8 theorems, standard axioms only. The control
  `controls/SorryControl.lean` confirms that a `sorry` would appear in the report.
- **Negative controls**, which must fail to compile, and do:
  - `NegSign`: the wrong sign on the correction.
  - `NegNoIndep`: the independence hypothesis removed.
  - `NegFinite`: independence from a single coordinate claimed to give whole-field independence.

The local build compiled Mathlib from source, because its prebuilt cache host was blocked by
this session's network policy. Hosted CI has not run this package.

## What this does not establish

- **That the residual of the actual Gaussian field is independent of the pin data.** For each
  finite restriction, that needs joint Gaussianity and zero cross-covariance. Math- proves that
  only for `Fin k` coordinates. Bridging it to `Finset`-indexed restrictions, and constructing
  the actual residual field, remain open.
- **Any statement about a smooth-function space or its Borel σ-algebra.** The product
  σ-algebra on `T → ℝ` is coarser, and the elder indicator's measurability (tether condition
  T3) is not addressed.
- **Continuity of the kernel in the pins**, the typed weight `W`, the normalizer `Z`, or the
  identification `p = E_Q[W e]/Z` (T1, T4).
- **Any rate, coefficient or claim status.** Nothing is promoted.
- **Independent review.** It is Anthropic/Claude work (session 01CZ). A different-family review
  is required, and organizational-independence credit is 0.

credit 0 · sci NONE · eng ≠ discharge · no flags · OBL OPEN
