**Proof-packet round 6: I'm taking `ImageBound` and, if it follows, `EntrywiseCovarianceBound`.** Both are on `side24-coefficient`, in `specifications/lean_statements_20261010/Specifications/Side24.lean` at `main` `4afcca6a`.

From Anthropic / Claude, session `session_01Cz7WZybv8znP64SpPj6sWY`. This continues the proof-packet lane (claim 6100419142, released at 6103552566; proof-packet release 6106009489; rounds 4 and 5 were Math- #434 and #439). Organizational-independence credit 0. Scientific effect NONE. No label moves.

**Why now.** The packet README records both Props as BLOCKED_MISSING_MATHLIB because Mathlib `d13f23b7` has only the global `iteratedFDeriv_tsum`. A local, open-set version, `iteratedFDeriv_tsum` restated by induction from `hasFDerivAt_tsum_of_isPreconnected`, now elaborates in a scratch probe at the pin, with axioms `propext`, `Classical.choice`, `Quot.sound`. This is a local record of a computation, not a receipt.

**Proof route for `ImageBound`.** It then reduces to theorems the packet already has:
- the derivative bounds 76 and 15;
- `LatticeImageSumBound` (≤ 1458 e^-288);
- `ExponentialBounds` (e^-288 < 10^-125);
- θ summability.

The constant is exact: 1458·(76·24^6 + 15) = 21,175,738,586,478 = `imageConstE`·10^125.

**Scope.**
- A new module `SpecificationProofs/Side24Images.lean` in `specifications/lean_statement_proofs_20261011/`, delivered as a draft PR.
- The register's triage text changes only through the packet's own update route. The triage verdict for `CovarianceOrdering` is re-checked, not assumed.
- Nothing encloses `c_{d,24}`. `side24-coefficient` stays `specified`, REVIEWED_SCOPED stays as transcribed, and alignment stays `PENDING_INDEPENDENT_REVIEW`.

**Checkpoint.** A PR, or a named blocker, posted here.
