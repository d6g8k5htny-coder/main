# Technical review record

This is source-exposed, same-provider technical review. Organizational
independence credit is zero. No review here changes scientific status.

## Energy argument

The reviewer checked the full-adjugate proof and implementation against
`projection.py` SHA-256
`7ccd0d4fda9c5637857ac11840cc5e9d1b0b515a0ecc59670f224bfbb520c4ba`.
No mathematical, cofactor-index, coordinate or error-budget defect was found.
The exact perturbation argument was also checked on 250 rational SPD matrix
pairs. Those controls are algebraic checks, not spatial-coverage evidence.
Nineteen separate synthetic tests cover the retained quadratic covariance
terms, mean-linear error, full height interval, adjugate transpose, transformed
height direction and determinant scaling.

## Parent composition

The final reviewer checked `check.py` SHA-256
`22d33ec25684c5a9c64f69e7e6e7b0be9f9489f7efaed32aec642476a0ea5197`,
the parent proof and both frozen component proofs. No reachable correctness
defect was found in the authenticated execution path:

- Exact atom occupancy excludes the old wedge's interior. Its pinned strict
  bound is added once after integrating the increment.
- Cell areas use the polar Jacobian. Cartesian containing-box overlap does
  not confer extra integration ownership.
- Actual original-coordinate determinant and full-height energy bounds feed
  the sharp majorant with density Jacobian one. The mark length and spatial
  Jacobian each appear once.
- Normalized stationary moments are freshly enclosed. Old and new source
  bindings must agree, with source identities checked before and after use.
- An inconclusive cell prevents a complete-cover total. A failed requested
  threshold never silently grants that threshold.
- The aggregate helper composes trusted in-memory results. It is not an
  arbitrary-JSON proof verifier; the authenticated runner establishes trust.

The reviewer did not rerun the entire sector. Full numerical evidence belongs
to the parent's completed runs, separately recorded in `VALIDATION.json`.

## Corrections and retained diagnostic history

An earlier parent helper accepted floating-point aliases and did not bind the
child rectangle and determinant/energy values to the aggregate entry. Review
found these gaps before any completed parent certificate. The first development
run was stopped after three cells, its outputs retained, and the helper fixed.
The production numerical certifier did not change. Twenty-two parent tests
exercise these contracts and polar accounting in normal and optimized Python.

The projection component separately retains eight unsuccessful region probes
and records two corrected initial synthetic test expectations. Failed search
does not refute the mathematical target, and tests never substitute for the
actual interval cover.
