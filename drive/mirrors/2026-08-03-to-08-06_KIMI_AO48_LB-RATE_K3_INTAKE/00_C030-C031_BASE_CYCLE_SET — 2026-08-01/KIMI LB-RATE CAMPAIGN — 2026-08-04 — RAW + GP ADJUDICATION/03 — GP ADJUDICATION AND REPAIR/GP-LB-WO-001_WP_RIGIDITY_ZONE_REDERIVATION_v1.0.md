# GP-LB-WO-001 — WP rigidity-zone re-derivation v1.0

Date: 2026-08-04

Status: OPEN / LOAD-BEARING FOR LB-RATE.

## Trigger

At `r = 0.025`, the delivered LB-1 run reports a rigidity-zone window-saddle integral of `1.29763e-5`; the C031 total is `0.213 r^3 = 3.328125e-6`. The subregion exceeds the total by a factor of 3.899.

The supplied `wp_rho` also omits the square root required by its stated Cauchy-Schwarz envelope:

- supplied: `p_grad * sqrt(E[det^2]) * min(Cantelli, P_window)`;
- required for that envelope: `p_grad * sqrt(E[det^2]) * sqrt(min(Cantelli, P_window))`.

The current numerical value is not a certified upper bound.

## Required object

Re-derive the exact typed pair-Palm/Kac-Rice WP contribution, including:

1. the determinant/type weight used by the Palm law;
2. the window indicator and its exact normalization;
3. collars and singular-near faces without deleting anisotropic factors;
4. the rigidity-zone, transition-zone, and far-zone partition with no overlap or omission;
5. a correct Hessian conditional-moment envelope;
6. a uniform-in-r bound on a declared interval `(0, r_WP]`;
7. an explicit coefficient `C_WP` in `WP(r) <= C_WP r^3`;
8. analytic or interval-certified mesh-to-continuum control;
9. fail-closed normal and `python -O` replay transcripts;
10. independent mutation tests targeting the missing-square-root defect and normalization errors.

## Acceptance tests

- Every displayed integrand is proven to dominate the exact typed intensity.
- The same object is used in proof, code, and ledger.
- The continuum enclosure is explicit; no unnamed formality factor.
- The result is uniform on `(0, r_WP]`, not only at isolated rungs.
- Every numerical constant has a full SHA-bound machine certificate.
- The previous 0.213 coefficient is either independently recovered or formally withdrawn.

## Status firewall

Until these tests pass, WP remains OPEN and KIMI-THM-023/KIMI-AUD-024 remain HOLD. This task changes no Boolean and does not affect the ratified upper-bound chain.
