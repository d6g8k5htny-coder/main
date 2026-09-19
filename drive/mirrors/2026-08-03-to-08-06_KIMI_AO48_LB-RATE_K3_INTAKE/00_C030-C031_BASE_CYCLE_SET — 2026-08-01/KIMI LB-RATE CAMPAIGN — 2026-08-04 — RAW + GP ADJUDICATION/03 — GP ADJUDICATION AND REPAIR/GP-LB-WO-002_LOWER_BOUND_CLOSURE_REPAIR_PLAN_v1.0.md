# GP-LB-WO-002 — SIDE24 lower-bound closure repair plan v1.0

Date: 2026-08-04

Goal: convert the received conditional LB-RATE assembly into a theorem-grade lower bound without changing the accepted upper-bound chain.

## Gates

1. **WP:** complete GP-LB-WO-001.
2. **gamma-LOC(ii-c):** prove a uniform conditional-measure tube-tail bound for the arch/rim escape configuration.
3. **Far channel:** replace measured `sup pbar = 0.0334` with a certified upper bound over the full parameter band.
4. **Lambda window:** replace measured `c_Lambda = 0.946` with a certified lower bound and grid-to-continuum enclosure.
5. **Ridge/exit:** prove sampled-field ridge, diversion, and exit bounds; finite-sample zero is corroboration only.
6. **Uniformity:** extend LB-1 and LB-2 from isolated rungs to a common interval `(0, r0]`.
7. **Remainders:** quantify the outer `1 - O(r^3)` factor and every residual channel.
8. **Composition:** remove the textual double subtraction of the far term.
9. **Identity:** regenerate the dependency table with full whole-file and body hashes; resolve `40596829…`.

## Constant policy

The current measured inputs give the candidate leading coefficient
`(1 - 0.0334) * 0.946 = 0.9144036`.

Do not state `c = 0.9144` until all finite-r and outer remainders are bounded below uniformly. The eventual theorem should select an explicit constant strictly below the verified common infimum and state the corresponding `r0`.

## Present status

- Lower bound: HOLD / conditional assembly only.
- Two-sided corollary: HOLD / conditional implication only.
- Upper bound under AO48-OPR-045: unchanged.
- P0.1: unchanged.
