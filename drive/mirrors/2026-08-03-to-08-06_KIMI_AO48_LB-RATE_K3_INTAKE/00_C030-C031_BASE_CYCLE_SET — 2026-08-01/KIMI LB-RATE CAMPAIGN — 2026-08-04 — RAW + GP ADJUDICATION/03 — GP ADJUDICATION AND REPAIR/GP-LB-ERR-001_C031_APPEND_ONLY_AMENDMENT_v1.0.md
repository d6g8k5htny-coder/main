# GP-LB-ERR-001 — C031 append-only amendment v1.0

Date: 2026-08-04

Targets: both frozen C031 LB-RATE carriers. Their bytes remain unchanged.

## Amendment A — R2 near-term denominator

The printed expression

`0.76 * (ell/2) * (25 - 6.25)/(9 - 6.25) ~= 2.1 * (ell/2)`

is arithmetically inconsistent: its coefficient is approximately 5.18, not 2.1.

The proposed source-geometry reading

`0.76 * (ell/2) * (25 - 6.25)/(9 - 2.25)`

has coefficient 2.111111…, which matches the printed 2.1 after rounding.

Status: **PROVISIONAL AUTHORING ERRATUM / SOURCE-GEOMETRY CONFIRMATION PENDING**. It is not a certified theorem constant, because an alternate reconstruction cited in the same campaign yields approximately 1.82 and the quoted `3.4e-6` does not equal `2.1(ell/2)` at `r = 0.025`.

## Amendment B — localized full-KR mechanism

The ledger language that the bare value factor
`exp(-(b-m)^2/(2v))` kills the intensity throughout the whole jet-cluster horizon is withdrawn. The delivered LB-1 computation exhibits conditional-mean ridge arcs above `b` and an `O(1)` critical-point mass near a ridge maximum.

The narrower admissible statement is:

> On the B3 pass zone covered by the finite-rung certificate, the displayed computational envelope combines gradient mismatch, value mismatch, and a type factor. This is finite-rung computational evidence for the localized above-b-saddle channel. It is not a proof of a zone-wide bare-value kill or an all-small-r continuum bound.

## Amendment C — WP is reopened

The rigidity-zone WP subintegral reported by LB-1 is `1.29763e-5` at `r = 0.025`, exceeding the printed total `0.213 r^3 = 3.328125e-6`. In addition, the supplied `wp_rho` routine uses `min(Cantelli, Pw)` where its cited Cauchy-Schwarz construction requires a square root. The existing WP value is therefore not controlling.

Status: `WP OPEN — RE-DERIVATION REQUIRED`.

## Non-effects

- Frozen C031 bytes are preserved.
- No lower-bound theorem is closed.
- No Boolean changes.
- AO48-OPR-045 is unaffected.
