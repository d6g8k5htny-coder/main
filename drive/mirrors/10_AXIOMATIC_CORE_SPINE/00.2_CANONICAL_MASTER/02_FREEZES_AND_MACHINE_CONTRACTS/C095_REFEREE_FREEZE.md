# C095 Freeze — Q0-REFEREE

**Target terminal state:** `EXTERNAL-REVIEW-TRACK`

## Candidate scope

0 <= 1-q(r,6/5) <= 4.35 r^3, 0<r<=0.025, L=24, program/verification grade

The finite lower decimal is not in the live core.

## Frozen gates

| Gate | Status | Pass criterion |
|---|---|---|
| `UB_G_RESIDUAL_UNIFORM` | OPEN | Explicit uniform coefficient budget for Gamma and collar residuals is present in the ASSEMBLY expression on the full r-domain. |
| `ND-MARKED-SOURCE-AUDIT` | OPEN | Source-level derivation identifies the two height marks, saddle types, pair-Palm weight, and six-pin transfer. |
| `GRID-INTERVAL` | OPEN | Measured moduli replaced by outward interval or analytic suprema. |
| `NINEPIN-ATLAS` | OPEN | One analytic regularized-frame atlas covers the declared radius. |
| `SARD-G-REFEREE` | OPEN | Chart coverage, differentiability, spectral injectivity, disintegration, and measurability survive adversarial review. |
| `TORUS-GLOBAL-PARTITION` | OPEN | Every covariance/Schur use has a transfer row covering its order, dimensions, region, and distance partition. |
| `CITATION-PROVENANCE` | PARTIAL-CLOSED | Every citation retrieved from a primary source; Ladgham, Azaïs–Delmas, and Nicolaescu wording follows C094. |
| `ADVERSARIAL-REFEREE-SIMULATION` | OPEN | Independent read-through produces an objection ledger; every objection is answered or classified as a known limitation. |

No package may display a theorem constant until ASSEMBLY, DOMAIN-INFIMUM, BAND-PROVENANCE, and all domain-specific gates pass.