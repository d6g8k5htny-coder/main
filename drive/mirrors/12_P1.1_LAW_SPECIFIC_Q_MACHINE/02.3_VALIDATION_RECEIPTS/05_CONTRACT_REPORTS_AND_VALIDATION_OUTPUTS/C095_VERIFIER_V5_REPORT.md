# C095 Verifier v5 Validation Report

**Cases:** 22  
**Passed:** 22  
**Failed:** 0  
**All expected checks pass:** True

## Gate discrimination

| Gate | Triggered in regression cases |
|---|---:|
| `ASSEMBLY` | 4 |
| `DOMAIN-INFIMUM` | 2 |
| `BAND-PROVENANCE` | 2 |
| `COMMON-MODE` | 1 |
| `HEURISTIC-BRIDGE` | 2 |
| `MEASURE` | 1 |
| `MARK` | 1 |
| `COVERAGE` | 1 |
| `DEPENDENCY-WARRANT` | 1 |
| `H4-PATH` | 1 |
| `UB_G_RESIDUAL_UNIFORM` | 1 |

## Key adjudications

- the current Q0 upper contract is blocked by `ASSEMBLY` and `UB_G_RESIDUAL_UNIFORM`;
- the external matrix-perturbation theorem validates outside the Q0 domain;
- 4.3 fails and 4.35 passes the base UB-G arithmetic regression;
- the finite lower 0.8411 assembly fails after explicit losses;
- sampled rungs fail `DOMAIN-INFIMUM`; 79.988 passes the full-domain certificate while 80 fails;
- missing or unfrozen uncertainty metadata fails `BAND-PROVENANCE`;
- shared-error agreement is provisional and fails the Certified grade cap;
- an independent implementation certificate passes `COMMON-MODE`;
- an unbridged quantitative analogy fails `HEURISTIC-BRIDGE`; a Plausible analogy remains provisional;
- measure and mark deployment drops fail without a transfer bridge;
- a target below uncovered-mass risk fails `COVERAGE`;
- an open H4-PATH gate blocks promotion;
- a Proven parent cannot silently depend on a Measured object.

## Second-domain root

```text
MATRIX_PERTURBATION_THEOREM
1f2c349058aa638ac7dcaf09079e3bf74007010e688ecf8f7333765a3c9eebd5
```