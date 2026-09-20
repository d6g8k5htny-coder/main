# C095 Gate Retrodiction Audit

**Frozen failure universe:** 19 entries

| Gate | Status | Hits | Silent fraction | Disposition |
|---|---|---:|---:|---|
| ASSEMBLY | QUALIFIED | 2 | 0.895 | POLARITY knows the direction but v4 did not evaluate the arithmetic expression or enforce display rounding. |
| DOMAIN-INFIMUM | QUALIFIED | 2 | 0.895 | DOMAIN checks metadata containment and ENDPOINT checks boundaries; neither proves the interior extremum of a numerically defined function. |
| BAND-PROVENANCE | QUALIFIED-WITH-SCOPE-CLARIFICATION | 2 | 0.895 | POLARITY guards side, not the construction or reproducibility of the band/tolerance. Gate scope includes stochastic numerical uncertainty. |
| COMMON-MODE | PROVISIONAL-EVIDENCE-BLOCKED | 0 | 1.000 | Implement the gate because its logic is sound; do not call the gate retrodiction-validated until the evidence package is supplied. |
| HEURISTIC-BRIDGE | PROVISIONAL-EVIDENCE-BLOCKED | 0 | 1.000 | Implement the gate and cap unbridged claims; keep the meta-validation claim provisional. |

## Qualified gates

### ASSEMBLY

- E-C094-2: upper bound rounded down from 4.35 to 4.3.
- E-C094-3: first-moment coefficient displayed before AO and Bonferroni losses.

### DOMAIN-INFIMUM

- E-C094-8: rung minimum exceeded 80 while the domain infimum was 79.9889203915… .
- C089-U-DOMAIN: discrete rung data was used toward a continuous interval claim without an extremum certificate.

### BAND-PROVENANCE

- E-C091-5: unexplained ±0.02 theorem band.
- E-C094-5: randomized orthant diagnostic changed because algorithmic state was not frozen.

## Provisional gates

COMMON-MODE and HEURISTIC-BRIDGE are implemented because their logical obligations are useful and low-cost. Their v1.0 claims of completed retrodiction remain provisional: the accessible release contains no named failure IDs and hashes for the examples asserted in Appendix A.

## Survivorship limitation

This audit covers only recorded failures. It cannot measure failures that were never detected and does not prove completeness of the gate set.