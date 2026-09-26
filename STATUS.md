# Scientific status snapshot

**Snapshot date:** 2026-09-26  
**Purpose:** one human-readable status page for the public front door.  
**Rule:** this page summarizes source-bound review outcomes; it is not an independent promotion register.

## ACCEPT — scoped

| Object | Accepted scope | Source and review | Explicit limits |
|---|---|---|---|
| **D2 — unrestricted lifetime remainder** | Theorem R at its stated existential `O(1)` remainder scope: candidate and elder densities have leading `c ell^(-1/3)` term with bounded remainder for sufficiently small lifetime | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) | No numerical remainder constant or numerical lifetime cutoff; no RN/JETMOD closure |
| **D3 — SIDE24 coefficient calculation** | The coefficient expression for SIDE24 in dimensions 2 and 3, including cone moments, all-direction periodization comparison, covariance/Schur transfer, and outward special-function arithmetic | [Proof/package](https://github.com/d6g8k5htny-coder/Math-/tree/main/coefficients/side24_v1) · [review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222) | Coefficient arithmetic acceptance is separate from any imported parent theorem and from a finite-radius error band |
| **D4 — fixed-remote RN count theorem** | Fixed positive spatial separation `rho`, fixed witness separation `eta`, between-pin height window; conditioned covariance, three determinant factors, endpoint normalizer, contact kernel, and fixed-separation factorial moments | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md) · [review](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) | Does not cover shrinking `rho` or `eta`, pin neighborhoods, intermediate scales, all remote heights, or global RN/24-jet closure |
| **D6 — P15 full-price theorem** | Realized disjoint capacity/clutter family with `d_i >= 2`, all independent probabilities, `c_v <= phi(p_v)`, same palette `K >= K_H(d)`, sharp `rho*=1/(3-log(3e-2))` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/full_price_20260924/PROOF.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/74#issuecomment-5842112010) | Demand one remains an obstruction; no arbitrary-downset or unrestricted prize-problem extension |

## AMEND / open

| Object | Current reason | Source |
|---|---|---|
| **D1 — parent quantitative Theorem-A selection chain** | Later review accepted the load-bearing §8–§15 interfaces, but the quantitative §§2–§7 selection chain was reopened for independent review. A displayed congruence defect in §5 has an additive erratum and the remaining A1–A7 chain is not reconciled. | [Review issue #63](https://github.com/d6g8k5htny-coder/main/issues/63) · [fail-closed reopening](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841830743) · [source-first erratum](https://github.com/d6g8k5htny-coder/main/issues/63#issuecomment-5841947606) |
| **D5 — pin neighborhoods / microdisk** | The public reconnaissance has an AMEND review; the inner microdisk bound and collar to the reviewed annulus are not complete, and the claimed summed pin-neighborhood bound remains AMEND. | [Math proof index](https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md) · [Math issue #58](https://github.com/d6g8k5htny-coder/Math-/issues/58) |
| **SARD-G A1/A6** | A1 remains AMEND as written; A6's slicing argument is conditionally valid but its source application remains AMEND until the required open-chart premise is repaired. | [Successor review](reviews/sard_g_successor_a1_a6_20260926/REVIEW.md) · [current PR](https://github.com/d6g8k5htny-coder/main/pull/122) |

## Engineering only

These are useful infrastructure, but they carry **no theorem-acceptance meaning**.

| Surface | What it provides |
|---|---|
| **[query-](https://github.com/d6g8k5htny-coder/query-)** | Public read-only package and exact-source lookup; default commit `76e1ca09` includes the package and the canonical no-dependency unittest command in CI |
| **[Universal-Law-Workspace](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace)** | Supporting federation/navigation map with `scientific_status_authority: false` |
| **main CI/navigation** | Link, structure, and engineering checks; green runs are not mathematical review |

## Reading rule

When a reviewed child result imports a parent that remains AMEND or open, keep those facts separate. “ACCEPT — scoped” never means “everything upstream and downstream is accepted.”
