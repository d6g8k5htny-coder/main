# Scientific status snapshot

**Snapshot date:** 2026-09-29  
**Purpose:** one human-readable status page for the public front door.  
**Rule:** this page summarizes source-bound review outcomes; it is not an independent promotion register.

## ACCEPT — scoped

| Object | Accepted scope | Source and review | Explicit limits |
|---|---|---|---|
| **D1 — parent lifetime theorem (Theorems A, B, C)** | Theorem A `0 <= 1-p_r <= C r^3`, uniform over compact births, compact positive gaps and all frames; Theorem B compact-mark candidate/elder densities `~ c_{B,K} ell^(-1/3)` with difference `O(ell^(2/3))`, for `B` and `K` of positive length; Theorem C unrestricted leading term `c_{d,L} ell^(-1/3)`. Fixed `d>=2` and `L`, existential constants, read with the congruence erratum, the Section 9 replacement v1.1, wording W1 and `r < L/(4 sqrt 2)` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) · [reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md) · [Math-#126](https://github.com/d6g8k5htny-coder/Math-/pull/126) | No numerical `C`, `r_*`, `z_*` or coefficient; no unrestricted difference rate; this row is confined to the Math- `8224783` reconciliation, whose lower composition is planar (`d=2`); the separately reviewed every-fixed-`d` lower bound `1-p_r >= c r^3` was integrated afterwards by [Math-#129](https://github.com/d6g8k5htny-coder/Math-/pull/129) at `e7f8aca41c308f8ad8614b587f75916943c86f27` and is not part of this row's reconciliation scope; no RN/24-jet certificate |
| **D2 — unrestricted lifetime remainder** | Theorem R at its stated existential `O(1)` remainder scope: candidate and elder densities have leading `c ell^(-1/3)` term with bounded remainder for sufficiently small lifetime | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) | No numerical remainder constant or numerical lifetime cutoff; no RN/JETMOD closure |
| **D3 — SIDE24 coefficient calculation** | The coefficient expression for SIDE24 in dimensions 2 and 3, including cone moments, all-direction periodization comparison, covariance/Schur transfer, and outward special-function arithmetic | [Proof/package](https://github.com/d6g8k5htny-coder/Math-/tree/main/coefficients/side24_v1) · [review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222) | Coefficient arithmetic acceptance is separate from any imported parent theorem and from a finite-radius error band |
| **D4 — fixed-remote RN count theorem** | Fixed positive spatial separation `rho`, fixed witness separation `eta`, between-pin height window; conditioned covariance, three determinant factors, endpoint normalizer, contact kernel, and fixed-separation factorial moments | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md) · [review](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) | Does not cover shrinking `rho` or `eta`, pin neighborhoods, intermediate scales, all remote heights, or global RN/24-jet closure |
| **D6 — P15 full-price theorem** | Realized disjoint capacity/clutter family with `d_i >= 2`, all independent probabilities, `c_v <= phi(p_v)`, same palette `K >= K_H(d)`, sharp `rho*=1/(3-log(3e-2))` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/full_price_20260924/PROOF.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/74#issuecomment-5842112010) | Demand one remains an obstruction; no arbitrary-downset or unrestricted prize-problem extension |

## AMEND / open

| Object | Current reason | Source |
|---|---|---|
| **D5 — pin neighborhoods / microdisk** | The public reconnaissance has an AMEND review; the inner microdisk bound and collar to the reviewed annulus are not complete, and the claimed summed pin-neighborhood bound remains AMEND. | [Math proof index](https://github.com/d6g8k5htny-coder/Math-/blob/main/PROOF_INDEX.md) · [Math issue #58](https://github.com/d6g8k5htny-coder/Math-/issues/58) |
| **SARD-G A1/A6** | A1 remains AMEND as written; A6's slicing argument is conditionally valid but its source application remains AMEND until the required open-chart premise is repaired. | [Successor review](reviews/sard_g_successor_a1_a6_20260926/REVIEW.md) · [current PR](https://github.com/d6g8k5htny-coder/main/pull/122) |

## Engineering only

These are useful infrastructure, but they carry **no theorem-acceptance meaning**.

| Surface | What it provides |
|---|---|
| **[query-](https://github.com/d6g8k5htny-coder/query-)** | Public read-only package and exact-source lookup; default commit `76e1ca09` includes the package and the canonical no-dependency unittest command in CI |
| **[Universal-Law-Workspace](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace)** | Supporting federation/navigation map with `scientific_status_authority: false` |
| **main CI/navigation** | Link, structure, and engineering checks; green runs are not mathematical review |

## Where to read — public source custody

The two human maps are this file and the [Math- proof index at the pinned public Math commit](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/PROOF_INDEX.md). Neither file replaces the underlying proof or review.

For exact source identity, use [`docs/public-math/sources.json`](docs/public-math/sources.json). It indexes **2,138 intended public text artifacts** and points to 14 JSON shards. Each artifact row carries:

`repository + path + 40-char commit + Git blob + bytes + sha256`.

The frozen inventory includes the public `Math-` and `main` mathematics selected for publication and explicitly excludes sandbox, quarantine, and personal paths. It is source visibility/custody only.

The machine catalog at [`meta-framework/registry.json`](https://github.com/d6g8k5htny-coder/meta-framework/blob/main/registry.json) currently contains only public GitHub sources (`Math-` plus one declared public `google-drive` replica). The query portable stubs likewise point only to public `Math-` commits. Candidate query stubs are not silently promoted into the catalog.

### Default-branch boundary

- Pinned Math reading checkout: `82247833b5dd58e04291d55353b52938d73ec614`.
- Pinned query checkout: `89456c5033173c69a304c5fffa781888185675cc`.
- [Math- PR #70](https://github.com/d6g8k5htny-coder/Math-/pull/70) landed nine byte-custody copies at `9d7b6802424fb4715b31999066aafca8ee2f3cca`; [PR #75](https://github.com/d6g8k5htny-coder/Math-/pull/75) added their proof-index pointer. These are public default-branch imports, **not newly accepted mathematics**. Source self-labels were not adopted.
- The `main#59` issue-body file in PR #70 is explicitly a transcription, not original source bytes, so it is not counted as byte custody.
- Open draft AMEND/review PRs remain visible but are not represented here as landed mathematics.

## Reading rule

When a reviewed child result imports a parent that remains AMEND or open, keep those facts separate. “ACCEPT — scoped” never means “everything upstream and downstream is accepted.”
