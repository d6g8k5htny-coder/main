# Mandatory D1 source reading rule and evidence ledger — v12

**Scope:** mathematically accepted/reviewed D1 proof components only; not an independent acceptance decision. Directly inspected GitHub-rendered pinned source and review text on 2026-10-24. The remote Git blob identities, byte counts and SHA-256 values below are *quoted from the project's reconciler*, **not independently rehashed from authenticated raw original source**.

| Role | Required file / read order | Source SHA-256 according to reconciliation | Admission |
|---|---|---|---|
| P | [Parent — `UNIFORM_MATRIX_CAP_AND_LIFETIME.md`](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) | `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` | **Only read with E1, E2, E3** |
| E1 | [Corrected endpoint congruence](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md) | `bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028` | Mandatory amended §5 A3 |
| E2 | [Replacement marked Kac–Rice §9 v1.1](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/reviews/d1_section9_borel_repair_20260925/REPAIR.md) | `845abf9f9c99d672c2a10a887b5a2e7206a3d2de3d876f35f75ff6e2dc13e62f` | Mandatory **replacement**, original §9 not to be consumed |
| E3 | [Reconciliation, §§1/5 and ledger](https://github.com/d6g8k5htny-coder/Math-/blob/9f3fb6918b28e6378c034ebc19e76d432f5391cb/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md) | Not supplied as a single byte-level SHA here | Mandatory wording W1, embedded radius, chain identity |

Reconciler explicitly declares D1 Theorems A–C accepted at existential/scoped level with no numerical C,r_*,z_* or leading coefficient. Its reading rule says original §9 is replaced by E2; the parent §5 uses E1; the parent §1/§5 convergence and embedding use E3. It notes original §9's inference to Borel elder mark had a missing written conditional-kernel and finite-measure argument. E2 adds it and was reviewed by two nonauthor providers at scoped C1–C6 interface level, **without organizational independence**.

## Graph corrections added

- `D1-CORR-CONG` → `D1-A` — corrected Hessian type/normalizer.
- `D1-CORR-SCOPE` → `D1-A` — in-law convergence and cylinder embedding.
- `D1-ELDER-MEAS` → `D1-A`, `D1-KR-BOREL` — global elder event is Borel.
- `D1-KR-KERNEL` → `D1-KR-BOREL` — C²-valued Gaussian conditional kernel.
- `D1-KR-BOREL` → `D1-KR-JACOB`, `D1-B`, `D1-C`, `D2-R` — corrected §9's marked pair intensity and selected elder bar count.
- `D1-KR-JACOB` → `D1-B`, `D1-KR-DOM`, `D2-R` — exact pair and lifetime measure.
- `D1-KR-DOM`, `D1-KR-FAR`, and `D1-A` → `D1-C` — near, far and selection for unrestricted leading theorem.
- `D1-KR-FAR` → `D2-R` — imported far-density O(1) bound.

All new edges are `source_explicit`, whereas editorial navigation links remain nonbinding. Existing v11 records and edges were not modified, and the user-facing status-filter still separates the 26 status-snapshot records from reviewed components.

## Important unresolved imports

The Armentano–Azaïs–León v3 Theorems 2.2/7.1 are identified in E2 as external mathematical background. Their exact v3 full-text hypotheses were **not independently checked in this pass**; a published paper's existence and the project's source-bound review do not constitute an independent reproof. C²-valued conditional smoothness, original Gaussian field covariance rank, all-mark moment bounds, and the global topology also remain independent obligations. Read [`PROOF_ADVANCES_v12_MARKED_KR.md`](PROOF_ADVANCES_v12_MARKED_KR.md).
