# EXTERNAL_REVIEW_CHECKLIST — PR-TAL-003 through PR-TAL-008

**Draft for:** prize-lane external review (MATH PUSH support)
**Folder:** `1wCojptlhvMs9kfjnaNjVAiGTJy4YkiEz`
**Controlling custody:** Overlap ZIP `Prize_Research_Phase05_Overlap_2026-09-16.zip` (folder copy `1Z4RvtapE1NDsUCKmSQprneqP-z2IrvSK`)
**Manifest:** `OVERLAP_MANIFEST.sha256` — verify hashes before math verdicts

## Hard status (do not change)

| Field | Value |
|-------|-------|
| `original_prizes_solved` | **0** |
| `original_prize_closed` | **false** |
| Grade | AUTHOR_SIDE_COMPLETE_ARGUMENTS; EXTERNAL_REVIEW_PENDING; NOVELTY_UNESTABLISHED |
| Unrestricted discrete Convexity | **OPEN** (out of scope) |
| UPPER2D / PKG merge | **FORBIDDEN** |

Ordinary same-author replay is **not** independent review. Return **PASS / AMEND / FAIL / CANNOT-VERIFY** per interface. Distinguish counterexample vs missing provenance vs uncertain novelty. Do not translate passing check counts into a confidence percentage.

---

## Reviewer metadata

- Reviewer name / lineage:
- Date (UTC):
- Objects read (titles + Drive IDs + SHA-256 if computed):
- Scripts / transcripts attached:
- Prior exposure disclosed:
- Independence statement (organizationally distinct from authoring line?):

---

## R1 — PR-TAL-003 (five-copy laminar)

Proof: `P05_01_five_copy_laminar.md` (`16PGGnpcFoP3rFUAGBJoSbQwLnrD9jR-_`)

| Check | Verdict |
|-------|---------|
| Exact zero-scope extraction | PASS / AMEND / FAIL / CANNOT-VERIFY |
| Residual probability factorization | |
| Redundancy pruning | |
| Log capacity budgets | |
| Four small-rank rational estimates | |
| All-r≥5 factorial bound + geometric summation | |
| Five-copy laminar conclusion | |
| Nested scopes not charged as independent; rank-zero cost retained | |

Notes / earliest defect:

---

## R2 — PR-TAL-004 (bounded-incidence crossing)

Proof: `P05_02_bounded_incidence_crossing.md` (`1NWnIX5Ya4BW3ZKz7-wYb7ltSNfLKa1sV`)

| Check | Verdict |
|-------|---------|
| Deterministic greedy coverage with K=d(t−1)+1 | |
| Off-by-one / +1 color and −1 prior-element count | |
| Positive-capacity requirement | |
| All nonchosen colors handled | |
| Inclusion vs equality of obstruction cover | |
| Crossing-triangle counterexample to dropping +1 (d=2) | |
| d measured after valid structural normalization | |

Notes / earliest defect:

---

## R3 — PR-TAL-004 / 006 (Holder–Finner / weighted incidence)

Proofs: P05_02 + `P05_04_weighted_budget_and_boundaries.md` (`1N6N-1LW5-8kvZ4b_wFljxL7oPudhUIc9`)

| Check | Verdict |
|-------|---------|
| Independent reconstruction of fractional Hölder/Finner for finite product measures | |
| u^d ≤ product(1−q_B); log directions | |
| Weighted incidence Λ; Λ=0 without division | |
| Joint independence from nonincident coordinates ≠ pairwise independence of scope events | |

Notes / earliest defect:

---

## R4 — PR-TAL-005 (bipartite capacity families)

Proof: `P05_03_bipartite_capacity_families.md` (`1sF4v9Kv6nBzZrYMuX9kx8xrSPb9c8mWO`)

| Check | Verdict |
|-------|---------|
| Clone splitting + regular bipartite completion | |
| Hall matching and collapse | |
| Parallel edges and zero capacities | |
| Exact k-fold decomposition criterion separate from probability budget | |
| Odd-cycle falsifies analogous criterion | |

Notes / earliest defect:

---

## R5 — PR-TAL-007 (heavy/light product measures)

Proof: `P05_05_heavy_light_product_measures.md` (`1TqejVLAWr_VlNN4P5UWNJu6DaL55SQy9`)

| Check | Verdict |
|-------|---------|
| Heavy/light restriction | |
| Existence of two-piece partition for heavy coordinates | |
| Lifted generator cost; additive vs multiplicative piece count | |
| Whole heavy part need not be feasible | |
| No general efficient algorithm claimed | |

Notes / earliest defect:

---

## R6 — PR-TAL-008 decomposition (Local Lemma / coloring)

Proof: `P05_06_high_capacity_overlap.md` (`1Q8HqJBp26qCsBNS4vmnWWXuCYLFg3ZPs`)

| Check | Verdict |
|-------|---------|
| Binomial coloring tail | |
| x_B = (3/4)^r; full dependency-neighbor sum; product lower bound | |
| Exact rank-100 inequality | |
| Asymmetric Lovász Local Lemma application | |
| Holds for arbitrary finite ground size / #scopes; ranks ≥100 | |
| Finite resampling examples not treated as the proof | |

Notes / earliest defect:

---

## R7 — PR-TAL-008 smallness / condition (H)

| Check | Verdict |
|-------|---------|
| Same normalized representation for weighted load (coloring and cost) | |
| No-dilution 20-piece theorem | |
| 22-piece nonuniform extension | |
| Exact zero contribution | |
| Explicit nonvacuous unbounded-incidence family | |
| Condition (H) treated as substantive restriction (not all downsets) | |

Notes / earliest defect:

---

## R8 — Code / custody

| Check | Verdict |
|-------|---------|
| Independent small subset partitions, product probs, generators, edge coloring | |
| Exact Fraction arithmetic; sentinel None for nondecomposability | |
| Overlarge K; graph validation; mutation semantics | |
| Every reported negative test exits 1 with mathematical counterexample (not parser/import failure) | |
| Frozen hashes in MANIFEST.sha256 reproduced | |

Notes / earliest defect:

---

## R9 — Scope / priority / novelty

| Check | Verdict |
|-------|---------|
| Compared to primary Talagrand / read-k / Local Lemma / matroid literature | |
| No original prize / arbitrary-downset / general finite-field / q0 closure inferred | |
| Classical ingredients credited | |
| Known results do / do not already imply exact subclass claims or constants | |

Novelty disposition: UNESTABLISHED / CREDITED PRIOR / OTHER:

---

## Terminal roll-up (per claim)

| Claim | Overall |
|-------|---------|
| PR-TAL-003 | PASS / AMEND / FAIL / CANNOT-VERIFY |
| PR-TAL-004 | |
| PR-TAL-005 | |
| PR-TAL-006 | |
| PR-TAL-007 | |
| PR-TAL-008 | |

**Required delivery with this checklist:** exact source identities, per-interface judgments, explicit objections/corrections, scripts/transcripts, exposure statement.

**Still not camera-complete until:** named reviewer invite, standalone `ENVIRONMENT_LOCK` in folder, operator “send now” ratification.

