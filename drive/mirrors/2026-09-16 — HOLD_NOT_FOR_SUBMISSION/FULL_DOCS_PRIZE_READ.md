# FULL_DOCS_PRIZE_READ — Prize Research PHASE01–10 deep read

**Written:** 2026-09-16 ~22:15 CT (America/Chicago)  
**Sources:** local extracts under `/workspace/drive_peer_review_triage/extracts/` (preferred); verified intake / routing under `/workspace/drive_peer_review_triage/prize_research/`; prior memos `LANE_PRIZE_RESEARCH.md`, `DRIVE_DEEP_FAMILIARIZATION_2026-09-16.md`, `packages/hold/HOLD_PRIZE_RECONNAISSANCE.md`.  
**Drive mutations this pass:** **NONE** (local read + this memo only).

---

## HARD RULE — ZERO PRIZES SOLVED; NO PROMOTIONS

| Field | Value (as written in sources) |
|---|---|
| `original_prizes_solved` | **0** |
| Monetary / prize eligibility | Not claimed |
| External independent review | PENDING / 0 across packages |
| Historical novelty | UNESTABLISHED (unless source says NO_CLAIM / credited prior work) |
| Readiness / peer-review promotion | **Forbidden** — HOLD reconnaissance only |

Controlling citations (local):

| Doc | Local path | Drive fileId (where known) |
|---|---|---|
| `00_READ_FIRST — VERIFIED_PHASES_AND_ROUTING.md` | `prize_research/00_READ_FIRST_VERIFIED_PHASES_AND_ROUTING.md` | `1zTOKVho4Cu08zq2wqAFzNE-MMAuCM-bo` |
| `CLAIM_REGISTRY_VERIFIED_INTAKE.json` | `prize_research/routing/CLAIM_REGISTRY_VERIFIED_INTAKE.json` | `1RTZ7rjQUzQdLtiPKatAUJs6qfZeTXv0a` |
| `CURRENT_STATE_VERIFIED_INTAKE.json` | `prize_research/routing/CURRENT_STATE_VERIFIED_INTAKE.json` | `1Wn6YUkAFibhCvXuFqJocq_INFdm-TRDo` |
| `ROUTING_LEDGER_VERIFIED_INTAKE.json` | `prize_research/routing/ROUTING_LEDGER_VERIFIED_INTAKE.json` | `18OtwgPPqn3sZ4-mHEVLIUXN98LT9uoiz` |
| HOLD | `packages/hold/HOLD_PRIZE_RECONNAISSANCE.md` | folder `1AyEOZP1C6ebBx7wxZoIT_RIbP2noYuJ5` |

From VERIFIED intake / READ_FIRST (quoted sense):

- “No original prize problem has been solved.” / `original_prizes_solved: 0`
- Grade: `AUTHOR_SIDE_COMPLETE_ARGUMENTS; EXTERNAL_REVIEW_PENDING; NOVELTY_UNESTABLISHED`
- Registry inventories scoped proof notes; **every** verified-intake claim has `original_prize_closed: false`
- HOLD: reconnaissance only; not a peer-review submission package

**Do not** treat frozen ZIP presence, check counts, author-side “complete,” or delivery receipts as prize closure or peer-review readiness.

**Verified-intake registry scope:** phases **1–5 only** (28 records). Phase06–10 claims live in their own CURRENT_STATE / CLAIM_REGISTRY / proofs and are **not** in `CLAIM_REGISTRY_VERIFIED_INTAKE.json`.

---

## Extract roots (this deep read)

| Phase | Local extract root |
|---|---|
| 01 | `extracts/Prize_Research_Phase01_2026-09-16/` |
| 02 | `extracts/Prize_Research_Phase02_2026-09-16/` |
| 03 | `extracts/Prize_Research_Phase03_2026-09-16/Prize_Research_Phase03_2026-09-16/` |
| 04 | `extracts/Prize_Research_Phase04_2026-09-16/` |
| 05 generic | `extracts/Prize_Research_Phase05_2026-09-16/Prize_Research_Phase05_2026-09-16/` |
| 05 **Overlap (controlling for 003–008)** | `extracts/Prize_Research_Phase05_Overlap_2026-09-16/` |
| 06 | `extracts/Prize_Research_Phase06_2026-09-16/Prize_Research_Phase06_2026-09-16/` |
| 07 FINAL v2 | `extracts/Prize_Research_Phase07_2026-09-16_FINAL_v2/Prize_Research_Phase07_2026-09-16/` |
| 08 | `extracts/Prize_Research_Phase08_2026-09-16/Prize_Research_Phase08_2026-09-16/` |
| 09 | `extracts/Prize_Research_Phase09_2026-09-16/Prize_Research_Phase09_2026-09-16/` |
| 10 | `extracts/Prize_Research_Phase10_2026-09-16/Prize_Research_Phase10_2026-09-16/` |

Mirror copies also exist under `prize_research/phase0N/` for earlier phases; prefer `extracts/` when both present.

---

## PHASE01 — Restricted multi-prize probes + failed routes

**What it is:** Exact restricted digital/AP certificates, adaptive base-11 domination, Sidon splice trap, Collatz descent obstructions, finite trace-moment positivity obstruction. Separate from q0. Explicit: no original prize solved.

**Key docs read:**

- `00_READ_FIRST.md`
- `STATUS.json` (`original_prize_problems_closed: 0`; full targets OPEN / REOPENED_FOR_FRONTIER_STUDY)
- `NEXT_WORK.md`, `MANIFEST.sha256`
- Proofs: `proofs/01_erdos3_digital.md` … `05_spectral_moments.md`

**Claims / status wording (sources):**

| ID | Status / grade (as written) |
|---|---|
| PR-AP-001 | STATUS: `AUTHOR_PROOF_COMPLETE`; registry: `AUTHOR_SIDE_PROOF_PRESENT`, `external_review: PENDING`, `original_prize_closed: false` |
| PR-AP-002 | STATUS: `AUTHOR_PROOF_COMPLETE_WITH_EXHAUSTIVE_LOCAL_CERTIFICATE`; same registry pattern |
| PR-AP-FINITE | STATUS: `EXACT_RESTRICTED_COMPUTATION` (finite optima; not a prize claim ID in verified registry) |
| PR-SID-001 | `AUTHOR_PROOF_COMPLETE` / registry PENDING / closed false |
| PR-COL-001 | `AUTHOR_PROOF_COMPLETE` / registry PENDING / closed false |
| PR-SP-001 | `AUTHOR_PROOF_COMPLETE` / registry PENDING / closed false |

**Sealed vs pending:** Frozen ZIP + five author-side proof notes + exact certificates/mutation receipts **sealed** as custody. **Pending:** external review; novelty; unrestricted Erdős #3/#39, Collatz, RH; uniform harmonic bound.

**Code pointers:** `code/verify.py`, `reproduce.py`, `run_digital_complete.py`, `digital_certificate.py`, `build_adaptive_certificate.py`, `run_finite_ap.py`, `run_mutations.py`, `independent_digit_counts.py`, `digital_probe.py` (heuristic — outside certified layer), `ap_exact.cpp` (via rebuild).

---

## PHASE02 — Adaptive mixed-radix AP / Walker menu

**What it is:** Author-side restricted-class proofs for adaptive mixed-radix 4-AP-free trees (bases 2..30), single-switch enclosure, explicit witnesses, unbounded-radix bottleneck, Walker base-55 menu optimality on an enlarged adaptive menu. Priority correction: Walker / related-work leads credited; no record claim.

**Key docs:** `00_READ_FIRST.md`, `STATUS.json` (`original_prize_problems_solved: 0`, `external_reviewers: 0`, `novelty: unestablished…`), `REVIEW_PACKET.md`, `NEXT_WORK.md`, proofs `01`–`05`.

**Claims:**

| ID | Source status wording |
|---|---|
| PR-AP-003 | Proof: “complete author-side proof… external review and novelty audit remain open. This is NOT a solution of Erdos #3 or #142.” Registry: `AUTHOR_SIDE_PROOF_PRESENT` / PENDING / closed false |
| PR-AP-004 | “Complete author-side analytic derivation… external review pending.” |
| PR-AP-005 | “no external review or novelty certification. No prize problem is solved.” STATUS: `not_a_literature_record: true` |
| PR-AP-006 | “Author-side proof; no novelty claim… not a uniform bound solving Erdos #3.” |
| PR-AP-007 | “external review and novelty unresolved. This is NOT an extremality claim among all four-AP-free sets…” |

**Sealed / pending:** Frozen ZIP + five proofs + Bellman/menu certificates sealed; external review, unrestricted harmonic record pending.

**Code:** `code/reproduce.py`, `verify_phase02.py`, `verify_menu_admissibility.py`, `mixed_radix_core.py`, `bellman_interval.py`, `threshold_certificate.py`, `explicit_constructions.py`, `walker_menu_certificate.py`, `walker_*` probes, `general_kempner.py`, `larger_radix_probe.py`, etc.

---

## PHASE03 — Beyond fixed digit menus

**What it is:** Carry certificates; sparse-tail replacement; finite-state sparsity / extremizers; DFA reciprocal enclosure; extremal-profile bridge (Sawin 2025 core credited).

**Key docs:** `README.md`, `CURRENT_STATE.json` (`original_erdos_3_solved: false`, `new_unrestricted_harmonic_record: false`, `external_reviews: 0`), `REVIEW_PACKET.md`, proofs `01`–`05`.

**Claims (CURRENT_STATE grades):**

| ID | Grade (as written) |
|---|---|
| PR-AP-008 | `AUTHOR_PROOF_WITH_EXACT_COMPANIONS`; novelty `UNESTABLISHED` |
| PR-AP-009 | `AUTHOR_PROOF`; novelty `UNESTABLISHED; classical construction used` |
| PR-AP-010 | `AUTHOR_PROOF_CONSUMING_CLASSICAL_INPUTS`; novelty `UNESTABLISHED` |
| PR-AP-011 | `AUTHOR_PROOF_AND_IMPLEMENTATION`; novelty `UNESTABLISHED` |
| PR-AP-012 | `AUTHOR_DERIVATION; CORE_PRIOR_WORK_IDENTIFIED`; novelty `core credited to Sawin2025; refinements unassessed` |

Registry (verified intake): all `AUTHOR_SIDE_PROOF_PRESENT` / PENDING / `original_prize_closed: false`.

**Sealed / pending:** Frozen ZIP + five author-side theorems + companions sealed; unrestricted M₄ / Erdős #3 pending.

**Code:** `run_reproduction.py`, `code/verify_phase03.py`, `carry_automaton.py`, `positive_carry.py`, `dfa_carry.py`, `dfa_harmonic.py`, `verify_dfa_harmonic.py`, `tail_replacement.py`, `verify_profile_bridge.py`, `certify_probe_leaders.py`, `run_mutations.py`, plus `baseline/` Phase02 carry-overs.

---

## PHASE04 — Weighted extremizers + laminar Talagrand start

**What it is:** Weighted AP extremizer / greedy / local-marginal obstruction results; first laminar discrete-convexity certificates (PR-TAL-001/002).

**Key docs:** `README.md`, `CURRENT_STATE.json` (`full_prize_problems_closed: 0`, `general_Talagrand_conjecture_closed: false`, `new_harmonic_M4_upper_bound: false`), `REVIEW_PACKET.md`, proofs `01`–`05` + `06_review_and_scope_audit.md`.

**Claims (CURRENT_STATE):** each of PR-AP-013..015, PR-TAL-001, PR-TAL-002 has `grade: AUTHOR_SIDE_COMPLETE`, `external_review: false`, novelty `UNESTABLISHED` or `NOT_CLAIMED` (015).

Proof headers: “not externally reviewed”; PR-TAL-001/002 “NOT the full / universal discrete Talagrand conjecture.”

**Sealed / pending:** Frozen ZIP + five claims with exact companions sealed; unrestricted Talagrand / M₄ / r₄ pending.

**Code:** `code/run_reproduction.py`, `verify_weighted.py`, `verify_local.py`, `verify_laminar.py`, `verify_intersections.py`, `laminar_certificate.py`, `common.py`.

---

## PHASE05 — Overlap-controlled discrete Talagrand (+ generic extras)

### Controlling package: Overlap ZIP

**Drive:** Overlap zip `1XB6HnA5YArrfN7QRYtLDIN0sTHF9fRXt` (sha256 `e976271ba1a3822bddf6247d72b5d6131bfeabdcbc610982e816d812dd2e5a90` in verified CURRENT_STATE).  
**Local:** `extracts/Prize_Research_Phase05_Overlap_2026-09-16/`

**What it is:** Six author-side subclass theorems PR-TAL-003–008 with executed code/fixtures. `CURRENT_STATE.json`: `original_prizes_solved: 0`, grade `AUTHOR_SIDE_COMPLETE_ARGUMENTS; EXTERNAL_REVIEW_PENDING; NOVELTY_UNESTABLISHED`, `controlling_package: Prize_Research_Phase05_Overlap_2026-09-16.zip`.

**Key docs:** `README.md`, `CURRENT_STATE.json`, `routing/00_READ_FIRST — CURRENT_PRIZE_RESEARCH.md`, `routing/CLAIM_REGISTRY.json`, `REVIEW_PACKET.md`, proofs `01`–`06`.

**Claims PR-TAL-003–008:** verified intake `declared_proof_grade: AUTHOR_SIDE_PROOF_PRESENT`, `verification_this_turn: proof derivation and exact finite companions`, `external_review: PENDING`, `original_prize_closed: false`. Proof headers: “Complete author-side… NOT independently reviewed… Historical novelty UNESTABLISHED… not a prize solution / not unrestricted Convexity.”

**Note — Overlap ZIP controlling PR-TAL-003–008:** Verified intake + ROUTING_LEDGER + package README Finalization identity all state that the Overlap bundle seals the six executed proofs; generic Phase05 archive’s six shared proofs are byte-identical but extras are not part of overlap execution.

### Generic Phase05 archive (custody + 009/010)

**Drive:** `1r1EVWCnOlh0TU7AwtpQw8HIDyTQ3lZJy`  
**Local:** `extracts/Prize_Research_Phase05_2026-09-16/Prize_Research_Phase05_2026-09-16/`

Adds **PR-TAL-009** (`SOURCE_DECLARED_AUTHOR_ARGUMENT`; “additional checker not replayed”) and **PR-TAL-010** (`SOURCE_DECLARED_FINITE_CERTIFICATE_NOTE`; checker not replayed). Package `CURRENT_STATE.json` lists 003–010; `original_prizes_solved: 0`.

**Sealed / pending:** Overlap seals 003–008; both archives frozen under PHASE05. Pending: external review; novelty; 009/010 full replay; **unrestricted discrete convexity**; AP/RH/Collatz/Sidon originals.

**Code (Overlap):** `run_reproduction.py`, `code/capacity.py`, `common.py`, `verify_analytic.py`, `verify_crossing.py`, `verify_bipartite.py`, `verify_heavy_light.py`, `verify_local_lemma.py`, `negative_controls.py`.  
**Code (generic extras):** same plus `verify_phase05.py`, `mutations.py`.

---

## PHASE06 — Modular composition + rank-stratified hybrid

**What it is:** Two author-side composition/hybrid theorems on Phase05 subclasses. Primary front: discrete Talagrand restricted subclasses. “No original prize is solved.”

**Key docs:** `CURRENT_STATE.md` (same text as `prize_research/routing/CURRENT_STATE_PHASE06.md`), `NEXT_WORK.md`, `REVIEW_PACKET.md`, proofs `01_modular_composition.md`, `02_rank_stratified_hybrid.md`.

**Claims (Phase06 package IDs — see collision note):**

| ID (in Phase06) | Topic / status wording |
|---|---|
| PR-TAL-011 | Modular composition of intersecting decreasing families with certified covers. Proof: “Author-side complete. External review and historical novelty unestablished… not the unrestricted discrete Convexity Conjecture.” |
| PR-TAL-012 | Rank-stratified hybrid (bounded-incidence low-capacity + Phase05 weighted-overlap high-capacity). Proof: “Author-side complete conditional on Phase05 PR-TAL-004, PR-TAL-007, and PR-TAL-008. No external review or novelty claim.” |

**Not in** `CLAIM_REGISTRY_VERIFIED_INTAKE.json`.

**Sealed / pending:** Frozen ZIP under PHASE06; two proof notes sealed. Pending: external review; novelty; registry intake; unbounded low-rank / common-cause regime.

**Code:** `code/verify_phase06.py`.

---

## PHASE07 FINAL v2 — Capacity-one + rank-≥3 witness hypergraphs

**What it is:** Minimal-witness hypergraph reformulation through bounded-codegree rank-three theorems. Package + Drive routing `P07_CURRENT_STATE_v4.md`. “No original prize conjecture is closed. No external review has occurred.”

**Key docs:** `CURRENT_STATE.md`, `NEXT_WORK.md`, `REVIEW_PACKET.md` (R1–R8; notes universal Ψₖ bound **OPEN**), proofs `01`–`08`.

**Claims (Phase07 package IDs — collide with Phase06 for 011–012):**

| ID (in Phase07) | Proof title / grade wording |
|---|---|
| PR-TAL-011 | Minimal-witness hypergraph reformulation — “author-side derivation; no external review; novelty unestablished” |
| PR-TAL-012 | Capacity-one overlap under weighted-degree control — “author-side complete derivation; no external review; novelty unestablished” |
| PR-TAL-013 | Common-cause core + light remainder — same grade pattern |
| PR-TAL-014 | Weighted deletion / colorability bridge — formulates OPEN universal Ψₖ question |
| PR-TAL-015 | Universal capacity-one via greedy common-cause — “author-side complete… historical novelty unestablished” (author-side subclass) |
| PR-TAL-016 | Dependency-light higher-rank — same |
| PR-TAL-017 | Linear 3-uniform witnesses — same |
| PR-TAL-018 | Bounded-codegree rank-three — same |

**Remaining frontiers (CURRENT_STATE):** high-dependency rank-≥3; unbounded pair-codegree. Universal Ψₖ bound OPEN (REVIEW_PACKET R7).

**Sealed / pending:** Frozen FINAL_v2 ZIP; eight author-side proof notes sealed. Pending: external review; novelty; claim-ID reconciliation vs Phase06; unrestricted discrete Convexity; **any prize**.

**Code:** `code/verify_phase07.py`.

---

## PHASE08 — Rank-three minimal-obstruction (PR-TAL-019)

**Drive:** zip `1BpYigWDjX4gQ8YoTYkbWkwjO0LhrwVF5` · parent PHASE08 `1SEyIArKRmWqbFWq6SI1mz6PPlht86dpj`

**What it is:** Author-side candidate: minimal forbidden size ≤3 and μ_p(D)≥3/4 ⇒ D^(400) is (p/4)-small; cover cost <106087/249600. If correct, supersedes 017/018 by removing linearity/codegree assumptions.

**Key docs:** `CURRENT_STATE.md` (“Grade: author-side theorem candidate. External independent reviews: 0. Historical novelty: unestablished. Original prize conjectures solved: 0.”), `REVIEW_PACKET.md`, `NEXT_WORK.md`, `proofs/01_rank3_universal.md`.

**Claim:** PR-TAL-019 — proof **Status:** “author-side theorem candidate; no external independent review; novelty unestablished.”

**Integrity note (from Phase09):** optimized-assert (-O) checker integrity defect on a Phase08 mutant copy — test-integrity issue, **not** a mathematical counterexample. Original Phase08 ZIP unchanged.

**Sealed / pending:** Frozen ZIP + checker sealed. Pending: hostile review; novelty; rank four / rank-independent compression; unrestricted Convexity; any prize.

**Code:** `code/verify_phase08.py`.

---

## PHASE09 — Rank reduction, bounded dilution, structured-coloring limits

**Drive:** zip `19NFQJMMQ4QUhaxT4sZD5GTEcqOzGam0b` · parent PHASE09 `1BkSn0R8gftkUlcrxiEFQ8YOvXiuX2Zn-` · delivery receipt `1gR2oWPGcAW5nSjK7oqbW3D8T8CMlC-4S`  
**Additive routing (local):** `prize_research/routing/P09_additive_routing.md` (prepared; historically “PREPARED_NOT_APPLIED” in-package — deep-fam notes remote ZIP + receipt landed later; do not invent promotion).

**What it is:** P09-A..G author-side candidates: robust rank≤3, link-moment recursion, rank≤4, all finite ranks with rank-dependent K_r, literature/frontier record, method-limit transversal, exact block compression. Prior-work gate: Park–Pham / Kahn–Kalai fixed-rank qualitative existence must be credited.

**Key docs:** `README.md`, `CURRENT_STATE.md` (in-zip still says local / “NOT uploaded” — treat as stale for remote custody only), `CLAIM_REGISTRY.json` (`original_prize_closures: 0`, `remote_promotion: false`), `REVIEW_PACKET.md`, proofs `00`–`07`.

**Claims (CLAIM_REGISTRY grades):**

| ID | Grade | Scope shorthand |
|---|---|---|
| P09-A | `AUTHOR_SIDE_CANDIDATE` | rank≤3; good ≥2/3; 1600 pieces; dilution4 |
| P09-B | `AUTHOR_SIDE_CANDIDATE` | link-moment recursion |
| P09-C | `AUTHOR_SIDE_CANDIDATE` | rank≤4; good ≥3/4; 6400 pieces; dilution8 |
| P09-D | `AUTHOR_SIDE_CANDIDATE` | all finite ranks; rank-dependent K_r; dilution6 under ≥3/4 |
| P09-E | `SCOPE_AND_PRIOR_WORK_RECORD` | frontier/literature — not an independent theorem |
| P09-F | `AUTHOR_SIDE_CANDIDATE` | method-limit; not conjecture counterexample |
| P09-G | `AUTHOR_SIDE_CANDIDATE` | exact block compression + conditional cover-transfer |

All: `novelty: UNESTABLISHED`, `external_reviews: 0`, `formal_validation: false`.

**Sealed / pending:** Frozen ZIP + delivery receipt under PHASE09 (remote custody present per deep-fam); proofs + checkers sealed. Pending: external review; novelty vs Park–Pham; rank-independent piece control; unrestricted conjecture; any prize.

**Code:** `code/verify_phase09.py`, `run_checks.py`, `verify_manifest.py`, `intake_phase08.py` (session intake; not portable final runner), `results/p08_assert_guard_mutant.py`.

---

## PHASE10 — Common-palette composition for disjoint threshold hierarchies

**Drive (citation):**  
- Zip `1V7JN0LQ6FpQAZ4FnygU02Wsk22wijwqo` · parent PHASE10 folder `1Lq-O44mcB2lTp3M3tSqdMfq6WCsjLy-x`  
- Delivery receipt `1zeLO_gLUFfF4GlHNLQ6mwfy9baouE3dV`  
- Drive deployment receipt `1QHAJ8O297cxlZxdQI07QFlkhKZVRYuVB`

**Local:** `extracts/Prize_Research_Phase10_2026-09-16/Prize_Research_Phase10_2026-09-16/`

**What it is:** Structural subclass theorems for read-once / disjoint-support threshold hierarchies (optional arity≤64 monotone zero-preserving gates): fixed 64-piece, **no dilution** cover bound; exceptional-cover extension. Explicitly **not** an all-decreasing-family theorem; does **not** supersede Phase09’s broader-but-rank-dependent scope. Path on 65 vertices is a **representation** counterexample, not a Talagrand counterexample.

**Key docs read:**

- `README.md` — “External mathematical review: 0. Formal-prover runs: 0. Novelty: UNESTABLISHED. Original prize closures: 0.”
- `CURRENT_STATE.md` — P10-A..D; “There is no external review, formal proof, novelty certification, original prize solution…”
- `CLAIM_REGISTRY.json` — `original_prizes_solved: 0`, `external_reviews: 0`, `formal_prover_runs: 0`
- `drive_apply/00_READ_FIRST_P10_CURRENT_ROUTING.md` — additive routing; unrestricted discrete-convexity remains open
- `REVIEW_PACKET.md` (R1–R9 hostile interfaces), `NEXT_WORK.md`
- Proofs: `01_COMMON_PALETTE_SUBSTITUTION.md` … `04_STABILITY_AND_SCOPE_BOUNDARIES.md`
- Receipts: `P09_DEPLOYMENT_READBACK.json`, `P09_EXECUTION_RECEIPT.json`, `P09_INTAKE.json`, `RECON_CUSTODY.json` (P09 custody reconciliation during P10 packaging)

**Claims:**

| ID | Grade (CLAIM_REGISTRY) | Scope |
|---|---|---|
| P10-A | `AUTHOR_SIDE_COMPLETE_WRITTEN_ARGUMENT` | Exact common-palette substitution on disjoint supports |
| P10-B | same | Scalar capped-hazard inequality, K≥64 |
| P10-C | same | Read-once threshold/small-gate hierarchies, 64 pieces, no dilution |
| P10-D | same | Exceptional-cover extension + representation boundaries |

Proof status lines: “Author-side derivation/complete… no external review… novelty unestablished/unresolved… not Talagrand's unrestricted…”.

**Sealed / pending:** Frozen ZIP under PHASE10 + detached delivery/deployment receipts (cite Drive IDs above; do not infer proof approval from upload). Pending: external/hostile review; novelty vs read-once/threshold literature; universal gate-level potential inequality; extension beyond disjoint supports; unrestricted Convexity; **any prize**.

**Code:** `code/run_reproduction.py`, `verify_phase10.py`, `tree_core.py`.

---

## Cross-phase claim index

Status wording is quoted/paraphrased **from sources**; no invented closures. Verified-intake rows include explicit `original_prize_closed: false`.

| Claim ID | Phase / package | Status wording (sources) |
|---|---|---|
| PR-AP-001 | 01 | AUTHOR_PROOF_COMPLETE / AUTHOR_SIDE_PROOF_PRESENT; PENDING; closed false |
| PR-AP-002 | 01 | AUTHOR_PROOF_COMPLETE_WITH_EXHAUSTIVE_LOCAL_CERTIFICATE; PENDING; closed false |
| PR-SID-001 | 01 | AUTHOR_PROOF_COMPLETE; PENDING; closed false |
| PR-COL-001 | 01 | AUTHOR_PROOF_COMPLETE; PENDING; closed false |
| PR-SP-001 | 01 | AUTHOR_PROOF_COMPLETE; PENDING; closed false |
| PR-AP-003 | 02 | complete author-side; review/novelty open; NOT Erdos #3/#142; PENDING; closed false |
| PR-AP-004 | 02 | author-side; external review pending; PENDING; closed false |
| PR-AP-005 | 02 | no prize solved; not literature record; PENDING; closed false |
| PR-AP-006 | 02 | author-side; no novelty claim; PENDING; closed false |
| PR-AP-007 | 02 | author-side; review/novelty unresolved; PENDING; closed false |
| PR-AP-008 | 03 | AUTHOR_PROOF_WITH_EXACT_COMPANIONS; UNESTABLISHED; PENDING; closed false |
| PR-AP-009 | 03 | AUTHOR_PROOF; UNESTABLISHED; PENDING; closed false |
| PR-AP-010 | 03 | AUTHOR_PROOF_CONSUMING_CLASSICAL_INPUTS; PENDING; closed false |
| PR-AP-011 | 03 | AUTHOR_PROOF_AND_IMPLEMENTATION; PENDING; closed false |
| PR-AP-012 | 03 | AUTHOR_DERIVATION; Sawin2025 core credited; PENDING; closed false |
| PR-AP-013 | 04 | AUTHOR_SIDE_COMPLETE; external_review false; UNESTABLISHED; PENDING; closed false |
| PR-AP-014 | 04 | AUTHOR_SIDE_COMPLETE; external_review false; UNESTABLISHED; PENDING; closed false |
| PR-AP-015 | 04 | AUTHOR_SIDE_COMPLETE; novelty NOT_CLAIMED; PENDING; closed false |
| PR-TAL-001 | 04 | AUTHOR_SIDE_COMPLETE; subclass NOT full Talagrand; PENDING; closed false |
| PR-TAL-002 | 04 | AUTHOR_SIDE_COMPLETE; not universal; PENDING; closed false |
| PR-TAL-003 | 05 **Overlap** | AUTHOR_SIDE_PROOF_PRESENT; companions executed; PENDING; closed false |
| PR-TAL-004 | 05 Overlap | same |
| PR-TAL-005 | 05 Overlap | same |
| PR-TAL-006 | 05 Overlap | same |
| PR-TAL-007 | 05 Overlap | same |
| PR-TAL-008 | 05 Overlap | same |
| PR-TAL-009 | 05 generic | SOURCE_DECLARED_AUTHOR_ARGUMENT; checker not replayed; PENDING; closed false |
| PR-TAL-010 | 05 generic | SOURCE_DECLARED_FINITE_CERTIFICATE_NOTE; checker not replayed; PENDING; closed false |
| PR-TAL-011 | **06** | Author-side complete; review/novelty unestablished — *modular composition* |
| PR-TAL-012 | **06** | Author-side complete conditional on P05; no novelty claim — *rank-stratified hybrid* |
| PR-TAL-011 | **07** | author-side derivation; no external review; novelty unestablished — *minimal-witness hypergraph* |
| PR-TAL-012 | **07** | author-side complete; no external review; novelty unestablished — *capacity-one weighted-degree* |
| PR-TAL-013 | 07 | author-side complete; no external review; novelty unestablished |
| PR-TAL-014 | 07 | bridge + OPEN universal Ψₖ question |
| PR-TAL-015 | 07 | author-side complete; historical novelty unestablished |
| PR-TAL-016 | 07 | author-side complete; no external review; novelty unestablished |
| PR-TAL-017 | 07 | author-side complete; no external review; novelty unestablished |
| PR-TAL-018 | 07 | author-side complete; no external review; novelty unestablished |
| PR-TAL-019 | 08 | author-side theorem candidate; reviews 0; novelty unestablished; prizes solved 0 |
| P09-A | 09 | AUTHOR_SIDE_CANDIDATE; UNESTABLISHED; external_reviews 0 |
| P09-B | 09 | AUTHOR_SIDE_CANDIDATE; UNESTABLISHED; external_reviews 0 |
| P09-C | 09 | AUTHOR_SIDE_CANDIDATE; UNESTABLISHED; external_reviews 0 |
| P09-D | 09 | AUTHOR_SIDE_CANDIDATE; UNESTABLISHED; external_reviews 0 |
| P09-E | 09 | SCOPE_AND_PRIOR_WORK_RECORD (not independent theorem) |
| P09-F | 09 | AUTHOR_SIDE_CANDIDATE; method-limit; not conjecture counterexample |
| P09-G | 09 | AUTHOR_SIDE_CANDIDATE; UNESTABLISHED; external_reviews 0 |
| P10-A | 10 | AUTHOR_SIDE_COMPLETE_WRITTEN_ARGUMENT; UNESTABLISHED; reviews 0 |
| P10-B | 10 | same |
| P10-C | 10 | same |
| P10-D | 10 | same |

**Aggregate:** Across PHASE01–10, sources consistently report **original prizes solved = 0**. No promotions recorded in this deep read.

---

## Claim-ID collisions

| Colliding ID | Phase06 meaning | Phase07 meaning |
|---|---|---|
| **PR-TAL-011** | Modular composition of certified restricted covers | Minimal-witness hypergraph reformulation (χ(H[S]) > k) |
| **PR-TAL-012** | Rank-stratified hybrid (P05 incidence + weighted-overlap) | Capacity-one overlap under weighted neighborhood mass |

**Always cite claim ID + phase/package path.** Phase07 also continues numbering 013–018 for different theorems; Phase08 uses PR-TAL-019; Phase09/10 switch to P09-*/P10-* namespaces (no collision with PR-TAL-* for those IDs).

Verified-intake registry does **not** list Phase06–10 IDs.

---

## Overlap ZIP controlling PR-TAL-003–008 (reminder)

- **Controlling archive:** `Prize_Research_Phase05_Overlap_2026-09-16.zip` / Drive `1XB6HnA5YArrfN7QRYtLDIN0sTHF9fRXt`
- Six proofs executed with companions; ROUTING_LEDGER records byte-identity match to generic archive’s shared six proofs
- Generic Phase05 zip holds extras PR-TAL-009/010 as **source-read / custody**; extra checker **not** replayed in verified overlap intake
- Do not silently equate archives or treat 009/010 as overlap-sealed execution evidence

---

## Code pointer index (path → purpose)

Paths relative to each phase extract root unless noted.

| Path | Purpose |
|---|---|
| **Phase01** `code/verify.py` | Main exact arithmetic/certificate verifier (normal + -O) |
| P01 `code/reproduce.py` | Fresh-copy reconstruction / compare |
| P01 `code/run_digital_complete.py` | Rebuild digital search artifacts |
| P01 `code/digital_certificate.py` / `build_adaptive_certificate.py` | Rational enclosures / adaptive 1024-case cert |
| P01 `code/run_finite_ap.py` + `ap_exact.cpp` | Finite weighted-AP optima |
| P01 `code/run_mutations.py` | Negative-control mutation batches |
| P01 `code/independent_digit_counts.py` | Independent digit-count cross-check |
| P01 `code/digital_probe.py` | Heuristic probe — **not** certified layer |
| **Phase02** `code/reproduce.py` | Separable part runner (core/enumeration/bellman/threshold/walker-*) |
| P02 `code/verify_phase02.py` / `verify_menu_admissibility.py` | Phase02 + menu residue admissibility |
| P02 `code/mixed_radix_core.py` / `bellman_interval.py` / `threshold_certificate.py` | Core DP / interval enclosure |
| P02 `code/explicit_constructions.py` | Explicit infinite witnesses |
| P02 `code/walker_menu_certificate.py` + walker_* probes | Walker menu bridge / exploratory |
| P02 `code/general_kempner.py` / `larger_radix_probe.py` | Summation / larger-base probes |
| **Phase03** `run_reproduction.py` | Full rebuild outside package |
| P03 `code/verify_phase03.py` / `run_mutations.py` | Verifier + mutations |
| P03 `code/carry_automaton.py` / `positive_carry.py` / `dfa_carry.py` | Carry AP decision |
| P03 `code/dfa_harmonic.py` / `verify_dfa_harmonic.py` | Infinite reciprocal enclosure |
| P03 `code/tail_replacement.py` / `verify_profile_bridge.py` / `certify_probe_leaders.py` | Tail / profile / leader certs |
| **Phase04** `code/run_reproduction.py` | Jobs: weighted/local/laminar/intersections + mutations |
| P04 `code/verify_*.py` / `laminar_certificate.py` / `common.py` | Exact Fraction companions |
| **Phase05 Overlap** `run_reproduction.py` | normal/optimized/mutations groups |
| P05O `code/capacity.py` / `verify_*.py` / `negative_controls.py` / `common.py` | Overlap executed checkers |
| **Phase05 generic** `code/verify_phase05.py` / `mutations.py` | Generic harness (+ 009/010 path) |
| **Phase06** `code/verify_phase06.py` | Finite companion checker |
| **Phase07** `code/verify_phase07.py` | Finite companion checker |
| **Phase08** `code/verify_phase08.py` | Rank-3 candidate checker (assert/-O defect noted in P09) |
| **Phase09** `code/verify_phase09.py` / `run_checks.py` / `verify_manifest.py` | Final checker + harness + manifest |
| P09 `code/intake_phase08.py` | Session archival intake (not portable runner) |
| P09 `results/p08_assert_guard_mutant.py` | Documented -O assert integrity mutant |
| **Phase10** `code/run_reproduction.py` | Ordinary/optimized + 12 mutations + intentional survivors |
| P10 `code/verify_phase10.py` / `tree_core.py` | Hierarchy checker / tree core |

---

## Cross-phase sealed vs pending (custody summary)

| Phase | Sealed (frozen / author-side package) | Pending |
|---|---|---|
| 01 | Multi-topic restricted proofs + failed-route receipts | External review; all original prizes |
| 02 | Adaptive menu / Walker-bridge author proofs | External review; unrestricted harmonic record |
| 03 | Carry / tail / profile-bridge author proofs | External review; unrestricted M₄ / Erdős #3 |
| 04 | Weighted AP + laminar Talagrand author proofs | External review; general Talagrand |
| 05 | **Overlap seals 003–008**; generic holds 009–010 read-only | External review; unrestricted convexity; 009/010 replay |
| 06 | Modular + hybrid author proofs | External review; low-rank overlap; registry intake |
| 07 | Capacity-one / rank-3 author proofs FINAL_v2 | External review; unbounded codegree; unrestricted conjecture |
| 08 | PR-TAL-019 frozen ZIP | Hostile review; novelty; rank-4 / rank-independent |
| 09 | P09-A..G ZIP + delivery receipt | External review; Park–Pham credit; rank-independent pieces |
| 10 | P10-A..D ZIP + delivery/deployment receipts | Hostile review; novelty; beyond disjoint supports; unrestricted Convexity |

**Across PHASE01–10: zero original prizes solved. HOLD / reconnaissance only. No promotions.**

---

## Ops confirmation

- Deep-read local extracts PHASE01–10 + Phase05 Overlap; verified intake JSON trio + READ_FIRST; HOLD prize note; lane + deep-fam memos (not contradicted).
- Wrote this memo to `/workspace/drive_peer_review_triage/FULL_DOCS_PRIZE_READ.md`.
- **No Drive create / copy / move / trash / update / share / upload.**
- **HARD RULE unchanged: `original_prizes_solved = 0`; no promotions.**
