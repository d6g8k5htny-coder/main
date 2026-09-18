# Research map

Lane-by-lane map of the research program as it stands in the Google Drive
source map of 2026-09-17 and the GP-REG-032-v1.2 register export of the same
date. Status labels are **as written by the sources**. Nothing here is a
promotion, a closure, or a review verdict.

Counts come from `drive/inventory.jsonl`; query it with
`python3 tools/drive_index.py`.

---

## 0. Shape of the corpus

| Lane (Drive root) | Items | Role |
|---|---:|---|
| `01_ACTIVE_RESEARCH_PACKAGES` | 3,408 | all live research |
| `02_LEGACY_Q0_ARCHIVE` | 346 | inspiration only; reverify from first principles |
| `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES` | 332 | protocols, registers, navigation, decisions |
| `90_QUARANTINE_AND_TRIAGE` | 191 | non-authoritative |
| `05_FOUNDATIONS_AND_PROTOCOL_HISTORY` | 151 | Fresh Start 2.0 history and archive |
| `03_PERSONAL_AND_EARLIER_RESEARCH` | 23 | zero evidentiary authority |
| `00_START_HERE` | 3 | Research Home + router |
| `06_SANDBOX_FRONTIER` | 2 | drafts, no authority |

Within the active lane:

| Sub-lane | Items |
|---|---:|
| `15_REVIEWS_RESPONSES_AND_CLOSURES` | 866 |
| `02_RESEARCH_CARRY_FORWARD_CANON` | 766 |
| `2026-08-03-to-08-06 — KIMI + AO48 LB-RATE / K3 INTAKE` | 468 |
| `2026-09-16 — PRIZE PROBLEM RECONNAISSANCE` | 295 |
| `01_RESEARCH_PLATFORM_AND_VERIFICATION_ARCHITECTURE` | 188 |
| `14_COORDINATION_AUTOMATION_SPINE` | 170 |
| `12_P1.1_LAW_SPECIFIC_Q_MACHINE` | 145 |
| `2026-09-16 — PEER_REVIEW_SUBMISSION_PACKAGES` | 127 |
| `2026-08-01-to-08-03 — SIDE24 RATIFICATION + P0.1` | 80 |
| `2026-09-16 — HOLD_NOT_FOR_SUBMISSION` | 79 |
| `2026-09-15 — KIMI FINAL INTAKE — UPPER2D STAGE E + H5 + ASSEMBLY` | 57 |
| `16_THEMATIC_RESEARCH_TRACKS` (T1–T5) | 49 |
| `13_POWER_PLANNING_CONTINUUM` | 42 |
| `10_AXIOMATIC_CORE_SPINE` | 41 |
| `11_P0.2_ADJACENCY_TRANSIT_TREE` | 19 |
| `2026-09-17_RN5_MOMENT_REPAIR_AND_REVIEW_ERRATUM` | 8 |
| `99_DO_NOT_OPEN` vault | 6 |

---

## 1. The frozen Q0 core and its successor chain

`Q0_MASTER.md` (1.83 MB, 36,063 lines, Drive `19nO3CU8B_YqGV9vTOwmvVe2Vyfeo0Jbz`)
is the consolidated spine — one of the **four-file canonical core** the
Research Carry-Forward Canon v1.0 (Drive `13ix3HD-AKBpxXKLydC1oMO9ask8SgreOOIOl7OeLMgI`)
names: `Q0_MASTER.md`, `Q0_LEDGER.md` (Drive `17gpxTn4MPsrWLf4yhDJsuBE4v3UqQXa3`,
253,067 B, `d1fea170…`), `q0_machine.json`, `q0_verify.py`; "these files
supersede the 488-file q0 archive for active use, but the historical archive
must not be deleted." `Q0_LEDGER.md` is the promotion channel: the Canon's
priority workspace "does not modify theorem status unless a result is approved
and entered through Q0_LEDGER", and the Formalization Board's canonical
boundary reads "Canonical mathematical promotion remains Dylan Roy's decision
through Q0_LEDGER.md or an equivalent exact operator record." The Canon writes
the live rate with its domain: "0 ≤ 1 − q(r,6/5) ≤ C_Q0 r³ for 0<r≤0.025, for
some finite C_Q0. Therefore q(r,6/5)→1 as r→0. No numerical value of C_Q0 is
certified." — model: exact normalized periodized Bargmann–Fock field on T² with
side L = 24, birth height b = 6/5, fold scaling ℓ = r³/6. Its binding claim
restrictions: "Do not promote the following as theorem constants: 4.3, 4.35,
0.8411, 0.84, 0.8501, 0.946, 0.97, 0.99, or 1.01." Its navigation companion
divides the spine into six parts:

* **Part I — Live theorem state (lines 20–2658).** `Q0_C101_QUALITATIVE_RATE_THEOREM`
  is the canonical theorem: `0 ≤ 1 − q ≤ C_Q0 r³`, `q → 1`, proved through defect
  decomposition, a global interceptor, Γ and collars, with **no typing division
  and no decimals**. `Q0_C104_THEOREM_B_PACKAGE` gives `ν_B(ℓ) = C*·ℓ^(−1/3)(1+o(1))`
  in conditional-MS form with **no numerical `C*`** — and its unconditional `PROVEN-HERE` status is **retracted**: `GP-AUD-187-v1.0` (2026-07-24, "CRITICAL SCOPE NARROWING — THEOREM NOT KILLED / NO PROMOTION") withdrew it, and the register's current status reads `CANDIDATE_UNCONDITIONAL_PROVEN_HERE_RETRACTED_EXACT_JACOBIAN_PROVED_CONDITIONAL_B0_PROVED` with "Do not cite Q0_C104 or Q0_MASTER historical PROVEN-HERE labels as current proof" (see `docs/OPEN_PROBLEMS.md` §G). `C102_REFEREE_OBJECTION_LEDGER`
  answers fourteen simulated adversarial objections.
* **Part II — Successor closure chain C094 → C101 (2659–4440).** How the frozen
  core's residues were closed or killed; the two cubic closures (near at 3598,
  collar at 4003) replaced numerical GRID/nine-pin obligations with exact
  scaled-frame algebra.
* **Part III — Frozen core C091–C093 (4441–7431).** `C091_PROOF_GATE_REDUCTION`
  line 4582 carries the marked-repulsion finding that conditionalized the lower
  theorem. `C092_FINAL_MASTER` §12.3 lists the **ten prohibited statements**.
* **Part IV — Gate framework (7432–9990).** v1.2 reconciled is live; v1.1
  superseded; a v1.1 filename variant retained under erratum E-C096-1.
* **Part V / V-b / V-c — Master v3.2 compiled set, Theorem-A lineage, 55
  inherited pre-C024 sources.** Files 01/03 carry pre-correction 4.3 and 0.8411
  displays, governed by Parts I–II (E-C094-2/-3, Q0-SHARP kill).
* **Part VI / VI-b — Ancestor program (15,390–36,063).** Reconstructed ancestor
  record plus 20 OCR-extracted PDF texts.

**Load-bearing rule from the core:** every finite decimal that was ever asserted
for `C_Q0` has a binding kill. The qualitative rate is existence of a finite
constant, not a numerical value.

---

## 2. UPPER2D — the D1 assembly (main active front)

Controlling body: `D1_ASSEMBLY_v2_2.md` — frozen **body** SHA-256
`490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6` (the
marker-delimited `BEGIN_FROZEN_BODY … END_FROZEN_BODY` body, 18,311 bytes, by
the file's own extraction rule); the whole Drive file
(`1v4z492iAzk5NcOrR47IJHGkIgfsRACpC`) is 20,078 bytes with SHA-256
`7ca114f0b38680d8bb987c097de10f3faf884ae3b05c3ca47215af5df081c174`. Hashing
the file and expecting the body digest is not drift.
Shipped as PKG-01. Intake authority:
`09152026OKComputer_Project_Gap_Closure.zip`, SHA-256
`a2136bc033f349382f9896896347da7a6dabde3334103276ad04db9205aa2b5b`
(2,029 archive members — the single largest carrier in the corpus).

* **Theorem (1)** — certified rung `r = 0.05`. Live clean totals
  `I_hi(v3) = 8.097558925 2e-2 = 647.8048·r³` (pin `8d7028e4…`).
* **Theorem (2)** — all-small-r, conditional on the five premises in the README.

**Layer discipline (do not collapse):**

1. Frozen v2.2 → five OPEN validity premises; PKG-01 ships here.
2. Register note + addenda → append-deltas effective at next issuance: B4LOC
   CLOSED, PERC RESTATED, normalizer DISCHARGED, BRANCH demoted. Reduced OPEN
   set becomes `OBL-D1-PROMOTE` (chart) + `D3-LEMMA-RN-UNIF`.
3. `D1_ASSEMBLY_v2_3_DRAFT` + `H5_ZBAND_CONSUMPTION` → **PROPOSED**, authority
   none until operator promotion.
4. HOLD checklist → still lists **all five** as blocking.

### H5 rung ladder (chart side)

| Rung `r` | Artifact | `I_hi/r³` | Note |
|---|---|---|---|
| 0.05 | frozen v1 / live v3 | 731.4311 / **647.8048** | v2 was contaminated by a rung-unscoped merge; v3 CLEAN, sha `8d7028e4…` |
| 0.035355 | `H5_RUNG3_2026-09-15` | **661.4712** | totals sha `808d6901…` |
| 0.025 | `H5_RUNG2_2026-09-15` | **664.3979** | totals sha `f7697bcf…`; C1 containment PASS; mutation 6/6 |
| 0.0177 | RUNG2 ladder | — | cells **42/70**, 2 shards resuming |
| 0.0125 | RUNG2 ladder | — | cells **21/70**, 2 shards resuming |

RUNG2/RUNG3 certify those rungs only. They do **not** discharge `OBL-D1-PROMOTE`,
`OBL-H5-JETMOD`, ZBAND-hi or REMOTE-THRESHOLD. The displayed modulus is dense
certified sampling plus an explicit fit; the certified **band enclosure**
(interval-`r` lattice sums giving `Î(r)/r³ ≤ F(G12-band)` per band, a finite
per-band computation, never a fitted exponent) remains `OBL-H5-JETMOD` OPEN.

### H3 band floor

For every `r ∈ (0, 0.05]`, `E[G_r] ≥ 2.30659559567154 > c_Z = 1.6154892676435024…`,
hence `Z_r ≥ c_Z·r²` uniformly (frozen body `281477c3…`). This discharges the
normalizer sub-part of `OBL-D1-PROMOTE` and supplies the lo side of
`OBL-H5-ZBAND`. The band table `Z_r/r² ∈ [2.3066, 3.7477]` is what the PROPOSED
ZBAND consumption certificate consumes.

---

## 3. RN — the remote/normalizer uniform lemma (`D3-LEMMA-RN-UNIF`)

Premise 2 of Theorem D1 v2.2(2). Two pieces; both **OPEN**; receipts carry
`lemma_closed: false`; CL-RNU-001/002/003 are all **PROPOSED** with authority
none. A session CLOSE, a smoke test, or a `max_err → 0` regeneration is **not**
lemma closure.

### RN3 (2026-09-17) — far region certified author-side

Fixed rung `r = 1/20`, fixed axis, side-24. Complete joint comparison of the
full nine-dimensional Hessian law against one product reference, then a sharp
Gaussian chi-square inequality and an exact saddle-typed second moment.

| Result | Value |
|---|---|
| Spatial cover | all 13,604 quarter-grid boxes meeting `5 ≤ |y| ≤ 17` |
| Mark coverage | every `v ∈ [6/5 − r³/6, 6/5]` |
| Intensity ratio | `0.88021 < ρ/ρ_ref < 1.12058` |
| Conditional far count | `2.22542 r³ < I_far < 2.83312 r³` |
| Arithmetic | Arb, 384 bits; both infinite tails enclosed |
| Audit | 294 checks, 24 full nine-pin Gaussian laws |
| Independence credit | **zero** (same-provider author-side) |

Proof body: 12,956 bytes, SHA-256
`0c9446b7cb49e6e1e45e43f84bb72ce680f022882e225fdff61c8e6edede1373`.

### RN5 (2026-09-17) — near-region moment defect found and repaired

**This is the most consequential recent finding.** The pinned `d3_perc.py`
function `envelope_v` returned `(EA⁴·EB⁴)^(1/4)·sqrt(EC⁴)` and described it as
Cauchy–Schwarz twice. The correct Hölder(4, 4, 2) bound is

```
E[|ABC| · 1{M max} · 1{S saddle} · 1{y saddle}]  ≤  (EA⁴ · EB⁴)^(1/4) · (EC²)^(1/2)
```

so the implementation used the **fourth** determinant moment where the **second**
was required — a substantive power error, not a typo. A typed, nondegenerate
Gaussian counterexample with exact rationals gives typed expectation
`≥ 0.2076750355 68` against an old envelope of `≈ 0.0625040624 9`, proving
`old envelope < 63/1000 < 207/1000 < typed expectation` by exact rational
comparison of fourth powers.

Affected and now scope-held: the `envelope_v` certification claim, its
annulus/remote consumers in `d3_amend_v2.py`, and CL-RNU-003 Section 3's
integrand-certification and forecast claims. **RN3 is outside the affected
scope** — it uses the correct second moment.

The repair supplies 65 point-law certificates on the historical 13-radius,
5-angle D4 net (all pass at 384 bits) and 10 genuine spatial-box certificates
via centered Taylor jets (order `N+2`, `N = 6`) with separate marginal whitening,
an L² remainder bound, and interval Cholesky pivots. Ten boxes **do not** cover
the annulus. Diagnostic polar sums: `≈ 2.34195 r³` corrected versus `≈ 17.67237 r³`
wrong-power; the older published `17.6804 r³` used a different mark-cap
heuristic. **None of these numbers closes Piece 2.**

`CANNOT_VERIFY` is recorded for the current CL executable and checkpoint
identity: `CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, `rnu_t4.py` and `rnu_spine.py`
were not returned by bounded exact-name lookup.

### RN3 ↔ CL-RNU-003 crosswalk (`RV-RN-ALIGN`, open)

Different methods, different majorants, different coverage. RN3's `0.12058`
far correction and CL-RNU-003's `0.67728` peak diagnostic **measure different
objects** and must not be compared until law, normalization, kappa definition
and domain are matched.

---

## 4. Percolation, branch control and the dam line

* **`PERC-DECAY`** — the `o(r³)` far-lane reading is **not reachable**; the
  recommended restatement is `Θ(r³)` with certified constants plus `PD-CONN`.
  Engine COMPLETE and FROZEN; `PD-CONN` is a named OPEN input.
* **`B4LOC-R1`** — dam line **CLOSED** at super-algebraic grade for the whole of
  B4 (loc + rem). The identification question was resolved **negatively**:
  cut-net ≢ nine-pin tube. The frozen v2.2 OPEN list still names the premise
  until the next issuance.
* **`OBL-B1-BRANCH(loop|B1)`** — OPEN in frozen v2.2; moved to REFINEMENT in the
  register note.

---

## 5. Lower side — LPW

| Object | Statement | Disposition |
|---|---|---|
| LPW qualitative | 2D side-24 exact six-pin typed pair-Palm elder pairing; `∃ c, r₀ > 0 : 1 − q ≥ c r³` | ACCEPTED AT REVIEW SCOPE. 37/37 payload hashes match; whole verdict hash matches; body-rule separator amendment needed |
| QC-RETURN03 fallback | `1 − q ≥ 10^(−1235) r³` for `0 < r ≤ 10^(−28)`, exact rational constants | ACCEPTED at received analytic review scope; preserved as a separate fallback |
| `LPW_CONSTANT` as delivered | exact fraction `260/(3790446482793·2⁴⁰·10²¹) = 6.23854270293559…e−44`; radius `1/2414592` | **AMEND REQUIRED**: reported headline `6.239e−44` exceeds the exact chain; the `E(|ξ|+|η|)⁴ = 12 + 16/π` identity is false; blanket-rounding explanation incomplete |
| R05 Rayleigh repair | proposed `1 − q ≥ 6.238e−44 r³` on `0 < r ≤ 1/2414592` using `ρ = √(ξ²+η²)`, `Eρ⁴ = 8` | author-side repair; does **not** inherit the external approve; 39 interval checks pass, oversized headline rejected in both modes |
| W8 v3 Lambda | limit-object difference ≈ `0.40371620975`, conditional on H-B3 | staged conditional progress; final transcripts END **FAIL-CLOSED**; no finite-`r` closure |
| Matching 2D upper | same-law `1 − q(r, 6/5) ≤ C r³` for all small `r` | **OPEN**, investigation authorized; 2D two-sided Θ **not admitted** |

PKG-04 ships the LPW review bundle. **Consequence 4 (the two-sided claim) is
WITHDRAWN.** The LPW-CONSTANT v4 brick is on HOLD, titled NOT READY.

---

## 6. P0.1 / P0.2 / P1.1 — adjacency and the q machine

* **P0.1 uniform adjacency positivity** — control record v1.10 / LS-CTL-003-v1.3.
  `FOUNDATION = APPLICATION = TRUE`; `INDEPENDENCE = INTEGRITY = FALSE`; state
  **HOLD**. B0's evidence condition is satisfied but its Boolean is unchanged and
  fold-ready; any successor fold must acknowledge the LS-WO-001 / CLWO-P01-001
  binding. Do not repeat obsolete E0/E2 or D0/I0 requests.
* **P0.2 adjacency-to-one cubic rate** — the controlling banner of its
  READ_FIRST (2026-07-30, Drive `1_hJQkq7Y8BFGZvGmCtvJR1UheV_YHmqXuimtYsYUDeA`)
  reads: "Counts remain five terminal / eight review-pending interfaces / ten
  exact objects. P0.2 remains OPEN / NONTERMINAL / NOT PROMOTED." and names
  `P02-RESOLUTION-REGISTER-v1.0` as the authoritative current-object register
  ("Use … before every further audit or review dispatch"). The document is
  append-heavy and says of itself that lower text stating earlier counts "is
  historical and nonoperative for current routing". The eight review-pending
  routes and their exact objects are in `docs/OPEN_PROBLEMS.md` §D and
  `claims/graph.json`. (Until 2026-09-18 this bullet reproduced the July 21–22
  gate layer — V1 discharge, the 864/864 T-B…T-E certificates, the
  effective-domain correction — which the banner supersedes.)
* **P1.1 law-specific q machine** — `q0_machine.json` + `q0_verify.py`;
  NONCANONICAL, fail-closed PASS, 12-of-12 negative tests. The 1.2.2 candidate
  repaired eight malformed-shape crashes (21 original + 13 shape + 2 CLI cases
  pass); distinct-provider review and adoption gates remain open.

---

## 7. K3 / LB-RATE — the 2D lower campaign: one assembly refuted, the campaign open

The lane (468 items; the 2026-08 KIMI + AO48 intake, the frozen C030/C031 base
cycle set, the WO-063 carriers, the K3 swarm delivery and the GP adjudication
chain) is the **2D lower-bound campaign** for `1 − q(r, 6/5)`. Its controlling
adjudication is `GP-LB-STAT-004` (hostile adjudication of the K3 partial
delivery, raw carrier Drive `1lfH7g57LcshqpLfNbrx-H7gbJckpzPqr`, 13,399 B,
`26a9c7ef…`), whose verdict on the assembly `K3-THM-001` (Form C,
`liminf (1−q)/r³ ≥ 0.9666·c_Λ`) reads, verbatim: **"REFUTED AS A THEOREM-GRADE OR
VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN"** — "The package
contains useful symbolic, forensic, numerical, minimality, and composition work.
It does not establish an all-small-r lower theorem, a valid fixed lower
constant" …; `K3-THM-001` itself: "REFUTED AS WRITTEN / NONCONTROLLING. No
theorem successor is minted here". `GP-LB-STAT-004A` is the W8 recovery
correction. Within the delivery, W2 is pointwise valid with a Lemma-17
amendment; W6 refuted as a rigorous modulus; W10 a valid schema with a load-path
gap; W13 a stale snapshot. Any successor needs an exact new identity **and**
every consumed weighted-law / Bonferroni / exit premise; the register says "Do
not mint a successor". The campaign's five P0 open questions (`OQ-014`,
`OQ-015`, `OQ-016`, `OQ-016-U1`, `OQ-016-U2`) are transcribed in
`docs/OPEN_PROBLEMS.md` §H. Until 2026-09-18 this section named the adjudicator
as the refuted object and read the assembly's verdict onto the whole lane.

---

## 8. Prize reconnaissance (independent track, HOLD)

`CLAIM_REGISTRY_VERIFIED_INTAKE.json` grades every PR-TAL carrier:
`declared_proof_grade: AUTHOR_SIDE_PROOF_PRESENT`, `external_review: PENDING`,
`historical_novelty: UNESTABLISHED`, `original_prize_closed: false`.

* **PR-TAL-003…008** — sealed in `Prize_Research_Phase05_Overlap_2026-09-16.zip`
  (SHA-256 `e976271ba1a3822bddf6247d72b5d6131bfeabdcbc610982e816d812dd2e5a90`,
  244 members). "Sealed" is archive-custody language, not theorem grade. Bodies
  are `P05_01_five_copy_laminar.md` … `P05_06_high_capacity_overlap.md`.
* **PR-TAL-001/002** — Phase04 priors. **PR-TAL-009/010** — generic Phase05,
  checker not replayed. **PR-TAL-011…018** — Phase06/07 with a known claim-ID
  collision across phases: always cite ID **plus** phase path.
  **PR-TAL-019** — Phase08 rank-3 source.
* **P14 / P15** — P15-A translates the optimized scalar-sandwich parameter to the
  established critical threshold `α(D)`; P15-B gives a setwise fixed-label
  palette-localization certificate; P15-C an even-cycle family with global
  threshold exactly `m(3s−2)/(2s)` growing unboundedly; P15-D a fixed
  `256·ceil(408κ)`-piece compatible cover independent of macro degree. All
  author-side; 0 external reviews; 0 formal prover runs.
* Phase carriers: Phase02 (237 members), Phase03 (342), Phase04 (468),
  Phase05 Overlap (244), Phase10 (272), Phase11 (620), P14 Fractional (191),
  Continuation Audit (685).

**No original prize problem has been solved.** Keep this on every public surface.

---

## 9. Platform and verification architecture

* `01_RESEARCH_PLATFORM_AND_VERIFICATION_ARCHITECTURE` (188 items) — metrics and
  defect interception (`RM-METRICS-001`), the formal/Lean research system with
  core lemmas `P02-LM-001/002/005/007/008`, `EC-014/015/021`, a replication
  tree, regression corpora and negative controls, and verification prototypes.
  The formal state, as the FORMALIZATION BOARD (Drive
  `10o4YRYOr8a2fB6rtnFnzMn7HkQMfv9-FZ5Mh0L-KF_o`) records it: the original
  `GP-FOR-001` bundle is "not present in Drive; its declared hash and commits
  remain historical provenance"; the additive `GP-FOR-189` source-recovery
  package is "STATIC-PASS / ROUNDTRIP-PASS / LEAN-CI-NOT-RUN / NONCANONICAL";
  "Lean build: NOT RUN. Lean/Lake unavailable"; every formal item is
  SORRY-PARTIAL, CI-PENDING or NOT COMPILED, and "Package approval is not claim
  promotion." The Board's OPERATOR PACKAGE DECISION of 2026-07-24 is the one
  operator sentence in the corpus about a Git repository: "REPOSITORY ROUTING:
  private Git repository creation and exact-history push are approved when
  platform access becomes available." (This repository is public; the
  visibility decision is the owner's — see `governance/GIT_ADAPTATION.md`.)
* `14_COORDINATION_AUTOMATION_SPINE` (170) — the live registers and the
  GP-AUTO-034 automation. Automation state: **R0.4 current development
  candidate, static T1–T35 + 4 fixtures pass, all write targets explicit,
  isolated sandbox prepared, runtime NOT executed, DO NOT INSTALL.**
* `07_MODEL_ACCESSIBILITY` (new, 2026-09-17) — 392 published reading copies,
  6,034 source-to-copy links, 137 retained exceptions. Mirrored here as
  `drive/source_map/`.

---

## 10. Archive carriers

77 carriers, 11,649 member occurrences, 4,020 distinct payloads. Largest:

| Carrier | Members |
|---|---:|
| `09152026OKComputer_Project_Gap_Closure.zip` | 2,029 |
| `SIDE24_G9_R2_RECOVERY_AND_RECONSTRUCTION_2026-08-01.zip` | 1,146 |
| `SepOKComputer_Project_Gap_Closure.zip` | 1,136 |
| `Prize_Research_Continuation_Audit_20260917.zip` | 685 |
| `Prize_Research_Phase11_2026-09-16.zip` | 620 |
| `Prize_Research_Phase04_2026-09-16.zip` | 468 |

Query with `python3 tools/drive_index.py archive <carrier substring>`.

---

## 11. What is explicitly not claimed

* No unconditional all-small-r 2D theorem.
* No closed two-sided 2D law; the 2D and 3D tracks are never composed.
* No sharp limiting constant `C`; every finite decimal for `C_Q0` has a kill.
* No numerical `C*` in Theorem B, whose unconditional status is retracted (`docs/OPEN_PROBLEMS.md` §G). The reviewed-scope `d = 3`, `L = 24` coefficient `c_(3,24)` of `LS-CLS-077-v1.0` ("CLOSED AT REVIEWED SCOPE", "Canonical impact: NONE BY ITSELF") sits on the 3D track and is not `C*`; the planar `C24` candidate in the Theorem B READ_FIRST carries `[[REVIEW:DEFERRED]] [[CANONICAL:UNCHANGED]]`. Neither is composed with anything here.
* `D3-LEMMA-RN-UNIF` is not closed.
* No original prize problem is solved.
* No external release is approved.
* Same-provider review earns **zero** organizational-independence credit.
