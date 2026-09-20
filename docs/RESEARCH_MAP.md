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
| `04_ACTIVE_GOVERNANCE_AND_DIRECTIVES` | 332 | the Drive's own description: "Rules, protected decisions, navigation, autonomous control, triage, and operational receipts" — the directives folder (64 items, incl. the R17 release bundle and OP-RECON-20260916 set), the open-questions folder (75), the self-healing / governance-history receipts (45), the evidence-gated autonomous-decision queue (41), the cross-line work-order inbox (23), the protocol standards and reference implementations (22), G0 (19), the navigation aids (17) and ~30 loose GP/CL records. It holds no registers: the live GP-REG-032 workbook is in `14_COORDINATION_AUTOMATION_SPINE` (§9). Until 2026-09-19 this row said "protocols, registers, navigation, decisions" |
| `90_QUARANTINE_AND_TRIAGE` | 191 | non-authoritative |
| `05_FOUNDATIONS_AND_PROTOCOL_HISTORY` | 151 | the Fresh Start 2.0 governance artifacts of 2026-07-29 (charter — mirrored byte-exact —, positive-admission allowlist, query/retrieval policy, the 17-file legacy quarantine manifest, contamination ledger, rebuild queue, execution report "PARTIAL — BLOCKERS REMAIN") and the historical governance records (the OP-PROT-002…010 lineage); R17 labels the old control documents history |
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

**How much of that this repository holds** is measured per lane in
[`drive/MIRRORS.md`](../drive/MIRRORS.md), generated from the manifests'
own Drive paths by `tools/mirrors_index_check.py`. At 2026-09-20: of the
inventory's 4,456 items, 1,813 are held byte-for-byte and a further 985 are
indexed tree-only — 40.7% held, 62.8% held or indexed. The three sub-lanes that
had been at zero of either — `12_P1.1_LAW_SPECIFIC_Q_MACHINE` (145 items),
`10_AXIOMATIC_CORE_SPINE` (41) and `11_P0.2_ADJACENCY_TRANSIT_TREE` (19), 205
items in all — were ported the same day and each now reads `neither = 0`.

The 1,658 items in neither column are not one kind of gap, and the index splits
them by reading each inventory record's own fields: **472 are folders**, which
have no payload for any manifest row to hold or index; **1,186 are native Google
Docs** the corpus declares no digest for anywhere, so the most this repository
can ever hold of one is a reading copy at `exact: false` — a text export, not the
object; and **none carries a digest**. That last number is the only one of the
three that measures work this repository could do and has not done, and after six
ports on 2026-09-20 it is **zero**: every item in the 4,456-entry inventory for
which a payload digest exists anywhere in the corpus is now either held
byte-exact here or carries a manifest row stating why it is not. All 26 Drive
lanes are at zero. What remains uncovered is what cannot be held byte-exact at
all — folders, which have no payload, and native Google Docs, for which the
corpus declares no digest to prove a copy against.
`99_DO_NOT_OPEN` (6 items) is at zero deliberately and
permanently: it is metadata only and is never opened.

Both paragraphs were corrected six times on 2026-09-20, because six ports landed
that day. The first gave 700 held and 873 indexed, named those three
sub-lanes as at zero coverage, and printed the remaining 2,883 as a single number
with no statement of what kind of gap they are; the second gave 861 held, 917
indexed and 1,020 digest-bearing, before the KIMI/AO48 LB-RATE intake's 330
digest-bearing items — 311 stored and 19 indexed — took that lane to zero; the
third gave 1,172 held, 936 indexed and 690 digest-bearing, before the reviews and
closures lane's 235 — 215 stored and 20 indexed — took that lane to zero too; and
the fourth gave 1,387 held, 956 indexed and 455 digest-bearing, before the
carry-forward canon's 174 — 172 stored and 2 indexed — did the same there; and the
fifth gave 1,559 held, 958 indexed and 281 digest-bearing, before the peer-review
packages, the HOLD lane, the SIDE24 ratification intake and the coordination
spine took theirs to zero together; and the sixth gave 1,789 held, 985 indexed
and 24 digest-bearing, before the last two dozen files closed the gap entirely.

The largest sub-lane, `15_REVIEWS_RESPONSES_AND_CLOSURES`, is where the
program's terminal closure records live: the documents behind every
`closure_log` row (`GP-CLS-141`, the `GP-CLS-BATCH-129` and `-137` batches,
`GP-PRP-130/131/132`, the observer relays), the twenty independent cold-review
packets of 2026-07-27 ("Terminal Exact Objects … Assign one packet to one
researcher"), the P0.1 post-ratification raw packages and the SIDE24
post-ratification theorem package, the Terminal Replication Capsule standard
chain, the PKG-SIDE24-001 external peer-review capsule (3D track), and the
Theorem B retraction-and-repair folder whose READ_FIRST controls Theorem B's
current status (§1). `drive/mirrors/15_REVIEWS_RESPONSES_AND_CLOSURES/` holds
the cold-review packets and the theorem package byte-exact and, as reading
copies (`exact: false`, no payload digest exists for a native Doc), the Theorem
B folder, the terminal closure records, the gate documents, the root documents
and the Theorem B registers (landed 2026-09-19). Until 2026-09-18 this table
gave the lane a row count and nothing else.

`16_THEMATIC_RESEARCH_TRACKS` (T1–T5) and `13_POWER_PLANNING_CONTINUUM` hold the
workspace charters, the T1 closure-and-reduction packages, the T2 SARD-G
working source and the 03.x leaf and live-state cards. Both lanes were ported on
2026-09-20: the thematic lane now holds 26 objects across 13 directories (24
byte-exact, two reading copies of native Docs), including all five workspace
charters, the T1 core pairing manuscript, the ten packages of its
closure-and-reduction directory, the T2 transversality manuscript with the chart
atlas and finite-jet certificate, the T3 continuum-validation adjudication and machine report, the T4
adjudications and the T5 extension manuscript; the power-planning lane holds 24
objects across its three 03.x directories (21 byte-exact) with 6 further items
indexed tree-only. Counts are generated per lane in
[`drive/MIRRORS.md`](../drive/MIRRORS.md). Until 2026-09-20 this paragraph read
"One object is held so far" and said the charters' fences and the leaf cards'
withdrawn claims were "not yet transcribed here"; both lane READMEs now quote
them, and each lane README states what it does not establish.
Note the name collision: the RN-UNIF lane's "T4 push" (§3, `LANE_RN_UNIF.md`)
is unrelated to the thematic track T4 (Lower Side). The `99_DO_NOT_OPEN` vault
(6 inventory items: the folder and five native Docs) sits inside
`01_ACTIVE_RESEARCH_PACKAGES` and is metadata only, never opened; the manifest
that describes it from outside (`00_DO_NOT_OPEN_MANIFEST`, at the lane root) is
held as a reading copy under `drive/mirrors/01_ACTIVE_RESEARCH_PACKAGES — ROOT
(the vault manifest)/` since 2026-09-19.

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

* **Theorem (1)** — certified rung `r = 0.05`: `1 − q(0.05, 6/5) ≤ Ĩ_hi +
  C_RN(0.05)·√Q(B1.dir) + P(B2) + P(B4)` with `Ĩ_hi = 8.1272827e-2 =
  650.1827·(0.05)³` and `C_RN(0.05) ≤ 3.46` (v2.2 §1). The H5 chart total it
  consumes is `I_hi(v3) = 8.0975589252e-2 = 647.8048·r³` (pin `8d7028e4…`);
  until 2026-09-19 this bullet gave that input as the theorem.
* **Theorem (2)** — all-small-r, conditional on the five premises in the README.

**Layer discipline (do not collapse):**

1. Frozen v2.2 → five OPEN validity premises; PKG-01 ships here.
2. Register note + addenda → append-deltas effective at next issuance: B4LOC
   CLOSED and B4.rem ADJUDICATED YES (whole-B4), PERC "CLOSED in its RESTATED
   form" with PD-CONN to the REFINEMENT register, normalizer DISCHARGED, BRANCH
   demoted. The OPEN set becomes `OBL-D1-PROMOTE` (chart side **and now also
   the uniform-band extension of the B4LOC/B2-far/far-route certificates**) +
   `D3-LEMMA-RN-UNIF` — reduced on one side, enlarged on the other (note §5).
3. `D1_ASSEMBLY_v2_3_DRAFT` + `H5_ZBAND_CONSUMPTION` → **PROPOSED**, authority
   none until operator promotion.
4. HOLD checklist → still lists **all five** as blocking.

### H5 rung ladder (chart side)

| Rung `r` | Artifact | `I_hi/r³` | Note |
|---|---|---|---|
| 0.05 | frozen v1 / live v3 | 731.4311 / **647.8048** | v2 was contaminated by a rung-unscoped merge; v3 CLEAN, sha `8d7028e4…` |
| 0.035355 | `H5_RUNG3_2026-09-15` | **661.4712** | totals sha `808d6901…` |
| 0.025 | `H5_RUNG2_2026-09-15` | **664.3979** | totals sha `f7697bcf…`; C1 containment PASS; mutation 6/6 |
| 0.0177 | RUNG2 ladder | — | cells **42/70** at the 2026-09-15 snapshot; v2.3 DRAFT: "FROZEN mid-flight (125 + 6 banked)", Kimi dark until 2026-09-30 |
| 0.0125 | RUNG2 ladder | — | cells **21/70** at the 2026-09-15 snapshot; v2.3 DRAFT: "FROZEN mid-flight (42 + 2 banked)" |

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
  binding. Do not repeat obsolete E0/E2 or D0/I0 requests. The registers carry
  two layers for P0.1 and both are transcribed: the Easy Closure Queue row
  (2026-07-30) still names `GP-DER-118-v1.9 / GP-DATA-236` ("EXACT BODY 28,257 B
  SHA-256 d894c0fe…"; "CARRIER-B v1.1 CONVERSION, INTEGRATED v1.9 CONVERSION, AND
  THIRD-FAMILY D0/I0/PZ0 OPEN / NONTERMINAL / NOT PROMOTED"), while the Autonomy
  Control row `P01_V110_CLASS3_ELIGIBILITY` (LS-CTL-003-v1.3, 2026-08-03) names
  the controlling `GP-DER-118-v1.10` ("frozen at 29,293 bytes / SHA
  9b7901e1…f014"; "ACTIVE / FOUNDATION TRUE / APPLICATION TRUE / HOLD / NOT YET
  ELIGIBLE"). The newer control layer governs routing; neither closes P0.1.
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
* **P1.1 law-specific q machine** — the active machine root is `q0_machine.json`
  (5,910,703 B, `c3a93bd2…`) + `q0_verify.py` (957,321 B, `eca1755d…`), which
  the register calls "authoritative and untouched" (artifact_index row P1.1:
  "No canonical impact; active q0_machine.json and q0_verify.py remain
  authoritative and untouched"). The labels "NONCANONICAL q0-law-specific/1.2
  FAIL-CLOSED PASS / 12-OF-12 NEGATIVE TESTS / INDEPENDENT REVIEW OPEN" belong to
  the additive **1.2 successor** (`GP-REG-188-v1.0` "NONCANONICAL / STRUCTURALLY
  VALID / ZERO ROOTS PROMOTABLE"; `GP-DATA-188-v1.0` "EXECUTED PASS / 12-OF-12
  ADVERSARIAL TESTS"; GP-AUD-188, 2026-07-24), and the earlier repair snapshot
  `GP-REG-051-v1.0` is stamped `[[STATUS:SUPERSEDED-REPAIR-CANDIDATE]]
  [[ACTIVE-MACHINE:UNCHANGED]] [[SUCCESSOR:1.2]] [[NO-PROMOTION]]`. The 1.2.1
  hardening package `LS-DATA-013-v1.0` ("NONCANONICAL SUCCESSOR CANDIDATE / EXACT
  ROUNDTRIP PASS"; source SHA `5202c5fa…`) is the review route RV-DQ-017's exact
  object — its 25,843-byte source is an archive member of two 2026-09-17
  carriers, not the 957 KB active verifier (see `recovery/LEDGER.json`
  ENB-04-S1) — and the dashboard's current word is "q0 verifier — 1.2.2
  CANDIDATE; ACCEPTANCE OPEN" (OQ-012: "Round 2 repaired eight malformed-shape
  crashes; 21 original +13 shape +2 CLI cases pass. Distinct-provider review and
  adoption gates remain."). Until 2026-09-19 this bullet attached the
  successor's labels to the active root.

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

## 7a. LIFETIME3D — the SIDE24 3D track: ratified at stated scope, never composed

The one result in the corpus whose register status is a ratification is on the
**3D track**, and it is the track the standing firewall (§11, `FW-2D-3D-SEPARATION`)
keeps apart from everything above. `operator_decisions` row `AO48-OPR-045`
(2026-08-02T16:10:54Z, subject `SIDE24_3D_RP_C_RP_S_THEOREM`, decided by Dylan
Roy) reads **RATIFIED-AT-STATED-SCOPE**, with the note "q0/P0.1 explicitly
unchanged". The record behind it (`AO48-OPR-045`, Drive
`1MfA94SaoYpnnAs7HYLNvowG9EHIR3Opk`, 3,397 B, `e48d7c27…`, mirrored byte-exact
under `drive/mirrors/2026-08-01-to-08-03 — SIDE24 RATIFICATION + P0.1
POST-RATIFICATION/`) is authored "Claude Opus 4.8 (Anthropic) — AO48, relaying
the operator", quotes the operator's one line ("I I Dylan Roy the human of this
the project sign. You may record it as the operator decision along with any
supporting changes.") and adopts the ratification text of `AO48-AUD-044` §D as
signed.

**What was accepted, at what scope** (the record's words): "the
compact-positive-mark elder-selection estimate sup_{t,b,κ}(1 − p_r(t,b,κ)) ≤
C r³, uniformly on compact (b,κ) subsets, for the normalized periodized
Bargmann–Fock field on the side-24 three-torus — completing the chain RP-A/RP-L
(Thm G.7.1), RP-F (G.8), RP-C/RP-S (V3.4 + envelope + audit chain) — and with it
the theorem ν₃,₂₄(ℓ) = c₃,₂₄ ℓ^(−1/3)(1+o(1)) with its closed-form constant and
10⁻¹⁸⁰ correction bound." **Carried dependencies, ratified as stated** (AUD-044
§C): the frozen V3.3 eigenfloor tables, the three absent V3.4 diagnostic
scripts ("non-blocking"), and the V3.3 Palm normalizer cr² ≤ Z_r ≤ Cr².
**Reopening conditions:** an exact counterexample to any audited display; failure
of a V3.3 eigenfloor table; a landed diagnostic script contradicting a
corroborated claim. **Evidence trail** the record names: `KIMI-AUD-006 → 006b
APPROVE` (the source calls it "independent third-family review"; the two Kimi
text carriers, `17c8eba9…` and `2a38f2d4…`, are on the Drive and not mirrored
here), `AO48-AUD-043` (both flagship displays confirmed from scratch in exact
arithmetic by the AO48 line) and `AO48-AUD-044` (verification ledger, including
the AO48 line's own boundary: it "did not independently re-read the envelope
file's full text this session"). Whether a Moonshot-family review earns
organizational-independence credit under R17 is the operator's determination;
the graph node records the provenance and assigns none.

**The firewall, in the record's words:** "this ratification concerns the SIDE24
3D track only. The 2D q0 program is untouched: P0.1 keeps every gate exactly as
LS-CTL-003-v1.1 states; LB-RATE remains measured-grade; no sealing, no release,
no cross-track inference." `ERRATA_AND_CLARIFICATIONS_2026-09-13` withdraws the
one composition that was ever attempted, and `tools/claims_check.py` fails the
build on any dependency edge between the tracks. The reviewed-scope coefficient
`c_(3,24)` of `LS-CLS-077-v1.0` ("CLOSED AT REVIEWED SCOPE", "Canonical impact:
NONE BY ITSELF") sits on this track and is not Theorem B's `C*` (§11).

The graph node is `SIDE24-3D-AO48-OPR-045` (grade token `RATIFIED_3D_ONLY`,
this graph's label for the register's word), bound field-for-field to the
register row, the inventory and the mirrored bytes by `tests/test_claims.py`.
The folder's `AO48-AUD-033` is a tree-only manifest row: two raw downloads
returned 10,155 bytes against the inventory's 10,154 and neither delta file
names the id. The lane's own README states that
`SIDE24_AUDIT_EVIDENCE_2026-08-01.zip` "is TRUNCATED / CORRUPT … DO NOT USE IT";
its five members are recovered byte-exact from mirror carriers in
`recovery/LEDGER.json`. Until 2026-09-19 this map described the 3D track only in
the firewall bullets of §11.

---

## 8. Prize reconnaissance (independent track, HOLD)

Lane `01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE —
INDEPENDENT TRACK` (295 items). Its status layer — the lane-root READ_FIRST, the
six grading files of `00_CURRENT_STATE_AND_ROUTING/` and the Erdős 142 scope
pointer — is mirrored byte-exact (inventory digests) under
`drive/mirrors/2026-09-16 — PRIZE PROBLEM RECONNAISSANCE — INDEPENDENT TRACK/`,
and since 2026-09-19 so is every other raw file of the lane (236 files in all;
the 19 zip carriers and the 3 quarantined items are tree-only rows and 9
byte-identical duplicates are pointer rows). Every grade
below is quoted from those bytes.

* **The lane about itself** — `CURRENT_STATE_VERIFIED_INTAKE.json` (as of
  2026-09-16): `"track": "independent prize research"`,
  `"current_primary_front": "Discrete Talagrand capacity-family subclass
  research"`, `"grade": "AUTHOR_SIDE_COMPLETE_ARGUMENTS; EXTERNAL_REVIEW_PENDING;
  NOVELTY_UNESTABLISHED"`, `"original_prizes_solved": 0`, `"q0_modified": false`.
  Its `forbidden_inferences` include "test counts are not independent
  confirmations" and "operator permission does not supply mathematical
  evidence". The claim graph files this lane under the track label
  `NUMBER_THEORY`; that label is the key the prize firewalls test, not a
  description — 53 of the lane's 72 proof notes are discrete Talagrand /
  hypergraph-threshold work.
* **The intake registry** — `CLAIM_REGISTRY_VERIFIED_INTAKE.json` grades 28
  claims: PR-AP-001…015, PR-SID-001, PR-COL-001, PR-SP-001 and PR-TAL-001…010.
  Twenty-six carry `declared_proof_grade: AUTHOR_SIDE_PROOF_PRESENT`; PR-TAL-009
  carries `SOURCE_DECLARED_AUTHOR_ARGUMENT` and PR-TAL-010
  `SOURCE_DECLARED_FINITE_CERTIFICATE_NOTE` ("source read and containing archive
  manifest; additional checker not replayed"). All 28: `external_review:
  PENDING`, `historical_novelty: UNESTABLISHED`, `original_prize_closed: false`.
  The registry's own note: "Scoped proof-note registry, not a count of novel
  publishable theorems; classical ingredients and reconstructed results are
  included." PR-TAL-011…019 are not in this registry; they are graded in the
  later phase registries inside the Phase06/07/08 carriers, which were not
  opened. Until 2026-09-18 this section said the registry "grades every PR-TAL
  carrier" at one grade.
* **PR-TAL-003…008** — the six Phase05 overlap notes (`verification_this_turn:
  "proof derivation and exact finite companions"`, author-side), sealed in
  `Prize_Research_Phase05_Overlap_2026-09-16.zip` (SHA-256
  `e976271ba1a3822bddf6247d72b5d6131bfeabdcbc610982e816d812dd2e5a90`, 98,175 B,
  **122 members**). The Drive holds two byte-identical copies of that zip (ids
  `1XB6HnA5…` under `90_FROZEN_PHASE_PACKAGES/PHASE05/` and `1Z4Rvtap…` under
  `EXTERNAL_REVIEW_PR_TAL_003-008/`) and the source map lists the members of
  each, which is where this section's earlier figure of "244 members" came from.
  "Sealed" is archive-custody language, not theorem grade. Bodies are
  `P05_01_five_copy_laminar.md` … `P05_06_high_capacity_overlap.md`. The folder
  name `EXTERNAL_REVIEW_…` records packaging for a review, not a review: every
  PR-TAL row reads `external_review: PENDING`.
* **PR-TAL-001/002** — Phase04 priors. **PR-TAL-011…018** — Phase06/07 with a
  known claim-ID collision across phases: always cite ID **plus** phase path.
  **PR-TAL-019** — Phase08 rank-3 source.
* **The four other sub-lanes** — `01_ERDOS_3_AND_169` (15 notes, PR-AP-001…015),
  `02_ERDOS_39` (PR-SID-001), `03_RIEMANN` (PR-SP-001), `04_COLLATZ` (PR-COL-001)
  and `05_ERDOS_142`, whose whole content is a scope pointer: "This lane has no
  new asymptotic formula or unrestricted upper bound for r_k(N). … A finite
  weighted maximum is not a cardinality asymptotic; an all-scale counting
  statement about weighted extremizers is not a bound on every progression-free
  set. The original target remains open in this project." Folder names such as
  "MILLENNIUM MOONSHOT" and "120M JPY MOONSHOT" are the source's aspirations, not
  claims. Until 2026-09-18 these sub-lanes were absent from this map.
* **P14** — `P14_CLAIM_REGISTRY.json`: `"grade": "AUTHOR_SIDE_COMPLETE"`,
  `"external_independent_reviews": 0`, `"historical_novelty": "UNESTABLISHED"`,
  `"prizes_solved": 0`. P14-A: "phi-compatible ceil(408kappa) cover under
  verified strict scalar sandwich"; P14-B: "exact fractional row cover supplies
  kappa; weak duality and directed rational repair"; P14-C: a many-necessary-rows
  family of "fractional width exactly2"; P14-D/E: `K_(m,m)` "has unbounded
  fractional and optimized scalar widths despite a proper2-coloring. A universal
  bounded-width inference is therefore false." The lane root: "The unrestricted
  Talagrand problem is still open in this project."
* **P15** — `P15_CLAIM_REGISTRY.json`: P15-A…D each `"grade":
  "AUTHOR_SIDE_COMPLETE"`, `"external_reviews": 0`, `"prize_closed": false`;
  file-level `"automatic_scientific_promotion": false`,
  `"external_independence_credit": 0`, `"q0_changes": 0`. P15-A translates the
  optimized scalar-sandwich parameter to the established critical threshold
  `α(D)` ("an attributed translation, not a new parameter") and obtains its
  palette bound "through the preceding P14 theorem"; P15-B gives a setwise
  fixed-label palette-localization certificate; P15-C an even-cycle family with
  global threshold exactly `m(3s−2)/(2s)` growing unboundedly; P15-D a fixed
  `256·ceil(408κ)`-piece compatible cover independent of macro degree. "No q0
  uniformity obligation is discharged." **P15 depends on P14**: the repository's
  own nonauthor reviews (`REV-P15-B`, `REV-P15-D`) say P15 "does not establish
  P14-A", and `claims/graph.json` now records P14 as a node and the P15 → P14
  edge; until 2026-09-18 it recorded P15 with no dependencies and no P14 node.
* Phase carriers, member counts from `drive/source_map/Archive_Members.csv`:
  Phase02 (237), Phase03 (342), Phase04 (468), Phase05 (129), Phase05 Overlap
  (122; two Drive copies), Phase10 (272), Phase11 — **two distinct carriers under
  one title**: the stored snapshot `860411bc…` (257,311 B, 436 members) and the
  `ATTACHED_ee4a16e1 — DISTINCT_AUTHOR_SNAPSHOT` (130,779 B, 184 members), which
  the source's continuation audit keeps apart ("Neither snapshot was overwritten
  or conflated") and which prove different constants — P14 Fractional (191),
  P15 Palette (149), Continuation Audit (685). Until 2026-09-18 this list
  summed the two Phase11 carriers into "620".

**No original prize problem has been solved.** Keep this on every public surface.

---

## 9. Platform and verification architecture

* `01_RESEARCH_PLATFORM_AND_VERIFICATION_ARCHITECTURE` (188 items, 130 of them
  native Google Docs with no payload digest) — `RM-METRICS-001` (metrics and
  defect interception; its §5 lists eleven named defects as the program's
  "initial regression corpus"); the Fresh Start 2.0 lane
  `02_FORMAL_AND_LEAN_RESEARCH_SYSTEM` (178 items), which its `00_READ_FIRST` and
  `FS2-SETUP-001` describe as an "INSTALLED / ACTIVE CONTROL PLANE" with the
  state machine `INGESTED → PREPPING → CANDIDATE → REVIEWED → CORE`, promotion
  gates PG-01…PG-10, and eight Core capsules `EC-014`, `EC-015`, `EC-021`,
  `P02-LM-001/002/005/007/008` (each a folder of `CURRENT_STATUS.json`,
  `statement.md`, `proof.md`, `independent_review.json`, `hostile_tests.py` and
  a review-packet PDF, to be read "CURRENT_STATUS.json before historical proof
  banners"; each records `parent_theorem_effect: NONE AUTOMATIC`); a
  `04_EVIDENCE` tree whose five output folders are empty in the snapshot; and two
  verification prototypes with register rows — `GP-VO-001` / DQ-060 ("COMPLETE /
  SANDBOX PROTOTYPE PASS / LIVE MIGRATION HOLD", dispatch "HOLD — DO NOT CLAIM")
  and DQ-059 ("PASS / COMPLETE / SANDBOX RELEASED"; "Scientific effect: none").
  **The lane holds zero `.lean` sources and zero prover receipts**: "LEAN" in its
  title is the lane's name (the program's Lean-prover material is `GP-FOR-001`,
  `GP-FOR-189` and `GP-REC-220`, below), and its formal content is Markdown
  statements, sympy hostile tests and PDFs. This repository indexes the lane
  (`drive/inventory.jsonl`) and nothing more: it carries none of its verifiers
  (`verify_obligations.py`, `math_integrity_gate.py`, the eight
  `hostile_tests.py`, the Apps Script executor `FS2_PCT003`), none of the
  capsules' operative status labels, and no git equivalent of its control plane
  (`governance/GIT_ADAPTATION.md` says so). Until 2026-09-18 this bullet
  described "regression corpora and negative controls" and "a replication tree"
  that are, in the snapshot, one two-item conversion-control folder and an empty
  output tree. The formal state, as the FORMALIZATION BOARD (Drive
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
* `14_COORDINATION_AUTOMATION_SPINE` (170) — the live registers
  (`04.1_LIVE_REGISTERS`: the GP-REG-032-v1.2 workbook, its leaf card, the
  deletion log), the automation lineage (GP-AUTO-034: **R0.4 current
  development candidate, static T1–T35 + 4 fixtures pass, all write targets
  explicit, isolated sandbox prepared, runtime NOT executed, DO NOT INSTALL**;
  the Run Log's 32 rows are receipts of manual connector sessions and closure
  watches and say so — "Automation not yet installed"), and, which the earlier
  one-line description hid, `04.3_CROSS_MODEL_LEDGERS`: the canon's promotion
  channel `Q0_LEDGER.md` (253,067 B, `d1fea170…`; cited by digest in §1 and held
  nowhere in this repository), the Cross-Model Review Ledger v1.1 and
  contribution registry, 33 digest-bearing C091–C108 cycle ledgers and
  manifests, the 2026-07-24 COMMAND CENTER (the fullest account of the
  automation lineage and of the Git-repository operator status), fifteen
  2026-07-25 cross-line status deltas, CL-REQ-225's standing automation
  protocol and the DQ-018 register-safety evidence. 44 digest-bearing objects
  (494,416 B) of this lane have no byte-exact port; the promotion channel is
  among them.
* `07_MODEL_ACCESSIBILITY` (new, 2026-09-17) — a five-part tree: a START HERE
  Sheet, three reading-copy folders holding the 392 published reading copies,
  and an audit folder of ten published files (`ACCESSIBILITY_COMPLETION_REPORT.md`,
  `EXTERNAL_RECON_ACCESSIBILITY.md`, `ACCESSIBILITY_VERIFICATION.json`,
  `Start_Here.csv`, `Files.csv`, `Reading_Links.csv` — the 6,034 source-to-copy
  links —, `Archive_Members.csv`, `Payloads.csv`, `Exceptions.csv` — the 137
  retained exceptions —, `Reading_Copies.csv`). `drive/source_map/` holds four
  of the ten (`Files`, `Archive_Members`, `Payloads`, `Exceptions`) and
  `drive/inventory.jsonl` is derived from `Files.csv`; the Sheet, the 392
  copies, `Reading_Copies.csv`, `Reading_Links.csv`, `Start_Here.csv`, the
  verification receipts, the completion report and the recon memo are not
  here. Until 2026-09-19 this bullet said the lane was "mirrored here".

The register's own operation view is worked actively on the git side. The
2026-09-18 export added `reusable_operations` (sheet 42: fifteen operations
OP01–OP15, all `Utility: UNMEASURED` and `Novelty: NOT_ASSESSED`) and an
intentionally empty `operation_trials` ledger (sheet 43), with the Drive's
guide mirrored under `drive/deltas/2026-09-18/2026-09-18_REUSABLE_OPERATIONS/`.
`engine/operations/REGISTRY.json` transcribes the fifteen cell for cell, and
marks separately (`git_side`) the four whose `Action / output` cell displays an
identity that exact arithmetic reproduces from the cell alone — OP02's power
ledger `(r^3/6) r^-5 r^2 = 1/6`, OP03's pushforward constant and exponent
`6^(2/3)/(3 κ^(2/3)) ℓ^(-1/3)`, OP04's Hölder conjugacy `1/4+1/4+1/2 = 1`, and
OP05's witness `LHS = 1/4`, wrong `RHS = 1/16` — and says why the other eleven
are not machine-checkable here (a statement about this repository's reach, not
about the operations). `engine/operations/trial.py` runs those four over
`fractions.Fraction` and appends one record per run, in the ledger's own
eighteen-column shape, under `engine/operations/trials/`; `tools/operations_check.py`
holds the registry to the register's words (never stronger on Utility or
Novelty, in a cell or in any sentence the repository wrote), holds every trial
to the closed result vocabulary, to the register's `Do not infer`, Utility and
Novelty cells verbatim, to no word from the status-word list and no
usefulness word without the register's `UNMEASURED` / `NOT_ASSESSED` (word
lists, not a reading of the sentence), to `Split`, `Arm` and every cost column
agreeing with the verdict, to a step-for-step re-run of its recorded
arithmetic against the catalog version it names, to `NOT_RUN` for a
non-checkable operation (and never for a checkable one), to the runner's fixed
`does_not_establish` sentences verbatim, to a run time no later than the
checker's clock nor than the commit that added the record, and to
byte-identity with `git HEAD`; the registry's own authority sentence,
`does_not_establish` and provenance blocks are pinned the same way. A trial is
a record of a computation, not evidence: it measures no utility, assesses no
novelty, and moves nothing.

---

## 10. Archive carriers

The accessibility completion report counts 77 original archive carriers, and the
number is recomputable from the committed files: the inventory marks 76 items
`ARCHIVE_INDEXED`, and `drive/source_map/Archive_Members.csv` lists 11,649 member
rows for 72 of them plus one carrier the inventory marks `BINARY_UNRENDERED`
(`S2-DATA-002-v1.0_result_carrier.zip`, `1mYHVSdVk57CM9h6L3NFR9_G2EFXKPwhj`);
the four `ARCHIVE_INDEXED` items without member rows are single-file `.gz`
uploads. Those 73 carriers with member rows have 69 distinct titles — four
titles are each shared by two carriers: the Phase05 Overlap copies, the two
Phase11 snapshots, `SIDE24_G9_R2_RECOVERY_AND_RECONSTRUCTION_2026-08-01.zip` and
`research_formal_core_r2.zip` — and 2,974 distinct member payload digests (the
five `READ_FAILED` members carry an empty digest cell, which until 2026-09-19 was
counted as a 2,975th). The
4,020 distinct payloads of `Payloads.csv` are counted over the whole source map,
not over the archives; until 2026-09-18 this sentence presented them as archive
payloads. Largest carriers by member count (a title shared by two carriers is
listed per carrier):

| Carrier | Members |
|---|---:|
| `09152026OKComputer_Project_Gap_Closure.zip` | 2,029 |
| `SepOKComputer_Project_Gap_Closure.zip` | 1,136 |
| `Prize_Research_Continuation_Audit_20260917.zip` | 685 |
| `SIDE24_G9_R2_RECOVERY_AND_RECONSTRUCTION_2026-08-01.zip` (one carrier, two byte-identical Drive copies: ids `1Uz1xeyn…` and `18PdDtAi…`; until 2026-09-19 this row summed them into 1,146) | 573 |
| `Prize_Research_Phase04_2026-09-16.zip` | 468 |
| `Prize_Research_Phase11_2026-09-16.zip` (stored snapshot `860411bc…`, id `1ef2RycJ…`) | 436 |
| `Prize_Research_Phase11_2026-09-16.zip` (`ATTACHED_ee4a16e1 — DISTINCT_AUTHOR_SNAPSHOT`, id `15CXT-wK…`) | 184 |

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
