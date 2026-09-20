# Open problems and the exact next actions

Everything below is copied from the sources' own "next exact action" language.
No item here is a new mathematical claim.

## A. Blocking the unconditional 2D upper theorem

### A1. `OBL-H5-JETMOD` — certified 24-jet band enclosure (chart-side lead)

**Obligation.** Certified interval bounds for the full 24-jet set — not just the
displayed `c₂` — over the r-bands `[r_{k+1}, r_k]`, with lattice-tail constants
re-certified **uniformly in the band** (the current LAT tail bound certifies at
point separations only). Content: for each jet `J` and band `B`,
`J(B)/r^{p_J} ∈` a certified interval.

**Proof step named by the source.** Evaluate the lattice sums with `r` as an
interval over the band. That yields G12-band enclosures, hence
`Î(r)/r³ ≤ F(G12-band)` for the whole band — a finite computation per band,
never a fitted exponent.

**Falsifier.** A band enclosure whose width exceeds the claimed modulus.

**Not a substitute.** RUNG2 (`r = 0.025`) and RUNG3 (`r = 0.035355`) certify
those rungs only.

*Repository state (code, not status):* `research/bands/` holds the machinery
this proof step needs — interval-`r` lattice sums with a tail bound proved
uniform over the band, the falsifier, and the ladder analysis — exercised on
reference kernels only. What it still lacks to bear on the obligation is named
in `research/bands/README.md`: a certified decay envelope for the program's
`kplane` at every order the 24-jet set reaches, the 24-jet definitions with
their powers `p_J`, the actual band endpoints `r_k`, and the six-pin
`r`-to-displacement geometry. `OBL-H5-JETMOD` is OPEN (display only).

### A2. Finish the rung ladder (engineering, not premise discharge)

RUNG2's 2026-09-15 snapshot put the cells at `r = 0.0177` at **42/70** and at
`r = 0.0125` at **21/70**, probes and patches complete; no later named `N/70`
exists. The lane's later state line records the rungs as **frozen, not
resuming**: `D1_ASSEMBLY_v2_3_DRAFT.md` (mirrored byte-exact) reads, under
"Running tasks at issuance (Kimi dark until 2026-09-30; states as of the
2026-09-15 13:11 snapshot)", "H5 rungs 0.0177 / 0.0125 — FROZEN mid-flight (125
+ 6 / 42 + 2 banked)". Raw JSONL line counts are not a status promotion, and
the sources themselves call this engineering. Until 2026-09-19 this section
said "two shards resuming each" and named no freeze.

### A3. `OBL-H5-ZBAND` hi side

The lo side rides the frozen H3 uniform certificate. The hi side is the H3
band **ceiling**. `H5_ZBAND_CONSUMPTION_2026-09-15.md` (Drive
`18R0wwSFHa--ZMdLRV3ElMO0T5u5YD3lk`, mirrored byte-exact) carries "STATUS:
PROPOSED · AUTHORITY: none" and states "OBL-H5-ZBAND: OPEN → **DISCHARGED
(consumption grade)**"; its hi-side row reads "`h3_band_ceil.py` b97c5428… →
`ceil_normal.txt` 26d08534… | E[G_r] = Z_r/r² ≤ 3.74767948915996". The sentence
"On the band: the two-sided normalizer is now bracketed uniformly" with the
bracket `2.30659559567154 ≤ Z_r/r² ≤ 3.74767948915996` on `(0, 0.05]`, and the
citation of `H3_closure/H3_BAND_CEIL.md` (body `cfe8a3a4…`; `ceil_normal.txt ≡
ceil_O.txt`), are the **v2.3 DRAFT**'s (`D1_ASSEMBLY_v2_3_DRAFT.md`, Drive
`1RQE2B3EY9IP5MGZyeAtXBpX4ahY5AhfC`, §0 table and §1), not the consumption
note's; the band table `docs/RESEARCH_MAP.md` §H3 cites is that bracket. The
discharge is **PROPOSED**, authority none, not operator-promoted over frozen
v2.2; the hi side stays OPEN. Until 2026-09-19 this section attributed the
DRAFT's sentences to the consumption note under the DRAFT's Drive id.

### A4. `OBL-H5-REMOTE-THRESHOLD`

Certify D3's remote bracket at the r-scaled threshold `d ≥ 2r`, or extend the
chart's machine cover to absolute `d₀`. Rides `D3-LEMMA-RN-UNIF`.

### A5. `D3-LEMMA-RN-UNIF` — the RN uniform lemma

* **Piece 1** OPEN. **Piece 2** OPEN — the annulus Riemann-sum driver is
  **unwritten**. Schedule it explicitly; do not hide it under a T4 push. ("T4"
  here is the RN-UNIF lane's T4 push region `d ∈ [5, 17]` from `LANE_RN_UNIF.md`;
  it is unrelated to the Drive's thematic track `T4 — Lower Side (Conditional,
  Marked-Repulsion Gate)` under `16_THEMATIC_RESEARCH_TRACKS`, which is a
  lower-side track and never filed under this lane.)
* **Next exact action from RN5:** build a complete non-overlapping spatial cover
  of `0.1 ≤ |y| ≤ 5`, retaining boundary-area bounds and every rejected cell;
  sum area × corrected cell supremum; verify no cell remains pending; then
  reassemble the remote budget. Treat the near-axis refinement cost explicitly —
  the present ten boxes are **not** a coverage certificate.
* Preserve the fixed-`r`, fixed-axis scope. All-small-`r` extension needs its own
  evidence.
* Prefer the ~9.7 KB `rnu_ds3.py` carrier; the 7.2 KB duplicate is SUPERSEDED.
* Build the whitened `env_form` orders 2–4 runnable smoke (currently missing);
  order-1 chi-squared white is a nearest neighbour only.

*Repository state (code, not status):* the annulus driver recorded as
unwritten now exists at `research/cover/`, with the accept / refine / reject /
pending ledger as its first-class output and `total()` refusing to return
while any cell is pending. Two exact facts it established about the recipe
above: the acceptance rule's cell-area factor cancels, so a zeroth-order sup
cover imposes a uniform oscillation bound per cell wherever the cell sits —
that is the near-axis refinement cost stated in one line — and a uniform
cover of `0.1 ≤ |y| ≤ 5` at Cartesian cell diameter 1/10 needs exactly
98 × 629 = 61,642 polar cells, the 50× inner/outer anisotropy being
`r_hi/r_lo` exactly. For the T4 region `d ∈ [5, 17]`, θ-halving alone cannot
tighten a radial integrand's enclosure past the shell's radial floor, however
much depth is granted. Every integrand exercised is a labelled reference; no
cell of the program's actual cover is certified, and the frozen engine
`engine/rn_engine/frozen/` is `mpmath` throughout. Both Pieces remain OPEN.

The new [SIDE24 point-law candidate](RN_SIDE24.md) separately assembles the
normalized field's derivative covariance at `y=(1,1)`, conditions on the exact
nine pins using positive interval LDL pivots, and replays the `M4/S4/y2`
moment certificates over the complete mark interval. The conditional
determinant factor is at most `1830559/250000000 = 0.007322236` there. This
point calculation is extended by the [density/window candidate](RN_SIDE24_DENSITY.md):
six-pin gradient density and gradient-conditioned height-window bounds give a
pointwise RN integrand upper `1888043/500000000000 = 0.000003776086`, conditional
on the imported H3 floor. The floor is hash-bound but not reproved. Neither
point calculation supplies a spatial area contribution or turns the generic
cover into an RN cover.

### A6. `PERC-DECAY`

Frozen v2.2: OPEN. The `o(r³)` far-lane reading is not reachable; what the note
layer carries is the `Θ(r³)` restatement with certified constants plus
`PD-CONN`. At that layer `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` §5 (mirrored
byte-exact) lists under "**CLOSED:**" "PERC-DECAY in its RESTATED form (the
certified far-lane inclusions + caps of §3 — validity content at the rung fully
absorbed into the E_w accounting of §4)" and moves `PD-CONN` to the
"**REFINEMENT/constants register:** PD-CONN (named; constants-not-order;
missing pieces (i)–(iii))"; the v2.3 DRAFT's table reads "REFUTED-AS-PHRASED →
RESTATED → ABSORBED; premise 3 removed; PD-CONN to refinement (upgrade-only)".
Both are effective only at the next issuance and no operator has promoted
v2.3. Until 2026-09-19 this section carried only "RESTATED" and called
`PD-CONN` a "named OPEN input" without the note's REFINEMENT wording.

### A7. `OBL-B1-BRANCH(loop|B1)`

OPEN in frozen v2.2; demoted to REFINEMENT for v2.3. Constant-level.

### A8. `B4.loc` wrap/remote reconciliation

`B4LOC-R1` closed the dam line for the whole of B4 and resolved the
identification negatively (cut-net ≢ nine-pin tube). On the wrap/remote
reconciliation the register note is explicit: §2 "ADJUDICATION of the
reconciliation question (lead's item 3): **YES**", with the consequence
"**B4.rem is CLOSED by B4LOC-R1 for the O(r³) validity grade (indeed
super-algebraic at the ladder rungs); PD-CONN is upgrade-only everywhere**",
and §5 lists "**CLOSED:** the B4.loc dam-line tube certificate AND B4.rem
(B4LOC-R1, super-algebraic at the ladder rungs; whole-B4 per §2)". That is the
note layer, effective at the next issuance; frozen v2.2 still names the premise
OPEN and no operator has promoted v2.3. Until 2026-09-19 this section said the
reconciliation "remains open per the capsule", a sentence taken from a
2026-09-16 reading document rather than from the register note.

---

## B. The matching 2D upper

Target: same-law `1 − q(r, 6/5) ≤ C r³` for all sufficiently small `r`, with
**no** combination with the 3D upper. Required: exhaustive failure-event
coverage, exact weighted / height-integrated bounds, eta-preserving conversions,
all-small-`r` control and spatial-regime control. C020/C021 historical
candidates were supplied offline and are **not** promoted by that intake.

---

## C. LPW constant repair

Freeze either the corrected exact fraction or `6.238e−44`, with end-to-end
headline tests. Kimi must review the R05 Rayleigh amplitude lemma, the full
tails / profile modulus, the conditional bounds, the exact fraction, and the
actual headline mutation. The delivered `6.239e−44` certificate must not be
admitted unqualified. Qualitative LPW is unchanged by this.

---

## D. Review queue — 25 routes, 22 unassigned

(Refreshed 2026-09-18 from the xlsx export of the register. An earlier version
of this heading said "24 routes, all unassigned", then "24 routes, 23
unassigned"; `registers/json/review_queue.json` now holds 25 rows, of which 22
carry `Reviewer / claim = UNASSIGNED`. The three that do not: `RV-LM009-MAIN`
"OPS4 nonauthor / exposed / OpenAI"; `RV-LM004-MAIN` "Prior OpenAI technical
reconciliation; ROUND5 author-line erratum; zero org credit"; `RV-RN-ALIGN`
"ROUND5 / OpenAI / author-side". The register is the source; the heading
transcribes it.)

From `registers/json/review_queue.json`. Nineteen originated in July routing
records and are 52–55 days old; the R17 ladder therefore puts them at
**ESCALATE**, except the two whose technical pass is same-line
(`RV-LM004-MAIN`, `RV-LM009-MAIN`), which the register puts at **EXTERNAL
ONLY**: only an organizationally distinct verdict remains for them. Aging never
approves anything.

| Key | Exact object | Technical status | Body bytes / SHA-256 |
|---|---|---|---|
| `RV-LM003-MAIN` | LCR-DER-014-v1.0 | NEEDS_RECONCILIATION | 6,874 / `5173d26d…` |
| `RV-LM004-MAIN` | LCR-DER-016-v1.1 | **PASS_TECHNICAL** (same-provider; "zero org credit") | 10,416 / `d43179f3…` |
| `RV-LM006-MAIN` | LCR-DER-019-v1.0 | NEEDS_RECONCILIATION | 9,147 / `239ed094…` |
| `RV-LM009-MAIN` | LCR-DER-027-v1.0 | **PASS_TECHNICAL** (same-provider) | 3,919 / `ccc07d95…` |
| `RV-LM010-MAIN` | LCR-DER-030-v1.1 | NEEDS_RECONCILIATION | 9,335 / `05e0f20f…` |
| `RV-LM011-MAIN` | LCR-DER-033-v1.0 (final synthesis) | NEEDS_RECONCILIATION | 9,721 / `17c37fa1…` |
| `RV-LM012-MAIN` | LCR-DER-043-v1.1 | NEEDS_RECONCILIATION | 11,543 / `68df3208…` |
| `RV-LM013-BOREL-GLOBALIZATION` | LCR-DER-061-v1.1 | NEEDS_RECONCILIATION | 8,928 / `8ec4386f…` |
| `RV-DQ-005` | GP-DER/DATA-212 full-matrix base mass | NEEDS_RECONCILIATION | hash unresolved |
| `RV-DQ-017` | LS-DATA-013 q0 verifier 1.2.1 | NEEDS_RECONCILIATION | source `5202c5fa…` |
| `RV-DQ-020` | GP-DATA-218 guarded register write | NEEDS_RECONCILIATION | source `199b7ce5…` |
| `RV-DQ-023` | LS-DER-021 TB-G2 contact intensity | NEEDS_RECONCILIATION | next path must be neither OpenAI nor Anthropic |
| `RV-DQ-027` | TRC-EC020-v0.4 + GP-DATA-233 validator | NEEDS_RECONCILIATION | three Drive-title mismatches to adjudicate |
| `RV-DQ-035` | LS-DER-026 TB-G4 near-diagonal tail | NEEDS_RECONCILIATION | 11,725 / `e4d8094e…` |
| `RV-DQ-038` | LS-DER-022 TB-G3 selection reduction | NEEDS_RECONCILIATION | 21,932 / `e5a61739…` |
| `RV-DQ-053` | LS-DER-032 TB-G2 covariance interface | NEEDS_RECONCILIATION | 12,106 / `3531e6e5…` |
| `RV-DQ-057` | LS-DER-036-v1.1 fixed-compact Palm loop | NEEDS_RECONCILIATION | 16,856 / `3416ddee…` |
| `RV-DQ-061` | LS-DER-038 minus-one-third component | NEEDS_RECONCILIATION | 10,410 / `5fd2413c…` |
| `RV-DQ-096` | LS-REQ-037 review of LS-DER-064/065/067/068/069 | NEEDS_RECONCILIATION | 7,013 / `e5ae1b6d…` |
| `RV-RN3` | RN3 joint far-zone `r = 1/20` | **READY** | 12,956 / `0c9446b7…` |
| `RV-RN-ALIGN` | CL-RNU-003 ↔ RN3 crosswalk | **AMEND** (author-side; "ZERO ORG CREDIT") | RN3 `0c9446b7…` / CL `59b8f002…` |
| `RV-P15` | P15-A/B/C/D structural localization | **READY** | archive manifest required |
| `RV-H5-REPAIR` | H5 corrected-kernel full consumer replay | **AMEND** | 11 archive-member hashes |
| `RV-OPS-R17` | OP-PROT-019-v1.1 implementation | **READY** | 14,073 / `04987ba4…` |
| `RV-RN5-MOMENT-REPAIR` | RN5 corrected moment and local spatial-box candidate | **READY** | 13,725 / `ac89f60b…` |

Status words that moved between the 2026-09-17 and 2026-09-18 exports, each
the register's own word: `RV-LM004-MAIN` NEEDS_RECONCILIATION → PASS_TECHNICAL
(reviewer line "Prior OpenAI technical reconciliation; ROUND5 author-line
erratum; zero org credit"; Independence status unchanged at
EXTERNAL_REVIEW_OPEN; the withdrawn A1 finding is quarantined as
`Q-RN5-LM004-A1`); `RV-RN-ALIGN` NEEDS_RECONCILIATION → AMEND (Independence
status NO_CREDIT_ASSIGNED → "AUTHOR_SIDE / ZERO ORG CREDIT"; next action
"Correct near-moment claim and complete spatial cover; locate named CL v5
archive / rnu_t4.py / rnu_spine.py before executable crosswalk"); new route
`RV-RN5-MOMENT-REPAIR` READY, UNASSIGNED, EXTERNAL_REVIEW_OPEN, prior exposure
"OpenAI author-side proof/replay; 66 checks; no independent review yet". A
PASS_TECHNICAL is a same-line technical pass at zero organizational
independence credit. Nothing here is promoted.

`RV-LM011` is the synthesis: it needs LM003, LM004-v1.1, LM006, LM009, LM010-v1.1,
LM012-v1.1 and LM013 Carrier-B-v1.1 plus the joint stack **first**.

---

## E. Scope holds and quarantine

* **`Q-R17-H5-01..11`** — eleven frozen H5 archive members under a
  path-and-hash consumption exclusion pending full corrected-kernel replay. Bytes
  intact. No H5 certificate chain may consume signed odd-frequency or missing
  denominator-tail bounds without replay.
* **RN5 scope holds** — the `envelope_v` certification claim, its annulus/remote
  consumers, and CL-RNU-003 §3's integrand-certification and forecast claims.
* **`CANNOT_VERIFY`** — current CL executable and checkpoint identity
  (`CL_ANTHROPIC_BUNDLE_2026-09-17_v5.zip`, `rnu_t4.py`, `rnu_spine.py`).
* 137 accessibility exceptions retained, including 8 `EMPTY_NATIVE_BODY`,
  5 `READ_FAILED`, 3 `ARCHIVE_READ_FAILURE`, 15 `ENCODED_BLOCK_FAILURE`.
  Missing source contents were **not invented**.

---

## F. Register defects found in this migration

Reported, not repaired — the export is kept faithful. See
`registers/KNOWN_FINDINGS.json`.

* Six duplicate artifact IDs in the Artifact Index (`GP-DER-118-v1.2`,
  `GP-AUD-119-v1.0`, `GP-PRP-121-v1.1`, `GP-AUD-144-v1.0`, `GP-DER-143-v1.1`,
  `GP-REQ-144-v1.1`), each pair carrying different status text. OP-CNS-001 §2
  requires collisions to be preserved and disambiguated in an append-only
  collision registry.
* Seven duplicate Transition IDs in the Transition Log (`TR-P12-007`,
  `TR-P02-011`, `TR-P02-013`, `TR-P01-006`, `TR-P01-011` ×3, `TR-P01-012`) with
  different types, objects and timestamps.
* Three Quarantine Index rows use class `EXISTING_CONTAINER`, which
  OP-PROT-019 §6 does not define. Proposal: add `CONTAINER_POINTER` to the table.
* Artifact Index row `GP-PRP-130-v1.1` (Modified UTC 2026-07-22, Status
  "HUMAN CLARIFICATION REQUIRED") predates the operator notice the same
  document now opens with — "OPERATOR DECISION NOTICE — 2026-07-23 HA-008 adopts
  EC-019 Independence Option A … [[STATUS:OPTION-A-ADOPTED]]
  [[QUEUE:EC-019-HUMAN-READY]] [[NO_TERMINAL_APPROVAL]]" — and Closure Log row
  `GP-CLS-EC019-20260730` records the later closure. Reported by the 2026-09-18
  coverage audit (Drive `1_Q1___VfRoN9IHvG5ueXcalNLQ1e7A4ynVI7dxHx4pY`).
  Proposal: a successor row, not an edit.
* Closure Log row `GP-CLS-137-A` carries the terminal label without the article
  "THE" that the closure document's own "Terminal label:" line contains
  ("… ROUTES TO THE CONSOLIDATED Q0 CANON"). Semantically inert; recorded as a
  transcription variance (Drive `15pRBxEcL5iOHAtjK9TWrqDX8382Ifmf4SCcwk56rlVs`).
  `GP-CLS-141-v1.0` and `GP-CLS-BATCH-129-A..E` match verbatim.
* **The 2026-09-17 markdown export was itself a truncated rendering** (found
  2026-09-18): the connector's markdown-table rendering returned only a prefix
  of seven large tabs — `file_catalog` 310 of 2,952 rows, `activity_log`
  241/518, `artifact_index` 206/761, `transition_log` 67/82, `review_ledger`
  101/138, `evidence_lineage` 137/485, `relations` 164/367 — so every derived
  file built before 2026-09-18 saw only those prefixes. A defect of the
  rendering, not of the register; repaired by the 2026-09-18 xlsx export
  (`registers/source/SOURCES.json`). The markdown export stays byte for byte.
* Seven further duplicate keys surfaced in the rows that export never
  delivered, all defects of the source workbook and none of the importer:
  Artifact Index `GP-DATA-168-v1.1` (rows 231/240), `LS-AUD-002-v1.0`
  (248/253), `LS-COR-001-v1.0` (249/254), `LS-AUD-003-v1.0` (250/255),
  `GP-REQ-194-v1.0` (325/373 — two *different* review requests under one ID),
  `LS-MAN-045-v1.0` (544/545), and Evidence Lineage `EV-LS-REQ030` (385/386).
  Recorded in `registers/KNOWN_FINDINGS.json` (section
  `findings_first_visible_in_2026-09-18_export`) with the rows and status text;
  OP-CNS-001 §2 requires the collisions to be preserved and disambiguated, not
  merged. Not yet covered by `registers/collision_proposal.json`, whose source
  of record is the markdown export; a numbered successor proposal against the
  xlsx export is required (the proposal is not edited in place).

## G. Theorem B — status retraction and repair program

*Status (transcribed from `registers/json/automation_config.json`,
`THEOREM_B_CURRENT_STATUS`):*
`CANDIDATE_UNCONDITIONAL_PROVEN_HERE_RETRACTED_EXACT_JACOBIAN_PROVED_CONDITIONAL_B0_PROVED`
— "Do not cite Q0_C104 or Q0_MASTER historical PROVEN-HERE labels as current
proof; preserve exact Jacobian and conditional B0; five analytic bridges remain
open" (sources named by the row: GP-AUD-187; OQ-011; HB-043).

The retraction: `GP-AUD-187-v1.0` (Activity Log 2026-07-24T21:25:00Z,
"AUD / STATUS CORRECTION", label "CRITICAL SCOPE NARROWING — THEOREM NOT KILLED
/ NO PROMOTION") "retracted current unconditional `PROVEN-HERE` label after
primary-source audit; preserved exact Jacobian and proved conditional
compact-mark B0; opened five analytic bridges and seven repair tasks." The
controlling routing banner is the READ_FIRST the register names
(`THEOREM_B_CURRENT_READ_FIRST_ID` = `1GCxx8Th9C5J8SrddCjwNa30O2LuYiwEB4pEYLxXK5pg`):
"Canonical Theorem B remains RETRACTED / NOT RESTORED / NOT PROMOTED. Package
remains UNSEALED." What survives, in its words: "Conditional compact-mark
pushforward Theorem B0: PROVED under explicit assumptions A1–A5"; the
unconditional full-κ positive-coefficient Theorem B for the actual persistence
lifetime density is "CANDIDATE / CANNOT VERIFY / former PROVEN-HERE status
RETRACTED".

**Open gates** (`THEOREM_B_OPEN_GATES`, "named restoration obligations for any
future unconditional theorem claim; no status promotion until the complete
chain is proved under one consistent exact object and independently reviewed"):
`TB_G1_MULTIPLICITY`, `TB_G2_CONTACT`, `TB_G3_SELECTION`, `TB_G4_TAILS`,
`TB_G5_OFF_FOLD`, `TB_G6_INDEPENDENT_RECONSTRUCTION`,
`TB_G7_CONTINUUM_FALSIFICATION`. The same-family successor
`LS-DER-040-v1.0` is frozen as "ACTIVE GLOBAL-BIRTH COMPLETE FINITE-TORUS
THEOREM CANDIDATE / REVIEW DEFERRED" (Frozen Objects). The help-board request
`HB-043` (P0) asks for the missing analytic bridges to be proved or falsified.

**Next exact action (the register's own dispatch, `THEOREM_B_TBG1_DISPATCH`):**
`DQ-009` — "Direct exact candidate-process and multiplicity derivation" for
TB-G1; "Does not restore theorem status."

**Not a substitute, and not composed:** `LS-CLS-077-v1.0` closes a *reviewed-scope*
absolute coefficient `c_(3,24)` for the `d = 3`, `L = 24` field ("MATHEMATICAL
SIDE-24 COEFFICIENT CHAIN CLOSED AT REVIEWED SCOPE"; "Canonical impact: NONE BY
ITSELF"; "Not closed or implied: other L; arbitrary Gaussian field classes;
arbitrary dimensions … canonical restoration, package sealing, publication
readiness, or release authorization"). It is a 3D-track object and is never
composed with the 2D tracks. The planar `C24` candidate in the READ_FIRST carries
`[[REVIEW:DEFERRED]] [[CANONICAL:UNCHANGED]]`. None of these is `C*`, and no
numerical `C*` is admitted (`FW-DECIMAL-KILL`).

*Repository state (code, not status):* until 2026-09-18 this repository's
README, research map and claim graph carried the historical `LIVE_ROOT_THEOREM`
label for `Q0-C104-THEOREM-B` while the registers it ships said otherwise; the
node now transcribes the register row, `tools/claims_check.py` refuses an
unconditional grade on any claim carrying a retraction
(`FW-RETRACTED-NOT-UNCONDITIONAL`), and a test binds the node's
`register_status` to the register row verbatim. No Theorem B computation exists
here.

## H. Lower-bound campaign (LB-RATE / K3) — OPEN / HOLD

The 2D lower-bound campaign for `1 − q(r, 6/5)` (`docs/RESEARCH_MAP.md` §7). Its
controlling adjudication `GP-LB-STAT-004` reads "REFUTED AS A THEOREM-GRADE OR
VALIDLY STATED CONDITIONAL ASSEMBLY / LOWER CAMPAIGN OPEN"; the assembly
`K3-THM-001` is "REFUTED AS WRITTEN / NONCONTROLLING" and the register says
"Do not mint a successor". The five P0 rows below are the Open Questions
register (`registers/json/open_questions.json`), transcribed verbatim; nothing
here is a new mathematical claim.

### OQ-014 — ANALYTIC / KAC-RICE / UNIFORMITY (P0)

* **Register status:** OPEN — MECHANISM REFUTED / QUANTITATIVE BOUND REPAIR REQUIRED
* **Current evidence / change:** KIMI-DER-025 confirms the bare value-kill mechanism is false where the conditional mean exceeds b, but its claimed CS upper bound omits sqrt(min(Cantelli,P_window)); 4.7602e-6 is numerical only, not a rigorous enclosure. GP-LB-STAT-003 additionally supersedes the later AO48 session handoff’s stale I_cs, rung-table, and 5.5e-3 r^1.6 upper-bound claims.
* **Live state:** WP OPEN / KIMI-THM-023 v1.0 AND v1.1 HOLD / NO BOOLEAN CHANGE
* **Next decisive action:** Re-derive the exact event inclusion and minimal WP rate; then repair the inequality and provide rigorous quadrature/spatial enclosures or a different uniform typed pair-Palm proof; rerun mutations and independent review.
* **Owner or capacity needed:** Analytic Kac-Rice specialist plus independent certificate reviewer
* **Source:** https://drive.google.com/file/d/11-H0mx2obq_wN7YYH0tL0BABlS-Nxm6y/view (reviewed 2026-08-05)

### OQ-015 — ANALYTIC / DYNAMICAL / GAUSSIAN TUBE (P0)

* **Register status:** OPEN — P-NMZ-gamma
* **Current evidence / change:** KIMI-DER-026 isolates gamma-LOC(ii-c) to one named near/moderate-zone C1 tube-clearance premise; the premise itself is not certified by delivered artifacts.
* **Live state:** gamma-LOC(ii-c) CONDITIONAL / NO GATE EFFECT
* **Next decisive action:** Certify the nominal third-saddle ascent tube on d≲3 and a sub-Gaussian C1 scale with clearance ratio diverging faster than sqrt(2 log(1/r)).
* **Owner or capacity needed:** Dynamical-systems and Gaussian-process tube specialist
* **Source:** https://docs.google.com/document/d/1vtKTJbunykXqJtTYmKsw1HwkTaw3_c-jP5L5jnmp2xs/edit (reviewed 2026-08-05)

### OQ-016 — THEOREM ASSEMBLY / COMPLETION / UNIFORMITY (P0)

* **Register status:** OPEN — DER-027B AND THM-023 V1.1 INCOMPLETE
* **Current evidence / change:** DER-027b has unfinished verdict/gap sections and no -O transcript. THM-023 v1.1 contains placeholders and PLACEHOLDER-BODY-HASH; its verifier later reads a missing 0.005 rung. Exact K3 Phase-0 raw carriers and W2/W3/W4 freeze receipts are not yet delivered; GP-LB-STAT-003 quarantines the inherited WP upper-bound branch.
* **Live state:** LOWER-BOUND THEOREM HOLD / DRAFTS NONCONTROLLING
* **Next decisive action:** Obtain hash-frozen W2/W3/W4 reports, sources, raw transcripts, exit receipts and mutations; then W12 blind adjudication. Separately settle WP’s minimal event-level target, P-NMZ-gamma, DER-027b, eta_r, and theorem verifier defects.
* **Owner or capacity needed:** Kimi authoring line plus independent theorem/certificate reviewer
* **Source:** https://drive.google.com/file/d/11-H0mx2obq_wN7YYH0tL0BABlS-Nxm6y/view (reviewed 2026-08-05)

### OQ-016-U1 — THEOREM ASSEMBLY / COMPLETION / UNIFORMITY (P0)

* **Register status:** OPEN — K3-THM-001 REFUTED AS WRITTEN / LOWER HOLD
* **Current evidence / change:** K3-THM-001 conflates measured 0.9666 with theorem tier 0.089569*P0, prints 0.8705 instead of 0.87003666, omits H-Bonf pair=o(r3), and consumes unproved W6 weighted-Palm and W8 H-B3 premises. W13 is a stale snapshot.
* **Live state:** LOWER-BOUND THEOREM OPEN / ALL DRAFTS NONCONTROLLING
* **Next decisive action:** Do not mint a successor. Prove Lambda positive uniform constant, exact weighted-Palm losses, eta_r, exit o(1), H-Bonf, analytic DER-027a/027c uniformity; complete W8 receipts; then new blind review and final-tree clean-room replay.
* **Owner or capacity needed:** Kimi or independent authoring line plus isolated theorem/certificate reviewer
* **Source:** https://drive.google.com/file/d/1lfH7g57LcshqpLfNbrx-H7gbJckpzPqr/view (reviewed 2026-08-05)

### OQ-016-U2 — THEOREM ASSEMBLY / W8 RECEIPT RECOVERY (P0)

* **Register status:** OPEN — W8 SOURCE RECOVERED / FINAL RECEIPTS ABSENT
* **Current evidence / change:** Outer-archive comparison recovered verify_lambda_grid_v2.py and mutate_lambda_grid.py outside K3, correcting one GP-LB-STAT-004 custody sentence. No W8 transcript_O, identity stamp, S6 excerpt, or W8 HASHES exists anywhere in the archive; H-B3 remains unproved.
* **Live state:** LOWER THEOREM OPEN / W8 NONCLOSED / K3-THM-001 REFUTED AS WRITTEN
* **Next decisive action:** Prove an analytic H-B3 replacement or a different Lambda-side positive uniform constant; generate complete normal/-O transcripts, exit receipts, identity/hash manifest and mutation evidence from one final immutable delivery; then independent review.
* **Owner or capacity needed:** Lambda/interval specialist plus independent certificate auditor
* **Source:** https://drive.google.com/file/d/1cdSyfIoV8WiHug7aJDv4Ng7Vq2GnRP_j/view (reviewed 2026-08-05)

*Repository state (code, not status):* no lower-campaign computation exists here.
`engine/carriers/MANIFEST.json` binds `jets.py` (`CR-JETS`, from the DER-027b
INCOMPLETE draft, mpmath, `certifying: false`) to this lane; until 2026-09-18 it was
bound to §A1 as if its Taylor jets were the upper chart's 24-jet set, which the
source never says. Nothing here rehabilitates `K3-THM-001`, closes or reopens the
campaign, or adjudicates between the AO48-layer and GP-layer labels beyond
recording that the register follows the GP layer.
