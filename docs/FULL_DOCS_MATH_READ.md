<!-- Reading copy of Google Drive file 12nId_WFjz10b_S0WlTSzQv2S0G8TjmSK
     (16,108 bytes, SHA-256 448adee5…). NOT byte-exact: this header is
     prepended, which is the whole difference. See governance/PROVENANCE.json.
     ("FULL_DOCS_MATH_READ.md", 2026-09-16). Author-side deep read of the math
     carriers; status labels are as written by their sources. No promotions. -->

# Full docs — math deep-read

**Written:** 2026-09-16 ~22:22 CDT (America/Chicago).  
**Output:** `/workspace/drive_peer_review_triage/FULL_DOCS_MATH_READ.md`  
**Discipline:** Claims/status **as written only**. No promotions. Packaging ≠ premise discharge; display ≠ certified band enclosure; PROPOSED/DRAFT ≠ operator-promoted; register note / v2.3 DRAFT do not edit frozen v2.2.  
**Built on:** `FULL_MATH_CLAIM_INVENTORY.md`, `LANE_MATH_MAP.md`, `CHART_SIDE_JETMOD_PLAN.md`, `OPEN_PREMISES_WORKLIST.md`, `PREMISE_BLOCKERS_D1_PROMOTE_AND_RN_UNIF.md`, `LANE_RN_UNIF.md`.  
**Local carriers:** `extract_sep15/`, `packages/pkg01|pkg03|hold/`, `extracts/_new/anthropic_cl_hub/{D1_v2_3_DRAFT,H5_ZBAND}/`, `extracts/RN_UNIF_2026-09-16/`. Drive MCP not required for this digest (critical text present locally).

---

## CoS Tier A1 — chart-side OBL-D1-PROMOTE (lead)

### 1) OBL-H5-JETMOD — band enclosure vs display modulus

**Obligation content** (`H5_PROMOTE.md` §3):

> **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set (not just the displayed c₂) over the r-bands [r_{k+1}, r_k], with lattice-tail constants re-certified uniformly in the band (LAT's tail bound currently certifies at point separations). Content: for each jet J and band B, J(B)/r^{p_J} ∈ certified interval; falsifier: a band enclosure whose width exceeds the claimed modulus.

**Proof step** (`H5_PROMOTE.md` §3(iii)):

> *Band certification (the proof step).* … evaluating those sums with r as an interval over the band yields G12-band enclosures, hence Î(r)/r³ ≤ F(G12-band) for the whole band — a FINITE computation per band, never a fitted exponent.

**Display ≠ certified enclosure** (`H5_PROMOTE_UPDATE_2026-09-15.md` §5):

> This is the displayed modulus (dense certified sampling + explicit fit); the CERTIFIED band enclosure (interval-r lattice sums) remains OBL-H5-JETMOD with its content unchanged.

Ledger (§1, assembly-time): JETMOD state **OPEN (display only)**.  
**As written:** modulus display is CERTIFIED sampling; **CERTIFIED band enclosure remains OPEN** under OBL-H5-JETMOD. RUNG2/3 do **not** discharge it.

### 2) Cell status at r = 0.0177 / 0.0125 (RUNG2 ladder N/70)

Freshest named N/70 (`H5_RUNG2_2026-09-15.md` § “Rung ladder”):

> r = 0.0177: cells **42/70** (2 shards resuming), probes/patches complete  
> r = 0.0125: cells **21/70** (2 shards resuming), probes/patches complete

Also on that ladder: r = 0.05 certified (frozen v1 731.4311 / live v3 clean 647.8048); r = 0.025 **THIS PACKAGE** 664.3979; r = 0.0354 stitch executing (cells 70/70 noted in RUNG2 body; later certified by RUNG3).  
`H5_RUNG3_2026-09-15.md` certifies r = 0.035355 and does **not** restate a new N/70 for 0.0177 / 0.0125.  
Earlier `H5_PROMOTE.md` §4 table had those rungs “queued / pending” — superseded for N/70 by RUNG2 ladder lines only.  
**Not claimed:** no later named cells N/70 after RUNG2; raw jsonl line counts are not a status promotion.

### 3) Normalizer CLOSED vs chart OPEN (H3 band-floor addendum)

`ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` §5:

> §1 OBL-D1-PROMOTE: **normalizer sub-part CLOSED** (discharge mechanism in §4 above); chart side OPEN (OBL-H5-JETMOD, OBL-H5-REMOTE-THRESHOLD unchanged; H5 rungs executing).

> §1 sub-obligation OBL-H5-ZBAND: the lo side now rides a FROZEN uniform certificate (Z_r ≥ c_Z·r² for ALL r ∈ (0, 0.05], body 281477c3…) — no longer an unbanked run; **the hi side (band LPW bracket) remains OPEN.**

Same addendum §7:

> the normalizer sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus; the chart side (H5 rung family + interpolation) remains OPEN

Certified floor statement (§2): for EVERY r ∈ (0, 0.05], E[G_r] ≥ 2.30659559567154 > c_Z = 1.615489267643502474…, hence Z_r ≥ c_Z·r² uniformly.  
**As written:** normalizer sub-part **CLOSED / DISCHARGED**; chart side **OPEN**.

### 4) What RUNG2 / RUNG3 do and do NOT discharge

| Package | r | Artifact claim (as written) | Does **not** discharge |
|---|---|---|---|
| `H5_RUNG2_2026-09-15.md` | **0.025** | I_hi/r³ = **664.3979**; C1 containment PASS; coverage PASS; mutation 6/6; A/B byte-identical; totals `h5_totals_r0.025_v2.json` sha256 `f7697bcf…` | OBL-D1-PROMOTE as validity premise; OBL-H5-JETMOD; OBL-H5-ZBAND hi; OBL-H5-REMOTE-THRESHOLD; r=0.0177/0.0125 merges |
| `H5_RUNG3_2026-09-15.md` | **0.035355** | I_hi/r³ = **661.4712**; same class checks; totals `h5_totals_r0.035355_v2.json` sha256 `808d6901…` | Same as above |

Also certified at rungs (artifact content): named-lemma re-cert in r-scaled form (H5-AXIS envelopes, patches 8/8, Z window live-recomputed, remote B_remote=19.55 symbolic). Modulus **display** in UPDATE §5 still labeled DISPLAY.  
H5_STATE title still: “OBL-D1-PROMOTE rung family **executing**”. Assembly §5 / addendum: chart side **OPEN**.  
Register note keeps uniform-band extension of B4LOC/B2-far/far-route on the same interpolation machinery as OPEN chart-side content — also not closed by RUNG2/3.

**Next named proof step (sources):** OBL-H5-JETMOD band enclosure. Concurrent engineering (not premise discharge): finish cells at r=0.0177 / 0.0125. Companions still OPEN: ZBAND hi (band LPW bracket), REMOTE-THRESHOLD (rides D3-LEMMA-RN-UNIF). No source names a single exclusive “then premise CLOSED” gate after RUNG2/3.

---

## D1 assemblies & registers

### Frozen controlling body — `D1_ASSEMBLY_v2_2.md`

- Local: `packages/pkg01/assembly/D1_ASSEMBLY_v2_2.md` (= extract `…/D1_assembly/`).  
- Frozen body SHA-256: **`490ad6b2f14176fe8cf5af363fb94dc73a8bc5523f608e5ab2a42ff749b235f6`** (PKG-01 controlling).  
- PKG-01 banner: READY FOR PEER REVIEW as **honest conditional / certified-rung**; **NOT** unconditional closed theorem.

**Theorem D1 v2.2(1):** certified rung r=0.05 (named hypotheses H5-RIM/AXIS; D3-LEMMA-RN-UNIF at rung — precisely stated, **NOT closed**). Live clean totals: I_hi(v3) = 8.0975589252e-2 = **647.8048·r³** (pin sha `8d7028e4…`).

**Theorem D1 v2.2(2):** all-small-r 1−q ≤ C r³, **CONDITIONAL** on five named VALIDITY premises.

**OPEN — VALIDITY premises of Theorem (2)** (assembly §3 / §5, quote):

> **OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD); D3-LEMMA-RN-UNIF (rung + uniform parts; foundations); PERC-DECAY (B1.dir-far, B2-far, B4.rem; percolation lane); OBL-B1-BRANCH(loop|B1) (branch-control lane; constant-level); **the B4.loc dam-line tube certificate** … INCLUDING the open identification … asserted-not-established — scope ruling V2 …

**CLOSED at assembly (selected):** OBL-B1-REG; H3 existential / rung floor; G.7-scope normalizer **AT THE RUNG** DISCHARGED; H4JC-R1 PASS-WITHSTOOD; OBL-B1-BRANCH(dir); remote bracket floor-consistent **modulo** D3-LEMMA-RN-UNIF; H5 rung interval (v3 totals).

### Register note — `D1_ASSEMBLY_v2_2_REGISTER_NOTE.md`

Header (quote):

> the frozen v2.2 body (490ad6b2…) is **NOT altered**; everything below takes effect at the next issuance (**v2.3**).

§5 (effective at next issuance):

> **CLOSED:** the B4.loc dam-line tube certificate AND B4.rem (B4LOC-R1…)  
> **RESTATED** PERC-DECAY …  
> **OPEN (validity premises of Theorem (2), reduced):** OBL-D1-PROMOTE — chart side … (the H3 band floor has already discharged the normalizer sub-part); D3-LEMMA-RN-UNIF …  
> **Theorem-statement consequence (for v2.3):** Theorem (2)'s open validity premises reduce to OBL-D1-PROMOTE and D3-LEMMA-RN-UNIF … **No frozen v2.2 line is edited by this note.**

OBL-B1-BRANCH(loop|B1) moved to **REFINEMENT** register in the note.  
Frozen body SHA of register note: `c1d5e95d97e019574d6b6b09e6e606fd6e0cbc1fe3aa2e0a54953de6e4f2c470`.

### DRAFT — `D1_ASSEMBLY_v2_3_DRAFT.md`

Local: `extracts/_new/anthropic_cl_hub/D1_v2_3_DRAFT/`.  
Header: **Status: PROPOSED.** “v2.2 remains the last Kimi-issued form; this v2.3 becomes the current strongest form **only upon operator promotion**.”  
Draft Theorem (2): **TWO** validity premises (OBL-D1-PROMOTE + D3-LEMMA-RN-UNIF); claims **OBL-H5-ZBAND DISCHARGED at consumption grade** (folds H5_ZBAND_CONSUMPTION).  
**As written:** DRAFT / PROPOSED — **not promoted** over PKG-01 / frozen v2.2.

### Obligation ledger & executive state (RETURN_06)

- `OBLIGATION_LEDGER.md`: §1 OBL-D1-PROMOTE **EXECUTING**; JETMOD OPEN (display only); ZBAND OPEN (assembly-time; lo later superseded by H3 addendum); REMOTE OPEN. §2 D3-LEMMA-RN-UNIF **NOT CLOSED**. §3 PERC-DECAY **OPEN** (assembly-time). §4 OBL-B1-BRANCH(loop|B1) **OPEN**. §5 B4.loc **OPEN** (assembly-time).  
- `00_EXECUTIVE_STATE.md`: five named VALIDITY premises listed; D3-LEMMA-RN-UNIF(r=0.05) “precisely stated, **NOT closed**”; H5 rungs EXECUTING (assembly-time cell counts superseded for 0.025/0.035355 by RUNG2/3 packages).  
- Later same-day **addenda** govern over ledger/exec for normalizer CLOSED, B4LOC CLOSED, PERC RESTATED — without editing those frozen files.

---

## H5 / ZBAND / promote

### H5_STATE / H5_PROMOTE / UPDATE

- `H5_STATE.md` title: “OBL-D1-PROMOTE rung family **executing**”; v3 = CLEAN live totals.  
- `H5_PROMOTE.md`: names three sub-obligations (JETMOD / ZBAND / REMOTE-THRESHOLD); body-sha `8d8b3538…`.  
- `H5_PROMOTE_UPDATE_2026-09-15.md`: display vs enclosure split (quoted in Tier A1 §1).

### Totals freeze / errata

- `H5_TOTALS_FREEZE_2026-09-15.md`: consumed artifacts **version-frozen**; never overwrite; drifted `h5_totals.json` lineage disclosed.  
- `H5_TOTALS_ERRATA_2026-09-15.md`: v2 I_lo contamination from rung-unscoped merge of r=0.025 patches into r=0.05; root-caused + fixed; **v3 CLEAN**: I_lo = 1.166721190e-08 (v1 pin restored); I_hi = 8.0975589252e-02 = **647.8047 r³**; sha256 `8d7028e4d0d49d54d3a7c4589a898289ec29eee6c2de9c774e99df0da46596bb`. Frozen docs not altered.

### OBL-H5-ZBAND / consumption (PROPOSED layer)

**Chart / addendum layer:** lo rides FROZEN H3 floor; **hi side (band LPW bracket) remains OPEN**.  

**Separate carrier** `H5_ZBAND_CONSUMPTION_2026-09-15.md` (local `extracts/_new/anthropic_cl_hub/H5_ZBAND/`):

> AUTHOR: Claude … STATUS: **PROPOSED**  
> AUTHORITY: **none** · CANONICAL IMPACT: OBL-H5-ZBAND OPEN → DISCHARGED (consumption grade) **upon promotion**; no theorem statement changes; no frozen carrier edited.

Consumes H3 floor + ceil (Z_r/r² ∈ [2.3066, 3.7477] band table); digest `cbe8603f…`. `FREEZE_H5_ZBAND.txt` pins scripts/transcripts.  
**Do not collapse:** chart addendum still says hi OPEN; consumption doc is PROPOSED discharge — not operator-promoted over frozen v2.2 / HOLD.

### OBL-H5-REMOTE-THRESHOLD

> certify D3's remote bracket at the r-scaled threshold d ≥ 2r, or extend the chart's machine cover to absolute d₀ … Rides with D3-LEMMA-RN-UNIF.  
> State: **OPEN** (ledger + addendum **unchanged**).

---

## SIDE24 manuscript & errata (brief)

| Carrier | Status as written |
|---|---|
| `SIDE24_pre_peer_review_manuscript_v2_2026-09-13` (+ PDF/md) | Sealed second edition body; manuscript review packaging |
| `ERRATA_AND_CLARIFICATIONS_2026-09-13.md` | Withdraws composing **3D ratified upper (AO48-OPR-045) with 2D lower**; “No 2D two-sided law at ratified grade”; standing firewall: changes **no** LS-CTL / theorem / RP / AO48 / q0 status |
| PKG-03 `00_START_HERE` | **BANNER:** Manuscript review package. **UPPER2D governed by D1 v2.2 (Sep 15)**, not older manuscript closure. **NOT** an unconditional closed 2D theorem |
| PKG-03 `03_GOVERNING_STATUS_NOTE` | Controlling UPPER2D = D1 v2.2; all-small-r remains **CONDITIONAL** on five open validity premises |
| Master Register R2 / V2–V5 snapshots | Historical protocol: snapshots never overwritten; self-“CLOSED” labels ignored per prior inventory maturity rule; RP-C/RP-S remain OPEN in R2 era |

Manuscript abstract still markets “two-sided cubic” / “ratified upper chain” — **ERRATA §1 governs** over that wording for peer-review packaging.

---

## Open premises & related

### HOLD checklist (still lists all five)

`packages/hold/HOLD_OPEN_VALIDITY_PREMISES.md`:

> These five premises block unconditional promotion of Theorem D1 v2.2(2). **Do not invent proofs.**

| # | Premise | Frozen v2.2 / HOLD | Register / addendum (append-delta; v2.2 untouched) |
|---|---|---|---|
| 1 | **OBL-D1-PROMOTE** | **OPEN** (chart + sub-obligations) | Still **OPEN** (chart); normalizer **CLOSED** per H3 addendum |
| 2 | **D3-LEMMA-RN-UNIF** | **NOT closed** / **OPEN** | Remains **OPEN** among reduced premises |
| 3 | **PERC-DECAY** | **OPEN** | **RESTATED**; engine COMPLETE+FROZEN; PD-CONN **NAMED INPUT (OPEN)** |
| 4 | **OBL-B1-BRANCH(loop\|B1)** | **OPEN** | **REFINEMENT** / demoted for v2.3 |
| 5 | **B4.loc dam-line** | **OPEN** (asserted-not-established ID) | **CLOSED 2026-09-15** (B4LOC-R1); wrap/remote reconciliation still OPEN under D1 per capsule |

### D3-LEMMA-RN-UNIF / RN_UNIF (fail-closed)

- Assembly / exec: rung part precisely stated, **NOT closed**.  
- `LANE_RN_UNIF`: Piece 1 **OPEN**; Piece 2 **OPEN** (annulus Riemann-sum driver unwritten).  
- Receipts: `lemma_closed: false`; CL-RNU-001/002/003 **STATUS: PROPOSED**; AUTHORITY none.  
- CL-GROK-CLOSE-001 closes **session work**, not the lemma / D1 v2.3 / any prize.  
- Local: `extracts/RN_UNIF_2026-09-16/`; also `d3_rn_unif.py` under `D3_percolation/`.

### PERC-DECAY / D3_PERCOLATION / B4LOC

- `PERC_DECAY.md`: o(r³) far-lane reading **NOT reachable**; recommended restatement Θ(r³) with certified constants + PD-CONN. ADDENDUM-2: **RESTATED**.  
- `D3_PERCOLATION.md`: remote preemption class A.rem lane (foundations for remote bracket; separate from RN-UNIF lemma close).  
- `B4LOC_DAMLINE.md` / ADDENDUM-2: **THEOREM B4LOC-R1** — dam line **CLOSED** at super-algebraic grade, whole B4 (loc+rem); identification resolved **NEGATIVELY** (cut-net ≠ 9-pin tube). Frozen v2.2 OPEN list still names the premise until next issuance.

### Chart companions (recap)

| Sub-obligation | Status as sources write |
|---|---|
| OBL-H5-JETMOD | **OPEN** (display only; enclosure pending) |
| OBL-H5-ZBAND lo | rides FROZEN H3 floor |
| OBL-H5-ZBAND hi | **OPEN** (band LPW bracket); consumption carrier **PROPOSED** only |
| OBL-H5-REMOTE-THRESHOLD | **OPEN**; rides D3-LEMMA-RN-UNIF |

---

## Cross-cutting layering (do not collapse)

1. **Frozen D1 v2.2** — five OPEN validity premises; PKG-01 ships here.  
2. **Register note + ADDENDA** — append-deltas; B4LOC CLOSED, PERC RESTATED, normalizer DISCHARGED, BRANCH demoted; reduced OPEN = OBL-D1-PROMOTE (chart) + D3-LEMMA-RN-UNIF; **v2.2 body not edited**.  
3. **D1 v2.3 DRAFT + H5_ZBAND_CONSUMPTION** — **PROPOSED**; AUTHORITY none until operator promotion.  
4. **HOLD checklist** — still lists **all five** as blocking unconditional promotion.  
5. **H5 RUNG2/3** — certified rung artifacts; **not** premise discharge; display ≠ JETMOD enclosure.  
6. **SIDE24 manuscript Sep-13** — manuscript review; UPPER2D authority = D1 v2.2 Sep-15; ERRATA firewall on 2D/3D compose.  
7. **RN_UNIF CL-* receipts** — infrastructure / candidates; `lemma_closed: false`.  
8. **Prize track** — independent; **0** original prizes solved; not this digest’s focus.

---

## Gaps / UNLOCATED

| Sought | Result |
|---|---|
| Local `packages/pkg01/17_CHART_SIDE_JETMOD_PLAN.md` | **UNLOCATED** on disk (Drive PKG-01 copy exists per LANE_MATH_MAP) |
| Standalone Drive titles for `B4LOC_DAMLINE.md`, `PERC_DECAY.md`, `OBLIGATION_LEDGER.md`, ADDENDA | **Not found** as separate Drive titles; present in local `extract_sep15/` / intake zip |
| Hub folders 03–09 under Active Research | **UNLOCATED** (tree jumps 02→10) |
| Later named N/70 for r=0.0177 / 0.0125 after RUNG2 | **None** found |
| Operator promotion of v2.3 / ZBAND consumption over frozen v2.2 | **Not found** — remain PROPOSED/DRAFT |
| FULL_DOCS_MATH_READ.md prior run | Was **missing**; this write creates it |

---

*End math deep-read. No premise or theorem status promoted. No proofs invented.*
