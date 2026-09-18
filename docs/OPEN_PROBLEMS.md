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

Cells at `r = 0.0177` stand at **42/70** and at `r = 0.0125` at **21/70**, two
shards resuming each, probes and patches complete. No later named `N/70` exists
after RUNG2. Raw JSONL line counts are not a status promotion.

### A3. `OBL-H5-ZBAND` hi side

The lo side rides the frozen H3 uniform certificate. The hi side needs the
**band** version of the LPW bracket. `H5_ZBAND_CONSUMPTION_2026-09-15` proposes
`OPEN → DISCHARGED (consumption grade)` but is **PROPOSED**, authority none,
and is not operator-promoted over frozen v2.2.

### A4. `OBL-H5-REMOTE-THRESHOLD`

Certify D3's remote bracket at the r-scaled threshold `d ≥ 2r`, or extend the
chart's machine cover to absolute `d₀`. Rides `D3-LEMMA-RN-UNIF`.

### A5. `D3-LEMMA-RN-UNIF` — the RN uniform lemma

* **Piece 1** OPEN. **Piece 2** OPEN — the annulus Riemann-sum driver is
  **unwritten**. Schedule it explicitly; do not hide it under a T4 push.
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

### A6. `PERC-DECAY`

The `o(r³)` far-lane reading is not reachable. Use the `Θ(r³)` restatement with
certified constants plus `PD-CONN`, which is a **named OPEN input**.

### A7. `OBL-B1-BRANCH(loop|B1)`

OPEN in frozen v2.2; demoted to REFINEMENT for v2.3. Constant-level.

### A8. `B4.loc` wrap/remote reconciliation

`B4LOC-R1` closed the dam line for the whole of B4 and resolved the
identification negatively (cut-net ≢ nine-pin tube). The wrap/remote
reconciliation under D1 remains open per the capsule.

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

## D. Review queue — 24 routes, 23 unassigned

(Corrected 2026-09-18. An earlier version of this heading said "all
unassigned"; `registers/json/review_queue.json` shows 23 rows with
`Reviewer / claim = UNASSIGNED` and one, `RV-LM009-MAIN`, carrying
"OPS4 nonauthor / exposed / OpenAI". The register is the source; the heading
was the transcription error.)

From `registers/json/review_queue.json`. Nineteen originated in July routing
records and are 51–54 days old; the R17 ladder therefore puts them at
**ESCALATE**. Aging never approves anything.

| Key | Exact object | Technical status | Body bytes / SHA-256 |
|---|---|---|---|
| `RV-LM003-MAIN` | LCR-DER-014-v1.0 | NEEDS_RECONCILIATION | 6,874 / `5173d26d…` |
| `RV-LM004-MAIN` | LCR-DER-016-v1.1 | NEEDS_RECONCILIATION | 10,416 / `d43179f3…` |
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
| `RV-RN-ALIGN` | CL-RNU-003 ↔ RN3 crosswalk | NEEDS_RECONCILIATION | bind both objects first |
| `RV-P15` | P15-A/B/C/D structural localization | **READY** | archive manifest required |
| `RV-H5-REPAIR` | H5 corrected-kernel full consumer replay | **AMEND** | 11 archive-member hashes |
| `RV-OPS-R17` | OP-PROT-019-v1.1 implementation | **READY** | 14,073 / `04987ba4…` |

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
