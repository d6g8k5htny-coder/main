# RESEARCH_STATUS_MEMO

**Drafted:** 2026-09-16 (America/Chicago) by agent 6; §2 filled ~16:12 CT after @5 artifacts landed  
**Discipline:** source-only. Quote or tight paraphrase from named carriers. **No status promotions. No invented proofs.**

**Sources used this draft:**
- `CHART_SIDE_JETMOD_PLAN.md` (primary for §1 chart side; @3)
- Cross-check skim: `PREMISE_BLOCKERS_D1_PROMOTE_AND_RN_UNIF.md`, `REPRO_CHECKS_H5_RN_UNIF.md`
- `DS3_RECEIPT_DRIFT.md` + `RN_UNIF_NEXT_SMOKE.md` (§2; @5 lane artifacts, mapped ~16:10 CT)

---

## 1. OBL-D1-PROMOTE — chart side (from CHART_SIDE_JETMOD_PLAN)

### Framing (normalizer CLOSED / chart OPEN)

Plan scope is chart side only. Normalizer stays CLOSED per sources the plan quotes:

> “§1 OBL-D1-PROMOTE: **normalizer sub-part CLOSED** (discharge mechanism in §4 above); chart side OPEN (OBL-H5-JETMOD, OBL-H5-REMOTE-THRESHOLD unchanged; H5 rungs executing).”  
> — `RETURN_06/ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` §5  
> *(via CHART_SIDE_JETMOD_PLAN)*

> “the normalizer sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus; the chart side (H5 rung family + interpolation) remains OPEN”  
> — same addendum §7 *(via plan)*

The plan deepens **chart-side** work only (H5 r-scaled rung family + band interpolation / named sub-obligations). It does not reopen the normalizer discharge.

### Named chart-side sub-obligations (as sources write them)

| Obligation | Content (paraphrase of plan quotes) | State as plan reports sources |
|---|---|---|
| **OBL-H5-JETMOD** | Certified interval bounds for the full 24-jet set over r-bands `[r_{k+1}, r_k]` with lattice-tail constants re-certified uniformly in band; falsifier = band enclosure width exceeding claimed modulus. Display modulus ≠ certified enclosure. | OPEN (display only vs certified) |
| **OBL-H5-ZBAND** | Z_r window bounds over bands; lo rides H3 `c_Z·r²` frozen uniform certificate; **hi side** needs band LPW bracket. | lo banked / frozen; **hi OPEN** |
| **OBL-H5-REMOTE-THRESHOLD** | Certify D3 remote bracket at r-scaled `d ≥ 2r`, or extend chart machine cover to absolute `d₀`; rides with D3-LEMMA-RN-UNIF. | OPEN; unchanged in addendum |

Parent framing (frozen assembly / register note, via plan): OBL-D1-PROMOTE remains among OPEN validity premises of Theorem (2); register note reduces to chart side + uniform-band extension of B4LOC/B2-far/far-route after normalizer discharge.

### What RUNG2 / RUNG3 certify vs what remains

**Artifact content (not premise discharge)** — plan table:

| Package | r | Headline I_hi/r³ | Totals digest (doc-cited; plan: repro PASS) | Body-sha256 (doc-claimed; plan: repro PASS) |
|---|---|---|---|---|
| `H5_RUNG2_2026-09-15.md` | 0.025 | 664.3979 | `f7697bcfa0fe32b5c87c8ef8adef5d4384ab1bda7cd4a7c4782fef8c99103fcb` | `91a34d83688b5110b886ab4fc168a55030343cc0da4a98e38347d3b0e6e48102` |
| `H5_RUNG3_2026-09-15.md` | 0.035355 | 661.4712 | `808d6901e3254a73181aa696904ed04567d401b6e58300454d489046c36f4f64` | `f4c3414fe8b072008f5f5c8c5676ec9e20944e115535e0f3dd9c13c1bb89c0d8` |

Also on the ladder (plan): r = 0.05 frozen v1 731.4311 / live v3 clean 647.8048; modulus **display** (κ = 1/8 fit) still labeled DISPLAY; certified enclosure remains OBL-H5-JETMOD. Drive mirrors of RUNG2/RUNG3 match local bytes per `REPRO_CHECKS_H5_RN_UNIF.md` (plan pointer).

**These packages do NOT (plan § “What these packages do NOT discharge”):**
- close OBL-D1-PROMOTE as a validity premise
- discharge OBL-H5-JETMOD (interval-r / G12-band 24-jet enclosure)
- discharge OBL-H5-ZBAND hi side (band LPW bracket)
- discharge OBL-H5-REMOTE-THRESHOLD
- certify merges/envelopes at r = 0.0177 or r = 0.0125
- close register note’s uniform-band B4LOC/B2-far/far-route extension

### Cell status at r = 0.0177 and r = 0.0125

Freshest named N/70 in H5 docs per plan (`H5_RUNG2_2026-09-15.md` rung ladder):

- r = 0.0177: cells **42/70** (2 shards resuming), probes/patches complete  
- r = 0.0125: cells **21/70** (2 shards resuming), probes/patches complete  

RUNG3 certifies r = 0.035355 and does **not** restate N/70 for 0.0177 / 0.0125. Plan: no later H5_*.md after RUNG2 names a new certified N/70 for those r; do not treat raw jsonl line counts as a status promotion beyond the RUNG2 ladder lines.

### Next proof obligation (ONE paragraph — from plan)

Grounded in `H5_PROMOTE.md` §3(iii) + named sub-obligations, ledger §1, register note chart-side OPEN, and the RUNG2 ladder: the chart-side blocking proof step remains **OBL-H5-JETMOD** — evaluate interval-r lattice sums to obtain **G12-band / 24-jet band enclosures** so Î(r)/r³ ≤ F(G12-band) on each band `[r_{k+1}, r_k]` (finite per-band computation; falsifier = enclosure width exceeding the claimed modulus), which the sources explicitly separate from the already-shipped c₂ **display**. Concurrent engineering named by the freshest ladder (not a substitute for that proof step) is finishing cells+stitch+merge at **r = 0.0177** and **r = 0.0125** (still 42/70 and 21/70 in RUNG2); still-OPEN chart companions named alongside JETMOD are **OBL-H5-ZBAND hi side (band LPW bracket)** and **OBL-H5-REMOTE-THRESHOLD** (r-scaled d ≥ 2r vs absolute d₀; rides with D3-LEMMA-RN-UNIF). No source names a single exclusive “do this next then premise CLOSED” gate after RUNG2/3; this paragraph does not invent one.  
— *verbatim substance of CHART_SIDE_JETMOD_PLAN § “Next proof obligation”*

### Key paths / Drive IDs (from plan table)

| Artifact | Local (under `drive_peer_review_triage/`) | Drive ID |
|---|---|---|
| CHART_SIDE_JETMOD_PLAN.md | `CHART_SIDE_JETMOD_PLAN.md` | — (also HOLD + PKG-01 mirrors per CoS) |
| H5 folder | — | `1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y` |
| H5_RUNG2 | `…/H5_closure/H5_RUNG2_2026-09-15.md` | `13eLq6qnVIxyGI-GE1L0T5TL5FKcj1qyZ` |
| H5_RUNG3 | `…/H5_closure/H5_RUNG3_2026-09-15.md` | `1IdE6t7TRBZqWLGRKNXBTi-pQfr2hVTcI` |
| H5_PROMOTE_UPDATE | `…/H5_closure/H5_PROMOTE_UPDATE_2026-09-15.md` | `1P-8dPVgy0yACgHghR-MN6nqWNMEvJ7cD` |
| D1_ASSEMBLY_v2_2 | `…/D1_assembly/D1_ASSEMBLY_v2_2.md` | `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` |
| Register note | `…/D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` |
| Premise blockers brief | `PREMISE_BLOCKERS_D1_PROMOTE_AND_RN_UNIF.md` | — |
| Repro checks | `REPRO_CHECKS_H5_RN_UNIF.md` | — |

---

## 2. @5 — DS3 receipt drift + RN-UNIF next smoke

Filled from `DS3_RECEIPT_DRIFT.md` and `RN_UNIF_NEXT_SMOKE.md` only. **No lemma close. No premise promotion.**

### 2.1 DS3 receipt drift (`DS3_RECEIPT_DRIFT.md`)

**Role (artifact):** fail-closed isolation / hypothesis, not closure. Explicit: does **not** claim D3-LEMMA-RN-UNIF closed; regen DS3 pass does **not** close the lemma.

**Locate facts:**
- `rnu_ds3.py` is **Drive-only** (absent from Sep-15 extract); Drive ID `1v7JAh_FbXHYMaL7W6DNwEoLnHPwQ5U21`, size 7201, sha256 `c3d5adc8b88788b7eed43bd68efe9381f7076775a7ec60817a20a8267d9116da`.
- Extract engine present: `d3_rn_unif.py` (second-order DS/DM, not DS3), sha256 `85d7725fab42eeb0e823226f44d17f142a5c57e5d89084b2b6edffe4a8f0c930`.
- Drive folder for receipts/scripts: `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG`.

**Numeric PARTIAL (Drive frozen receipt vs regen with current Drive `rnu_ds3.py` + extract `d3_rn_unif`):**

| Field | Drive receipt | Regen |
|---|---|---|
| `steps.A_ds3.max_abs_err` | `"2.1746255"` | `"0.0"` |
| `steps.A_ds3.pass` | **false** | **true** |
| `lemma_closed` | **false** | **false** |
| `status` | `PROPOSED` | `PROPOSED` |

Evidence note (artifact): `nstr(0.8 * exp(1), 8) == "2.1746255"` matches the receipt string; current Drive `ds3_selftest()` returns max err `0.0`.

**Cause hypothesis (artifact; evidence-backed, not proven):** receipt/script vintage mismatch — stale DS3 fail frozen beside a later-clean `rnu_ds3.py` (or different `rnu_ds3` on `sys.path` at receipt generation than the uploaded file). Batch upload timestamps within ~2 s are consistent with a pre-built folder mix.

**Non-claims carried forward:** `lemma_closed: false` on Drive and regen; regen `A_ds3.pass: true` does not overturn Drive receipt for fail-closed ledger purposes; no FREEZE / MUT-RN / premise promotion. Receipt still lists `next_required_for_freeze` including whitened `env_form` orders 2–4, DS3 through `kappa_far`, etc.

### 2.2 RN-UNIF next smoke (`RN_UNIF_NEXT_SMOKE.md`)

**What sources name as step 1:** whitened `env_form` bounds on residual covariances at orders 2–4 (CL-RNU-001 §5 / receipt `next_required_for_freeze[0]`). Order-1 whitened χ² exact gradient is named DONE (`rnu_chi2_white_v2.py`).

**Runnable inventory (artifact conclusion):** whitened `env_form` residual-cov bounds orders 2–4 **do not exist as runnable code** in extract or current Drive RN-UNIF folder. Nearest existing entrypoints smoked:

| Smoke | Command / entrypoint | Exit | Result as artifact states |
|---|---|---|---|
| A — whitened χ² order-1 | `python rnu_chi2_white_v2.py` (sandbox) | `0` (~29 s) | **PASS** as order-1 FD check; not orders 2–4; does not close lemma |
| B — native unwhitened `env_form` qord 0–4 | `d3_rn_unif.env_form(...)` probe | `0` | entrypoint live; **not** whitened residual-cov certification |
| C — `rnu_execute.py` suite | prior repro | `0` | `LEMMA_CLOSED=NO`; T4 `INTERNAL-NOT-ENV_FORM` |

### 2.3 H5 cell harness note (from same smoke artifact)

Executable harness **YES** under extract `H5_closure/`: `h5_run_r0p0177.py`, `h5_run_r0p0125.py`, `promote_run.py`, `supervisor_rung.sh`, etc. RUNG2 ladder language still 42/70 and 21/70 at those r (package snapshot — not re-certified in the smoke). Observational local jsonl counts (70 / 42 unique cells) may disagree with RUNG2 text; artifact: treat as raw log inventory, **not** OBL-D1-PROMOTE discharge.

---

## Explicit non-claims (this memo)

- Does **not** promote OBL-D1-PROMOTE / chart-side / Theorem (2) status.
- Does **not** claim D3-LEMMA-RN-UNIF closed, FREEZE, or any new discharge.
- RUNG2/3 language above is **artifact content** as reported by the chart-side plan — not premise discharge.
- §2 reports only what `DS3_RECEIPT_DRIFT.md` / `RN_UNIF_NEXT_SMOKE.md` state: drift hypothesis, smoke inventory, harness presence — not lemma closure or premise discharge.
- Local H5 jsonl cell counts are observational inventory only (per smoke artifact).

---

*Complete draft for current sources. §1 from CHART_SIDE_JETMOD_PLAN; §2 from DS3_RECEIPT_DRIFT + RN_UNIF_NEXT_SMOKE. No status promotions.*
