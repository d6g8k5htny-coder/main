# Chart-side OBL-D1-PROMOTE — JETMOD plan

**Mapped:** 2026-09-16 (America/Chicago).  
**Scope:** chart side of OBL-D1-PROMOTE only. Normalizer stays CLOSED per sources (quoted below).  
**Discipline:** no invented proofs; no status promotions. Statements and cell counts are quoted or tightly paraphrased from named carriers; where a later named N/70 is absent, that absence is stated.

**Normalizer CLOSED / chart OPEN (quote only):**
> “§1 OBL-D1-PROMOTE: **normalizer sub-part CLOSED** (discharge mechanism in §4 above); chart side OPEN (OBL-H5-JETMOD, OBL-H5-REMOTE-THRESHOLD unchanged; H5 rungs executing).”  
> — `RETURN_06/ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` §5

> “the normalizer sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus; the chart side (H5 rung family + interpolation) remains OPEN”  
> — same addendum §7

This plan deepens **chart-side** work only (H5 r-scaled rung family + band interpolation / named sub-obligations). It does not reopen or re-litigate the normalizer discharge.

---

## Exact statements (as sources write them)

### OBL-H5-JETMOD

From `H5_PROMOTE.md` §3 (“Named sub-obligations (exact content, not absorbed)”):

> **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set (not just the displayed c₂) over the r-bands [r_{k+1}, r_k], with lattice-tail constants re-certified uniformly in the band (LAT's tail bound currently certifies at point separations). Content: for each jet J and band B, J(B)/r^{p_J} ∈ certified interval; falsifier: a band enclosure whose width exceeds the claimed modulus.

Ledger §1 (same content, state as of assembly-time ledger):

> **OBL-H5-JETMOD:** certified interval bounds for the full 24-jet set over r-bands [r_{k+1}, r_k] with lattice-tail constants re-certified uniformly in band. Falsifier: a band enclosure whose width exceeds the claimed modulus. State: OPEN (display only, see above).

`H5_PROMOTE_UPDATE_2026-09-15.md` §5 distinguishes display vs certified enclosure:

> This is the displayed modulus (dense certified sampling + explicit fit); the CERTIFIED band enclosure (interval-r lattice sums) remains OBL-H5-JETMOD with its content unchanged.

Proof step named in `H5_PROMOTE.md` §3(iii):

> *Band certification (the proof step).* … evaluating those sums with r as an interval over the band yields G12-band enclosures, hence Î(r)/r³ ≤ F(G12-band) for the whole band — a FINITE computation per band, never a fitted exponent.

### OBL-H5-ZBAND (hi side / band LPW bracket)

From `H5_PROMOTE.md` §3:

> **OBL-H5-ZBAND:** Z_r window bounds over bands (point-rung windows are banked; lo rides H3's c_Z·r² theorem-grade uniform bound; the hi side needs the band version of the LPW bracket).

Addendum delta (governs over ledger’s assembly-time ZBAND RUNNING note for the lo side):

> §1 sub-obligation OBL-H5-ZBAND: the lo side now rides a FROZEN uniform certificate (Z_r ≥ c_Z·r² for ALL r ∈ (0, 0.05], …) — no longer an unbanked run; **the hi side (band LPW bracket) remains OPEN.**  
> — `ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` §5

Do **not** strengthen this: sources do not claim a certified band hi-side LPW bracket.

### OBL-H5-REMOTE-THRESHOLD

From `H5_PROMOTE.md` §3:

> **OBL-H5-REMOTE-THRESHOLD:** the remote is consumed at D3's grade (19.55·r³, I_ann 17.02 + I_far 2.5283) with threshold d ≥ 2r at the rungs; D1's prescription names an ABSOLUTE d₀. Content: certify D3's remote bracket at the r-scaled threshold d ≥ 2r, or extend the chart's machine cover to absolute d₀ per rung. Rides with D3-LEMMA-RN-UNIF (foundations).

Ledger §1:

> **OBL-H5-REMOTE-THRESHOLD:** certify D3's remote bracket at the r-scaled threshold d ≥ 2r per rung, or extend the chart's machine cover to an absolute d₀. Rides with D3-LEMMA-RN-UNIF. State: OPEN.

Addendum §5: OBL-H5-REMOTE-THRESHOLD **unchanged** (still OPEN with chart side).

### Parent premise framing (chart side)

Frozen assembly OPEN list (`D1_ASSEMBLY_v2_2.md` §3 / §5):

> **OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD)

Register note (append-delta; frozen v2.2 body untouched) §5:

> **OPEN (validity premises of Theorem (2), reduced):** OBL-D1-PROMOTE — chart side (H5's r-scaled rung family + interpolation; …) and now also the uniform-band extension of the B4LOC/B2-far/far-route certificates … (the H3 band floor has already discharged the normalizer sub-part)

---

## What RUNG2 / RUNG3 already certify vs what remains

### Certified by the rung packages (artifact content — not premise discharge)

| Package | r | Headline claim | Totals digest (doc-cited; repro PASS) | Body-sha256 (doc-claimed; repro PASS) |
|---|---|---|---|---|
| `H5_RUNG2_2026-09-15.md` | **0.025** | I_hi/r³ = **664.3979**; C1 containment PASS; coverage self-test PASS; mutation 6/6; A/B byte-identical | `h5_totals_r0.025_v2.json` sha256 `f7697bcfa0fe32b5c87c8ef8adef5d4384ab1bda7cd4a7c4782fef8c99103fcb` | `91a34d83688b5110b886ab4fc168a55030343cc0da4a98e38347d3b0e6e48102` |
| `H5_RUNG3_2026-09-15.md` | **0.035355** | I_hi/r³ = **661.4712**; same class of fail-closed checks | `h5_totals_r0.035355_v2.json` sha256 `808d6901e3254a73181aa696904ed04567d401b6e58300454d489046c36f4f64` | `f4c3414fe8b072008f5f5c8c5676ec9e20944e115535e0f3dd9c13c1bb89c0d8` |

Also on the ladder (not these packages’ “THIS PACKAGE” headlines):

- **r = 0.05:** frozen v1 731.4311 / live v3 clean 647.8048 (cited in both rung docs).
- Named-lemma re-certification **at the certified rungs** (r-scaled form): H5-AXIS envelope constants from banked edge probes; patches 8/8; Z window live-recomputed; remote `B_remote = 19.55` carried symbolically (RUNG2/3 bodies).
- Modulus **display** (not band enclosure): dense c₂ sampling / κ = 1/8 fit in `H5_PROMOTE_UPDATE` §5 — still labeled DISPLAY; certified enclosure remains OBL-H5-JETMOD.

Drive mirrors of RUNG2/RUNG3 match local bytes/sizes (see `REPRO_CHECKS_H5_RN_UNIF.md`).

### What these packages do **NOT** discharge for the premise

- They do **not** close OBL-D1-PROMOTE as a validity premise (assembly §3/§5 still OPEN; addendum: chart side OPEN; H5_STATE title still “rung family executing”).
- They do **not** discharge **OBL-H5-JETMOD** (interval-r / G12-band 24-jet enclosure) — display ≠ certified band bounds (`H5_PROMOTE` §3; UPDATE §5; ledger “OPEN (display only)”).
- They do **not** discharge **OBL-H5-ZBAND hi side** (band LPW bracket) — addendum §5: remains OPEN.
- They do **not** discharge **OBL-H5-REMOTE-THRESHOLD** (r-scaled d ≥ 2r vs absolute d₀) — OPEN; rides with D3-LEMMA-RN-UNIF.
- They do **not** certify merges / envelopes at **r = 0.0177** or **r = 0.0125** (RUNG2 ladder still lists those cells incomplete; RUNG3 does not claim those rungs certified).
- Register note’s additional chart-side item (uniform-band extension of B4LOC/B2-far/far-route on the same interpolation machinery) is likewise not closed by RUNG2/3.

---

## Cell status at r=0.0177 and r=0.0125

**Named ladder status (freshest named N/70 in H5 docs):** `H5_RUNG2_2026-09-15.md` § “Rung ladder”:

> r = 0.0177: cells **42/70** (2 shards resuming), probes/patches complete  
> r = 0.0125: cells **21/70** (2 shards resuming), probes/patches complete

`H5_RUNG3_2026-09-15.md` certifies r = 0.035355 and updates the three-rung envelope table; it **does not** restate N/70 for 0.0177 / 0.0125.

Earlier / coarser status (superseded for those two r values by RUNG2’s ladder line, not a promotion):

- `H5_PROMOTE.md` §4 table: r = 0.0177, 0.0125 — cover “queued”, envelope “pending”.
- Ledger §1 (assembly-time): focuses on r = 0.025 / 0.035355 execution lines; does not give 0.0177/0.0125 N/70.

**Pointers:**

| Item | Local | Drive |
|---|---|---|
| Named 42/70 & 21/70 | `…/H5_closure/H5_RUNG2_2026-09-15.md` | `13eLq6qnVIxyGI-GE1L0T5TL5FKcj1qyZ` |
| RUNG3 (no N/70 update for these r) | `…/H5_RUNG3_2026-09-15.md` | `1IdE6t7TRBZqWLGRKNXBTi-pQfr2hVTcI` |
| Cell result logs (raw; not a named N/70 certificate) | `h5_results_r0.0177_*cells*.jsonl`, `h5_results_r0.0125_*cells*.jsonl` | *(no standalone Drive titles in H5 folder listing)* |

**Missing / not claimed:** no later H5_*.md after RUNG2 names a new certified cells N/70 for r = 0.0177 or 0.0125; no rung-merge package for those r values; do not treat raw jsonl line counts as a status promotion beyond the RUNG2 ladder lines.

---

## Next proof obligation (ONE paragraph)

Grounded in `H5_PROMOTE.md` §3(iii) + named sub-obligations, ledger §1, register note chart-side OPEN, and the RUNG2 ladder: the chart-side blocking proof step remains **OBL-H5-JETMOD** — evaluate interval-r lattice sums to obtain **G12-band / 24-jet band enclosures** so Î(r)/r³ ≤ F(G12-band) on each band [r_{k+1}, r_k] (finite per-band computation; falsifier = enclosure width exceeding the claimed modulus), which the sources explicitly separate from the already-shipped c₂ **display**. Concurrent engineering named by the freshest ladder (not a substitute for that proof step) is finishing cells+stitch+merge at **r = 0.0177** and **r = 0.0125** (still 42/70 and 21/70 in RUNG2); still-OPEN chart companions named alongside JETMOD are **OBL-H5-ZBAND hi side (band LPW bracket)** and **OBL-H5-REMOTE-THRESHOLD** (r-scaled d ≥ 2r vs absolute d₀; rides with D3-LEMMA-RN-UNIF). No source names a single exclusive “do this next then premise CLOSED” gate after RUNG2/3; this paragraph does not invent one.

---

## Paths & Drive IDs table

| Artifact | Local path (under `drive_peer_review_triage/`) | Drive ID |
|---|---|---|
| H5 folder | — | `1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y` (`03_H5_PROMOTION_AND_RUNG_CERTIFICATES`) |
| H5_STATE.md | `extract_sep15/K3_SIDE24_LB/UPPER2D/H5_closure/H5_STATE.md` | `1aR5OpFZsYCCeE-irltKpqiUchroz_3yg` (size 1583; verified) |
| H5_PROMOTE.md | `…/H5_closure/H5_PROMOTE.md` | *(local; not a separate file in Drive H5 folder listing)* |
| H5_PROMOTE_UPDATE_2026-09-15.md | `…/H5_closure/H5_PROMOTE_UPDATE_2026-09-15.md` | `1P-8dPVgy0yACgHghR-MN6nqWNMEvJ7cD` (size 5234; verified) |
| H5_RUNG2_2026-09-15.md | `…/H5_closure/H5_RUNG2_2026-09-15.md` | `13eLq6qnVIxyGI-GE1L0T5TL5FKcj1qyZ` (size 5311; verified) |
| H5_RUNG3_2026-09-15.md | `…/H5_closure/H5_RUNG3_2026-09-15.md` | `1IdE6t7TRBZqWLGRKNXBTi-pQfr2hVTcI` (size 3157; verified) |
| D1_ASSEMBLY_v2_2.md | `…/D1_assembly/D1_ASSEMBLY_v2_2.md` | `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` |
| D1_ASSEMBLY_v2_2_REGISTER_NOTE.md | `…/D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` |
| OBLIGATION_LEDGER.md §1 | `extract_sep15/K3_SIDE24_LB/RETURN_06/OBLIGATION_LEDGER.md` | *(in zip `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`; no standalone Drive title)* |
| ADDENDUM_2026-09-15_H3_BAND_FLOOR.md | `…/RETURN_06/ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` | *(zip / local)* |
| Totals r0.025 v2 | `…/H5_closure/h5_totals_r0.025_v2.json` | *(local; hash in RUNG2)* |
| Totals r0.035355 v2 | `…/H5_closure/h5_totals_r0.035355_v2.json` | *(local; hash in RUNG3)* |
| Premise blockers brief | `PREMISE_BLOCKERS_D1_PROMOTE_AND_RN_UNIF.md` | — |
| Repro checks (H5 hashes) | `REPRO_CHECKS_H5_RN_UNIF.md` | — |

**Verified Drive folder children (MCP `parentId = 1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y`):** H5_STATE, H5_PROMOTE_UPDATE, H5_RUNG2, H5_RUNG3, H5_TOTALS_ERRATA (`1s0Bmos-a4T2horx3GAj8vKFmLs6cNRu1`), H5_TOTALS_FREEZE (`1nEqNSXXzLg_oVk1AGQI-HnW_EfaRCEV8`).

---

*End of plan. No premise status promoted. No proofs invented beyond named source obligations.*
