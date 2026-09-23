# Premise blockers — OBL-D1-PROMOTE & D3-LEMMA-RN-UNIF

Brief source note (no promotions): statuses and next steps below are quoted or tightly paraphrased from named local Sep-15 extract carriers and Drive Sep-15/16 intake only. No proofs invented; no premise status promoted. Where assembly-time and later same-day carriers differ on *execution progress*, both are reported; the frozen v2.2 OPEN list is not edited by later notes.

Mapped 2026-09-16 (America/Chicago) from `OPEN_PREMISES_WORKLIST.md` + deep read of the local lanes and Drive IDs listed there.

---

## OBL-D1-PROMOTE

### Already CLOSED (only what sources say is closed)

- **Normalizer sub-part DISCHARGED (uniform modulus).** ADDENDUM_2026-09-15_H3_BAND_FLOOR.md §4–§5:
  > “The NORMALIZER sub-part of OBL-D1-PROMOTE is DISCHARGED with uniform modulus”
  > “§1 OBL-D1-PROMOTE: **normalizer sub-part CLOSED** …; chart side OPEN”
- **G.7-scope normalizer AT THE RUNG** (pointwise R2 floor consumption) — CLOSED in frozen assembly §3 (separate from the continuum band routed into this premise):
  > “**G.7-scope normalizer AT THE RUNG (R2: …) … discharged**”
- **Rung-interval execution progress (not premise discharge):** H5_RUNG2 / H5_RUNG3 packages certify ladder rungs `r = 0.025` and `r = 0.035355` (I_hi/r³ = 664.3979 / 661.4712); assembly §3 also records “the H5 rung interval (v3 totals …)” among CLOSED *at-the-rung* items. These do **not** close OBL-D1-PROMOTE as a validity premise.

### Still OPEN (as sources name)

- Frozen `D1_ASSEMBLY_v2_2.md` §5:
  > “**OPEN — VALIDITY premises of Theorem (2):** OBL-D1-PROMOTE (H5 lane + D1; sub-obligations OBL-H5-JETMOD / OBL-H5-ZBAND / OBL-H5-REMOTE-THRESHOLD)”
- Ledger §1: “**Execution state (fresh):** EXECUTING”; sub-obligations named OPEN (JETMOD display-only; ZBAND hi-side; REMOTE-THRESHOLD). H3 addendum delta (governs over ledger §1 for the normalizer): chart side remains OPEN — “OBL-H5-JETMOD, OBL-H5-REMOTE-THRESHOLD unchanged; H5 rungs executing”; OBL-H5-ZBAND lo side now rides frozen H3 band floor, **hi side (band LPW bracket) remains OPEN**.
- Register note §5 (append-delta; v2.2 body untouched):
  > “**OPEN (validity premises of Theorem (2), reduced):** OBL-D1-PROMOTE — chart side (H5's r-scaled rung family + interpolation; …) and now also the uniform-band extension of the B4LOC/B2-far/far-route certificates … (the H3 band floor has already discharged the normalizer sub-part)”
- H5_STATE title: “OBL-D1-PROMOTE rung family executing”. H5_PROMOTE: certified band enclosure remains **OBL-H5-JETMOD** (modulus display ≠ certified interval-r band).

### Exact local paths + Drive file IDs (verified)

| Artifact | Local path | Drive ID |
|---|---|---|
| Assembly (frozen OPEN list) | `extract_sep15/K3_SIDE24_LB/UPPER2D/D1_assembly/D1_ASSEMBLY_v2_2.md` | `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` |
| Register note | `…/D1_assembly/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` |
| H5 folder (Drive) | — | parent `03_H5_PROMOTION_AND_RUNG_CERTIFICATES` `1fVdPlwhtL8hxMiRoD6TxNJaNAHOCDy4y` |
| H5_STATE.md | `…/UPPER2D/H5_closure/H5_STATE.md` | `1aR5OpFZsYCCeE-irltKpqiUchroz_3yg` |
| H5_PROMOTE.md | `…/H5_closure/H5_PROMOTE.md` | (local; Drive mirror not separately required by worklist) |
| H5_PROMOTE_UPDATE_2026-09-15.md | `…/H5_closure/H5_PROMOTE_UPDATE_2026-09-15.md` | `1P-8dPVgy0yACgHghR-MN6nqWNMEvJ7cD` |
| H5_RUNG2_2026-09-15.md | `…/H5_closure/H5_RUNG2_2026-09-15.md` | `13eLq6qnVIxyGI-GE1L0T5TL5FKcj1qyZ` |
| H5_RUNG3_2026-09-15.md | `…/H5_closure/H5_RUNG3_2026-09-15.md` | `1IdE6t7TRBZqWLGRKNXBTi-pQfr2hVTcI` |
| Obligation ledger §1 | `…/RETURN_06/OBLIGATION_LEDGER.md` | *(no standalone Drive title; in zip `1vSI-evINWskhXVyiZ0slt-rLT74sPpXH`)* |
| H3 band-floor addendum | `…/RETURN_06/ADDENDUM_2026-09-15_H3_BAND_FLOOR.md` | *(zip / local)* |
| HOLD checklist | `packages/hold/HOLD_OPEN_VALIDITY_PREMISES.md` | `1g3ixQx5rsPK07PSVwo8miBb78Q-LVOqt` |

### Next concrete proof/engineering step (source-grounded)

- **Assembly-time plan** (`RETURN_06/RUNNING_TASKS.md` §1, ~08:15 CT): “Next milestone: close cells+stitch at r = 0.025 → rung merge → r = 0.035355 merge.”
- **Later same-day packages supersede that milestone’s execution status** (not the premise): H5_RUNG2 / H5_RUNG3 certify those two rungs complete; H5_RUNG2 ladder still lists incomplete lower rungs — `r = 0.0177: cells 42/70 …`; `r = 0.0125: cells 21/70 …`.
- **Proof-side step named in H5_PROMOTE §3(iii):** “Band certification (the proof step)” — evaluate interval-r G12-band enclosures so Î(r)/r³ ≤ F(G12-band); content of **OBL-H5-JETMOD** (certified 24-jet band bounds). Register note additionally keeps the **uniform-band extension** of B4LOC/B2-far/far-route on the same interpolation machinery as OPEN chart-side content.
- No single source names one exclusive “do this next then premise CLOSED” gate after RUNG2/3; the grounded engineering next from the freshest ladder is continue/finish cells at **r = 0.0177 / 0.0125**, and the grounded proof next remains **OBL-H5-JETMOD band enclosure** (+ ZBAND hi / REMOTE-THRESHOLD as still-OPEN sub-obligations).

---

## D3-LEMMA-RN-UNIF

### Already CLOSED (only what sources say is closed *within* the item)

- Frozen assembly §1(b) / ledger §2 — **closed at the rung within the item**, not the lemma:
  > “Closed at the rung WITHIN the item: all station κ pieces (floor-certified), the exact far-zone main term (576 − 25π)J(ℓ), monotone-decay exact-kernel evidence.”
- Remote bracket floor-consistent consumption is recorded CLOSED **modulo** this lemma (assembly §3).

### Still OPEN

- Exec state / assembly §1(b):
  > “**(b) D3-LEMMA-RN-UNIF(r = 0.05)** (rung part precisely stated, **NOT closed**)”
- Ledger §2:
  > “**State:** NOT CLOSED (rung part precisely stated; uniform part frozen).”
- Register note §5: remains among reduced Theorem (2) validity OPEN premises.
- Drive CL-RNU-001 header: “CANONICAL IMPACT: NONE — the lemma is NOT closed by this document.”
- Drive RNU_EXECUTE_RECEIPT.md: “STATUS: PROPOSED. D3-LEMMA-RN-UNIF is NOT closed.” JSON: `"lemma_closed": false`.

### Exact local paths + Drive file IDs (verified)

| Artifact | Local path | Drive ID |
|---|---|---|
| Assembly §1(b), §5 | `…/D1_assembly/D1_ASSEMBLY_v2_2.md` | `1v4z492iAzk5NcOrR47IJHGkIgfsRACpC` |
| Register note | `…/D1_ASSEMBLY_v2_2_REGISTER_NOTE.md` | `1oFNNK4D6BEVuBjTL9jERKynXxnNL_f4t` |
| D3_PERCOLATION.md | `…/UPPER2D/D3_percolation/D3_PERCOLATION.md` | *(no standalone Drive title; in zip `1vSI-ev…`)* |
| d3_rn_unif.py (rung engine) | `…/D3_percolation/d3_rn_unif.py` | *(local / zip; described in CL-RNU-001)* |
| Ledger §2 / Exec state | `…/RETURN_06/OBLIGATION_LEDGER.md`, `00_EXECUTIVE_STATE.md` | *(zip / local)* |
| RN_UNIF_2026-09-16 folder | — | `1FmKnSQRpHc6EYCUEIl07FQDECUZsiEqG` |
| CL-RNU-001 closure plan | — | `16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6` |
| RNU_EXECUTE_RECEIPT.md | — | `1G8XM7uRYuT8mtK3JtJQuV1EH5MhZo6Hn` |
| RNU_EXECUTE_RECEIPT.json (same folder) | — | `1OXS646pxXoS_mOSSvjpPG3vDoZokMHLx` |

### Next concrete proof/engineering step (source-grounded)

From **CL-RNU-001 §5** (ordered remaining work to close Piece 1) and **RNU_EXECUTE_RECEIPT.json `next_required_for_freeze`** (verbatim list):

1. whitened `env_form` bounds on residual covariances at orders 2–4  
2. DS3 wired through `kappa_far` (not only FD of Hessian)  
3. valid T4 from `env_form`, not measured-scale  
4. full polar cover `d ∈ [5,17]` with θ-halving  
5. both-mode transcript + MUT-RN-1..5 + FREEZE rule-id  

CL-RNU-001 §5 item 1 names the immediate construction: **whitened χ² derivative bounds at orders 2–4** (exact first derivative already validated in receipts; Piece 2 Riemann-sum driver still unwritten, “smaller of the two”). Receipt STATUS remains PROPOSED — freeze / lemma close not claimed.

---

## Layering note

Frozen v2.2 vs register note — quote, don’t reconcile inventively:

- **Frozen body** (`D1_ASSEMBLY_v2_2.md` §5): five OPEN validity premises for Theorem (2), including OBL-D1-PROMOTE, D3-LEMMA-RN-UNIF, PERC-DECAY, OBL-B1-BRANCH(loop|B1), and the B4.loc dam-line tube certificate.
- **Register note** (header + §5):
  > “the frozen v2.2 body … is NOT altered; everything below takes effect at the next issuance (v2.3).”
  > “**Theorem-statement consequence (for v2.3):** Theorem (2)'s open validity premises reduce to OBL-D1-PROMOTE and D3-LEMMA-RN-UNIF … No frozen v2.2 line is edited by this note.”
- **HOLD** checklist still lists all five as open math (not packaging). This brief does not collapse the five-vs-two discrepancy; both layers stand as written.
