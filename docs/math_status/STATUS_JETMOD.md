# STATUS — OBL-H5-JETMOD interval-r multi-jet push

**As of:** 2026-09-21 ~11:30 CT (America/Chicago)  
**Discipline:** fail-closed. No premise flips. Display ≠ certified enclosure.  
**RUNG2/3 do NOT discharge JETMOD.**

| Flag | Value |
|---|---|
| **OBL-H5-JETMOD** | **OPEN (display only)** |
| `discharges_OBL_H5_JETMOD` | **false** |
| `lemma_closed` | **false** |

---

## Jet count

| Stage | jets_done | Notes |
|---|---|---|
| Prior prototypes | **1** | `c2` / `c2_f` only |
| `jetmod_multi_jet_band.py` | **6** | MS-diag scaled: `c2_f,c2_fx,c2_fy,c2_fxx,c2_fxy,c2_fyy` |
| `jetmod_g12_ext_named.py` (this turn) | **8** | + `kappa_c2`, `s_f_fx` (source-named; **not** a 24-list invention) |

Receipts:

- `code_prototypes/jetmod_multi_jet_band_receipt.json` (6 MS-diag; **not re-run** this turn)
- `code_prototypes/jetmod_g12_ext_named_receipt.json` (delta +2)

---

## Enclosure widths vs modulus (diagnostic)

Band B0 = `[0.035355, 0.05]`, `n_sub=64`.  
Display residual bound `3.4e-6` (UPDATE §5). Struct κ=1/8 halfwidth ≈ `7.81e-5`.

| Quantity | subdiv width | ≫ struct hw? | Point mid ~ |
|---|---|---|---|
| `c2_f` | ≈ 0.01290 | yes | 0.4998 |
| `c2_fx` | ≈ 0.03869 | yes | 1.499 |
| `c2_fy` | ≈ 0.01290 | yes | 0.4998 |
| `c2_fxx` | ≈ **0.19341** (max among 6) | yes | 7.49 |
| `c2_fxy` | ≈ 0.03869 | yes | 1.499 |
| `c2_fyy` | ≈ 0.03871 | yes | 1.4995 |
| `kappa_c2` = (½−c2)/r² | ≈ 10.322061 | yes | → 0.125 |
| `s_f_fx` = G[0,4]/r | ≈ 0.01289 | yes | → −1 |

**Read:** falsifier *shape* exercised (widths ≫ claimed display/struct modulus). **Does not close OBL.**

---

## What H5 sources allow past G12 MS-diag (no invented 24-list)

| Candidate | Source grounding | Action |
|---|---|---|
| Six MS-diag `(G_ii−G_iS)/r²` | PinFrame G12 + promote c2 display | **Done** (multi_jet) |
| `kappa_c2` / next SD coeff | H5_PROMOTE.md:62; UPDATE §5 κ=1/8 | **Done** (ext) |
| Off-diag `G12[0,4]/r` etc. | PinFrame G12 product; G12-band step §3(iii) | **Done** one exemplar |
| Order **>2** jets (`fxxx`…) | — | **BLOCKED**: `DER`/`JETS_MS`/`JETS_H` stop at order 2 (`h5_kernel.py:33-35`) |
| y-station **18-jet** spatial | StationBox TM at **fixed** r | Out of scope for interval-r MS-band |
| Full **24-jet roster + p_J** | Named in OBL text only | **BLOCKED — UNENUMERATED** (Drive H5_ANALYTIC_ADVANCE; PROMOTE never lists them) |
| interval-r PinFrame T/A/Neumann | `self.r=mpf(r)` at `:461` | **BLOCKED** without cancelled Laurent form |
| `Î/r³ ≤ F(G12-band)` cover | PROMOTE §3(i)–(iii) | **BLOCKED** — no band-G12 cover runner |

**Hessian / higher:** PinFrame **does** admit Hessians (already in the 6). It does **not** admit higher-than-Hessian field jets.

---

## STOP condition (honored)

Further named “jets” toward 24 without a source roster would **invent** the 24-list. Hard blocker memo: roster + `p_J` missing; order>2 API missing; F(G12-band) missing.

**OBL-H5-JETMOD remains OPEN.** No promotions.

---

## Artifacts

| Path | Role |
|---|---|
| `code_prototypes/jetmod_multi_jet_band.py` | 6 MS-diag scaled jets |
| `code_prototypes/jetmod_multi_jet_band_receipt.json` | widths; discharges=false |
| `code_prototypes/jetmod_g12_ext_named.py` | +κ, +s_f_fx; inventory; stop |
| `code_prototypes/jetmod_g12_ext_named_receipt.json` | 6→8; discharges=false |
| `code_prototypes/STATUS_JETMOD.md` | this file |

---

## Walls recorded 2026-09-22 evening CT

**As of:** 2026-09-22 evening CT (America/Chicago)  
**OBL-H5-JETMOD remains OPEN.** These lines do not discharge OBL-H5-JETMOD.  
`discharges_OBL_H5_JETMOD` stays **false**. `lemma_closed` stays **false**.  
Figures below are recorded walls. This note does not re-derive them, and a display is not a certified enclosure.

Recorded partial, not a discharge: cancelled A/Ainv on B0 via LAT.k1 Taylor–Lagrange remainder tails is recorded at theorem grade, with Neumann ρ≈9.77e-4 < 1/2. The partial does not discharge OBL-H5-JETMOD.

Still refused, straddling, or absent:

- F(G12)/Lip **REFUSED**
- cover pipeline F **REFUSED** (point-r / `mpf(r)` wall)
- `KernelSeriesIv` accepts interval-r
- StationBox TM straddles 0 even at r-width 1e-9
- CoarseBox cancelled detgg **REFUSED**
- `eval_F(G12_box)` **REFUSED**
- Drive and local hunt for `explicit_interval_map_F_G12box_to_Rplus` is **EMPTY**. On hand is structural H5_PROMOTE wording only (`Î(r)/r³ = F(G12(r))`, and the band step `Î(r)/r³ ≤ F(G12-band)`), not that explicit interval map. This workspace tree has no object under that name.
- joint-(r,y) `cancelled_detgg_s_t2` identity **ABSENT** from the corpus
- documented detgg=ad-c^2 still straddles 0 (width ≈ 1.26)

### Chart factor (sharpened wall, still OPEN)

Recorded from existing cancelled series. This note does not re-derive the figure. The chart identity det(A)=det(G6)det(T)^2 is recorded as checked at point-r with rel_err ~4.5e-19. Cancelled-A negative Laurent poles cancel. That identity does not discharge OBL-H5-JETMOD.

A chart det(A_reg) that is positive on a thin r-subcell is not a StationBox detgg enclosure.

Missing object, absent from the corpus: a bridge φ such that det(Σ_gg)=φ(det A, det T, det G6, A_ser). Inventing φ/r^α is refused.

Receipt cited by name only, from local triage, not vendored in this tree: `jetmod_lat_k1_detgg_factor_probe_receipt.json`. This workspace tree has no file under that name. The name is not a certificate.

**OBL-H5-JETMOD remains OPEN.** No promotions. RUNG2/3 do NOT discharge JETMOD. `discharges_OBL_H5_JETMOD` stays **false**.

---

## Sibling sweep CLOSED EMPTY — 2026-09-22 evening CT

**As of:** 2026-09-22 evening CT (America/Chicago)  
OBL-H5-JETMOD stays **OPEN**. D3-LEMMA-RN-UNIF stays **OPEN**.  
`lemma_closed` stays **false**. `discharges_OBL_H5_JETMOD` stays **false**. `prizes_solved` stays false. `certified_C_H=false`.  
Sibling sweep CLOSED EMPTY does not discharge OBL-H5-JETMOD or D3-LEMMA-RN-UNIF. It does not FREEZE either. It is **NOT FREEZE-grade**.  
Recorded by name only. This note does not re-derive these walls. Receipts below are not vendored in this tree. A receipt name is not a certificate.

1. Interval Schur via Ainv: **REFUSED_IA_STRADDLES**. Receipt: `jetmod_interval_schur_detgg_via_ainv_receipt.json`. Point-r OK; positive-width (r,y) detgg straddles. Primary still missing: `correlated_C_Ainv_Ct_cancellation_under_joint_r_y` / `cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM`.
2. `eval_F(G12_box)` sibling: **REFUSED**. Exact missing object: `explicit_interval_map_F_G12box_to_Rplus`. Receipt: `jetmod_eval_F_G12box_sibling_probe_receipt.json`.
3. Joint (r,y) cancel rewrite corpus hunt: **EMPTY**. Receipt: `jetmod_joint_ry_cancel_rewrite_hunt_receipt.json`. Structural Schur `S1=Grr−C Ainv C^T` documented. No joint cancel formula.
4. φ(det A)→detgg bridge: still **ABSENT** (prior).

**OBL-H5-JETMOD remains OPEN.** D3-LEMMA-RN-UNIF remains OPEN. No promotions.

---

## Inventable probes (REFUSED receipts only) — 2026-09-23 CT

**As of:** 2026-09-23 ~11:26 CT (America/Chicago)  
**Runner:** `docs/math_status_probes/inventable_jetmod_probes.py`  
**Index:** `docs/math_status_probes/INVENTABLE_PROBES_INDEX.json`

Named walls only (sibling sweep CLOSED EMPTY — **no new walls**). Each probe *attempts* an inventable shortcut and writes an honest refusal:

| Named wall | Inventable attempt | Receipt status | File |
|---|---|---|---|
| Interval Schur via Ainv | invent φ≡0 / fantasy non-straddle | **REFUSED_IA_STRADDLES** | `inventable_interval_schur_ainv_REFUSED_receipt.json` |
| eval_F(G12_box) | invent max-norm / Lip·width as F | **REFUSED** | `inventable_eval_F_G12box_REFUSED_receipt.json` |
| Joint (r,y) cancel rewrite | invent cancel from point-r scales | **EMPTY** | `inventable_joint_ry_cancel_EMPTY_receipt.json` |
| φ(det A)→detgg bridge | invent φ/r^α | **ABSENT** | `inventable_phi_bridge_ABSENT_receipt.json` |

All four keep `discharges_OBL_H5_JETMOD: false`, `lemma_closed: false`, `inventable_attempt_accepted: false`.  
**OBL-H5-JETMOD remains OPEN.** These receipts do not discharge OBL-H5-JETMOD. Green ≠ discharge.
