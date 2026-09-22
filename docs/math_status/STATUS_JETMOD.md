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
