# HUNT BANK — joint (r,y) cancel / F-map / φ bridge
**As of:** 2026-09-23 ~11:25 CT · agent 6 · CoS 48h research lane  
**Scope:** H5-JETMOD / RN-UNIF phrasing hunt  
**STATUS draft?** **No** — no *new* wall or SoT find vs `code_prototypes/STATUS_JETMOD.md` walls digest dated **2026-09-22 18:18 CT**.

## Non-claims (fail-closed)
- Does **not** discharge OBL-H5-JETMOD.
- Does **not** claim lemma_closed / FREEZE / MUT-RN / premise promotion.
- Does **not** invent F-map, joint cancel identity, or φ(det A)→detgg bridge.
- Green CI / proxy stubs ≠ discharge.

## 1. Joint (r,y) cancel

| Source | Quote / state |
|---|---|
| `code_prototypes/jetmod_cancelled_detgg_s_t2_joint_r_y_stationbox_tm_receipt.json` | `status`: **REFUSED_MISSING_JOINT_R_Y_CANCEL_IDENTITY**; `cancellation_works`: false; exact missing: `cancelled_detgg_s_t2_under_joint_r_y_for_StationBox_TM` |
| Same receipt, corpus | CoarseBox y-only NUM/det **present** (`joint_r_y: false`); chart cancelled T/A **present** but ≠ StationBox detgg; Laurent coeffs of detgg in joint (r,y) **absent** |
| `code_prototypes/jetmod_joint_ry_cancel_rewrite_hunt_receipt.json` | `status`: **EMPTY_NO_DOCUMENTED_JOINT_RY_CANCEL_REWRITE_FOUND**; `found`: false |
| `code_prototypes/STATUS_JETMOD.md` § cancelled_detgg… / CURRENT HONEST WALLS | Positive-width (r,y) detgg_t.g0 **straddles** (width ≈ 1.263); y-only NUM/det ≠ joint (r,y) |

**Corpus phrasing to keep hunting:** documented series/factorization restoring non-straddling detgg/s_t2 under joint (r,y); not inventable from point-r ~1e-11 scale.

## 2. F-map

| Source | Quote / state |
|---|---|
| `code_prototypes/STATUS_JETMOD.md` § F-map SoT hunt (2026-09-22 18:17 CT) | receipt `jetmod_F_G12box_algebraic_IA_definition_hunt_receipt.json`; **found: no**; missing `explicit_interval_map_F_G12box_to_Rplus` |
| `code_prototypes/jetmod_eval_F_G12box_dedicated_receipt.json` | `option_B_eval_F_G12box_found`: false; H5_PROMOTE §3 is structural dependence, not algebraic map |
| `code_prototypes/jetmod_certified_F_G12_band_and_Lip_F_G12_receipt.json` | `status`: **REFUSED_MISSING_F_MAP_AND_LIP_OBL_OPEN**; `F_map_found`: false |
| Walls digest | Cover-pipeline F **REFUSED_COVER_APIS_FORCE_POINT_R**; named-8 / Ainv / LAT.k1 **not** composed into F |

**Corpus phrasing to keep hunting:** formula-grade interval map F: G12-box → R₊ with certified Lip; no fabrication from widths/endpoints.

## 3. φ bridge (det A → detgg)

| Source | Quote / state |
|---|---|
| `code_prototypes/jetmod_phi_bridge_detA_to_detgg_hunt_receipt.json` | `status`: **EMPTY_NO_DOCUMENTED_PHI_BRIDGE_FOUND**; `documented_phi_bridge`: false |
| `jetmod_interval_schur_detgg_via_ainv_receipt.json` | KEY INSIGHT: Schur uses **Ainv**, not φ(det A); applying documented Schur still straddles under joint (r,y); missing `det(Σ_gg)=φ(det A,…)` |
| STATUS_JETMOD walls | Invented r^α / φ(det A) / roster **refused** |

**Note:** RN-UNIF CL-RNU-002 “Φ” is Faà-di-Bruno / normal CDF symbol class — **not** the H5 φ(det A)→detgg bridge. Do not conflate.

## RN-UNIF gate (unchanged; not a new find)

From STATUS_JETMOD RN-UNIF gate note (same digest):
- `rnu_env.py` / LEMMA-RNU-ENV-RESCOV: **ABSENT / BLOCKED**
- ALLCELL-FDZ-Q4 package: **ABSENT**
- CL_ANTHROPIC_BUNDLE cites: **ABSENT** (pending operator upload)
- Gate chain ENV-RESCOV → … → CH-LIFT: **blocked**; no certified C_H

Local `repo_honesty_hunt/docs/RN_INNER_WEDGE.md` exists as honesty-lane notes — **not** promoted here as RN-UNIF SoT discharge.

## Drive glance (2026-09-23)

Newer Drive objects seen in metadata (not fully extracted this pass): RN_INNER_WEDGE_20260921 delivery zip; DG-SEARCH-RN-FAMILIES / RESEARCH_EXECUTION / Q0_TWELVE deliveries ~2026-09-20. **No STATUS promotion** until a formula-grade hit on the three targets is verified fail-closed inside those zips.

## Verdict
| Question | Answer |
|---|---|
| New wall vs 2026-09-22 18:18 CT STATUS? | **No** |
| New SoT find (joint cancel / F-map / φ bridge)? | **No** — all three still EMPTY/REFUSED |
| PR-ready STATUS under `docs/math_status/`? | **Not warranted** — hunting bank only |
| Next | Keep hunting corpus/Drive zips for documented joint cancel identity **or** eval_F(G12_box) formula **or** φ bridge; report only on real new wall/SoT find |

