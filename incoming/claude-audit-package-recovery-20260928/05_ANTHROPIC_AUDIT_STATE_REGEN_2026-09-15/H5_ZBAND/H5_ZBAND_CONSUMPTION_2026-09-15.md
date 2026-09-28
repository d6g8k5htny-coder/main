# H5_ZBAND_CONSUMPTION — OBL-H5-ZBAND discharged at consumption grade (2026-09-15)

AUTHOR: Claude (Anthropic) · CLASS: OBL-DISCHARGE (H5 promotion lane, sub-obligation) · STATUS: PROPOSED
AUTHORITY: none · CANONICAL IMPACT: OBL-H5-ZBAND OPEN → DISCHARGED (consumption grade) upon promotion; no
theorem statement changes; no frozen carrier edited.

## Obligation (as frozen, H5_PROMOTE.md)
"OBL-H5-ZBAND: Z_r window bounds over bands (point-rung windows are banked; lo rides H3's c_Z·r² theorem-grade
uniform bound; the hi side needs the band version of the LPW bracket)."

## What discharges it
Two frozen H3 carriers, both no-whitening / no-MC / fail-closed, both-mode byte-identical, both in H3's MANIFEST:

| side | carrier (pinned by sha256 in the certificate) | certified statement, uniform on (0, 0.05] |
|---|---|---|
| lo | `h3_band_floor.py` a907eeed… → `band_normal.txt` dcb3c8e6… | E[G_r] = Z_r/r² ≥ 2.30659559567154 (per-cell floors in the table) |
| hi | `h3_band_ceil.py` b97c5428… → `ceil_normal.txt` 26d08534… | E[G_r] = Z_r/r² ≤ 3.74767948915996 (per-cell ceilings in the table) |

Direction discipline verified in `h5_run.py`/`h5_kernel.py`: `ctx.Z_lo` feeds every upper bound (I_hi) via 1/Z_lo
→ needs a floor; `ctx.Z_hi` feeds only I_lo (containment display) → needs a ceiling. The floor is therefore
load-bearing for Theorems (1)/(2); the ceiling only for the C1 containment ck.

## Band table (H3 cells → H5 promotion bands; min floor / max ceiling over covering cells)

| H5 band (r_{k+1}, r_k] | covering H3 cells | Z_r/r² ≥ | Z_r/r² ≤ |
|---|---|---|---|
| (0.035355, 0.05] | (0.03,0.04] ∪ (0.04,0.05] | 2.30659559567154 | 3.74767948915996 |
| (0.025, 0.035355] | (0.02,0.03] ∪ (0.03,0.04] | 2.41792545239159 | 3.72414748332433 |
| (0.0177, 0.025] | (0.01,0.02] ∪ (0.02,0.03] | 2.44022641168325 | 3.71138266685765 |
| (0.0125, 0.0177] | (0.01,0.02] | 2.45593199230367 | 3.69805856644731 |
| (0, 0.0125] | (0,0.0025] ∪ … ∪ (0.01,0.02] | 2.45593199230367 | 3.69805856644731 |

Note: the band floor (2.3066) is stronger than the c_Z = 1.6155 the H5 code currently takes as its H3 input
(factor 1.4278); consuming it tightens 1/Z_lo wherever the H3 floor rather than the rung's own bracket binds.

## Certificate
`h5_zband_consume.py` — parses every number from the pinned frozen transcripts (nothing typed from prose);
checks per cell floor < ceiling, floor > c_Z, ceiling ≤ U = 4; checks the seven cells tile (0, 0.05]; builds the
band table; pins both-mode identity of the consumed carriers. **exit 0**, normal ≡ -O byte-identical,
`H5-ZBAND-CONSUME PASS digest=cbe8603f1819227b03463f1cb6dd5171678882f3cffedad321cd09f3db9c869f`.
Mutations (each must fail, all do): `--mut=swap` (floor consumed as ceiling) → FAIL CELL-0.0025-floor<ceil;
`--mut=inflate_floor` → FAIL UNIFORM-floor; `--mut=band_floor_above_cell` → FAIL MUT-band-floor-above-cell.

## Register action (proposed)
OBL-H5-ZBAND: OPEN → **DISCHARGED (consumption grade)**. Remaining H5 sub-obligations: OBL-H5-JETMOD (display
only), OBL-H5-REMOTE-THRESHOLD (rides with D3-LEMMA-RN-UNIF). The next H5 rung issuance should cite this
carrier and set `C_Z_H3` per band from the table (or keep c_Z and cite the band floor as the uniform fallback).

## Falsification
Any r ∈ (0, 0.05] at which Z_r/r² lies outside [floor(cell), ceil(cell)] of its H3 cell refutes the consumed
carriers, and with them this discharge.
