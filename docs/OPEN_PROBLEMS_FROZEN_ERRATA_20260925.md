# OPEN_PROBLEMS frozen errata — 2026-09-25

This page is a reading aid. It is not a source of truth, it does not discharge
anything and it changes no flag. Scientific effect: **NONE**.

## Why the corrections live here

[`OPEN_PROBLEMS.md`](OPEN_PROBLEMS.md) is **byte-frozen** at 28,417 bytes,
sha256 `8f404f87e44d2861770eb86deed79710be3bc072aebeaff74d232304670dfb4d`.
Two pins hold it:

1. `SUPPLEMENTAL_DEPENDENCIES` in
   [`tools/twelve_project_check.py`](../tools/twelve_project_check.py).
2. `lambda_uniform/DEPENDENCIES.json` inside the hash-pinned archive
   [`research/campaigns/q0_twelve_20260920_v1.zip`](../research/campaigns/q0_twelve_20260920_v1.zip).
   The archived `lambda_repair.py` enforces that pin during CI replay. The
   archive members `tb_contact` and `bonferroni_eta` list the file as well.

An in-place edit would therefore break the pinned replay. The file stays
untouched. Corrections to its wording are recorded on this page instead. Line
numbers below refer to the frozen bytes above.

The wider downstream RN crosswalk is
[`DOWNSTREAM_RN_CROSSWALK_20260925.md`](DOWNSTREAM_RN_CROSSWALK_20260925.md).
The errata below are consistent with it and add no status.

## E1 — A5, lines 116–120 (SIDE24_CELL)

Frozen text, `OPEN_PROBLEMS.md` lines 116–120:

> The [local spatial candidate](RN_SIDE24_CELL.md) extends the full-mark bound
> to the square `[1999/2000,2001/2000]^2`: one refused parent is retained and
> four verified children form a complete partition with integral upper less
> than `6e-12`. This is a nonzero-area actual RN result. It covers only this
> local square, not `0.1 <= |y| <= 5`. The N0 L2 transport becomes too wide near

**Corrected scoped reading.**
[`RN_SIDE24_CELL.md`](RN_SIDE24_CELL.md) claims only the declared square
`C = [1-1/4000, 1+1/4000]^2` (note line 19). On `C` the full-mark integrand
upper is below `559/10^8 = 0.00000559` (note line 23; the stored exact
rational is about `5.5890032754e-6`). That bound is conditional on the imported
H3 floor.

The rectangle `[1999/2000,2001/2000]^2`, its four-cell split and the integral
upper `<= 3/500000000000` (below `6e-12`) are the engineering replay of
[`tools/rn_side24_spatial_check.py`](../tools/rn_side24_spatial_check.py)
(`RECTANGLE` at line 31). The note says so itself at lines 254–270. That replay
records `near_annulus_covered: false`, `independence_credit: 0` and
`h3_floor_status: EXPLICIT_IMPORTED_HYPOTHESIS`. The replay is not a claim of
the note and it is not discharge. For the note, the frozen text's domain
`[1999/2000,2001/2000]^2` is wrong: the note's domain is `C`.

**Complement (not covered by the note).** The rectangle minus `C` is
engineering replay only. The near annulus `0.1 <= |y| <= 5` is not covered.

## E2 — A5, lines 89–92 (Piece-2 annulus driver)

Frozen text, `OPEN_PROBLEMS.md` lines 89–92:

> *Repository state (code, not status):* the annulus driver recorded as
> unwritten now exists at `research/cover/`, with the accept / refine / reject /
> pending ledger as its first-class output and `total()` refusing to return
> while any cell is pending. Two exact facts it established about the recipe

**Corrected scoped reading.**
[`research/cover/`](../research/cover/README.md) is a region-generic
engineering engine that runs on reference integrands. It is **not** the Piece-2
annulus driver of CL-RNU-001. That driver is a certified Riemann sum for
`rho_spine = p_grad * window-cap / Z_lo` with per-cell error bounds (CL-RNU-001
§1 lines 20–21 and §5 lines 150–151; Drive
`16o9u_KgysJv9lnR93PNXCqc-MdrW9QL6`). The driver remains **REQUIRED CARRIER
ABSENT**. This agrees with
[`STATUS.md`](math_status/STATUS.md) lines 17, 53 and 80,
[`STATUS_RN_UNIF.md`](math_status/STATUS_RN_UNIF.md) line 27 and
[`PACKET.json`](math_status/PACKET.json) line 32
(`"piece2_annulus_driver": "UNWRITTEN"`).

### ABSENT / REPLACED crosswalk

| Node | Reading |
|---|---|
| Historical `rnu_env.py`, `rnu_white.py`, `rnu_t4.py` and `rnu_spine.py` | **ABSENT** ([SHA keyword addendum](context/SHA_KEYWORD_DISCOVERY_ADDENDUM_20260920.md)). Stop replaying them. |
| Reconstruction `rnu_white.py` (13,344 B, sha256 `859e1963b1b7c84f2c1f14b4b163fb5575ad41c5394d27c51c0d1ceff70d2dc3`) | A separate node. It is not a fill of the historical file. |
| Whitened log-q orders 2–4 | **REPLACED** by `MATH-20260917-b9c2_RN_WHITENED_JET_THEOREM.md` (Drive `1O4hMhhUhVtmCvvkRNqaqevByyTwnWe3b`) plus RN-FIELD-001 `CLOSE-20260917-b9c2_RN_FIELD_PROOF.md` (Drive `1xwENAPr6OCmn0AcfEEkljUbBSejYiq0B`; kernel [`rn_field.py`](../research/parallel/c2/sources/rn_field.py), sha256 `d9167ae821684f716fdbae11cd39eb13818422f50b6566f677d62c292e4a93c8`). Scope: `5 <= \|y\| <= 17` at `r = 1/20` only. Author-side, independence credit 0. |
| Complement of that replacement | Near annulus `0.1 <= \|y\| <= 5`; `r != 1/20`; `sqrt(chi^2)` derivatives of order `>= 2`; `kappa_y`; `kappa_cross`; the full composition `C_comp`. All open. |
| Next successors | **S1:** a proved floor `chi^2 >= delta > 0` on the far annulus (theorem §6 item 3). Then **S2:** `C_comp` (CL-RNU-003 T4 push, "Still required for a freeze" item 1). |
| STATUS.md "CERTIFIED q=2" (line 14) | Carrier absent. See PR #101. |

---

Scientific effect: **NONE**. OBL-H5-JETMOD stays **OPEN**. Source of truth
for the objects above stays **ABSENT**. No flag flips. Engineering is not
discharge.
