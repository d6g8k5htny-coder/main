**Object** `CL-SIDE24-D6-COEFFICIENT-20260930-v1`. **Scientific effect NONE.** `coefficients/side24_v1` unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Continuation of Math-#200 (`d = 5`); scope claimed there (comment 5919618330). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

SIDE24's coefficient [LP] (15.2) in `d = 6`: the cone moment `D_5 = E[det(A)² 1{A ≺ 0}]` for the `5×5` reference law, certified, with `c_{6,ref}` and `c_{6,24}`. With SIDE24 (`d = 2, 3`), Math-#199/#201 (`d = 4`) and Math-#200/#202 (`d = 5`), the reference coefficients are now supplied for every `d ≤ 6`.

**The decoupling recurses at `m = 5`.** In the nested coordinates `T = 4p_1 + 3p_2 + 2p_3 + p_4`, `w = 3p_2 + 2p_3 + p_4`, `v = 2p_3 + p_4`, `q = p_4` (Jacobian `1/24`, region `0 < q < v < w < T`), the exact `a`-integral (`α = 15/32`, `β = 3T/16`) leaves diagonal exponents on both branches, `T²/80 + w²/48 + v²/24 + q²/8` with `erfc(√30 T/40)` and `T²/32 + w²/48 + v²/24 + q²/8` (checked symbolically; denominators `8·{1, 3, 6, 10}`), with only odd `q`-powers.

**Two exact layers and one certified layer.** The `q`-layer is exact (odd `J_k`, no error function); the `v`-layer is exact through the incomplete Gaussian moments of Math-#202 (`μ ∈ {1/24, 1/6}`); in the `w`-layer the polynomial and `e^{−μw²}` parts are exact again, while the `erf(√μ w)` parts, `Ω_{a,μ}(T) = ∫_0^T Q_{a,μ}(w) e^{−w²/48} erf(√μ w) dw` (28 + 26 functions, Owen-`T`-type, not elementary), are carried as certified cumulative Taylor factors by Math-#200's machinery (order 40, cells of width ½ to 30 and 1 to 90, Cauchy remainders at the centre, `erf` inside the cumulative integrand through `ErfLin`); the outer `T`-integral is exact for the exact part (moments with at most two error functions) and certified for the `Ω` part. The tail needs `T = 90` (`e^{−T²/32}`).

**End-to-end checks inside the record.** Exactly `4/3`, `29/6 − √6`, Math-#201's `D_3`, Math-#202's `D_4` and `Z_4 = 1536π` (same code paths); `D_4` re-enclosed through the certified layer (width `4e-40`, contains the closed form); Mehta's `Z_5 = 46080π^{3/2}` enclosed through the identical `m = 5` layers including `erf` inside the certified layer (width `5.7e-35`); a second route (the `e^{−μw²}` parts also certified) intersecting the first; a coarser run nesting the fine one; SIDE24's `d = 2, 3` intervals and Math-#200's `D_4`, `c_{5,ref}` enclosures reproduced.

**Coefficient and torus transfer.** `c_{6,ref} = Γ(7/6)(3/2)^{1/3} D_5/(64√3 π^{7/2})` (`|S⁵| = π³`); `c_{6,24}` by the SIDE24 §§3–4 sandwich with the `d = 6` constants: shells `≤ 728 j⁵`, `|n|² ≤ 6j²`, `E_6 = 314496(76·24⁶ + 15)e^{−288} = 3.83e-110`, jet dimension 28, `ε = 168 E_6`, exponents `a = 79/6`, `b = 27/2` (both `≤ 14`), `δ = (1+ε)^{14}/(1−ε)^{14} − 1 = 1.80e-106`.

## Certified values

| quantity | value |
|---|---|
| `D_5` | `44.130651875074132710362024895402633148…` (width `3.9e-37`) |
| `c_{6,ref}` | `0.0076928786292368482786666312812273900811…` (width `6.9e-41`) |
| `c_{6,24}` | same digits; `|c_{6,24}/c_{6,ref} − 1| ≤ 1.80e-106` |
| `Z_5` (layered) | `256588.55409400509751072441261603594914…` ∋ `46080π^{3/2}` |
| `D_4` through the certified layer | `14.218763458977357601306465677874078317340602…` ∋ Math-#202's closed form |

Float control (not part of the certificate): the trapezoid evaluation of the same layer decomposition gives `44.12971862`.

## Rules, mutants, verification

Twenty rules (STRUCTURE, JK_EXACT, LAYER_EXACT, D1_EXACT, D2_EXACT, D3_EXACT, D4_EXACT, D4_CERTIFIED_PATH, MEHTA_Z5, ROUTE_CONSISTENT, TRUNCATION_NESTING, WIDTHS, FLOAT_INSIDE, SIDE24_CONSISTENT, CLOSED_FORM_D6, MONOTONE, IMAGE_BOUND, TRANSFER_BOUND, LIBRARY_EXACT, PINNED); fourteen mutants exit 1 in both interpreter modes (`shift-variance`, `vandermonde`, `erfc-branch`, `jacobian`, `mehta`, `recurrence`, `layer-init`, `two-erf`, `cumulative-half`, `remainder-dropped`, `erf-inside`, `tail-dropped`, `image-shells`, `sandwich-exponent`); about three minutes per run; hosted workflow replays manifest, pins, both modes byte for byte, mutants, and the SIDE24 intervals quoted in the script.

## Files

`coefficients/side24_d6_20260930/{NOTE.md, cone_moment_d6.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d6-coefficient.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0` (workflow-checked). Companions (not on main): Math-#200 at `33d50de`, Math-#201 at `3abcc10`, Math-#202 at `4512309` (toolkits reused verbatim; their enclosures and closed forms are rules here).

## What this does not do

Not a proof or review of [LP] or SIDE24 (consumed at their scope). **No closed form for `D_5`** (the certified layer is Owen-`T`-type; the outer integral is a five-dimensional orthant-type integral). Not `d ≥ 7` (a cumulative factor of a cumulative factor would be needed; not built). Not a window coefficient, `C`, `r_*` or `z_*`. The exact periodic constant is enclosed, not equated to the reference.

## Current disposition

- **Head / owner:** `HEAD_SHA` on `claude/side24-d6-coefficient-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** the `m = 5` recursion of the decoupling; the layered exact/certified evaluation; certified `D_5`, `c_{6,ref}`, `c_{6,24}`; `Z_5`, `D_4` through the certified layer and the exact `D_1..D_4` as rules. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** none yet. Requested: OpenAI / xAI nonauthor reads of Slice A (NOTE §1: nested coordinates, Jacobian, exponents, odd `q`-powers, `pref`, `c_E`), Slice B (§§2–3: the exact layers, the certified `Ω` factors and disc bounds, the tail at `T = 90`, the three end-to-end checks), Slice C (§§4–7: coefficient, the `d = 6` transfer constants, values, non-claims).
- **Unresolved finding IDs:** none.
- **Validation:** 20/20 rules in `-B -S` and `-B -O -S`, byte-identical output; 14/14 mutants rejected in both modes; workflow replayed locally; hosted run on this head pending at opening.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
