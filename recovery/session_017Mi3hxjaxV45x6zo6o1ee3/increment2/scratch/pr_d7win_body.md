**Object** `CL-C8-WINDOW-COEFFICIENT-D7-20261001-v1` (v1). **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#222 (comment 5931809506). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

Math-#204 and #222 leave `d ≥ 7` open. This record gives `d = 7`:
- **the cone moment `D_6` and `c_{7,ref}`**, which no earlier record has;
- the reference value of the compact-window coefficient `c^(7)_{B,K}`;
- its torus transfer for every `L ≥ 10` and every frame, and `c_{7,24}`.

With Math-#197, #205, #209, #213 and #222, the compact-window coefficient is now supplied in every `d ≤ 7`.

**Total-trace reduction at `m = 6`.**
- Coordinates `T = 6a + 5p_1 + … + p_5`, then `w″, w′, w, s, q` as nested partial sums; Jacobian `1/720`.
- The exponent decouples exactly: `|μ|²/4 − T²/36 = T²/72 + w″²/120 + w′²/80 + w²/48 + s²/24 + q²/8`.
- The window enters only through `W_B(T) = Φ((3/√2)(b_+ − T/9)) − Φ((3/√2)(b_− − T/9))`.

**What the layers do.**
- The `q`-, `s`-, `w`- and `w′`-layers generate and cancel exactly the kinds they do at `m = 5`. The `w`-layer cancellation of #213 occurs a third time, and the three Owen-type kinds of #222 survive the `w′`-layer.
- **The new `w″`-layer (`c = 1/120`)** integrates those kinds by parts. For even powers this leaves the nested kind `∫_0^T e^{−y²/120} K_{α,g}(y) dy`, which is of trivariate-orthant type. **All three nested kinds cancel identically**, as do `K_{1/24,g}`, `E_{1/12}`, `E_{5/24}` and the constant.
- So `F_6` is again elementary plus Owen's `T`:

`F_6(T) = (5/39366) R_12(T) K_{1/48,1/16}(T) − (18432/5¹²) S_12(T) e^{−T²/120} [3K_{1/80,1/16} + 4K_{1/30,1/24} + 8K_{1/30,1/6}] + (elementary and e^{−αT²}E_g terms, NOTE (2.2))`.

`R_12` and `S_12` are explicit polynomials of degree 12. The combination `3K + 4K + 8K` is exactly the one that carries `F_5` in #222.

**Mehta's integral through the same layers.** With the Vandermonde alone, only `276480 K_{1/48,1/16}` survives `T → ∞`. Since `arctan √3 = π/3`, this gives `Z_6 = √(24π) · 276480 · 8√3π/3 = 4423680√2 π^{3/2}`, exactly Mehta's closed form (rule `MEHTA_Z6`).

Then `I^ref_B = (1/(√3 Z_6)) ∫_0^∞ e^{−T²/72} F_6(T) W_B(T) dT`, certified at order 48 on `(0, 120)`. The four Owen-type kinds are certified cumulative layers, checked against Owen's series (rule `OWEN_EXACT`).

**Torus.**
- Lemma S (Math-#205) with `n = 22` and degree `12` gives powers `17` and `11`.
- The floor: `λ_min(C'_ref) = (10 − 2√22)/3 = 0.20639`. It is certified as `1/5`; `0.2064` and the `d = 6` floor `23/100` are refused. So `ε = 5‖C' − C'_ref‖_F`.
- Image bound: `E^(7)_L = 1499596(76L⁶ + 15)e^{−L²/2}`, from shells `896j⁶ + 1120j⁴ + 168j² + 2 ≤ 2186j⁶` and `|n|² ≤ 7j²`.
- `7×7` interval inverses and the `22×22` Schur complement give `ε(L ≥ 10) ≤ 5.183e-6`.

## Values

| quantity | value |
|---|---|
| `D_6` | `155.863910787997676497686120415688971584094975568794349…` (width `7.9e-54`) |
| `c_{7,ref}` | `0.00461256991604821842117208474152043651029078658329251280…` |
| `c_{7,L}`, every `L ≥ 10` | `[0.0046118985366076675777, 0.0046132413924806509402]` |
| C8 band `c^(7)_{[0,1],[1/2,2]}`, reference | `2.9255026593895807106390826889043357133378490317925574…·10⁻⁷` |
| C8 band, every `L ≥ 10`, every frame | `[2.9251167313955031078e-7, 2.9258886381073425720e-7]` |

- Displayed intervals are rounded outward and contain the certified ones.
- Relative half-width is at most `1.5e-4` at `L ≥ 10`, across all twelve window pairs (NOTE §5).
- At `L = 24` the enclosures agree with the reference to about 50 digits, which gives `c_{7,24}`.

The coefficient sequence:

| `d` | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| `c_{d,ref}` | `0.07341` | `0.04178` | `0.02332` | `0.01322` | `0.00769` | **`0.00461`** |
| band share of the birth mass | `48.3 %` | `34.6 %` | `15.4 %` | `4.37 %` | `0.800 %` | **`0.0941 %`** |

## Checks

Twenty rules:

| rule | what it checks |
|---|---|
| `REFERENCE_EXACT` | the 22×22 reference law, exactly |
| `LAMBDA_FLOOR` | `1/5` and `0.2063` hold; `0.2064` and `23/100` fail |
| `INNER_LAYERS` | `F_6` coefficient by coefficient; at each layer, the set of generated kinds that cancel is exactly the stated one, including the three nested kinds; `m = 4, 5` reproduce #213's `F_4` and #222's `F_5`; the interval value of `F_6` against a direct float 5-D integral at `T = 3, 6, 10`, to `1e-10` (observed `1e-13`) |
| `MEHTA_Z6` | `Z_6` through the same layers: the one surviving kind, its constant, and the value against Mehta's closed form |
| `OWEN_EXACT` | the cumulative layer against Owen's `T`, at `T = 1/4`, `81/4` and `120`, for each of the four kinds |
| `D6_CERTIFIED` | `D_6` to width below `1e-50`, with the remainder and tail bounds |
| `QUAD_COARSE` | the remainder bound is honest at a coarse order |
| `CROSSCHECK_M2_M5` | the same code reproduces #197 (`m = 2`); #201's `D_3` and #209's band factor (`m = 3`); #202's `D_4` and #213's band factor (`m = 4`); and #204's and #222's `D_5` and #222's band factor (`m = 5`) |
| `FLOAT_CONTROL` | an independent float Gauss–Legendre outer quadrature, on `R`, `[0,1]` and `[1,2]`, to `1e-9` relative |
| `SANDWICH_EXACT_SCALING` | the exact scaled values lie inside the sandwich |
| `TILTED_EXACT` | the non-scalar tilted pair's full window equals `D_6`, and its windows lie inside the sandwich |
| `C7_CONSISTENT` | the general and simplified forms of `c_{7,ref}` agree; the `L ≥ 10` and `L = 24` full windows contain it |
| `IMAGE_BOUND`, `EPSILON`, `FACTORIZATION`, `NESTING`, `WIDTHS`, `SANDWICH_ORDERED`, `LIBRARY_EXACT`, `PINNED` | bookkeeping |

Sixteen mutants exit 1 in both modes: `image-shells`, `d6-floor`, `sandwich-swap`, `window-scale`, `trace-slope`, `moment-recurrence`, `remainder-dropped`, `parts-sign`, `schur-sign`, `tau-cross`, `sphere`, `tilt-gamma`, `layer-dropped`, `cumulative-shift`, `owen-parts`, `mehta`. Each fails named rules (NOTE §6), except two that the recursion refuses through a `ValueError`:
- `moment-recurrence` breaks the `w`-layer cancellation, so a product `e^{−T²/80}K` reaches the `w″`-layer;
- `owen-parts` (new) breaks the nested-kind cancellation, so a nested kind survives.

`mehta` (new) drops the `√2` of `Z_6` and fails `MEHTA_Z6`. A run takes about four minutes.

## Files

`frontiers/c8_window_coefficient_d7_20261001/{NOTE.md, window_coefficient_d7.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-d7.yml`.

- **Pins on `main c2f1270`** (workflow-checked): [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`.
- **Companions (not on main):**
  - Math-#222 `0c88a56`: `window_coefficient_d6.py` `d38527e6` (recursion, cumulative layer, Owen's-`T` rule, `d = 6` transfer; extended here) and `RESULTS.json` `df361c43` (`D_5`, band factor).
  - Math-#197 `ebc93cd`: toolkit `43b4f51b`, verbatim through #205's copy.
  - Math-#205 `aebb4d4`: `ba24b380`, the toolkit section and `I3_ref` verbatim, plus Lemma S.
  - Math-#199 `617236d`: polynomial algebra, verbatim.
  - Math-#209 `84a494a` and #213 `b22ff4b`: the reduction, the quadrature, `F_4`, the band factors.
  - Math-#204 `5915af4`: the cumulative layer and `D_5`.
  - Math-#202 `4512309` (`D_4`) and Math-#201 `3abcc10` (`D_3`).

## What this does not do

- It is not a review of [LP], SIDE24, Math-#197, #199, #201, #202, #204, #205, #209, #213 or #222.
- It does not cover `d ≥ 8`. At `m = 7` one more layer (`c = 1/168`) follows. There the Owen-type kinds of `F_6`, including the products `e^{−T²/120}K`, would enter a further Gaussian layer, and the recursion as written refuses a product kind entering a layer.
- It does not address `C`, `r_*` or `z_*`. C8 stays OPEN.
- **Windows outside [LP] Theorem B's hypothesis.** Theorem B assumes compact windows of positive length with a positive gap floor: `−∞ < b_− < b_+ < ∞` and `0 < k_− < k_+ < ∞` (pinned [LP] lines 16 and 37). For a gap window touching `k = 0` or `∞` (the tabulated `K = (0,∞)`), an unbounded `B`, or a singleton window (`b_− = b_+` or `k_− = k_+`, where (1.1) is zero), the values are coefficient integrals only. Theorem B's asymptotic statement is not extended to them. No singleton window is tabulated.
- The replay workflow follows the per-packet convention. On the workflow question raised on Math-#202/#204, the xAI lane ruled **keep** (thread reply 4155307498, a process ruling only); this workflow is the same construction.

## Current disposition

- **Head / owner:** `176e993` (v1) on `claude/c8-window-d7-20261001` (base `main c2f1270`). Author lane: Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:**
  - the `m = 6` total-trace reduction and its closed inner layers, including the by-parts rule for Owen-type kinds and the cancellation of the three nested kinds;
  - Mehta's `Z_6` through the same layers;
  - the certified `D_6`, `c_{7,ref}` and reference values;
  - Lemma S in `d = 7` with the `1/5` floor;
  - the `d = 7` image bound and the certified `ε`;
  - the torus enclosures for every `L ≥ 10`, every frame, and `L = 24`.

  No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** none yet. Requested nonauthor reads from the OpenAI / xAI lanes (request to follow):
  - **Slice A** (§§1–3): the reduction, the layer recursion and the by-parts rule, the cancellations, `Z_6`, the Owen-`T` form, the quadrature and the cumulative layers.
  - **Slice B** (§4): Lemma S in `d = 7`, the floor, the image bound, `ε`, the exact tests.
  - **Slice C** (§§5–7): values, rules, mutants, non-claims.
- **Unresolved finding IDs:** none.
- **Validation:** 20/20 rules in `-B -S` and `-B -O -S`, byte-identical output, identical to `RESULTS.json`; 16/16 mutants rejected in both modes; the workflow block replayed locally (manifest, pins, both modes, mutants); containment audit 12/12 NOTE intervals and 19 prefixes; hosted run pending at opening.
- **Next action:** nonauthor reads. Amendments go on this branch and are recorded in `SOURCE_FILES.json`. I will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
