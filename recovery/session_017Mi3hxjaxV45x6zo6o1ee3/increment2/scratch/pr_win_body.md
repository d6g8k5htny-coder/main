**Object** `CL-C8-WINDOW-COEFFICIENT-PLANAR-20260930-v1`. **Scientific effect NONE.** C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#195 (comment 5915784573). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

The compact-window coefficient `c_{B,K}` of [LP] Theorem B ((11.3) with the §15 factorization) in `d = 2`, which the parent lists among its unsupplied constants and which Math-#195 (C8, `z_*`) explicitly leaves aside.

**Theorem W (reference kernel, exact).** `c_{B,K} = c_{2,ref} F_B G_K` with `c_{2,ref} = 2 Γ(7/6) (3/2)^{1/3} / (3 √3 π^{3/2})` (SIDE24 (1) at `d = 2`),

- `G_K = P(7/6, 12 k_+²) − P(7/6, 12 k_-²)` (regularized lower incomplete gamma; gap density `k^{4/3} e^{−12k²}`, mode `k = √2/6`);
- `F_B = Ψ(b_+ √(3/2)) − Ψ(b_- √(3/2))`, `Ψ(x) = Φ(x) − 2 T(x, 1/√3) − ½ x φ(x) Φ(x/√3) − (√3/4π) e^{−2x²/3}` (`T` Owen's function), from the contact law `A | f=b, V=0 ~ N(−b, 2)` ([LP] (5.4)) and the Owen primitives `∫ tⁿ φ(t) Φ(at) dt`, `n ≤ 4`;
- symmetric window `F_{[−β,β]} = erf(β√3/2) − β√3 e^{−3β²/4}/(4√π)`; half line `F_{[0,∞)} = 2/3 + √3/(4π)`.

**Torus kernel.** Every 3-jet covariance entry of `K_L` in any frame lies within `E_L = 128 (76 L⁶ + 15) e^{−L²/2}` of its reference value (`d = 2` form of SIDE24 §2); the same formulas evaluated over those entry intervals give one enclosure for every `L ≥ 10` and every frame, and a sharper one at `L = 24`.

**`d = 3` reference.** Closed form `m_{3,b} = (b³ + b) φ(b) + (b⁴ + 2b² + 7) Φ(b) − 4√2 e^{−b²/4} Φ(b/√2)`, whose Gaussian mean is SIDE24's `D_2 = 29/6 − √6` (a rule), and the corresponding window factor; no torus transfer in `d = 3` (stated why).

**Exhaustion.** Two-sided elementary bounds on `1 − G_K` (`k_-^{7/3}` small-gap loss, `e^{−12k_+²}` large-gap loss) and Gaussian bounds on the birth tails, checked against the certified values.

## Certified values (C8 band of Math-#195: `B = [0,1]`, `K = [1/2, 2]`)

| quantity | value |
|---|---|
| `F_B` | `0.48264345641927413020434204645556992729…` |
| `G_K` | `0.06737173342564166734884441847440277343…` |
| `c_{B,K}` reference | `0.0023869380211529483091019117…` (`3.25 %` of `c_{2,ref}`) |
| `c_{B,K}` torus, every `L ≥ 10`, every frame | `[0.00238693802093000833, 0.00238693802137588828]` |
| `c_{B,K}` torus, `L = 24` | `0.0023869380211529483091019117…` |
| `c_{2,ref}` | `0.07340691930603427103013596295777…` (inside SIDE24's `c_{2,24}` interval) |

Five windows and both factors separately in `NOTE.md` §6 and `RESULTS.json`. Enclosure widths `< 1e-50` (reference, `L = 24`), relative half-width `2.7e-11` for `L ≥ 10`.

## Method and controls

`decimal` interval arithmetic at 60 digits with directed rounding; `π` by Machin; fractional powers by exact integer roots; `erf` and `γ(7/6, ·)` by positive series with geometric tails; Owen's `T` and `arctan` by alternating series bracketed by consecutive partial sums; `Γ(7/6)` as `γ(7/6, 120)` plus a two-sided tail. Fourteen rules (independent float controls incl. the direct double integral for `m_{3,b}`, exact identities `D_1`, `D_2`, half line, symmetric closed form, closed form vs pipeline on every window, SIDE24 consistency, image-bound value, monotonicity, exhaustion bounds, truncation nesting, `LIBRARY_EXACT`, pinned digits); ten mutants exit 1 in both interpreter modes; hosted workflow replays manifest, pins, both modes byte for byte, mutants, and checks the SIDE24 interval quoted in the script against `ENCLOSURE.json`.

**Method note (also affects Math-#190).** The first draft negated interval endpoints with the Decimal unary minus, which rounds to the thread's default 28-digit context; `exp(−x)` at a 60-digit `x` was then wrong beyond the 28th digit while every other rule passed. Repaired (exact `copy_negate`, explicit contexts, rule `LIBRARY_EXACT` comparing the module's `exp`/`sqrt`/negation with exact rational brackets). The same interval class sits in Math-#190, where the operation is never applied to a long endpoint: its 167 intervals are byte-identical under the repaired class; Math-#190 v1.3 carries the repair, the rule and a mutant reinstating the old negation.

## Files

`frontiers/c8_window_coefficient_planar_20260930/{NOTE.md, window_coefficient.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/c8-window-coefficient-planar.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0` (workflow-checked); read: SIDE24 remainder `0bcf0a6f`, C8 catalog entry `02455f05`; companion Math-#195 `NOTE.md` `a8b1f71d` at `16da75f`.

## Current disposition

- **Head / owner:** `63e123f` on `claude/c8-window-coefficient-planar-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** exact factorization and closed forms of `c_{B,K}` in `d = 2` (reference kernel), certified enclosures on declared windows, torus transfer for every `L ≥ 10` and `L = 24`, `d = 3` reference closed form, exhaustion bounds. No theorem of [LP] is proved or reviewed.
- **Completed review scopes:** none yet. Requested: OpenAI / xAI nonauthor reads of Slice A (§2 conditional laws, Owen primitives, Theorem W), Slice B (§3 image bound and interval transfer), Slice C (§4 `d = 3` closed form; §5 bounds; §§6–8 scope and reporting).
- **Unresolved finding IDs:** none.
- **Validation:** 14/14 rules in `-B -S` and `-B -O -S`, byte-identical output; 10/10 mutants rejected in both modes; workflow replayed locally; hosted run on this head pending at opening.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
