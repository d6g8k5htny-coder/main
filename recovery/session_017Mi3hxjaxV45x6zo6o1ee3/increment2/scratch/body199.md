**Object** `CL-SIDE24-D4-COEFFICIENT-20260930-v1` (v1.2 at `617236d`). **Scientific effect NONE.** `coefficients/side24_v1` unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#197 (comment 5916664092). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

SIDE24 (`coefficients/side24_v1`) evaluates [LP] (15.2) for `L = 24` in `d = 2, 3`; the audit's row 19.3 and catalog entry C8 list `c_{d,L}` beyond that as unsupplied. This record supplies `d = 4`.

**Reduction (exact).** For the reference law `A = Q + √(2/3) Z I_m` (`Q` the `m×m` GOE of density `∝ exp(−tr Q²/4)`), integrating the shared shift `Z` against the GOE eigenvalue density gives

`D_m = √(3/(m+3)) Z_m⁻¹ ∫_{μ<0} ∏ μ_i² ∏_{i<j} |μ_i − μ_j| exp(−|μ|²/4 + (Σμ_i)²/(4(m+3))) dμ`,

`Z_m` Mehta's integral (`Z_1 = 2√π`, `Z_2 = 8√(2π)`, `Z_3 = 48√2π`, the last re-derived in closed form and certified as a rule). `m = 1` gives `4/3`, `m = 2` gives SIDE24's `29/6 − √6` (rule D2_EXACT, through the same code path as `m = 3`). For `m = 3`, in `μ_3 = −a, μ_2 = −a−p, μ_1 = −a−p−q` the `a`-integral is exact (`erfc` and polynomial branches via `G_{n+1} = (n G_{n−1} − β G_n)/(2α)`), and in `s = 2p + q` both remaining quadratic forms decouple (`s²/24 + q²/8`, `s²/12 + q²/8`), so the `q`-integral is exact through `J_k(s) = ∫_0^s q^k e^{−q²/8} dq` and `D_3 = I_3/(16π)` with `I_3` a one-dimensional integral of polynomials × `exp` × `erf(s/(2√2))` × `erfc(s/(2√6))`.

**Certified quadrature.** Order-40 Taylor expansion at each cell centre (cells of width 1/2 up to `T = 60`); the omitted tail is bounded by Cauchy's estimate at the centre on discs of radius 6 with the geometric factor and elementary modulus bounds for every factor; tails by incomplete-gamma bounds.

**Coefficient and torus transfer.** `c_{4,ref} = Γ(7/6)(3/2)^{1/3} D_3 / (8√3 π^{5/2})` from (15.2) with `|S³| = 2π²` (the general-`d` formula reproduces SIDE24's `d = 2, 3` intervals, rule SIDE24_CONSISTENT); `c_{4,24}` by the SIDE24 §§3–4 covariance sandwich with the `d = 4` constants written out: image bound `E_4 = 10240(76·24⁶ + 15)e^{−288} = 1.2462e-111`, jet dimension 15, `ε = 90 E_4`, ratio exponents `a = 20/3`, `b = 7`, `δ = (1+ε)⁷/(1−ε)⁷ − 1 = 1.5702e-108`.

## Certified values

| quantity | value |
|---|---|
| `D_3` | `5.3231802688898968949231987396872606357843885294…` (width `6.5e-47`; Math-#201's closed form `(50π + 200 arctan 2 − 228)/(9π)` lies inside, 47 common digits) |
| `c_{4,ref}` | `0.023321666002952835094521194952856885153101372363…` |
| `c_{4,24}` | same digits; `|c_{4,24}/c_{4,ref} − 1| ≤ 1.5702e-108` |
| `D_2` | encloses `29/6 − √6` to `1e-50`; `c_{2,ref}`, `c_{3,ref}` inside SIDE24's intervals |

Float controls (not part of the certificate): orthant midpoint quadrature of the `m`-dimensional integral (`5.3098` on a `90³` grid), the `(p,q)` quadrature after the `a`-integration only (`5.32335` at `1200²`; Richardson-extrapolated offline from `2400²`/`4800²` to `5.3231802687`, within `2e-10` of the certified value), a Monte Carlo of the matrix definition (`5.03 ± 0.47`), and a cube quadrature of `Z_3`.

## Rules, mutants, verification

Thirteen rules (FLOAT_INSIDE, D1_EXACT, D2_EXACT, WIDTHS, SIDE24_CONSISTENT, CLOSED_FORM_D4, TRUNCATION_NESTING, MONOTONE, IMAGE_BOUND, TRANSFER_BOUND, MEHTA_Z3, LIBRARY_EXACT, PINNED); nine mutants exit 1 in both interpreter modes (`shift-variance`, `vandermonde`, `erfc-branch`, `remainder-dropped`, `mehta`, `sphere-area`, `image-shells`, `sandwich-exponent`, `recurrence`); hosted workflow replays manifest, pins, both modes byte for byte, mutants, and checks the SIDE24 intervals quoted in the script against `ENCLOSURE.json`. About thirty seconds per run.

## Files

`coefficients/side24_d4_20260930/{NOTE.md, cone_moment_d4.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d4-coefficient.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0` (workflow-checked); read: SIDE24 remainder `0bcf0a6f`, C8 catalog entry `02455f05`.

## What this does not do

Not a proof or review of [LP] or of SIDE24 §§1–4 (consumed at their scope); not `d ≥ 5` (Math-#200 supplies `d = 5`); not a window coefficient, `C`, `r_*` or `z_*`; no closed form for `D_3` claimed here (Math-#201 supplies one). The exact periodic constant is enclosed, not equated to the reference constant.

## Revisions

- **v1.1 (`4fc6d15`, Codex review on `54dd458`, two P2 findings taken):** Cauchy's estimate applied at the centre with the geometric factor (thread 4148001628); `Z_3` decimal typo. Values unchanged.
- **v1.2 (`617236d`):** rule TRANSFER_BOUND in the proving direction, certifying `13ε ≤ δ ≤ 15ε` for every represented value and recording the intersection with `14ε` (the same construction Codex flagged on Math-#200, thread 4148778791); NOTE §9 Revisions added. No certified value changed.

## Current disposition

- **Head / owner:** `617236d` on `claude/side24-d4-coefficient-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** exact reduction of the `d = 4` cone moment to one dimension; certified `D_3`, `c_{4,ref}`, `c_{4,24}`; `d = 2, 3` cross-checks. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** Codex code review at `54dd458` (two P2 findings, taken in v1.1, threads resolved). Requested: OpenAI / xAI nonauthor reads of Slice A (§§1–2 reduction: shift integration, Mehta constant, `a`-integral recurrence, decoupling, `J_k`), Slice B (§3 quadrature rigor: Cauchy remainders, disc bounds, tails; `LIBRARY_EXACT`), Slice C (§4 coefficient formula and the `d = 4` transfer constants; §§5–7 values and scope).
- **Unresolved finding IDs:** none.
- **Validation:** 13/13 rules in `-B -S` and `-B -O -S`, byte-identical output; 9/9 mutants rejected in both modes; workflow replayed locally at `617236d`; hosted run 4/4 green at `4fc6d15`, run at `617236d` pending.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
