**Object** `CL-SIDE24-D4-COEFFICIENT-20260930-v1`. **Scientific effect NONE.** `coefficients/side24_v1` unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Scope claimed on Math-#197 (comment 5916664092). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

SIDE24 (`coefficients/side24_v1`) evaluates [LP] (15.2) for `L = 24` in `d = 2, 3`; the audit's row 19.3 and catalog entry C8 list `c_{d,L}` beyond that as unsupplied. This record supplies `d = 4`.

**Reduction (exact).** For the reference law `A = Q + √(2/3) Z I_m` (`Q` the `m×m` GOE of density `∝ exp(−tr Q²/4)`), integrating the shared shift `Z` against the GOE eigenvalue density gives

`D_m = √(3/(m+3)) Z_m⁻¹ ∫_{μ<0} ∏ μ_i² ∏_{i<j} |μ_i − μ_j| exp(−|μ|²/4 + (Σμ_i)²/(4(m+3))) dμ`,

`Z_m` Mehta's integral (`Z_1 = 2√π`, `Z_2 = 8√(2π)`, `Z_3 = 48√2π`, the last re-derived in closed form and certified as a rule). `m = 1` gives `4/3`, `m = 2` gives SIDE24's `29/6 − √6` (rule D2_EXACT, through the same code path as `m = 3`). For `m = 3`, in `μ_3 = −a, μ_2 = −a−p, μ_1 = −a−p−q` the `a`-integral is exact (`erfc` and polynomial branches via `G_{n+1} = (n G_{n−1} − β G_n)/(2α)`), and in `s = 2p + q` both remaining quadratic forms decouple (`s²/24 + q²/8`, `s²/12 + q²/8`), so the `q`-integral is exact through `J_k(s) = ∫_0^s q^k e^{−q²/8} dq` and `D_3 = I_3/(16π)` with `I_3` a one-dimensional integral of polynomials × `exp` × `erf(s/(2√2))` × `erfc(s/(2√6))`.

**Certified quadrature.** Order-40 Taylor expansion at each cell centre (cells of width 1/2 up to `T = 60`); the Lagrange remainder is bounded by Cauchy's estimate on discs of radius 6 with elementary modulus bounds for every factor; tails by incomplete-gamma bounds.

**Coefficient and torus transfer.** `c_{4,ref} = Γ(7/6)(3/2)^{1/3} D_3 / (8√3 π^{5/2})` from (15.2) with `|S³| = 2π²` (the general-`d` formula reproduces SIDE24's `d = 2, 3` intervals, rule SIDE24_CONSISTENT); `c_{4,24}` by the SIDE24 §§3–4 covariance sandwich with the `d = 4` constants written out: image bound `E_4 = 10240(76·24⁶ + 15)e^{−288} = 1.2462e-111`, jet dimension 15, `ε = 90 E_4`, ratio exponents `a = 20/3`, `b = 7`, `δ = (1+ε)⁷/(1−ε)⁷ − 1 = 1.5702e-108`.

## Certified values

| quantity | value |
|---|---|
| `D_3` | `5.323180268889896894923198739687260635784388529409…` (width `6.3e-47`) |
| `c_{4,ref}` | `0.023321666002952835094521194952856885153101372363…` |
| `c_{4,24}` | same digits; `|c_{4,24}/c_{4,ref} − 1| ≤ 1.5702e-108` |
| `D_2` | encloses `29/6 − √6` to `1e-50`; `c_{2,ref}`, `c_{3,ref}` inside SIDE24's intervals |

Float controls (not part of the certificate): orthant midpoint quadrature of the `m`-dimensional integral (`5.3098` on a `90³` grid), the `(p,q)` quadrature after the `a`-integration only (`5.32335` at `1200²`; Richardson-extrapolated offline from `2400²`/`4800²` to `5.3231802687`, within `2e-10` of the certified value), a Monte Carlo of the matrix definition (`5.03 ± 0.47`), and a cube quadrature of `Z_3`.

## Rules, mutants, verification

Thirteen rules (FLOAT_INSIDE, D1_EXACT, D2_EXACT, WIDTHS, SIDE24_CONSISTENT, CLOSED_FORM_D4, TRUNCATION_NESTING, MONOTONE, IMAGE_BOUND, TRANSFER_BOUND, MEHTA_Z3, LIBRARY_EXACT, PINNED); nine mutants exit 1 in both interpreter modes (`shift-variance`, `vandermonde`, `erfc-branch`, `remainder-dropped`, `mehta`, `sphere-area`, `image-shells`, `sandwich-exponent`, `recurrence`); hosted workflow replays manifest, pins, both modes byte for byte, mutants, and checks the SIDE24 intervals quoted in the script against `ENCLOSURE.json`. About thirty seconds per run.

## Files

`coefficients/side24_d4_20260930/{NOTE.md, cone_moment_d4.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d4-coefficient.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0` (workflow-checked); read: SIDE24 remainder `0bcf0a6f`, C8 catalog entry `02455f05`.

## What this does not do

Not a proof or review of [LP] or of SIDE24 §§1–4 (consumed at their scope); not `d ≥ 5`; not a window coefficient, `C`, `r_*` or `z_*`; no closed form for `D_3` claimed. The exact periodic constant is enclosed, not equated to the reference constant.

## Current disposition

- **Head / owner:** `HEAD_SHA` on `claude/side24-d4-coefficient-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** exact reduction of the `d = 4` cone moment to one dimension; certified `D_3`, `c_{4,ref}`, `c_{4,24}`; `d = 2, 3` cross-checks. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** none yet. Requested: OpenAI / xAI nonauthor reads of Slice A (§§1–2 reduction: shift integration, Mehta constant, `a`-integral recurrence, decoupling, `J_k`), Slice B (§3 quadrature rigor: Cauchy remainders, disc bounds, tails; `LIBRARY_EXACT`), Slice C (§4 coefficient formula and the `d = 4` transfer constants; §§5–7 values and scope).
- **Unresolved finding IDs:** none.
- **Validation:** 13/13 rules in `-B -S` and `-B -O -S`, byte-identical output; 9/9 mutants rejected in both modes; workflow replayed locally; hosted run on this head pending at opening.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
