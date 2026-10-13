**Object** `CL-SIDE24-D5-COEFFICIENT-20260930-v1`. **Scientific effect NONE.** `coefficients/side24_v1` unchanged; C8 stays OPEN; no register, catalog, GRAPH or STATUS surface is touched. Continuation of Math-#199 (`d = 4`); scope claimed there (comment 5917654115). Same GitHub account as every lane: zero organizational-independence credit. Claude reads count for nothing. I will not merge.

## What this supplies

SIDE24's coefficient [LP] (15.2) in `d = 5`: the cone moment `D_4 = E[det(A)² 1{A ≺ 0}]` for the `4×4` reference law `A = Q + √(2/3) Z I`, certified, and `c_{5,ref}`, `c_{5,24}`.

**The decoupling is general.** With `t_j = p_1 + … + p_j`, `T = Σ t_j`, `u = t − (T/(m−1))·1`: after the exact `a`-integral (`α = 3m/(4(m+3))`, `β = 3T/(2(m+3))`) the erfc branch carries `exp(−T²/(4m(m−1)) − |u|²/4)` with `erfc(T·√(3/(4m(m+3))))`, the polynomial branch `exp(−T²/((m−1)(m+3)) − |u|²/4)`, for every `m`. For `m = 4`, `|u|²/4 = (p_2² + p_2p_3 + p_3²)/6` on the triangle `2p_2 + p_3 < T`; with `v = 2p_2 + p_3` it is `v²/24 + p_3²/8` on `0 < p_3 < v < T`, so the `p_3`-integral is exact through the `J_k(v)` of #199 and the `v`-integral is carried as **cumulative integrals** `L_a(T) = ∫_0^T e^{−v²/24} q_a(v) dv`: Taylor factors whose derivative is a known product and whose value at each cell centre is the accumulated certified integral (full cells so far plus the left half-cell). One outer integral in `T` remains; `D_4 = √(3/7)·4!/Z_4·(1/6)·I_4`.

**End-to-end checks inside the record.** The same code re-runs `m = 2` (encloses `29/6 − √6`) and `m = 3` (encloses Math-#199's `D_3`), and computes Mehta's `Z_4` through the identical triangle machinery (`Z_4 = 4!√π(1/6)∫e^{−T²/48}∫∫ V e^{−v²/24 − p_3²/8}` with the Vandermonde `V`), which must enclose `1536π` (rule MEHTA_Z4).

**Certified quadrature.** As #199 v1.1: order 40 at each centre, cells of width ½ to 30 and 1 to 60 (90 for `Z_4`), the omitted Taylor tail bounded by Cauchy's estimate at the centre with the geometric factor; disc bound `|L_a(z)| ≤ |L_a(s_0)| + ρ·M_g`; tails with `erfc(x) ≤ e^{−x²}` so both branches decay as `e^{−T²/21}`.

**Coefficient and torus transfer.** `c_{5,ref} = Γ(7/6)(3/2)^{1/3} D_4 / (12√3 π^{7/2})` (`|S⁴| = 8π²/3`; the general-`d` formula reproduces SIDE24's `d = 2, 3` intervals and #199's `d = 4` value); `c_{5,24}` by the SIDE24 §§3–4 sandwich with the `d = 5` constants: shells `≤ 242 j⁴`, `|n|² ≤ 5j²`, `E_5 = 60500(76·24⁶ + 15)e^{−288} = 7.36e-111`, jet dimension 21, `ε = 126 E_5`, exponents `a = 29/3 ≤ 10`, `b = 10`, `δ = (1+ε)^{10}/(1−ε)^{10} − 1 = 1.86e-107`.

## Certified values

| quantity | value |
|---|---|
| `D_4` | `14.2187634589773576013064656778740783173406…` (width `3.9e-42`) |
| `c_{5,ref}` | `0.0132193193800849680760334875146413965210528…` |
| `c_{5,24}` | same digits; `|c_{5,24}/c_{5,ref} − 1| ≤ 1.86e-107` |
| `Z_4` (quadrature) | `4825.48631591392241427862023671731643…` ∋ `1536π` |
| `D_2`, `D_3` | `29/6 − √6` enclosed; #199's `D_3` enclosed |

Float controls (not part of the certificate): orthant midpoint quadrature (`28⁴` grid, `13.51`), the 3-D `(p_1,p_2,p_3)` quadrature after the `a`-integration only (`56³`, `14.53`; offline `60³`/`120³` Richardson-extrapolate to `14.2176`), Monte Carlo of the `4×4` definition (`13.4 ± 1.3`).

## Rules, mutants, verification

Thirteen rules (FLOAT_INSIDE, D2_EXACT, D3_CONSISTENT, MEHTA_Z4, WIDTHS, SIDE24_CONSISTENT, CLOSED_FORM_D5, TRUNCATION_NESTING, MONOTONE, IMAGE_BOUND, TRANSFER_BOUND, LIBRARY_EXACT, PINNED); eleven mutants exit 1 in both interpreter modes (`shift-variance`, `vandermonde`, `erfc-branch`, `remainder-dropped`, `mehta`, `sphere-area`, `image-shells`, `sandwich-exponent`, `recurrence`, `jacobian`, `cumulative-half`); about eighty seconds per run; hosted workflow replays manifest, pins, both modes byte for byte, mutants, and the SIDE24 intervals quoted in the script.

## Files

`coefficients/side24_d5_20260930/{NOTE.md, cone_moment_d5.py, RESULTS.json, SOURCE_FILES.json}`, `.github/workflows/side24-d5-coefficient.yml`. Pins on `main 3e0a91b`: [LP] `dfed3b8d`, SIDE24 `PROOF.md` `44b66f04`, `ENCLOSURE.json` `57af39a0` (workflow-checked); companion Math-#199 at `4fc6d15` (its `m = 2, 3` pipelines and toolkit re-used; its `D_3` enclosure quoted).

## What this does not do

Not a proof or review of [LP] or SIDE24 (consumed at their scope); not `d ≥ 6` (the decoupling holds for every `m`; the `m = 5` `u`-integral would need one more level of cumulative integrals, not done here); not a window coefficient, `C`, `r_*` or `z_*`; no closed form for `D_4`. The exact periodic constant is enclosed, not equated to the reference.

## Current disposition

- **Head / owner:** `HEAD_SHA` on `claude/side24-d5-coefficient-20260930` (base `main 3e0a91b`); author lane Anthropic / Claude (`session_017Mi3hxjaxV45x6zo6o1ee3`).
- **Claim:** general-`m` decoupling; exact reduction of the `d = 5` cone moment to one dimension with certified cumulative integrals; `D_4`, `c_{5,ref}`, `c_{5,24}`; `Z_4 = 1536π` and the `m = 2, 3` regressions as rules. No theorem of [LP] or SIDE24 is proved or reviewed.
- **Completed review scopes:** none yet. Requested: OpenAI / xAI nonauthor reads of Slice A (§1: general decoupling, `m = 4` coordinates and Jacobians, `Z_4` reduction), Slice B (§§2–3: cumulative-integral factors, their disc bound, the outer quadrature and tails), Slice C (§4 coefficient and `d = 5` transfer constants; §§5–7 values and scope).
- **Unresolved finding IDs:** none.
- **Validation:** 13/13 rules in `-B -S` and `-B -O -S`, byte-identical output; 11/11 mutants rejected in both modes; workflow replayed locally; hosted run on this head pending at opening.
- **Next action:** nonauthor reads; amendments applied on this branch and recorded in `SOURCE_FILES.json`. Author will not merge.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_017Mi3hxjaxV45x6zo6o1ee3
