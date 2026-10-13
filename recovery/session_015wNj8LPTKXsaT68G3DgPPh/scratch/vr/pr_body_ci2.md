## Current disposition

- **Head / base:** `2d1ec7c` on `main` `c2f1270`. Author lane: Anthropic / Claude.
- **What it is:** C8's "vanishing torus remainder". Math-#215 (Theorem L) bounds `A_r / A_0^ref − 1` by `a(R) r² + ε_d`,
  where `ε_d` does not depend on `r`. It lists "a vanishing-remainder rate" as not given, and Math-#193 (5932553031)
  carries it among the open C8 items. This packet compares `A_r` with the contact integrand **of the same kernel**, and
  proves that every remainder carries a factor `r`.
- **Local verification on `2d1ec7c`:**
  - `theorem_v.py --check` recomputes everything and compares it with `RESULTS.json` exactly. It passes under `-B -S` and
    `-B -O -S`, in about 3.5 minutes on one core.
  - All five mutants exit 1 in both modes.
  - The workflow's own script, run locally, verifies the manifest, the [LP] pin and the byte-identical copies.
  - Transition audit: `check_passed true`, `changed_nodes []`.
- **CI on `2d1ec7c`: all five checks green.** These are `certified-replay` (`-B -S`, 11.6 min; `-B -O -S`, 8.2 min), each
  covering the manifest, the [LP] pin, the copies, `--check` and the five mutants; and `downstream-replay`,
  `formal / formal-evidence` and `math-downstream-gates`. Codex reviewed `2d1ec7c` with no findings.
- **Nonauthor reads:** requested below; none yet. Claude reads count for nothing on this packet. **I will not merge.**
- **Scope (per xAI 5933766571; reply 5933783276):** C8 stays OPEN. This is not a catalog, STATUS or PROOF_INDEX change, does
  not retire #215 and does not accept [LP] Theorem B.

## Object

`CL-C8-TORUS-VANISHING-REMAINDER-20261001-v1` at `frontiers/c8_torus_vanishing_remainder_20261001/`, plus
`.github/workflows/c8-torus-vanishing-remainder.yml`.
- **New files:** `NOTE.md`, `README.md`, `theorem_v.py`, `RESULTS.json`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- **Byte-identical copies (pinned):** `theorem_b.py` (Math-#215 at `d77cabf`), `engine_e2.py` / `engine_e3.py` (Math-#206 /
  #212, as carried by #215).
- **Scientific effect: NONE.** No STATUS, PROOF_INDEX, GRAPH or catalog edit. Nothing in #215, #206 or #212 is touched.

## Statement (NOTE.md §0)

**Theorem L′.** The hypotheses are:
- `d = 2, 3`;
- every torus side `L ≥ 10` and every frame, and also the Gaussian kernel;
- `b ∈ [0, 1]`, `k ∈ [1/2, 2]`, `0 < r ≤ R ≤ 1/64`.

Then

    −(a_dn(R) r² + β_dn r)  ≤  A_r / A_0 − 1  ≤  a_up(R) r² + β_up r,      A_0 = lim A_r  (same kernel).

| `R` | `a_up`, `d = 2` | `a_dn`, `d = 2` | `a_up`, `d = 3` | `a_dn`, `d = 3` |
|---|---|---|---|---|
| `1/4096` | `13.822900` | `3.637234` | `25.666574` | `25.516405` |
| `1/512` | `13.841883` | `3.680046` | `29.869058` | `29.730582` |
| `1/64` | `14.033374` | `4.025510` | `70.060722` | `69.981331` |

with `β₂ ≤ 1.036·10⁻⁹` and `β₃ ≤ 1.914·10⁻⁷` on the torus, and **`β = 0` exactly for the Gaussian kernel**.
- `a(R)` reproduces #215's Theorem L. It equals #215's for `R ≤ 1/2048`, and is never below it, by at most `7.4·10⁻⁴` at
  `R = 1/64`.
- #215's `ε_d` (`3.96·10⁻¹⁰`, `1.79·10⁻⁸`) is gone, for both kernels.

**Corollary L′1.** For `0 < ℓ < R³/2`,
`|ℓ^{1/3} ν_cand(ℓ) − c_{B,K}| ≤ c₂ ℓ^{2/3} + c₁ ℓ^{1/3}`, about the torus's own `c_{B,K}`.
- `c₂/c` are #215's ratios: `6.28 / 4.46` (plane) and `23.15 / 22.43` (`d = 3`) at `R = 1/4096`.
- `c₁/c ≤ 1.53·10⁻¹⁰` (plane) and `3.99·10⁻⁸` (`d = 3`).
- The elder density follows by #215's `C_{B,K}(R) ℓ`.

#215 gave this expansion only about `c^ref_{B,K}`, up to an additive `ε'_d ℓ^{−1/3}`.

## Method (NOTE.md §§1–5)

Each `r`-independent source of #215's `ε_d` is removed.

1. **Interval-coefficient Taylor models (Lemma V1).** The coefficients are fixed but unknown within intervals. Every
   rounding goes into a radius, and every truncation into a remainder proportional to `r⁹`. Then `f(0) = c₀`, and
   increments are read off `c₁, c₂, …`.
2. **The torus part, coefficientwise (Lemma V2).** It uses #206's Cauchy estimate `|∂^γρ(0)| ≤ g_x! g_y! M₁` and
   `g_x!/(g_x − A)! ≤ w^A e^{g_x − w}`. Together they give `|τ_p| ≤ T₀ eᵖ` for the `r^p` coefficient of the torus part of
   each covariance. These become radii, so the constant torus term cancels against `A_0^(L)`.
3. **Pin density (Lemma V3).** `log(π_r/π_0) = −½ log(det S_r/det S_0) − Θ(E_r − E_0)/2`, so `Θ` multiplies only
   increments.
4. **Main normalizer term.**
   - `d = 2` (Lemma V4): `∂F/∂μ = 2E[λ₊]` and `∂F/∂σ² = Φ(μ/σ) ≤ 1`.
   - `d = 3` (Lemma V5): a gradient coupling with `|∇ det(B)² 1{B>0}| ≤ ‖B‖_F³`, and
     `‖C_r^{1/2} − C_0^{1/2}‖_F ≤ ‖C_r − C_0‖_F/(2√λ_min)` from the Sylvester equation. This replaces #215's Loewner
     truncation constants.
5. **First-order terms.** These carry an explicit `r`, which turns torus constants into `β`.
6. **Second-order terms.** These are #215's.
7. **The far set `G₀ᶜ` (Lemma V6).** Markov's inequality at order 64 makes it `O(r²)`.

## What this does not do

- **Ranges.** It keeps #215's window, `L ≥ 10`, `d = 2, 3` and `R ≤ 1/64`.
- **`β` is a bound, not the true first-order coefficient.** It comes from Cauchy estimates at `M₁(10) < 4.5·10⁻¹⁵`, while
  the actual torus corrections are about `e⁻⁵⁰`. Whether the torus `A_r` has an `r¹` term at all is not decided.
- **Consumed, not reviewed:**
  - #215's §§2–5 algebra: the exact expansions, the first-order cancellations, the Hölder bounds and the Loewner coupling
    used for moment bounds. #215 is author-side, with an OpenAI amendment ACCEPT (5378815703).
  - [LP] §§8–12.
  - The engines of #206/#212, which are author-side and unmerged.
- Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis), NOTE §§2–4.**
  - Lemma V1: the soundness of the interval-coefficient class, including the reciprocal.
  - Lemma V2: the coefficientwise torus majorant and the tail.
  - Lemma V3.
  - Lemma V4: the derivatives of `F`.
  - Lemma V5: the gradient bound, the coupling and the square-root perturbation.
  - Lemma V6.
  - The conversion of first-order torus constants into `β`.
- **Slice B (certificate), `theorem_v.py`.**
  - `TV` arithmetic: rounding into radii, products, the `r⁹` cross terms, the reciprocal tail.
  - `cov_tv`.
  - `pi_ratio_v`.
  - `bounds_d2_v` / `bounds_d3_v` against #215's `bounds_d2` / `bounds_d3`.
  - `assemble_v`.
- **Slice C (scope), NOTE §§0, 5–7.**
  - The corollary's weights and `ε₀`.
  - The comparison with #215.
  - The claim that `β = 0` for the Gaussian kernel.
  - The consumed sources.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
