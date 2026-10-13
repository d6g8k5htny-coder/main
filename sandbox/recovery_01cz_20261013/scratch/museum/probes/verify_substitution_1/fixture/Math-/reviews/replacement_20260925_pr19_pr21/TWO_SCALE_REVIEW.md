# Additive review of the two-scale addendum

**Claim.** This is the one next assignment on the existing session, after delivery of the PR19/PR21 review at `e34077f3f2482e3c9850afd26085f37afc7472a4`. That review is unchanged. This file does not reopen it.

**Immutable source.** Math- PR22 file `frontiers/rn_thin_tube_20260925/TWO_SCALE_ADDENDUM.md` at commit `b2e1652f1374c3b45759324a1ad1fd4177458500`, blob `89cae3a9734f2ec7172cd0b6b0b3af3ddd355d73`, 15902 bytes, SHA256 `079f9399aef401d58749b3684f3acbb7f49ffdcbcac9f501b79e06c28a7f4e7d`.

The PR22 branch head has since advanced. This review does not read `FIXED_ANNULUS_CANDIDATE.md`, later PR22 files, or PR28, and it does not transfer these dispositions to any later blob.

**Scope.** Sections S6–S21, together with the candidate bounds (S3)–(S5) that those sections are used to prove: finite-r subtraction, aspect-uniform `O(δ)` remainders, all three contact minors, the rare Gaussian target, extra-conditioned triple-Hessian moments, the original endpoint-only normalizer, the weighted count, and the power-width versus logarithmic-width quantifier order.

**Scientific effect: NONE.** ACCEPT means the stated interface follows from the stated hypotheses and from the cited weighted Kac–Rice theorem. It does not close a fixed scaled annulus, a global RN statement, or any status flag.

## Provenance

| Item | Value |
|---|---|
| Reviewer model | Grok 4.7, xAI (`grok-4.7-high-fast`) |
| Agent | Same Cursor cloud session `bc-ff620630-782c-4f77-b994-35d0967e5dce` |
| Provider | xAI. The addendum is an OpenAI-lane candidate. This is not an OpenAI session. |
| Source exposure | The addendum at the blob above, `two_scale.py` and `test_two_scale.py` at the same commit, and ar5iv HTML of arXiv:2103.10853 for Theorem 29. The owner’s same-provider algebra notes were not used as acceptance. |
| Prior verdict | PR19 `e93eade` and PR21 `b420099` remain ACCEPT on the interfaces already scored. Those arguments were rederived here only where the addendum repeats them. |

## Dispositions

Coordinates: witness `(x, z) = (r u, r t)`, `δ = √(r² + t²)`, `α = r/δ`, `β = t/δ`, with `A ≤ |u| ≤ B`, `A > 1`, and `δ ≤ δ₀`. The closed semicircle `α ≥ 0`, `α² + β² = 1` is a compactification of the aspect ratio. Physical points keep `α > 0`.

| Interface | Disposition | Ground |
|---|---|---|
| S6 | **ACCEPT** | The midpoint Hermite jet equals `(S0 − a² D1/2, (3 D0 − S1)/2, D1, 3(S1 − D0)/a², L(0), L'(0))`. Evaluation recovers the six observations, so the map is invertible. The label `S1` inside this display is the average slope. |
| S7 | **ACCEPT** | The six pins produce `v_r = (b − k r³/2, −(3/2) k r², 0, 12k, 0, 0)`, `L_r = 0`, and `H'(r u) = 6 k r² (u² − 1/4)`. |
| S8–S9 | **ACCEPT** | `Y` subtracts `r t L'(0)` before any pin substitution. For the monomial `x z` that term is the whole transverse contribution to `f_x`; without it the normalized component is order `1/r` or worse. Under the pins, `L_r = 0` and `grad f = diag(r² δ, r δ) [Y + (6k(u² − 1/4)/δ, 0)]`, with Jacobian determinant `r³ δ²`. |
| S10–S12 | **ACCEPT** | Cubic Hermite reproduction gives `g'(r u) − H'(r u) = r³ a(u) Q + O(r⁴)` with `a(u) = u(u² − 1/4)/6`. Affine interpolation gives `h'(r u) − L'(0) = r u T + O(r²)` and `h(r u) − L(r u) = r² b(u) T + O(r³)`, `b(u) = (u² − 1/4)/2`. Taylor expansion in `z = r t` then produces `Y = M(Q, T, S)ᵀ + O(δ)`, where `M` is the displayed 2×3 matrix, uniformly in the aspect ratio. Each remainder quotient is bounded by a constant times `(r² + r|t| + t²)/δ` or `(r² + r|t| + r t²)/δ`, and both are at most `3 δ` once `δ₀ ≤ 1`. There is no factor `1/α` or `1/β`. Value-interpolation coefficients are `O(1/r)` and derivative coefficients stay `O(1)` for bounded `u`, including `|u| > 1/2`. |
| S13–S14 | **ACCEPT** | The three minors are `a b α²`, `a α β`, and `u β²`. Cauchy–Binet gives `det(M Mᵀ) = a² b² α⁴ + a² α² β² + u² β⁴`. On `A ≤ |u| ≤ B`, both `a(u)² b(u)²` and `u²` are bounded below by a positive `m₀`. Hence `det(M Mᵀ) ≥ m₀(α⁴ + β⁴)`. For `α² + β² = 1`, `α⁴ + β⁴ ≥ 1/2`, with equality at `α² = β² = 1/2`. The trace of `M Mᵀ` is bounded on that compact set, so the least eigenvalue is uniformly positive at both aspect endpoints and between them. |
| S15 | **ACCEPT** | The nine jets `(U₀, Q, T, S)` are linearly independent in every frame: a nontrivial symbol polynomial cannot vanish on every rotated lattice frequency while every Fourier weight of `K_L` is positive. Compactness of `O(2)` and of the semicircle upgrades the rank of `M` to uniform eigenvalue bounds for `Cov(U₀, Y₀)`. The `O(δ)` convergence of Section 3 passes those bounds to `(U_r, Y)` for small `δ`, and the Schur complement stays between positive finite multiples of the identity. The conditional mean of `Y` stays bounded because `v_r` is bounded for compact marks. |
| Rare target and (S3) | **ACCEPT** | The zero-gradient condition is `Y = τ = (−d(u)/δ, 0)` with `d(u) = 6k(u² − 1/4) ≥ d₀ = 6 k₋ (A² − 1/4) > 0`. For small `δ`, `|τ − m_Y| ≥ d₀/(2δ)`. The Gaussian density bound under `λ I ≤ Σ ≤ Λ I` is `C exp(−c/δ²)`. Division by the Jacobian `r³ δ²` is (S3). |
| S16 | **ACCEPT** | Given only the pins, the nine Hessian entries at the two endpoints and the witness have bounded conditional moments. Cauchy–Schwarz and `Σ ≤ Λ I` bound `Cov(H, Y)`. A further condition `Y = τ` shifts the conditional mean by `O(1/δ)` and does not increase the conditional covariance. Each 2×2 determinant is quadratic, so the product of the three absolute determinants is `O(|H|⁶)`. The conditional sixth moment of a Gaussian with mean `O(1/δ)` and bounded covariance is `O(δ⁻⁶)`. |
| S17 | **ACCEPT** | With `f_{xxx}(0) = 12k + O(r²)` and `f_{xx}(0) = O(r²)`, evaluation at `x = ±r/2` gives `f_{xx}(M)/r = −6k + O(r)` and `f_{xx}(S)/r = 6k + O(r)`. The transverse pins force `f_{xz}` at both pins to be `O(r)` in `Lᵖ`, because `f_{xxz}(0)` has bounded moments. Multiplying by `f_{zz} = A_r + O(r)` gives the two determinant identities. |
| S18 | **ACCEPT** | `Var(A_r | U_r)` stays between two positive constants for small `r`, and the conditional mean stays bounded on the compact mark set. Thus `P(−2 ≤ A_r ≤ −1) ≥ p₀ > 0`. On that event the main terms are at least `6 k₋` in absolute value. Markov removes an `O(rᵖ)` probability where the scaled errors exceed `3 k₋`. The remaining event has probability at least `p₀/2` and lies in the index set `{H_M negative definite, det H_S < 0}`, with both absolute determinants at least `3 k₋ r`. Therefore `Z_r ≥ c_Z r²`. |
| S19 and (S4)–(S5) | **ACCEPT** | Stecconi, arXiv:2103.10853, Theorem 29, puts a measurable weight inside the conditional expectation of the Kac–Rice density. Its proof extends from bounded continuous positive weights to nonnegative Borel weights by monotone convergence; truncation covers the unbounded endpoint weight. For each fixed `r > 0` the witness set stays a positive distance from the pins, and the periodic field with full Fourier support keeps a nondegenerate gradient/Hessian jet there. Packing the witness Jacobian once into `F_j(H_w)` and the endpoint determinants into `W_r` produces (S19). Combining (S3), (S16), and `Z_r⁻¹ ≤ C r⁻²` produces (S4): the powers are `r⁻³ · r⁻² · δ⁻² · δ⁻⁶ = r⁻⁵ δ⁻⁸`. Physical area `dx dz = r² du dt` then produces the integral (S5). |
| S20 | **ACCEPT** | For `0 < γ ≤ 1`, `r ≤ 1`, and `ε = K r^γ`, one has `K r^γ ≤ √(r² + ε²) ≤ √(1 + K²) r^γ`. The map `s ↦ s⁻⁸ exp(−c/s²)` increases on a sufficiently small interval `(0, δ₀]` because its logarithmic derivative is positive for `s² < c/4`. The integral is then at most a constant times `r^γ · r⁻⁸γ exp(−c_γ / r^{2γ})`, and (S5) contributes another `r⁻³`. The resulting power is `−3 − 7γ`. For `γ = 1` the physical width is `O(r²)` and the count power is `r⁻¹⁰`. For `γ = 1/2` the physical width is `O(r^{3/2})` and the power is `r⁻¹³/²`. |
| S21 | **ACCEPT** | The quantifiers are `for each N there exists D_N`, in that order. For `ε = D_N / √log(1/r)` and small `r`, `s_max² ≤ 2 D_N² / log(1/r)`, so the exponential factor is at most `r^{c/(2 D_N²)}`. The increasing envelope and the interval length contribute `D_N⁻⁷ (log(1/r))^{7/2}`. The full bound is a constant times `r^{c/(2 D_N²) − 3} (log(1/r))^{7/2}`. The choice `D_N² ≤ c / [2(N+4)]` makes the exponent of `r` at least `N + 1`, and `r (log(1/r))^{7/2} → 0` absorbs the logarithm. One fixed `D` is not asserted to work for every `N`. |

## Computations performed

Python 3.12.3, standard library. `check_two_scale.py` checks the normalization identities

```text
(r³ a Q + r² t u T) / (r² δ) = α a Q + β u T,
(r² b T + r t S) / (r δ) = α b T + β S,
```

the Jacobian `r³ δ²`, the drift `6(u² − 1/4) = 45/2` at `u = 2`, the circle bound `α⁴ + β⁴ ≥ 1/2`, the count ledger `−3 − 2 + 2 = −3` and `−2 − 6 = −8`, and the powers `−10` and `−13/2`.

The same session fetched ar5iv HTML for arXiv:2103.10853. Theorem 29 is the weighted counting formula (4.3), and its proof reduces nonnegative Borel weights to the bounded continuous case by monotone convergence. Printed PDF page numbers are not recovered from that HTML rendering; the theorem statement and the nonnegative extension match the use in S19.

Author unit tests at this commit were not treated as analytic evidence.

## Left open

The addendum’s own exclusions stay open: a fixed positive scaled width `ε`, sites near the pins or near `u = 0`, intermediate distances, witness collisions, dimension greater than 2, the limit `k₋ = 0`, uniformity in `L`, numerical values of `C`, `c`, and `D_N`, and any elder or lifetime theorem. The later fixed-annulus candidate is not part of this blob and is not accepted.
