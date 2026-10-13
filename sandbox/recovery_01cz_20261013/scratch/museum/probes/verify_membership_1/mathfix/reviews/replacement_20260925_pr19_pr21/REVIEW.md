# Replacement review of the finite-r Hermite repair and the inner-axial density

**Claim.** This is the one bounded replacement review requested in [Math- issue 23](https://github.com/d6g8k5htny-coder/Math-/issues/23). It is a fresh session. It does not retry Cursor agent 91fa or follow-up comment 5839326176, and it does not call that follow-up active.

**Sources actually read.**

| Target | Identity |
|---|---|
| Math- PR19 | head `e93eade078cac2e8b14aa39d402a3043b7cdb1c7` |
| C6 supplement | `reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md`, 8054 bytes, blob `2f8dc9badeea`, SHA256 `8260e567e46cad8892442a5baf59cc0cbcbbeaf860edb1e234379e666e035bad` |
| Repair note | `REPAIR.md`, blob `40c7ff79a1ca`, SHA256 `cc70ee30943492ac57512a12ff74b2907ebf9e64bcb36fff3f7a8d1dd6928413` |
| Math- PR21 | head `b420099b2440da3a8ba62f7000feacc9fc88069b` |
| Density proof | `frontiers/axial_density_20260925/PROOF.md`, 10858 bytes, blob `a72418d78fd3`, SHA256 `8b9376a69fda9f0f231d9502be839ae45d8006d1891ff3e700fa521cc96f6cb4` |

Both heads match the issue text, including the PROOF.md byte count and SHA256. Relative to merge-base `baca69c394ab42130c61771bee74e808703f1ce7`, each head only adds its own package and workflow. Later main commits that add the downstream transition gate are absent from these heads because the branches predate them.

**Scope.** Interfaces M1–M7 and S1–S4 from the C6 supplement, and equations (2)–(8) of the density proof, including the compact positive-mark quantifiers. The density statement (1) is scored only as the corollary of those equations. PR22’s weighted-tube extension was not read and is not accepted. No author file on either head was edited. No scientific register, `lemma_closed` flag, prize, or premise is changed by this review.

**Scientific effect: NONE.** An ACCEPT below means the stated interface follows from the stated hypotheses. It does not close a Gaussian count, a thin belt, or an RN/24-jet obligation.

## Provenance

| Item | Value |
|---|---|
| Reviewer model | Grok 4.7, xAI (`grok-4.7-high-fast`) |
| Agent | Cursor cloud session `bc-ff620630-782c-4f77-b994-35d0967e5dce` |
| Provider relative to the targets | xAI. Both targets are OpenAI-lane author candidates committed by `d6g8k5htny-coder`. This session is not an OpenAI session. |
| Source exposure | The author proofs, their tests, and the PR19/PR21/issue 23 comments were read. The derivations below were recomputed from the six pin equations and from the stated periodized Gaussian. This is an exposed technical review, not a sealed blind review. |
| Independence claimed | Provider-level nonauthor status relative to the OpenAI author lane. Different Cursor bcIds are not the ground for that claim. |
| Computations | Python 3.12.3, standard library only. Recorded in `check_interfaces.py`. |

## PR19 — finite-r C6 interfaces

Hypotheses used throughout: `B >= 1`, `0 < r <= 1`, `f` of class C6 on a neighborhood of the square `|x| <= B r`, `|z| <= B r`, with every partial of total order at most 6 bounded by one finite `M` on that square, and the six pins

```text
f(-r/2, 0) = b,  f(r/2, 0) = b - k r^3,
f_x(-r/2, 0) = f_x(r/2, 0) = f_z(-r/2, 0) = f_z(r/2, 0) = 0.
```

The square contains the midpoint, both pins, and every segment used by Taylor’s formula. A bound supported only on an annulus that omits those segments does not justify the expansions below.

Write `h = r/2` and `g(t) = f(t, 0)`. Lagrange remainders with the derivative ceiling `M` give the following, which were expanded and collected independently of `finite_r_contact.py`.

| ID | Disposition | Ground |
|---|---|---|
| M1 | **ACCEPT** | Average `g'(±h) = 0` through order 3. The odd terms cancel and `|g'(0) + (h^2/2) g'''(0)| <= M h^4/24`, which is `M r^4/384`. |
| M2 | **ACCEPT** | Difference of the order-4 expansion of `g'`, whose remainder uses `g^{(6)}`. Division by `2h` produces `M h^4/120 = M r^4/1920`. |
| M3 | **ACCEPT** | `g(h) - g(-h) = -8 k h^3` equals `2h g'(0) + (h^3/3) g'''(0) + E` with `|E| <= M h^5/60`. Substitute M1. The collected coefficient is `(3/20) M h^2 = (3/80) M r^2`. |
| M4 | **ACCEPT** | The pin average is `b - k r^3/2`. The order-3 even expansion plus M2 and `h <= 1/2` produce `M h^4 (1/8 + h^2/240) <= (121/960) M h^4 = (121/15360) M r^4`. |
| M5 | **ACCEPT** | Combine M1 and M3. `3/640 + 1/384 = 7/960`. |
| M6 | **ACCEPT** | The same averaging argument as M1, applied to `w(t) = f_z(t, 0)`, which vanishes at both pins. Order 4 of `w` is order 5 of `f`. |
| M7 | **ACCEPT** | The same difference argument as M2, applied to `w`. Order 5 of `w` is order 6 of `f`. |
| S1 | **ACCEPT** | Expand `f_x` through degree 2. The scaled defect is the M5 drift, the M3 cubic error, the two midpoint rows `f_xx` and `f_xz`, and a remainder at most `M r (2B)^3/6`. M2 and M7 give `|f_xx(0)|, |f_xz(0)| <= (27/640) M r^2` for `r <= 1`, and the two rows together contribute `27B/320`. The displayed `C_x` is that sum. The `r^2 <= r` absorption is inside the hypothesis `r <= 1`. |
| S2 | **ACCEPT** | Expand `f_z` through degree 1. M6 contributes `49/384`, M7 contributes `27B/640`, and the degree-2 remainder contributes `2 B^2`. |
| S3 | **ACCEPT** | Expand `f_z(ru, 0)` through degree 2 and keep the M6 term `-q/8` after division by `r^2`. The three contributions are `1/384`, `27B/640`, and `B^3/6`. |
| S4 | **ACCEPT** | Expand `f` through degree 3. The exact cubic and transverse leading terms cancel against `r H` and `a v^2/2` once M3–M6 are inserted. The remaining pieces are M4 (`121/15360`), the M5 and M6 evaluation errors (`7B/960 + B/384 = 19B/1920`), the midpoint quadratic rows (`81 B^2/1280`), the M3 error in `f_xxx` (`B^3/160`), and the degree-4 remainder (`2 B^4/3`). |

The multivariate remainder used for S1–S4 is `|R_m| <= M (|x| + |z|)^m / m!`. It follows by applying the one-variable Lagrange formula to `t -> f(tx, tz)` and the binomial bound on that derivative. On the square this is at most `M (2 B r)^m / m!`.

The constants are sufficient bounds. M1, M2, M6, and M7 are sharp for the triangle-inequality ceiling: the degree-5 longitudinal monomial, after the six pins are solved at `r = 1`, `k = 1`, has `|∂_x^5 f| = 120` and ceiling `M = 120`, and the M1 defect equals `120/384`. S1–S4 are not claimed to be optimal, and the fixture scan did not saturate them.

The pin-preserving quartic `(x^2 - r^2/4)^2` has vanishing value and gradient at both pins. Its scaled longitudinal derivative is `4 r u (u^2 - 1/4)`, hence `30 r` at `u = 2`. Under a derivative ceiling independent of `r`, this defect is not `O(r^2)`. The stated S1 rate is the rate the C6 hypothesis supports.

All six pins are used: the two height pins in M3 and M4, the two longitudinal derivative pins in M1 and M2, and the two transverse pins in M6 and M7. The scaled rows call those midpoint bounds. Positivity of `k` is not required for M1–M7 or S1–S4.

## PR21 — equations (2)–(8)

The model is the centered, variance-one periodization `K_L` on the torus of side `L`, conditioned on the same six pins, with `|b| <= B0`, `k ∈ [k0, k1]`, `k0 > 0`, `A <= |u| <= B`, `A > 1`, and `|w| <= W`. The witness is `R(r u, r^2 w)`. Gradients are components in the orthonormal frame `R`.

| ID | Disposition | Ground |
|---|---|---|
| (2) | **ACCEPT** | A general cubic `p + q x + s x^2/2 + t x^3/6` was matched to arbitrary value and derivative data at `±a` by rational elimination. The jet `(p, q, s, t)` equals `(S0 - a^2 D1/2, (3 D0 - S1)/2, D1, 3(S1 - D0)/a^2)`. Evaluation of that cubic recovers the four longitudinal observations, so the map is invertible for every `r > 0`. The linear interpolant of the two transverse values is the last pair in (2). |
| (3) | **ACCEPT** | Substituting the six pins produces `U_r = (b - k r^3/2, -(3/2) k r^2, 0, 12 k, 0, 0)` and `H'(r u) = 6 k r^2 (u^2 - 1/4)`. |
| (4) | **ACCEPT** | The displayed normalization is linear and unconditional. For the monomial `x z`, omitting `r^2 w L'(0)` leaves the witness value `w/r`. With the subtraction, the same monomial is normalized to `(0, 0)`. The subtraction is required before conditioning. |
| (5) | **ACCEPT** | Read as `‖U_r - U_0‖_2 = O(r^2)` and `‖Y_r - Y_0‖_2 = O(r)`, uniformly for frames in `O(2)` and `(u, w)` in the stated compact set. Cubic Hermite reproduction gives the quartic defect `(Q/24)(x^2 - a^2)^2`; its derivative at `r u` is `r^3 α(u) Q` with `α(u) = u(u^2 - 1/4)/6`. The next longitudinal contribution is `O(r^4)` before division by `r^3`. Linear interpolation of `h` contributes `r^2 β(u) T` with `β = (u^2 - 1/4)/2`, and `h'(r u) - L'(0) = r u T + O(r^2)`. The transverse Taylor step of width `r^2 w` then produces exactly `Y_0 = (α Q + u w T, β T + w S)`. Both components of `Y_r - Y_0` are `O(r)`. Fourier coefficients of `K_L` are positive and decay as `exp(-2 π^2 |n|^2/L^2)`, so every fixed derivative supremum has finite moments of every order, uniformly in the frame. The Hermite value coefficients at scale `r` grow at most like `1/r`, and the derivative coefficients stay bounded on `A <= |u| <= B`, so the Taylor defects are `O(r)` in `L^2` with a uniform implicit constant. Covariance convergence follows from Cauchy–Schwarz because the observations are centered. |
| (6) | **ACCEPT** | In coordinates `(Q, T, S)` the limit rows are `(α, u w, 0)` and `(0, β, w)`. Their `(Q, T)` minor is `α β = u (u^2 - 1/4)^2/12`, nonzero for `|u| >= A > 1`. Thus the two limit rows are independent modulo the six midpoint jets. A nontrivial linear combination of the nine jets `(f, f_x, f_xx, f_xxx, f_z, f_xz, Q, T, S)` has a nonzero symbol polynomial in the rotated frequency. Composition with an invertible rotation is still a polynomial, and a polynomial vanishing on `Z^2` is identically zero, so the symbol cannot vanish on every lattice frequency. Every Fourier weight is strictly positive, so the nine-dimensional covariance is positive definite in every frame. |
| (7) | **ACCEPT** | The joint covariance of `(U_0, Y_0)` is continuous in the frame and in `(u, w)`, and positive definite on the compact set `O(2) × {A <= |u| <= B} × {|w| <= W}`. Its least eigenvalue therefore has a positive minimum and its greatest eigenvalue is finite. `L^2` convergence (5) makes `Cov(U_r, Y_r)` uniformly close to that limit for small `r`. The Schur complement is continuous on the set where `Cov(U_r)` stays invertible, and `Cov(U_0)` is uniformly invertible, so `λ I <= Σ_r <= Λ I` for some `λ, Λ` independent of the frame and of `(u, w, b, k)` inside the stated ranges. The minimizing representation `v^T Σ v = min_u (u, v)^T Cov(u, v)` supplies the lower bound from the joint matrix; the upper bound is the `Y` block. The conditional mean `m_r = Cov(Y_r, U_r) Cov(U_r)^{-1} v_r` stays bounded because `v_r` stays bounded for `|b| <= B0` and `k <= k1`. |
| (8) | **ACCEPT** | Under the pins, `L_r = 0` and `grad f(X) = (6 k r^2 (u^2 - 1/4) + r^3 Y_{r1}, r^2 Y_{r2})`. The Jacobian determinant in `Y` is `r^5`. Therefore the conditional density at the origin in gradient space is `r^{-5}` times the conditional `Y`-density at `(-6 k (u^2 - 1/4)/r, 0)`. |

**Corollary (1): ACCEPT as an existence bound.** Let `δ = 6(A^2 - 1/4) > 0`. The target in (8) has length at least `δ k / r`. For `0 < r <= r_*` small enough that the covariance bounds of (7) hold, that `r_* <= δ k0 / (2 M0)` when `M0 > 0`, and that the whole pin-witness configuration sits inside one injectivity chart of the torus,

```text
|t - m_r| >= δ k / (2 r).
```

The two-dimensional Gaussian density with `λ I <= Σ <= Λ I` is at most `(2 π λ)^{-1} exp(-|t - m|^2 / (2 Λ))`. This is (1) with `C = (2 π λ)^{-1}` and `c = δ^2 / (8 Λ)`. These constants exist and depend only on `L, B0, k0, k1, A, B, W`. They are not numerical enclosures. The hypothesis `k0 > 0` is required to dominate the bounded regression mean uniformly.

The resulting object is the conditional density of the two-dimensional gradient at one witness, with respect to Lebesgue measure on gradient space. It includes no pin-density factor, no spatial Jacobian, and no endpoint Palm weight. The region is the inner belt `|z| <= W r^2`. A fixed-width scaled belt `|v| <= ε` is outside this estimate.

The centered finite-r vector `(S0, D0, D1, 3(S1 - D0)/a^2, L(0), L'(0))` differs from `U_r` by the shear `(V0 - r^2 V2/8, V1 - r^2 V3/24)` on the first two coordinates, with the other four fixed. That shear has determinant one, so it is another coordinate description of the same six observations.

## Computations performed

Python 3.12.3, standard library only. `check_interfaces.py` repeats the following exact checks.

- The fraction identities `3/640 + 1/384 = 7/960`, `1/24 + 1/1920 = 27/640`, `121/960 / 16 = 121/15360`, `7/960 + 1/384 = 19/1920`, and `27/1280 + 27/640 = 81/1280`.
- Closed-form Hermite jet against an independent 4×4 elimination, for generic rational data.
- Pin specialization (3), the determinant-one shear, the minor (6), and the Jacobian `r^5`.
- Quartic increment `30 r` at `u = 2`.
- M1 equality for the solved `x^5` fixture at `r = 1`.
- All eleven interfaces on 360 pin-solved polynomials: radii `1, 4/5, 1/2, 1/5, 1/31`, box scales `B ∈ {1, 2, 3}`, and the 22 free monomials through degree 6 together with two dense coefficient vectors. The comparison ceiling is the triangle bound on every partial of order at most 6, which is a valid `M`. Every ratio was at most 1. Largest observed ratios: M1, M2, M6, M7 equal to 1; M3 `2/3`; M4 `40/121`; M5 `1/14`; S1 `19902/113659`; S2 `540/4381`; S3 `6510/14443`; S4 `103680/841357`.
- Exact Lagrange extrapolation of the normalized monomial map through degree 6, on a small `(u, w)` grid, against `(α Q + u w T, β T + w S)`.
- The `x z` normalization with and without the `L'` subtraction.

A separate floating-point check at `L = 4`, summing lattice modes through radius 10, produced a positive-definite 9×9 axis-frame jet covariance (least eigenvalue about `0.302`) and, at `(u, w) = (1.5, 0.4)`, a positive-definite Schur complement (eigenvalues about `2.22` and `9.54`). That is one numerical instance. The general frame argument is the symbol argument in (6) and (7).

The author unit files were also executed from extracted copies of the two heads: 21 tests in the Hermite package and 15 tests in the density package, all passing. Those runs check the finite algebra in the fixtures. They are not the reason for the dispositions above.

## Imported gaps, left open

These items are outside the accepted interfaces. This review does not prove them and does not treat them as consequences.

- Conditional moments of a random derivative ceiling `M` after all six pins. The pathwise C6 inequalities do not license a deterministic `M` inside a Gaussian integral.
- Any weighted Kac–Rice count, height-conditioned density, Hessian-moment bound, endpoint Palm weight, or full normalizer.
- The belt `W r < |v| < η`, pin-witness collisions, intermediate scales, dimension 3, and numerical values of `C`, `c`, or `C_H`.
- PR22’s two-scale weighted-tube candidate. It was not reviewed.
- RN/24-jet closure, `lemma_closed`, and every campaign status flag.
- Effectivity of `r_*`, `λ`, and `Λ`. The density argument proves existence on a sufficiently small interval.

## Limits of this review

The Taylor and regression arguments above are analytic. The polynomial scan covers a finite family of degree at most 6; it corroborates the constants and does not replace the Lagrange derivation. The Gaussian bound is an existence result for one conditional gradient density on the inner belt.
