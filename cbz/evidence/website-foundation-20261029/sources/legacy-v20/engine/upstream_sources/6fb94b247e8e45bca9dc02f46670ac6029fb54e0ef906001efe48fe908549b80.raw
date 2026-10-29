# D5 in every fixed dimension: punctured pin balls, the compact collar, intermediate height-window shells, and the global window first moment

**Object:** CL-D5-DIMENSION-LIFT-20260929-v1.
**Author:** Anthropic Claude (Claude Code session `session_015wNj8LPTKXsaT68G3DgPPh`), 29 September 2026.
**Disposition:** author-side proof candidate. **Nonauthor analytic review is required.**
**Scientific effect:** NONE. No `STATUS`, `PROOF_INDEX` verdict, `GRAPH` node, claim, `lemma_closed` flag, prize or source body changes.

## 0. Why this note exists

Every D5 estimate of the last three days is planar. The punctured endpoint disk [PP], the compact collar [CP], the intermediate shells and the global first moment (I5) [IW], and the planar halves of Corollary F [RC] and Theorem L [LOCAL] all fix `d = 2`. The fixed-remote Theorem A [RM] and the lower bounds of [RC] (F-) and [EDL] (A4) hold in every fixed `d`. [RC] states the consequence exactly: for `d > 2` the matching upper bound on the window event "stays conditional on a dimension-matched global first moment", and no source supplies one. The program's own application is `d = 3` (SIDE24, `L = 24` in [LP] §1).

This note supplies that first moment in every fixed `d >= 2`. It lifts the three planar D5 components and their composition. The planar proofs are followed step by step; every step that is not literally dimension-free is replaced here by a written `d`-dimensional argument. Five devices do all the new work (§3). Nothing else changes: the model, the pins, the weight, the normalizer, the Kac–Rice framework and the regression estimates are those of the sources.

## 1. Setting and results

### 1.1 Model

Fix `d >= 2`, `m = d - 1`, and `L > 0`. `f` is the centered variance-one Gaussian field on `X = R^d/(L Z^d)` with the exact normalized periodized Gaussian covariance of [LP] §1. Fix compact `B = [b_-, b_+]` and `K = [k_-, k_+]` with `0 < k_-`. Frames `R` range over `O(d)`; `u = R e_1` is the axial direction and `y in R^m` are the transverse coordinates. For small `r > 0` the pins are

    M = -(r/2) u,  S = (r/2) u,  f(M) = b,  f(S) = b - k r^3,  grad f(M) = grad f(S) = 0.

`Q_r` is Gaussian regression on these `2(d+1)` observations (not an adjacency or elder law). For a symmetric matrix `H`, `F_j(H) = |det H|` if `H` is nonsingular with `j` negative eigenvalues, and `0` otherwise. Retain the original weight, the full normalizer and the tilted law

    W_r = F_d(H_M) F_(d-1)(H_S),   Z_r = E_(Q_r) W_r,   dQ_r^W = (W_r/Z_r) dQ_r,

and the between-pin height window `I_r = (b - k r^3, b)`. `N_j(B)` counts index-`j` critical points of `f` in a Borel set `B` at all heights; `N_(r,j)(B)` counts those with height in `I_r`. Constants `C, c, r_*, s_0` depend on `d, L, B, K` and on the fixed chart radii named in each statement; never on `r, s, b, k, R, j` or the Borel set. No constant is evaluated numerically. Nothing is uniform as `k -> 0`, as marks grow, or as `d` or `L` vary.

Throughout, `K` also denotes a fixed multiple of `1 + ||f||_(C^6(X))` (the sources use the same letter for the mark interval; the meaning is always clear from context), and `L <= K` a Hessian Lipschitz constant on the relevant chart.

### 1.2 Statements

Write a point of the scaled chart around `M` as `X = M + r (p, q)` with `p in R`, `q in R^m`, and put

    D = { (p, q) : 0 < p^2 + |q|^2 <= 1/16 },
    omega_d(p, q) = max(|q|, r |p|)^(2-d)      (omega_2 = 1).                       (1.1)

**Theorem P_d (all-height punctured pin ball).** There are `C, r_* > 0` such that for `0 < r <= r_*`, all `b, k, R, j` and every Borel `E` contained in `D`,

    E_(Q_r^W) N_j(M + r E) <= C r^3 integral_E omega_d(p, q) dp dq <= C' r^3.        (1.2)

The same holds for the punctured ball about `S` (§4.9). For a nested ball `p^2 + |q|^2 <= kappa^2 r^2` (physical radius `kappa r^2`), the right side is `O(r^5)`. For `d = 2` this is [PP] (P2).

**Theorem C_d (compact collar).** In midpoint scaled coordinates `X = r (u, v)`, `u in R`, `v in R^m`, fix `R >= 1` and `0 < eta <= 1/4`, and let `C(eta, R)` be the closed ball of radius `R` with the two open balls of radius `eta` about `(-1/2, 0)` and `(1/2, 0)` removed. There are `C, r_* > 0` such that for every Borel `E` contained in `C(eta, R)`,

    E_(Q_r^W) N_j(r E) <= C r^3 |E|.                                                 (1.3)

Consequently, for every Borel subset `E` of the scaled ball of radius `R` with the two pins removed,

    E_(Q_r^W) N_j(r E) <= C r^3 (1 + integral_(E cap pin balls) omega_d) <= C' r^3.  (1.4)

For `d = 2` these are [CP] (C1), (C2).

**Theorem I_d (intermediate height-window shells).** There are `C, s_0 > 0` such that for `0 < r <= s/4` and `0 < s <= s_0`,

    E_(Q_r^W) N_(r,j)({ s <= |X| <= 2 s }) <= C r^3 [ (r/s)^2 + s^2 ],               (1.5)

and hence, uniformly for `A_0 >= 4` and `A_0 r < rho <= s_0`,

    E_(Q_r^W) N_(r,j)({ A_0 r <= |X| <= rho }) <= C r^3 ( A_0^(-2) + rho^2 ).       (1.6)

For `d = 2` these are [IW] (I3), (I4).

**Theorem G_d (global height-window first moment).** There are `C, r_* > 0` such that

    E_(Q_r^W) N_(r,j)( X minus {M, S} ) <= C r^3.                                    (1.7)

For `d = 2` this is [IW] (I5).

**Corollary F_d.** Let `A_r` be the event that some critical point other than `M, S` has height in `I_r`. In every fixed `d >= 2`,

    c r^3 <= Q_r^W(A_r) <= C r^3.                                                    (1.8)

The lower half is [RC] Corollary (F-), which holds in every fixed `d`. The upper half is Markov's inequality applied to (1.7), summed over the `d+1` indices. This discharges the sentence in [RC] §1 that leaves the `d > 2` upper half conditional.

**Corollary M_d (window multiplicity in every dimension).** Let `N_r` count all window critical points other than `M, S`. In every fixed `d >= 2`,

    c r^3 <= Q_r^W{ N_r >= 2 } <= C r^3,        E_(Q_r^W)[ N_r (N_r - 1) ] >= c r^3.  (1.9)

The lower bounds are [EDL] (A4). The upper bound is `1{N >= 2} <= N/2` and (1.7). In particular a torus-wide `O(r^5)` second factorial moment is excluded in every `d`, the global window point process is at total-variation distance `Theta(r^3)` from its exact-mean Bernoulli approximation (the elementary identity of [LOCAL] §7 is dimension-free), and no upper bound on the second factorial moment is claimed.

### 1.3 What the results are not

They are first moments and event probabilities. They are not an all-height intermediate or global count, a torus-wide second factorial moment, a shrinking-separation collision estimate outside the fixed remote region, an elder-selection statement, a numerical constant, a 24-jet certificate, or a status transition. §8 lists the non-claims.

## 2. Sources consumed, with the exact interface

| Tag | Source (identities in `SOURCE_MAP.json`) | What is consumed |
|---|---|---|
| [LP] | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | §2 positive Fourier spectrum and distinct-site jet rank; §3 contact frame `U_r`, uniform regression; (4.1) uniform `C^q` moments of every order under `Q_r`; (5.1) `|alpha_i|, ||beta_i|| <= M_3/2`; (5.3) `det H_i / r = alpha_i det A_i - r beta_i^T adj(A_i) beta_i`; (5.5) `Z_r >= z_* r^2`. All in every fixed `d`; accepted as A1–A4 in the D1 reconciliation. |
| [PP] | `reviews/d5_local_collar_20260928/PUNCTURED_PIN_PROOF.md` | The planar template for Theorem P_d: (P3)–(P20). |
| [CP] | `reviews/d5_local_collar_20260928/COLLAR_PROOF.md` | The planar template for Theorem C_d: (C3)–(C20). |
| [IW] | `frontiers/intermediate_window_20260928/PROOF.md` | The planar template for Theorems I_d, G_d: (I6)–(I34). |
| [RM] | `frontiers/remote_window_20260924/PROOF.md` | Theorem A in every `d`, reviewed (main#76): the outer tile in §7. |
| [RC] | `frontiers/remote_collision_20260928/PROOF.md` | (F-) lower bound; the conditional sentence for `d > 2`. |
| [EDL] | `frontiers/elder_dimension_lift_20260928/PROOF.md` | (A4) `Q^W(N_r >= 2) >= c r^3` in every `d`. |
| [LOCAL] | `frontiers/window_multiplicity_laws_20260928/LOCAL_MULTIPLICITY.md` | The total-variation identity of §7 only. |

External framework: Armentano, Azaïs and León, arXiv:2304.07424v3, Theorems 2.1, 2.2 and 7.1 with Remarks 7–8, exactly as in [LP] §9, [RM] §5, [PP] §9, [CP] §9 and [IW] §9. The zero-counted field is `grad f : X -> R^d`; parameter and value dimensions agree.

Planar steps that are dimension-free are cited by their source label and not reproved. A reviewer should read each cited step in its source; this note is not self-contained without them.

## 3. Five devices

The planar proofs use, at several places, the two-dimensional facts that a `2 x 2` matrix with two small columns has a small determinant, that a `2 x 2` symmetric matrix is a scalar plus a small correction, and that a bounded intensity integrates to `O(r^3)` over a disk of area `r^2`. The following replace them.

**Lemma D1 (two soft directions).** Let `H` be a real `d x d` matrix with `||H||_op <= K`, and let `e, e'` be unit vectors with `sigma = ||e wedge e'|| > 0` (the sine of their angle). Then

    |det H| <= ||H e|| ||H e'|| K^(d-2) / sigma,                                        (3.1)

and every unit vector `w` in `span(e, e')` satisfies `||H w|| <= ( ||H e|| + ||H e'|| ) / sigma`.

*Proof.* Let `0 <= sigma_1 <= ... <= sigma_d` be the singular values of `H`. Then `|det H| = prod sigma_i <= sigma_1 sigma_2 K^(d-2)`. The singular values of the second exterior power `Lambda^2 H` are the products `sigma_i sigma_j`, `i < j`, so its smallest singular value is `sigma_1 sigma_2`. Apply it to the unit decomposable 2-vector `omega = (e wedge e')/sigma`: `sigma_1 sigma_2 <= ||Lambda^2 H omega|| = ||H e wedge H e'|| / sigma <= ||He|| ||He'|| / sigma`. For the second claim write `w = a e + b e'`; in the plane, Cramer's rule gives `a = ||w wedge e'|| / ||e wedge e'||`, so `|a| <= 1/sigma`, and likewise `|b| <= 1/sigma`. ∎

For `d = 2` and `e = e_x`, (3.1) is the planar two-column bound of [PP] §8.1 and [CP] (C18). The finite control `X` checks `det(H)^2 ||u_1 wedge u_2||^2 <= ||Hu_1 wedge Hu_2||^2 ||H||_F^(2(d-2))` on random rational matrices and the equality case on a diagonal family.

**Lemma D2 (one soft direction).** For a unit vector `e`, `|det H| <= ||H e|| K^(d-1)`. *Proof.* Hadamard's inequality in an orthonormal basis containing `e`; or `sigma_1 <= ||He||`. ∎

**Lemma D3 (jet functionals on `R^m`).** Let `beta in R^m`. Coordinates on symmetric matrices, symmetric 3-tensors, and vectors are their independent entries.

- The linear map `S -> S beta`, `Sym_m -> R^m`, has Gram matrix `G_beta` (over independent entries) with `G_beta >= (|beta|^2 / 2) I_m`.
- The functional `C -> (1/2) beta^T C beta` on `Sym_m` has squared norm `>= |beta|^4 / 4`.
- The functional `D -> D[beta, beta, beta] / 12` on symmetric 3-tensors has squared norm `>= |beta|^6 / 144`.

*Proof.* For `xi in R^m`, `xi . S beta = sum_i xi_i beta_i S_ii + sum_(i<j) (xi_i beta_j + xi_j beta_i) S_ij`, so

    xi^T G_beta xi = sum_i xi_i^2 beta_i^2 + sum_(i<j) (xi_i beta_j + xi_j beta_i)^2
                   >= (1/4) sum_(i,j) (xi_i beta_j + xi_j beta_i)^2
                   = (1/4) || xi beta^T + beta xi^T ||_F^2 = (1/2)( |xi|^2 |beta|^2 + (xi . beta)^2 ) >= (1/2) |beta|^2 |xi|^2.

For the other two, an independent entry with multiplicity `mu in {1, 2}` (resp. `{1, 3, 6}`) carries the coefficient `mu beta_i beta_j / 2` (resp. `mu beta_i beta_j beta_k / 12`); since `mu^2 >= mu`, the squared norm is at least `(1/4) sum_(all ordered) (beta_i beta_j)^2 = |beta|^4/4` (resp. `|beta|^6 / 144`). ∎

The finite control `N` checks all three on random rational vectors.

**Lemma D4 (transverse block determinant).** Let `S in Sym_m`, `hat v` a unit vector, `epsilon = hat v^T S hat v`, `w = S hat v - epsilon hat v` (so `w` is orthogonal to `hat v`), and `S'` the compression of `S` to `hat v^perp`. Then

    det S = epsilon det S' - w^T adj(S') w,        |det S| <= |epsilon| ||S||^(m-1) + |S hat v|^2 ||S||^(m-2),   (3.2)

with `det S' = 1`, `adj(S') = 0` when `m = 1`, and `adj(S') = 1` when `m = 2`.

*Proof.* In an orthonormal basis beginning with `hat v`, `S = [[epsilon, w^T], [w, S']]`, and the first identity is the cofactor expansion (Schur's formula when `S'` is invertible, and by continuity otherwise). Then `||adj(S')||_op <= ||S'||^(m-2)` and `|w| <= |S hat v|`. ∎

The finite control `TB` checks the identity and the bound on random rational matrices for `m = 2, 3, 4`.

**Lemma D5 (pin weight).** For `d >= 2` and `0 < a <= 1/4`,

    integral_(|q| <= a) |q|^(2-d) dq = |S^(m-1)| a  (with |S^0| = 2),  hence
    integral_D omega_d <= (1/2) |S^(m-1)| (1/4),   integral_(p^2+|q|^2 <= kappa^2 r^2) omega_d <= 2 kappa r |S^(m-1)| kappa r.

*Proof.* Polar coordinates in `R^m`: the radial exponent is `(2-d) + (m-1) = 0`. Since `omega_d <= |q|^(2-d)`, both bounds follow. ∎

The finite control `W` checks the radial exponent and the nested-ball count exponent `3 + 2 = 5` for `d = 2, ..., 6`.

## 4. Proof of Theorem P_d

We follow [PP] §§3–10. Coordinates: translate so that `M = 0` and `S = (r, 0)`; `X = (rp, rq)`, `|p|, |q| <= 1/4`, `(p, q) != 0`.

### 4.1 Uniform endpoint regression ([PP] §3)

Let `g(x) = f(x, 0)` and `h(x) = grad_y f(x, 0) in R^m`. The unconditional observation vector

    E_r = ( g(0), g'(0), h(0), [g'(r) - g'(0)]/r, (12/r^3)[ g(r) - g(0) - (r/2)(g'(r) + g'(0)) ], [h(r) - h(0)]/r )

has `2 + m + 1 + 1 + m = 2(d+1)` coordinates, is invertibly equivalent to the raw pins for each `r > 0`, has target `e = (b, 0, 0, 0, -12k, 0)` independent of `r`, and contact limit `E_0 = (f, f_x, grad_y f, f_xx, -f_xxx, grad_y f_x)(0)`. Taylor's integral formula gives `E_r - E_0 = O_(L^s)(r)` for every finite `s`, uniformly over frames and locations. The contact functionals are distinct one-site derivative monomials, so [LP] §2 gives positive definite covariance; compactness of `O(d)` makes the floor uniform. The regression coupling `f_(Q_r) = f + Cov(f, E_r) Cov(E_r)^(-1) (e - E_r)` and [LP] (4.1) give, for every fixed `n, s`, uniformly bounded `s`-th moments of the `C^n` norm on the whole torus.

Define the midpoint jets (all at `0 = M`)

    J = ( A_4, T_3, S_0, C_3 ) = ( f_xxxx,  grad_y f_xx in R^m,  D_y^2 f in Sym_m,  D_y^2 f_x in Sym_m ).

Their independent entries are distinct derivative monomials of orders `4, 3, 2, 3`, none in `E_0`. So the covariance of `J` after conditioning on `E_r` is uniformly sandwiched and its mean is bounded: a Schur-complement statement for the exact field, as in [PP].

### 4.2 Relative remainders ([PP] §4)

[PP] (P4) is a statement about the axial function `g` alone and is unchanged:

    g'(rp) = 6 k r^2 p(p-1) + (r^3 A_4 / 12) p(p-1)(2p-1) + O(r^4 |p| K).

Since `h(0) = h(r) = 0` and `h''(0) = T_3`,

    h(rp) = (r^2/2) p(p-1) T_3 + O(r^3 |p| K),      h'(rp) = r (p - 1/2) T_3 + O(r^2 K),

componentwise, by the same two-point Hermite argument. Expanding in `y = rq`,

    f_x(X)/r^2 = 6k p(p-1) + (r A_4/12) p(p-1)(2p-1) + (p - 1/2) q . T_3 + (1/2) q^T C_3 q + epsilon_1,
    grad_y f(X)/r = (r/2) p(p-1) T_3 + S_0 q + epsilon_2,                                            (4.1)

with the pathwise bounds

    |epsilon_1|  <= C K ( r^2 |p| + r |q| + r |p| |q|^2 + r |q|^3 ),
    ||epsilon_2|| <= C K ( r^2 |p| + r |p| |q| + r |q|^2 ),                                       (4.1')

which are [PP] (P7) with `|q|` the Euclidean norm of `q in R^m`. Since (P7) is written for two scalars, the
`d`-dimensional remainders are derived here rather than cited. Let `h(x) = grad_y f(x, 0) in R^m`, so `h(0) = h(r) = 0`
and `h''(0) = T_3`. Every bound below is pathwise, `K` dominating the derivatives of `f` through order five on the
chart.

- *Axial rows.* [PP] (P4) concerns the scalar `g(x) = f(x, 0)` alone and is unchanged:
  `g'(rp) = 6 k r^2 p(p-1) + (r^3 A_4/12) p(p-1)(2p-1) + O(r^4 |p| K)`.
- *Transverse rows along the axis.* Componentwise, `h_i(0) = h_i(r) = 0` gives `h_i(x) = x(x - r) h_i''(xi_i)/2` with
  `xi_i` in the convex hull of `{0, r, x}`, hence `|xi_i| <= r` for `x = rp`, `|p| <= 1/4`, and
  `h(rp) = (r^2/2) p(p-1) T_3 + O(r^3 |p| K)`. For the derivative, `h(r) = 0` and Taylor's formula at `0` give
  `h'(0) = -(r/2) T_3 + O(r^2 K)`, so `h'(rp) = r (p - 1/2) T_3 + O(r^2 K)`. Both are vector statements: each
  component obeys the one-dimensional estimate and the norm of a vector is at most `sqrt(m)` times its largest
  component.
- *Transverse Taylor expansion.* With `y = r q`, expand in `y` about `y = 0` at fixed `x = rp`:

      f_x(X) = g'(rp) + y . h'(rp) + (1/2) y^T D_y^2 f_x(rp, 0) y + R_3,      |R_3| <= (1/6) |y|^3 sup ||D_y^3 f_x|| <= C K r^3 |q|^3,
      grad_y f(X) = h(rp) + D_y^2 f(rp, 0) y + R_2,                           ||R_2|| <= (1/2) |y|^2 sup ||D_y^3 f|| <= C K r^2 |q|^2,

  the suprema over the segment from `(rp, 0)` to `X`, and `D_y^2 f_x(rp, 0) = C_3 + O(r |p| K)`,
  `D_y^2 f(rp, 0) = S_0 + O(r |p| K)` by the Lipschitz bound over the axial distance `r|p|`.
- *Collecting.* Insert the four expansions and divide by `r^2` and `r` respectively:

      epsilon_1 = O(r^2 |p| K) + O(r |q| K) + O(r |p| |q|^2 K) + O(r |q|^3 K),
      epsilon_2 = O(r^2 |p| K) + O(r |p| |q| K) + O(r |q|^2 K),

  where the four terms of `epsilon_1` come from the axial remainder, `r q . O(r^2 K)/r^2`, `(1/2) q^T O(r|p|K) q` and
  `R_3/r^2`, and the three terms of `epsilon_2` from the axial remainder `O(r^3|p|K)/r`, `O(r|p|K) q` and `R_2/r`.
  This is (4.1'). The only place the dimension enters is the constant, through `sqrt(m)` and the operator norms of
  the tensors `C_3`, `S_0`, `D_y^3 f`; the powers of `r`, `|p|`, `|q|` are those of (P7).

The ratios used in §4.3, `r^2|p|/Delta <= r`, `r|q|/Delta <= r`, `r|p||q|^2/Delta <= r|p||q|`, `r|q|^3/Delta <= r|q|^2`,
`r|p||q|/Delta <= r|p|`, `r|q|^2/Delta <= r|q|`, all follow from `Delta >= |q|` and `Delta >= r|p|`, exactly as in the
planar continuum record `reviews/d5_punctured_pin_continuum_claude_20260929/REVIEW.md` (its note n2). The finite
control `RO` checks the expansions (4.1) exactly on a pinned cubic in `d = 3` (there the remainders vanish).

### 4.3 A stable gradient frame on the whole punctured ball ([PP] §5)

Set `Delta = (|q|^2 + r^2 p^2)^(1/2) > 0`, `alpha = r p / Delta in [-1, 1]`, `beta = q / Delta in R^m`, so `alpha^2 + |beta|^2 = 1`, and

    Y = ( f_x(X) / (r^2 Delta),  grad_y f(X) / (r Delta) ) in R^d.

From (4.1),

    Y = ( 6k p(p-1) / Delta, 0 ) + B J + R,       ||R||_(L^s(Q_r)) <= C_s r,                         (4.2)

where the `d x (independent entries of J)` matrix `B = B(p, alpha, beta, Delta)` acts by

    row 1:      a(p) alpha A_4 + c(p) beta . T_3 + (Delta/2) beta^T C_3 beta,
    rows 2..d:  b(p) alpha T_3 + S_0 beta,
    a = (p-1)(2p-1)/12,   b = (p-1)/2,   c = p - 1/2.                                           (4.3)

The remainder is uniform even when `p` or `q` is arbitrarily small, exactly as in [PP]: `Delta >= |q|` and `Delta >= r|p|` give `r^2|p|/Delta <= r` and `r|q|/Delta <= r`.

**Rank and floor.** On `|p| <= 1/4`, `|a| >= 1/32`, `|b| >= 3/8`, `|c| >= 1/4` and `b^2 <= 25/64`. Let `B_core` be `B` without the `C_3` block. The `C_3` block only adds a positive semidefinite term to `B B^T`, so `B B^T >= B_core B_core^T`. The `A_4` column contributes `a^2 alpha^2` to the `(1,1)` entry; the `T_3` block contributes the rank-one-plus-diagonal matrix `[[c^2 |beta|^2, c b alpha beta^T], [c b alpha beta, b^2 alpha^2 I_m]]`; the `S_0` block contributes `diag(0, G_beta)` with `G_beta >= (|beta|^2/2) I_m` by Lemma D3. Hence

    B_core B_core^T >= N := [[ a^2 alpha^2 + c^2 |beta|^2,  c b alpha beta^T ], [ c b alpha beta,  (b^2 alpha^2 + |beta|^2/2) I_m ]].

`N` is positive definite on `alpha^2 + |beta|^2 = 1`, and its Schur complement on the first coordinate is

    a^2 alpha^2 + c^2 |beta|^2 - c^2 b^2 alpha^2 |beta|^2 / (b^2 alpha^2 + |beta|^2/2)
      = a^2 alpha^2 + c^2 |beta|^4 / (2 b^2 alpha^2 + |beta|^2)
      >= min{ a^2/2,  c^2 / (4(b^2 + 1)) } >= 1/2048,

by the case split `|beta|^2 >= 1/2` (second term at least `c^2 (1/4)/(b^2 + 1)`) or `alpha^2 >= 1/2` (first term). Since `det(N) = (Schur complement) (b^2 alpha^2 + |beta|^2/2)^m` and `b^2 alpha^2 + |beta|^2/2 >= min(b^2, 1/2) >= 9/64`,

    det(B B^T) >= det(B_core B_core^T) >= det(N) >= (9/64)^(d-1) / 2048.                     (4.4)

For `d = 2` the right side is `9/131072`, the constant of [PP] (P9). The entries of `B` are bounded on the chart, so (4.4) gives a uniform floor `lambda_min(B B^T) >= c_d > 0` on the whole compact parameter set `{|p| <= 1/4} x S^(d-1)`. The finite control `G` checks rank `d`, the block-Schur lower bound and (4.4) on random rational points of the sphere, including `alpha = 0` and `beta = 0`.

By §4.1, `Cov_(Q_r)(B J) = B Cov(J) B^T >= c B B^T >= c c_d I_d`. The centered `L^2` triangle inequality applied to (4.2), with `r_*` reduced to absorb the `O(r)` remainder, gives

    c_0 I_d <= Sigma := Cov_(Q_r)(Y) <= C_0 I_d.                                                  (4.5)

The axis `q = 0`, `p != 0` (`beta = 0`) is covered by the `A_4, T_3` columns, and the transverse direction `p = 0` (`alpha = 0`) by the `T_3, S_0` blocks. This is a `2(d+1) + d` observation all-height problem; no witness value is pinned.

### 4.4 Density and the full conditional moment ([PP] §6)

Put `chi = |p| / Delta`. The mean of `Y` is `dvec + O(1)` with `dvec = (6k p(p-1)/Delta, 0)`, and `(9 k_-/2) chi <= |dvec| <= C chi`. The raw gradient map `grad f(X) -> Y` has Jacobian `(r^2 Delta) (r Delta)^m = r^(d+1) Delta^d`. With `|dvec + z|^2 >= |dvec|^2/2 - |z|^2` and (4.5) in the `d`-dimensional Gaussian density,

    p_( grad f(X) | pins )(0) <= C r^(-(d+1)) Delta^(-d) exp(-c chi^2).                            (4.6)

Let `K` dominate the Hessian supremum and its Lipschitz constant on the whole chart. For every fixed `n`,

    E_(Q_r)[ K^n | grad f(X) = 0 ] <= C_n (1 + chi^n).                                             (4.7)

The argument is [PP] (P12) verbatim: the cross-covariance of `f` and its derivatives through order three with `Y` is bounded by Cauchy–Schwarz and (4.5); the Gaussian residual `f - m_f - C(x) Sigma^(-1)(Y - m_Y)` is independent of `Y` with uniformly bounded `C^3` moments; conditioning `Y = 0` replaces the mean by a function of `C^3` norm at most `C(1 + chi)`.

### 4.5 The original normalizer, once only

[LP] (5.5): `Z_r >= z_* r^2` in every fixed `d`. (For `d = 2` [PP] (P14) is an alternative proof.) It is never replaced by a witness-conditioned quantity.

### 4.6 Two pathwise determinant estimates ([PP] §8)

Let `A = M`, `B = S`, `C = X` be three exact critical points, `|B - A| = r`, `|C - A| =: d_X = r (p^2 + |q|^2)^(1/2) <= r/4`, `ell = |B - C| <= 5r/4`. The critical-segment identity [PP] (P15), `||H_U e|| <= L |V - U| / 2` for `e = (V-U)/|V-U|`, is one-dimensional along the segment and holds in every `d`.

**Angle-sensitive bound.** Let `sigma = |sin angle(B - A, C - A)| > 0`, i.e. `q != 0`. Apply Lemma D1 at `A` with `e = e_AB`, `e' = e_AC`: `|det H_A| <= (Lr/2)(L d_X/2) K^(d-2) / sigma`, and `||H_A w|| <= L (r + d_X)/(2 sigma)` for unit `w` in the triangle plane `P`. At `B`, `||H_B e_BA|| <= Lr/2` by the segment `BA`; for the unit vector `w` in `P` orthogonal to `e_BA`, transport gives `||H_B w|| <= ||H_A w|| + L r <= L(r + d_X)/(2 sigma) + L r <= 13 L r / (8 sigma)`; so by Lemma D1 with `sigma = 1`, `|det H_B| <= (13/16) L^2 r^2 K^(d-2) / sigma`. At `C`, the sine of the angle is `r sigma / ell`, and the segments `CA`, `CB` give `|det H_C| <= (L d_X/2)(L ell/2) K^(d-2) ell / (r sigma) <= (25/64) L^2 r d_X K^(d-2) / sigma`. Multiplying, and using `sigma <= 1`,

    |det H_M det H_S det H_X| <= L^6 r^4 d_X^2 K^(3(d-2)) / sigma^4
                              = L^6 K^(3(d-2)) r^6 |q|^2 ( 1 + p^2/|q|^2 )^3.                     (4.8)

For `d = 2` this is [PP] (P16). Only the `K^(3(d-2))` factor is new; it is the price of the `d - 2` directions in which no Hessian is forced small.

**Angle-free bound.** Lemma D2 with the short segment `AC` at `A` and at `C`, and the segment `BA` at `B`:

    |det H_M det H_S det H_X| <= (1/8) L^3 K^(3(d-1)) r d_X^2 = (1/8) L^3 K^(3d-3) r^3 (p^2 + |q|^2). (4.9)

This holds on collinear triangles. The two factors of `d_X` are load-bearing, as in [PP].

### 4.7 Weighted Kac–Rice ([PP] §9)

Unchanged: the counted field is `grad f : X -> R^d`; the nonnegative mark is `W_r 1{H_X in O_j}`, where `O_j` is the open set of nonsingular symmetric matrices of index `j`; the witness determinant enters once through the Jacobian factor; the conditional gradient covariance is positive definite on compact punctures by distinct-site nondegeneracy ([LP] §2), and continuous Gaussian regression supplies the remaining hypotheses of arXiv:2304.07424v3 Theorem 7.1 (Remarks 7–8). Truncation and monotone convergence remove the growth restriction. Thus the intensity per unit physical volume is

    rho_j^W(X) = Z_r^(-1) p_( grad f(X) | pins )(0) E_(Q_r)[ W_r F_j(H_X) | grad f(X) = 0 ].       (4.10)

### 4.8 The crossover, and the integrable weight ([PP] §10)

**Region I: `|q| >= r |p|`.** Here `q != 0`. Put `t = |p|/|q|`, so `r^2 t^2 <= 1`, `chi^2 = t^2/(1 + r^2 t^2) in [t^2/2, t^2]`, and `|q|^2 Delta^(-d) <= |q|^(2-d)` because `Delta >= |q|`. Insert (4.8), dropping types, and (4.6), (4.7) with `n = 3d`, and [LP] (5.5) into (4.10):

    rho_j^W(X) <= C r^(-2) r^(-(d+1)) Delta^(-d) e^(-c chi^2) r^6 |q|^2 (1 + t^2)^3 (1 + chi^(3d))
               <= C r^(3-d) |q|^(2-d) sup_t (1 + t^2)^3 (1 + t^(3d)) e^(-c t^2/2)
               <= C' r^(3-d) |q|^(2-d).                                                          (4.11)

**Region II: `|q| < r |p|`.** Here `p != 0`, `Delta >= r|p|`, `Delta^2 <= 2 r^2 p^2`, `chi^2 in [1/(2r^2), 1/r^2]`, and `p^2 + |q|^2 <= 2 p^2 = 2 (r|p|)^2 / r^2`. Insert (4.9) instead:

    rho_j^W(X) <= C r^(-2) r^(-(d+1)) (r|p|)^(-d) e^(-c chi^2) r^3 (r|p|)^2 r^(-2) (1 + chi^(3d))
               <= C r^(-2-4d) e^(-c/(2r^2)) (r|p|)^(2-d)
               <= C r^(3-d) (r|p|)^(2-d),                                                        (4.12)

the middle step using `1 + chi^(3d) <= 2 r^(-3d)`, and the last step `e^(-c/(2r^2)) <= n! (2/c)^n r^(2n)` with `2n >= 5 + 3d`. The longitudinal axis `q = 0` is included; no division by `q` is made.

**Integration.** In both regions `rho_j^W(X) <= C r^(3-d) omega_d(p, q)` with `omega_d` from (1.1). Apply (4.10) on increasing compact punctures and use monotone convergence. Since `dX = r^d dp dq`,

    E_(Q_r^W) N_j(M + rE) = integral_(rE) rho_j^W <= C r^3 integral_E omega_d(p, q) dp dq,

which is (1.2). Lemma D5 gives the finite right side and the `O(r^5)` nested-ball bound. For `d = 2`, `omega_2 = 1` and this is [PP] (P2) with a bounded intensity; for `d >= 3` the intensity is unbounded on the axis but its weight is integrable, which is exactly why the count still has order `r^3`. ∎

### 4.9 Reflection to `S`

[PP] §10.1 applies verbatim: use the reflected local field `tilde f(x, y) = f(S - x e_1 + R_y y)` for any orthogonal map `R_y` of the transverse space (a reflection of one transverse axis keeps the orientation if desired; only orthogonality matters), with `tilde b = b - k r^3`, `tilde k = -k`. Sections 4.1–4.4 use only boundedness of the targets and `|k| >= k_-`; the deterministic estimates are index-free; the weighted Kac–Rice argument uses the relabeled original mark `F_(d-1)(H_0) F_d(H_r)`, the same random variable as `W_r`; and the original `Z_r` is retained. The two physical balls of radius `r/4` about `M` and `S` are disjoint.

## 5. Proof of Theorem C_d

We follow [CP] §§3–10 in midpoint coordinates `X = r(u, v)`, `u in R`, `v in R^m`, with the exact endpoint cubic `H_r(x) = b - k r^3/2 - (3k r^2/2) x + 2k x^3` and the midpoint jets `T_3 = grad_y f_xx(0) in R^m`, `C_3 = D_y^2 f_x(0) in Sym_m`, `S_0 = D_y^2 f(0) in Sym_m`.

### 5.1 Degree-five value-and-gradient interpolation at three sites ([CP] §3)

Let `a = (-1/2, 0)`, `b = (1/2, 0)`, `c = (u, v)` be the three sites of the collar; pairwise distances are at least `min(1, eta)`. Let `P_5` be the polynomials on `R^d` of total degree at most five; `dim P_5 = C(d+5, 5) >= 3(d+1)`. Consider

    B_c : P_5 -> R^(3(d+1)),   P -> ( P(a), grad P(a), P(b), grad P(b), P(c), grad P(c) ).

**Lemma.** `B_c` is onto for every `c` in `C(eta, R)`, including collinear configurations and `c = 0`, and its smallest row singular value has a uniform positive lower bound on `C(eta, R)` in the fixed monomial basis.

*Proof.* Choose a unit direction `w` such that the three projections `t_i = w . x_i` are distinct: for each of the three pairs the directions orthogonal to `x_i - x_j` form a great subsphere of `S^(d-1)`, a closed set with empty interior, so such `w` exists. Complete `w` to an orthonormal basis `w, w_1, ..., w_m` and write `z_j = w_j . x`, `z_(i,j) = w_j . x_i`. With the degree-two Lagrange polynomials `L_i(t)` at `t_1, t_2, t_3`, the polynomials

    h_i(t) = [ 1 - 2 L_i'(t_i)(t - t_i) ] L_i(t)^2,
    d_i(t) = (t - t_i) L_i(t)^2,
    e_(i,j)(t, z) = (z_j - z_(i,j)) L_i(t)^2,     j = 1, ..., m,

have total degree at most five. At the three sites, `h_i` has value `delta_(ij)` and zero gradient; `d_i` has zero value, `w`-derivative `delta_(ij)` and zero transverse derivatives; `e_(i,j)` has zero value and `w`-derivative and transverse derivative `delta_(ij)` in the `w_j` direction. These `3(1 + 1 + m) = 3(d+1)` polynomials prescribe arbitrary values and gradients, so `B_c` is onto in the original coordinates. Its entries are polynomials in `(u, v)`; surjectivity at every point of the compact collar gives a positive minimum of the least eigenvalue of `B_c B_c^T`. A single continuous choice of `w` is not needed. ∎

### 5.2 The coarse uniform floor ([CP] §4)

Collect the `3(d+1)` observations `V_r = ( f(ra), r grad f(ra), f(rb), r grad f(rb), f(rc), r grad f(rc) )`. With `J_5` the `C(d+5, 5)` midpoint Taylor coefficients `D^alpha f(0)/alpha!` and `D_r = diag(r^|alpha|)`, Taylor's formula through total degree five gives `V_r = B_c D_r J_5 + R_r`, `||R_r||_(L^s) <= C_s r^6` (the gradient entries carry `r . O(r^5)`), using [LP] (4.1) for the sixth derivatives. `Cov(J_5)` is uniformly positive definite (distinct one-site monomials, [LP] §2, compactness in the frame). For a unit row `t`, `std(t B_c D_r J_5) >= c r^5 ||t B_c|| >= c' r^5`, and the `L^2` triangle inequality gives `Cov(V_r) >= c r^10 I_(3(d+1))` for small `r`. Delete the auxiliary witness value (a principal submatrix keeps the floor), then pass from the scaled entries `r grad f` to the physical entries `grad f` by the congruence `V |-> diag(1, r^(-1) I_d, ...) V`, which multiplies the covariance by `r^(-2) >= 1` on the gradient block and, by the Schur variational identity `Cov(grad f(X) | pins) = min over affine predictors`, keeps the conditional covariance of the physical witness gradient at least `c r^10 I_d` (this is a statement about the congruence and the variational formula, not eigenvalue monotonicity under an arbitrary entrywise scaling), and take the Schur complement on the `2(d+1)` pins. Hence

    Sigma_X := Cov_(Q_r)( grad f(X) ) >= c r^10 I_d,     Sigma_X <= C I_d.                      (5.1)

Only `2(d+1) + d` observations are conditioned; no witness value and no height window appear.

### 5.3 The pinned drift ([CP] §5)

With `D(u) = u^2 - 1/4`, the endpoint pins and Taylor's formula give, uniformly on `|u|, |v| <= R`,

    G_1 := f_x(ru, rv)/r^2 = 6k D(u) + u v . T_3 + (1/2) v^T C_3 v + O_(L^s)(r),
    G_2 := grad_y f(ru, rv)/r = S_0 v + O_(L^s)(r),                                           (5.2)

exactly as [CP] (C7), with `v . T_3`, `v^T C_3 v` and `S_0 v` replacing the planar products. Since (C7) is stated for
two scalars, the vector form is derived in §6.3: (5.2) is (6.3) with `s = r` and `epsilon = 1` (so `d_e = D(u)`), on
the compact chart `|u|, |v| <= R`, whose `O(sK)` and `O(s^2 K)` remainders are `O(rK)`; the terms `(r/2) D(u) T_3`,
`r u C_3 v` and `(r/2) D_3[v, v]` of the transverse line are absorbed into the `O_(L^s)(r)` of the second line of
(5.2), and the inputs are the vector-valued endpoint identities (6.2). Hence `|E_(Q_r) G_1 - 6k D(u)| <= C(|v| + r)` and `Var_(Q_r)(G_1) <= C(|v|^2 + r^2)`. On `|v| <= v_0 <= eta/2` the collar gives `|D(u)| >= eta^2/4`, so `|E_(Q_r) G_1| >= d_0 > 0` after reducing `v_0` and `r_*`. The one-coordinate Mahalanobis bound `m^T Sigma^(-1) m >= m_1^2 / Sigma_11` [CP] (C10) supplies the factor `exp[-c/(r^2 + |v|^2)]` in the joint `d`-dimensional density.

### 5.4 Very near the axis: `|v| <= r^(1/3)` ([CP] §6)

By (5.1) the density prefactor is at most `C r^(-5d)`, so

    p_( grad f(X) | pins )(0) <= C r^(-5d) exp[-c r^(-2/3)].

Regress the whole field on `grad f(X)`: `||Sigma_X^(-1)|| <= C r^(-10)`, bounded cross-covariances, bounded targets, and the independent residual give `E_(Q_r)[ K^(3d) | grad f(X) = 0 ] <= C r^(-30d)`. The product of three absolute determinants is at most `K^(3d)`. Divide once by [LP] (5.5):

    rho_j^W(X) <= C r^(-2-35d) exp[-c r^(-2/3)] <= C r^(3-d),

by the `n`-th term of the exponential series with `2n/3 >= 5 + 34d`. The exact midpoint and the whole axis `v = 0` are covered.

### 5.5 The rest of the near-axis strip: `r^(1/3) <= |v| <= v_0` ([CP] §7)

Write `G = (G_1, G_2) = (6k D(u), 0) + B J + R`, `J = (T_3, C_3, S_0)`, `||R||_(L^2) <= C r`, with

    row 1: u v . T_3 + (1/2) v^T C_3 v,      rows 2..d: S_0 v.

The `C_3` block acts only on row 1 and the `S_0` block only on rows `2..d`, so by Lemma D3

    B B^T >= diag( |v|^4/4,  (|v|^2/2) I_m ) >= (|v|^4/4) I_d       (|v| <= sqrt 2),

and `trace(B B^T) <= C_R |v|^2`. The conditional covariance of `J` is uniformly sandwiched (§4.1, now at the midpoint), so `Cov(B J) >= c |v|^4 I_d`. Since `r/|v|^2 <= r^(1/3)`, the centered remainder is at most half the leading standard deviation for small `r`, and

    c |v|^4 I_d <= Cov_(Q_r)(G) <= C |v|^2 I_d.                                                  (5.3)

The physical gradient map is `diag(r^2, r I_m)` with determinant `r^(d+1)`. Using §5.3 and (5.3),

    p_( grad f(X) | pins )(0) <= C r^(-(d+1)) |v|^(-2d) exp[-c/|v|^2],                          (5.4)
    E_(Q_r)[ K^(3d) | grad f(X) = 0 ] <= C |v|^(-12d),                                           (5.5)

the latter by regression on `G` with `||Cov(G)^(-1)|| <= C |v|^(-4)`.

**Three soft Hessians in the collar.** The segment `MS` gives `||H_M e_x|| <= Lr/2`. From `grad f(X) = grad f(M) = 0`, `||H_M (X - M)|| <= L |X - M|^2 / 2 <= C_R L r^2`; since `X - M = r((u + 1/2) e_x + v)`, subtracting the controlled `e_x` component gives `||H_M v|| <= C_R L r`, hence `||H_M hat v|| <= C_R L r / |v|`. The plane `P = span(e_x, hat v)` is orthonormal because `v` is transverse. Lemma D1 with `sigma = 1` gives `|det H_M| <= C_R L^2 r^2 K^(d-2) / |v|`, and for every unit `w in P`, `||H_M w|| <= C_R L r / |v|` (using `|v| <= R`). Transport to `S` and to `X` (distances at most `(R+1) r`) keeps `||H_S w||, ||H_X w|| <= C_R L r / |v|` on `P`, so `|det H_S|, |det H_X| <= C_R L^2 r^2 K^(d-2) / |v|^2`. Therefore

    |det H_M det H_S det H_X| <= C_R K^(3d) r^6 |v|^(-6)                                          (5.6)

(we used `|v|^(-5) <= 2 |v|^(-6)` for `|v| <= 2` and `L <= K`). For `d = 2` this is [CP] (C18). Combining [LP] (5.5), (5.4), (5.5), (5.6):

    rho_j^W(X) <= C r^(3-d) |v|^(-(14d+6)) exp[-c/|v|^2] <= C' r^(3-d).                          (5.7)

The radius powers are `-2 - (d+1) + 6 = 3 - d`. No height-window factor appears.

### 5.6 Transverse compact remainder: `|v| >= v_0` ([CP] §8)

`B` has a fixed positive singular-value floor, the `O(r)` error is absorbed uniformly, the reduced mean, inverse covariance and conditional `C^3` moments are bounded, the density is `O(r^(-(d+1)))`, (5.6) gives `O(r^6)`, and division by `Z_r` leaves `rho_j^W <= C r^(3-d)`. No assertion that `D(u)` is separated from zero is needed here.

### 5.7 Integration and synthesis ([CP] §§9–10)

Weighted Kac–Rice on an open neighbourhood of the compact collar (distinct-site nondegeneracy, or the local polynomial floor), with the mark `W_r 1{H_X in O_j}`, gives (4.10) there. Since `dX = r^d du dv` and `rho_j^W <= C r^(3-d)` on all three strips, `E_(Q_r^W) N_j(rE) <= C r^3 |E|`, which is (1.3).

For (1.4), take `eta = 1/4`, partition a Borel subset of the scaled ball into its intersections with the two punctured pin balls of scaled radius `1/4` (Theorem P_d in endpoint coordinates; translation preserves volume) and the remaining collar (this theorem); assign boundaries once. Take the minimum of the radius cutoffs and the maximum of the constants. Choosing `R` larger than the inner radius of a scaled annulus supplies an overlapping cover of it. Every constant and cutoff of §5 depends on the fixed `d`, `R`, `eta`, `L`, `B` and the mark compacts, as stated in §1.1; nothing in this section is uniform as `d` grows. ∎

## 6. Proof of Theorem I_d

We follow [IW] §§3–10 with `epsilon = r/s in (0, 1/4]`, the endpoint frame `U_r` of [IW] (I7) extended by the `2m` transverse rows `L_r(0) in R^m`, `L_r'(0) in R^m` (affine interpolants of `grad_y f` at `+-r/2`), the scaled vector

    V_(r,s,w) = ( S_s U_r, f(sw), s grad f(sw) ),   S_s = diag(1, s, s^2, s^3, s I_m, s^2 I_m),   w = (u, v) in A = {1 <= |w| <= 2},

which has `2(d+1) + (d+1)` coordinates, and the endpoint-law inputs [IW] (I6), which are [LP] (4.1) and (5.5) in every `d`.

### 6.1 Polynomial row rank through `epsilon = 0` ([IW] §3.1)

For `epsilon > 0` the three sites `(-epsilon/2, 0)`, `(epsilon/2, 0)`, `w` are distinct and §5.1 gives rank `3(d+1)` on `P_5`, hence rank `3(d+1)` for the transformed rows (the Hermite transform is invertible). At `epsilon = 0` the `2(d+1)` contact rows `(f, f_x, f_xx, f_xxx, grad_y f, grad_y f_x)(0)` are independent. Polynomials of degree at most five annihilated by all of them are spanned by the monomials other than `1, x, x^2, x^3, y_j, x y_j`. To control the remaining `d + 1` rows:

- if `v != 0`, rotate the transverse frame so `v = |v| e_1` (the contact functionals are covariant under transverse rotations), and use `y_1^2, y_1^3, x y_1^2, y_j y_1^2 (j = 2..m)`. Their value/gradient matrix at `w` is block triangular: the `y_j`-derivative block is `|v|^2 I_(m-1)`, and the `3 x 3` block on `(value, d/dx, d/dy_1)` has determinant `-|v|^6`, so the total determinant is `-|v|^(2d+2) != 0`;
- if `v = 0`, then `|u| >= 1`, and `x^4, x^5, x^2 y_j` have value/gradient determinant `u^8 . u^(2m) = u^(2d+6) != 0`.

So the `(3(d+1)) x dim P_5` matrix `B_(epsilon, w)` has full row rank also at confluence. Its entries are polynomial in `epsilon`, hence continuous at `epsilon = 0`; compactness of `[0, 1/4] x A` supplies a uniform positive smallest row singular value. Degree four fails: on the axis, the annihilated monomials of degree at most four have proportional value and `x`-derivative at `(u, 0)` (only `x^4` contributes), so the witness rows have rank at most `d < d + 1`. The finite control `R` checks the two explicit families, a general rational transverse direction with a rational basis of its orthogonal complement, the degree-five rank `d + 1` and the degree-four rank `d` on the axis, for `d = 2, 3, 4`.

### 6.2 Taylor remainder, floor and conditional floor ([IW] §3.2)

[IW]'s divided-difference argument is one-dimensional along the axis and applies to each transverse component: every scaled endpoint coordinate is a bounded functional on `C^3` uniformly for `0 <= epsilon <= 1/4`, with the mass-one kernel identity for `H'''(0)` unchanged. Hence `V = B_(epsilon, w) D_s J_5 + R`, `||R||_(L^p) <= C_p s^6`, the standard-deviation floor `c s^5 ||a B||`, and

    Cov(V_(r,s,w)) >= c s^10 I,     Sigma_X := Cov_(Q_r)( grad f(X), f(X) ) >= c s^10 I_(d+1),     (6.1)

by unscaling and the Schur variational identity, exactly as [IW] (I10), (I11).

### 6.3 Midpoint expansions and the height coordinate ([IW] §4)

Let `T_3 = grad_y f_xx(0) in R^m`, `C_3 = D_y^2 f_x(0) in Sym_m`, `D_3 = D_y^3 f(0)` (symmetric 3-tensor), `S_0 = D_y^2 f(0) in Sym_m`. The endpoint constraints give, pathwise with `K` a fixed multiple of `1 + ||f||_(C^6)`,

    |f(0) - b_bar| <= C r^4 K,   f_x(0) = -3k r^2/2 + O(r^4 K),   grad_y f(0) = -r^2 T_3/8 + O(r^4 K),
    |f_xx(0)| + ||grad_y f_x(0)|| <= C r^2 K,   f_xxx(0) = 12k + O(r^2 K),                     (6.2)

by the same averaged endpoint identities as [IW] (I12), applied to each transverse component. Since (I12) is stated for
two transverse scalars, the vector-valued identities are derived here. Write `g(x) = f(x e_1)` and
`h(x) = grad_y f(x e_1) in R^m`, so the pins read `g(-r/2) = b`, `g(r/2) = b - k r^3`, `g'(±r/2) = 0`, `h(±r/2) = 0`.
Taylor's formula at `0` through order four, with `K` bounding the fifth derivatives, gives for each of `g` and each
component `h_i`:

    phi(r/2) - phi(-r/2) = r phi'(0) + (r^3/24) phi'''(0) + O(r^5 K),
    phi(r/2) + phi(-r/2) = 2 phi(0) + (r^2/4) phi''(0) + O(r^4 K),
    phi'(r/2) - phi'(-r/2) = r phi''(0) + O(r^3 K),
    phi'(r/2) + phi'(-r/2) = 2 phi'(0) + (r^2/4) phi'''(0) + O(r^4 K).

For `phi = g`: the third line gives `f_xx(0) = O(r^2 K)`; the fourth gives `g'(0) = -(r^2/8) g'''(0) + O(r^4 K)`;
inserting this into the first, whose left side is `-k r^3`, gives `-(r^3/12) g'''(0) = -k r^3 + O(r^5 K)`, hence
`f_xxx(0) = 12k + O(r^2 K)` and `f_x(0) = -3k r^2/2 + O(r^4 K)`; the second gives `f(0) = b - k r^3/2 - (r^2/8) f_xx(0)
+ O(r^4 K) = b_bar + O(r^4 K)`. For `phi = h_i`, both left sides vanish: the first line gives `h_i'(0) = O(r^2 K)`,
that is `grad_y f_x(0) = O(r^2 K)`, and the second gives `h_i(0) = -(r^2/8) h_i''(0) + O(r^4 K)`, that is
`grad_y f(0) = -(r^2/8) T_3 + O(r^4 K)`. Each is a one-dimensional identity applied to one component; the vector norm
costs a factor `sqrt(m)` in the constant. This is (6.2).

The expansions (6.3) below hold for any bounded `(u, v)` and any `epsilon <= 1`, with constants depending on the
bound; §5.3 uses them with `s = r`, `epsilon = 1` on `|u|, |v| <= R`. With `d_e = u^2 - epsilon^2/4`, at `X = s(u, v)`,

    f_x(X)/s^2 = 6k d_e + u v . T_3 + (1/2) v^T C_3 v + O(sK),
    grad_y f(X)/s = S_0 v + (s/2) d_e T_3 + s u C_3 v + (s/2) D_3[v, v] + O(s^2 K),
    [f(X) - b_bar]/s^3 = k(2u^3 - 3 epsilon^2 u/2) + v^T S_0 v/(2s) + (d_e/2) v . T_3 + (u/2) v^T C_3 v + (1/6) D_3[v,v,v] + O(sK).   (6.3)

The row operation `Z = ( f_x/s^2, grad_y f/s, [f - b_bar - (s/2) v . grad_y f]/s^3 )` removes the `S_0/s` term and the `C_3` term from the height row:

    Z = ( 6k d_e, 0, k(2u^3 - 3 epsilon^2 u/2) ) + B J + O_(L^p)(s),   J = (T_3, C_3, D_3, S_0),
    row 1:      u v . T_3 + (1/2) v^T C_3 v,
    rows 2..d:  S_0 v,
    row d+1:    (d_e/4) v . T_3 - (1/12) D_3[v, v, v].                                          (6.4)

The map from `Z` to `(f_x, grad_y f, f - b_bar)` is triangular with determinant `s^(2 + m + 3) = s^(d+4)`. The target at height `y in I_r` and zero gradient is `tau = (0, 0, (y - b_bar)/s^3)`, `|tau_(d+1)| <= (k_+/2) epsilon^3`, bounded through `epsilon -> 0`. The finite control `RO` checks (6.3)–(6.4) exactly on a pinned cubic in `d = 3`: no negative power of `s`, `S_0` and `C_3` absent from the height row, and the displayed coefficients.

### 6.4 Euler suppression and the transverse determinants ([IW] §5)

Suppose the pins hold, `grad f(X) = 0` and `f(X) = y in I_r`. Taylor expansion about `0` through degree three and Euler's identity for homogeneous polynomials give, in every `d`,

    3[f(X) - f(0)] - X . grad f(X) = 2 grad f(0) . X + (1/2) X^T H_0 X + O(s^4 K),                (6.5)

the cubic terms cancelling because their Euler degree is three (finite control `EU` in `d = 3`). Here `(1/2) X^T H_0 X = (s^2/2)[ u^2 f_xx(0) + 2u grad_y f_x(0) . v + v^T S_0 v ]`. By (6.2), `|y - b_bar| <= k_+ r^3/2` and `r <= s/4 <= 1`, every term other than `(s^2/2) v^T S_0 v` is `O(K(r^2 s + s^4))`, so for `v != 0`, with `hat v = v/|v|`,

    | hat v^T S_0 hat v | <= C K (r^2/s + s^2) / |v|^2 <= C K (r + s^2) / |v|^2.                    (6.6)

This is [IW] (I18) for the quadratic form in the direction `v`. It controls one entry of `S_0`, not the matrix. The second piece of information is the transverse gradient equation: from `grad_y f(X) = 0` and the second line of (6.3), `S_0 v = O(sK)`, that is

    | S_0 hat v | <= C s K / |v|.                                                                  (6.7)

**Endpoint determinants.** Let `A_i = D_y^2 f(i)` for `i in {M, S}`; then `A_i = S_0 + O(rK)` by the Lipschitz bound over distance `r/2`, so `|hat v^T A_i hat v| <= C K (r + s^2)/|v|^2` (absorbing `rK <= 4 rK/|v|^2`) and `|A_i hat v| <= C s K/|v|`. Lemma D4 gives

    |det A_i| <= C K^(d-2) (r + s^2)/|v|^2 + C K^(d-3) s^2 K^2/|v|^2 <= C K^(d-1) (r + s^2)/|v|^2

(the second term is absent when `d = 2`, where `A_i` is the scalar `epsilon_i`). With [LP] (5.3) and (5.1), `|det H_i| = r |alpha_i det A_i - r beta_i^T adj(A_i) beta_i| <= r K |det A_i| + r^2 K^d`, hence

    |det H_M|, |det H_S| <= C r K^d (r + s^2) / |v|^2.                                             (6.8)

For `d = 2` this is [IW] (I19). The second term of Lemma D4, of size `s^2/|v|^2`, is of the same order as the `s^2` part of the Euler bound; that coincidence is what makes the lift work without loss.

**Witness determinant.** As in [IW] (I20), `||H_M e_x|| <= Lr/2` and `||H_M (X - M)|| <= C K s^2` give `||H_M hat v|| <= C K s/|v|`; transport over distance at most `3s` gives `||H_X e_x|| <= C K s` and `||H_X hat v|| <= C K s/|v|`. Lemma D1 on the orthonormal pair `(e_x, hat v)`:

    |det H_X| <= C s^2 K^d / |v|.                                                                  (6.9)

Combining (6.8), (6.9), dividing once by [LP] (5.5) and using `r^2/s <= r`,

    W_r |det H_X| / Z_r <= C K^(3d) s^2 (r + s^2)^2 |v|^(-5).                                        (6.10)

**Axis-safe bound, valid throughout the strip `|v| <= s^(1/8)`, including `v = 0`.** The two endpoint short columns `H_M e_x`, `H_S e_x`, of size at most `Lr/2` by [LP] (5.3), do not involve the witness position, so Lemma D2 (Hadamard with `e_x`) gives `|det H_M|, |det H_S| <= C r K^(d-1)` for every `X`; with the trivial `|det H_X| <= K^d`, `W_r |det H_X| <= C r^2 K^(3d)` and, dividing once by [LP] (5.5), `W_r |det H_X| / Z_r <= C K^(3d)` on the whole strip, with no division by `|v|`. A bound stated only at `v = 0` would not control nonzero `v`; it is this uniform bound that §6.5 uses.

### 6.5 Small axial strip ([IW] §6)

On `A` and `|v| <= v_0`, `|u|` is bounded below and `d_e >= c > 0`; the first line of (6.3) gives `|E_(Q_r)[f_x(X)/s^2]| >= c_0` and `Var <= C(|v|^2 + s^2)`; the one-coordinate Mahalanobis bound applies to the full `(d+1)`-dimensional exponent. On `|v| <= s^(1/8)`, (6.1) gives the joint density bound `C s^(-5(d+1)) exp[-c s^(-1/4)]`, regression on `Y_X = (grad f(X), f(X))` with `||Sigma_X^(-1)|| <= C s^(-10)` gives `E[K^(3d) | grad f(X) = 0, f(X) = y] <= C s^(-30d)`, and with the axis-safe bound of §6.4 (`W_r |det H_X| / Z_r <= C K^(3d)` throughout the strip) the intensity per unit volume and height is bounded by `C s^(-N) exp[-c s^(-1/4)] <= C`. The strip has volume `O(s^d) <= O(s^2)` and the window length is `k r^3`, so it contributes at most `C r^3 s^2` to the shell count.

### 6.6 Transverse region: `s^(1/8) <= |v| <= 2` ([IW] §7)

In (6.4) the three jet blocks `C_3`, `S_0`, `D_3` act on disjoint row sets, so by Lemma D3

    B B^T >= diag( |v|^4/4,  (|v|^2/2) I_m,  |v|^6/144 ),   det(B B^T) >= c |v|^(2d+8),   sigma_min(B) >= c |v|^3,

and `||B|| <= C|v|`. The conditional covariance of `J` is uniformly sandwiched, so the leading standard deviation of `a . Z` is at least `c ||aB|| >= c |v|^3 ||a||`, while the centered `O(s)` error is at most `C s ||a||`; since `s/|v|^3 <= s^(5/8)`, the error is at most half the leading term for small `s`, giving the relative bound

    Cov_(Q_r)(Z) >= c B B^T,   det Cov_(Q_r)(Z) >= c |v|^(2d+8),   ||Cov_(Q_r)(Z)^(-1)|| <= C |v|^(-6),   Cov_(Q_r)(Z) <= C |v|^2 I.   (6.11)

With the Jacobian `s^(d+4)` and the Gaussian penalty `exp(-c/|v|^2)` from §6.5 (or a constant for `|v| >= v_0`),

    p_( grad f(X), f(X) | pins )(0, y) <= C s^(-(d+4)) |v|^(-(d+4)) exp(-c/|v|^2),                  (6.12)
    E_(Q_r)[ K^(3d) | grad f(X) = 0, f(X) = y ] <= C |v|^(-18d),                                    (6.13)

the latter by regression on `Z` with (6.11) and the bounded target `tau`.

### 6.7 The shell ledger ([IW] §8)

Insert (6.10) into the conditional expectation of the weighted density and apply (6.12), (6.13):

    Lambda_(r,j)(X, y) <= C s^(-(d+2)) (r + s^2)^2 |v|^(-(19d+9)) exp(-c/|v|^2) <= C' s^(-(d+2)) (r + s^2)^2.   (6.14)

Every inverse power of `|v|` is absorbed by its Gaussian factor uniformly down to `s^(1/8)`. Integrate over the physical shell (volume `O(s^d)`) and the window (length `k r^3`):

    E N_(r,j)(shell, transverse) <= C r^3 (r + s^2)^2 / s^2 <= 2C r^3 [ (r/s)^2 + s^2 ],             (6.15)

by `(a + b)^2 <= 2a^2 + 2b^2`; equivalently `2[(r/s)^2 + s^2] - (r + s^2)^2/s^2 = (r/s - s)^2 >= 0` (finite control `L`). Adding the axial contribution of §6.5 proves (1.5). The `r` powers are `-2 + 2 + 3 = 3` and the `s` powers `-(d+4) + 2 + d = -2` in every `d`.

### 6.8 Kac–Rice with height disintegration, and the dyadic sum ([IW] §§9–10)

For each `r, s > 0` the closed shell is separated from the pins; (6.1) gives nondegeneracy of `(grad f(X), f(X))`; the marked critical-point formula with the mark `W_r 1{H_X in O_j} 1{f(X) in I_r}` and height disintegration gives

    E_(Q_r^W) N_(r,j)(B) = integral_B integral_(I_r) Lambda_(r,j)(X, y) dy dX,
    Lambda_(r,j)(X, y) = Z_r^(-1) p_( grad f(X), f(X) | pins )(0, y) E_(Q_r)[ W_r F_j(H_X) | grad f(X) = 0, f(X) = y ],

exactly as [IW] (I33), with the same framework citation. The dyadic sum [IW] (I34) is dimension-free: `sum_j (r/s_j)^2 <= 4/(3 A_0^2)` and `sum_j s_j^2 <= (4/3) rho^2` for `s_j = 2^j A_0 r`. This proves (1.6). ∎

## 7. Proof of Theorem G_d and the corollaries

Use one `(Q_r, W_r, Z_r)` throughout, distances from the midpoint, and the tiling of [IW] §10 and its review (N1):

| Region | Source | Parameters | Heights |
|---|---|---|---|
| `|X| <= 4r`, pins removed | Theorem C_d, (1.4) with `R = 4` | fixed | all heights, which contain `I_r` |
| `4r <= |X| <= s_0` | Theorem I_d, (1.6) with `A_0 = 4`, `rho = s_0` | fixed `s_0` | `I_r` |
| `|X| >= s_0` | [RM] Theorem A, "fix `0 < rho < L/4`", reviewed ACCEPT in every fixed `d` (main#76) | `rho = s_0` | `I_r` |

Boundaries are assigned once and `r` is taken below all three cutoffs. No annulus is left uncovered, no regional normalizer is multiplied, and no pin-density Jacobian is inserted. Summing the three bounds gives (1.7).

*Corollary F_d.* `Q_r^W(A_r) <= E_(Q_r^W) sum_j N_(r,j)(X minus {M, S}) <= (d+1) C r^3` by Markov; the lower bound is [RC] (F-), whose proof uses only [RM] (A) and the fixed-remote second factorial moment (D), both valid in every fixed `d`. ∎

*Corollary M_d.* [EDL] (A4) gives `Q_r^W(N_r >= 2) >= c r^3` and `E[N_r(N_r - 1)] >= c r^3` in every fixed `d`. For the upper bound, `1{N >= 2} <= N/2` and (1.7). The total-variation statement is [LOCAL] §7 (an identity for any finite point process with mean mass at most one) applied with (1.7) and (A4). ∎

## 8. What this note does not establish

- An all-height intermediate or global `O(r^3)` count. The height window in (6.6) is load-bearing, as [IW] and its reviews record; the all-height shell method gives `O(r)`, not `O(r^3)`.
- A torus-wide second factorial moment upper bound, or any collision estimate outside the fixed remote region of [RC].
- Elder selection; D1 Theorem A covers it separately.
- Uniformity as `k -> 0`, as marks grow, as `L` or `d` vary, or any numerical `C`, `c`, `r_*`, `s_0`.
- Any recovery of historical RN/JETMOD/24-jet carriers, or any change to their `BLOCKED_ABSENT` / `OPEN_HISTORICAL` classifications. Theorem G_d in `d = 3` is an analytic, existential-constant statement about the same kind of object those carriers were meant to certify numerically; whether the registers should record a regional supersession is a separate source-bound reconciliation, not something this note asserts.
- Organizational independence: every lane shares one GitHub account.

The candidate's own review obligations are listed in `README.md`. A defect in any of the five devices, or in any planar step cited here as dimension-free, blocks consumption of the corresponding theorem.

## 9. Finite controls

`lift_exact_check.py` (standard library, exact rational and polynomial arithmetic) checks the nine groups listed in its docstring and rejects ten deliberate mutations: the exterior-power order, the two frame blocks, the interpolation degree, the row-operation constant, the Euler coefficient, the adjugate sign, one soft direction instead of two, a non-summable shell factor, and a non-integrable pin weight. `RESULTS.json` is its exact output, identical in normal and optimized modes. These checks verify identities, ranks, floors and exponents. They do not verify the Gaussian, compactness, regression, or Kac–Rice steps, which are the written arguments above and in the sources.

## 10. Attribution and reconnaissance

The mechanisms are those of the planar sources; the contribution here is the written `d`-dimensional replacement of each planar-specific step and the five devices of §3, which are classical linear algebra (exterior powers, Hadamard, Schur complements) applied at the right places. Nothing is claimed as new beyond this explicit application. `RECONNAISSANCE.md` records the limited external check and its failure mode.
