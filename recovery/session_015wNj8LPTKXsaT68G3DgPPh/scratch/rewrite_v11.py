import pathlib
import os
os.chdir('/home/user/Math-/frontiers/c6_cluster_law_20260929')
import pathlib
p = pathlib.Path('PROOF.md'); s = p.read_text()
start = s.index("### 3.2 The shear chart and the singular slice")
end = s.index("## 5. The middle and far regions, and the absence of cross terms")
new = r'''### 3.2 The spectral chart and the singular slice

For `d = 2`, `A = f_yy(0)` is a scalar; put `sigma = A`, and there is no `D`, no `U` and no `w`. For `d >= 3` write the
eigenvalues of `A` as `lambda_1, ..., lambda_m` and let `sigma` be the eigenvalue of least modulus; `Q_r`-almost
surely it is unique and simple, since `A` has a Lebesgue density ([LP] (3.5)). Let `u_1` be a unit eigenvector for
`sigma`, `U in O(m)` an orthogonal matrix with first column `u_1`, and `D = diag(lambda_2, ..., lambda_m)` the
remaining eigenvalues in a fixed order, so that

    A = U diag(sigma, D) U^T,      |sigma| <= |lambda_j|  (j >= 2).                                                 (3.2)

The Weyl integration formula for real symmetric matrices (e.g. Anderson, Guionnet, Zeitouni, *An Introduction to Random
Matrices*, §2.5) writes Lebesgue measure on `Sym_m` in these coordinates as

    dA = c_m  prod_(j >= 2) |sigma - lambda_j| . Delta(D)  dsigma dD dU,      Delta(D) = prod_(2 <= i < j) |lambda_i - lambda_j|,   (3.3)

with `dU` the Haar measure on `O(m)` and `c_m` a fixed positive constant (the restriction `|sigma| <= |lambda_j|`
selects one of the `m` eigenvalue labellings). The density factor is a polynomial in `(sigma, D)`, and it equals
`c_m |det D| Delta(D)` on the singular slice `{sigma = 0}`.

Rotate the transverse coordinates by `U`: `f~(x, z, w) = f( R diag(1, U) (x, z, w) )`, with `z in R` along `u_1` and
`w in R^n`, `n = m - 1`. Since `diag(1, U)` is orthogonal, `f~` has the same `C^4` norm as `f`, its transverse Hessian
at `0` is `diag(sigma, D)` (so `f~_zw(0) = 0`, `f~_ww(0) = D`), and the physical ball `|X| <= A r` is the scaled ball
`X^2 + Z^2 + |rW|^2 <= A^2` in the coordinates `(x, z, w) = (rX, rZ, r^2 W)` used below. Let `tau` be the third
derivatives of `f~` at `0` other than `f~_xxx`; `tau` is the image of `T` under a fixed orthogonal representation of
`diag(1, U)`, so `|tau| = |T|` and `T -> tau` has Jacobian one for fixed `U`. Write `j = (sigma, U, D, tau)` and
`chart(j)` for the raw jet `(A, T)`.

The chart replaces the shear of [EDL] (A8)–(A10) by a rotation; no unbounded coefficient enters, at the price of the
polynomial density (3.3). Configurations in which a second eigenvalue is also small (`|lambda_2|` small) are not
excluded from the chart; they are handled by the domination of §4, whose dominating function does not involve
`D^(-1)` (Lemma 4.3), and by the almost-sure nonsingularity of `D` in the pointwise limits.

### 3.3 The normal form

Let `F_r(X, Z, W) = [ f~(rX, rZ, r^2 W) - b ] / r^3` and

    s = sigma / r,   a_3 = tau_xxz,   beta = tau_xzz,   c_3 = tau_zzz,   Q_j(X, Z) = (tau_xxw_j / 2)(X^2 - 1/4) + tau_xzw_j X Z + (tau_zzw_j / 2) Z^2.

**Lemma 3.1 (normal form, [EDL] (A17), (A26)–(A28); [LM] (L6)–(L7)).** For every `R_0 >= 1` there is `C = C(d, R_0)`
such that on `{|(X, Z)| <= R_0, |W| <= R_0}`, uniformly in `b, k, R`, `0 < r <= 1`,

    || F_r(., ., 0) - F_0 ||_(C^2) <= C r K_4,                                                                     (3.4)
    r^(-2) grad f~(rX, rZ, r^2 W) = ( partial_X F_0, partial_Z F_0, D W + Q(X, Z) ) + O( r K_4 (1 + |D|) ),          (3.5)

with `F_0` as in (1.2) (the pins are exactly retained: `F_0(-+1/2, 0) = 0, -k` and `grad F_0(-+1/2, 0) = 0`, checked
exactly in the finite control `NF`). Moreover, for `|(X, Z)| <= R_0` and physical `|w| <= R_0 r`,
`|partial_w f~| >= (sigma_min(D) - C r K_4 R_0)|w| - r^2 |Q| - C r^3 K_4`, so if `D` is nonsingular and
`r <= r_0(K_4, R_0, D)`, every critical point of `f~` in the physical ball of radius `R_0 r` has
`|W| <= C |Q(X,Z)| / sigma_min(D) + 1`.

*Proof.* The Taylor expansion of `f~` at the midpoint through order three, with the pin identities of [LM] §3
(`f~_xxx(0) = 12k + O(r K_4)`, `f~_x(0) = -(r^2/8) f~_xxx(0) + O(r^3 K_4)`, `f~_z(0) = -(r^2/8) f~_xxz(0) + O(r^3 K_4)`,
`f~_xx(0), f~_xz(0) = O(r^2 K_4)`, `f~(0) - b = -(r^3/24) f~_xxx(0) + O(r^4 K_4)`, and, from the two transverse gradient
pins, `f~_w(0) = -(r^2/8) f~_xxw(0) + O(r^3 K_4)`, `f~_xw(0) = O(r^2 K_4)`, while `f~_zw(0) = 0` exactly), is (A17) of
[EDL] on `w = 0` and gives (3.4). Differentiating in `w` and using `D_w^2 f~(0) = D`, `D_(zw) f~(0) = 0` gives (3.5),
the last block being `r^2 (D W + Q) + O(r^3 K_4)` before division. The lower bound on `|partial_w f~|` is (3.5) read at
physical `w = r^2 W` with `|W| <= R_0 / r`: the quadratic remainder `K_4 |w|^2 <= K_4 R_0 r |w|` is relative to the
linear term `D w`. ∎

Consequently, when `D` is nonsingular and `r <= r_0(K_4, R_0, D)`, the critical points of `f~` in the physical ball of
radius `R_0 r` are in bijection with the zeros of the reduced planar gradient

    grad G_r(X, Z),   G_r(X, Z) = F_r(X, Z, W_*(X, Z)),   W_*(X, Z) = the unique solution of the third block of (3.5) = 0,

`W_* = -D^(-1) Q(X, Z) + O(r K_4 (1 + |D^(-1)|)^2)`, and

    || G_r - F_0 ||_(C^2({|(X,Z)| <= R_0})) <= C r K_4 (1 + |D^(-1)|)^2 (1 + |tau|)^2.                              (3.6)

Indeed `F_r(X, Z, W) - F_r(X, Z, 0) = r (Q . W + W^T D W / 2) + O(r^2 K_4)` on bounded `W`, and the implicit function
`W_*` is `C^2` with the stated bound because the third block of (3.5) has derivative `D + O(r K_4)` in `W`. The index of a
critical point of `f~` equals the index of the corresponding critical point of `G_r` plus the index of `D` ([EDL] §9:
the full Hessian is block triangular up to `O(r)` with diagonal blocks the planar Hessian of `G_r` and `D`), and its
height satisfies `(f~ - b)/r^3 = G_r` at the point. The bound (3.6) is used only for the pointwise limits of §4.2, at
fixed jets with `D` nonsingular; it is not used for domination.

### 3.4 The planar cubic

**Lemma 3.2 (at most two extra critical points).** For every `(k, s, a_3, beta, c_3)` with `k > 0`, the resultant of
`partial_X F_0` and `partial_Z F_0` with respect to `Z` equals `(X^2 - 1/4) Q_2(X)` identically in
`(X, s, a_3, beta, c_3, k)`, where

    Q_2(X) = c_2 X^2 + s c_1 X + c_0,
      c_2 = a_3^3 c_3/4 - 3 a_3^2 beta^2/16 - (9/2) a_3 beta c_3 k + 3 beta^3 k + 9 c_3^2 k^2,
      c_1 = -a_3^2 beta/4 - 3 a_3 c_3 k + 6 beta^2 k,
      c_0 = 3 s^2 beta k - ( a_3 beta/8 - 3 c_3 k/2 )^2                                                         (3.7)

(finite control `RS`, exact polynomial identity). Let `E_k` be the set of `(s, a_3, beta, c_3)` for which the two
quadratics `partial_X F_0, partial_Z F_0` have a common factor, or `Q_2` vanishes identically, or some common zero other
than the pins is degenerate, or has height in `{-k, 0}`. `E_k` is a proper algebraic subset of `R^4`, hence
Lebesgue-null, and off `E_k` the critical points of `F_0` are isolated, at most four with multiplicity (Bézout for two
conics), two of them the pins; so the extra real critical points number at most two, they are nondegenerate, and
`N_oo in {0, 1, 2}`. On `E_k` put `N_oo := 0` (the value is irrelevant: `E_k` is null for every `k`).

*Proof.* The resultant identity is an exact computation (the Sylvester determinant of two quadratics in `Z`), verified
symbolically in `RS`. The pins are the zeros of `partial_X F_0 = 6k X^2 - 3k/2 + a_3 X Z + (beta/2) Z^2` and
`partial_Z F_0 = s Z + (a_3/2)(X^2 - 1/4) + beta X Z + (c_3/2) Z^2` on `Z = 0`, which explains the factor `X^2 - 1/4`.
Off `E_k`, for each root `X_*` of `Q_2` the two quadratics in `Z` are not proportional (they have no common factor), so
they share at most one root `Z`; if they are proportional at `X_*`, the extra points are the at most two zeros of
`partial_X F_0(X_*, .)` (the [LM] configuration, where control `EX` exhibits `X_* = -3/4` as a double root of `Q_2`).
Bézout's theorem bounds the number of isolated common zeros of two conics by four. `E_k` is defined by polynomial
equations in the jets (common factor: vanishing of the resultant as a polynomial in `X`; degeneracy and boundary
heights: vanishing of the Hessian determinant, or of `F_0` or `F_0 + k`, at a common zero, eliminated through the
resultant), and it is proper because the configurations of §7 (exact rational examples with `0`, `1` and `2` extra
critical points in the window, all nondegenerate) lie outside it. The example `(0, 0, 0, 0)`, where the lines
`X = -+1/2` consist of critical points, lies in `E_k`. ∎

### 3.5 Large soft curvature excludes near critical points

**Lemma 3.3 (exclusion, full transverse space).** There are `C = C(d)` and `r_0(kappa, R_0) > 0` such that, if
`K_4 <= kappa`, `r <= r_0(kappa, R_0)`, and the transverse Hessian `A` at the midpoint satisfies `|lambda_j| >= S r`
for every eigenvalue, with

    S >= S_*(tau, R_0, kappa, k) := C (1 + |tau|)^2 (1 + R_0)^3 (1 + kappa)^2 / k^2,                               (3.8)

then the only critical points of `f` in the physical ball of radius `R_0 r` about the midpoint are the two pins.
Equivalently, on `{K_4 <= kappa}`, `N^(R_0) >= 1` forces `|s| = |sigma|/r <= S_*(tau, R_0, kappa, k)` for `r <= r_0`.

*Proof.* Work in the original pin frame (no rotation is needed), with scaled coordinates `x = rX`, `y = rY`,
`|(X, Y)| <= R_0`. From the two transverse gradient pins and the pin identities ([LM] §3, [EDL] (A17), the same
computation as (3.4) in every transverse direction),

    grad_y f(rX, 0) = (r^2/2)(X^2 - 1/4) grad_y f_xx(0) + O(r^3 K_4 (1 + R_0)^3),      D_y^2 f(rX, 0) = A + O(r K_4 R_0),

so at a critical point, `0 = grad_y f(rX, rY) = r^2 [ (1/2)(X^2 - 1/4) grad_y f_xx(0) ] + (A + O(r K_4 R_0)) r Y + O(r^2 K_4 R_0^2)`.
Dividing by `r` and using `|A Y| >= S r |Y|`,

    (S r - C r K_4 R_0) |Y| <= C r (1 + |tau|)(1 + R_0)^2 + C r K_4 R_0^2,     i.e.     |Y| <= delta := C (1 + |tau| + kappa R_0^2)(1 + R_0)^2 / (S - C kappa R_0),

which is small for `S` large. In the axial equation, `f_x(rX, rY)/r^2 = 6k X^2 - 3k/2 + X grad_y f_xx(0) . Y + O((1 + |tau|) |Y|^2) + O(r K_4 (1 + R_0)^3)`
([LM] §3, [EDL] (A17): the term `f_xy(0) . Y / r` is `O(r K_4 |Y|)` by the pin identity `f_xy(0) = O(r^2 K_4)`), so
`|6k X^2 - 3k/2| <= C (1 + |tau|) R_0 delta + C (1 + |tau|) delta^2 + C r kappa (1 + R_0)^3`. Since
`|6k X^2 - 3k/2| = 6k |X - 1/2| |X + 1/2|`, for `delta` and `r` small this forces `X` within `C (1 + |tau|) R_0 delta / k`
of `-+1/2`. Hence every critical point in the scaled ball lies within scaled distance
`rho := delta + C (1 + |tau|) R_0 delta / k` of a pin.

At a pin `P`, `grad f(P) = 0` exactly and, for `|xi| <= 1`, `|grad f(P + r xi)| >= sigma_min(H_P) r |xi| - K_4 r^2 |xi|^2 / 2`.
The Hessian `H_P` is `r` times the block matrix with axial entry `-+6k + O(r K_4)`, mixed axial–transverse row
`O(1 + |tau|)` (the entries `-+(1/2) grad_y f_xx(0) + O(r K_4)`), and transverse block `A/r + O(K_4)` whose eigenvalues
have modulus at least `S - C kappa`. For `S >= C (1 + |tau|)^2 / k + C kappa`, the Schur complement of the transverse
block is `-+6k + O((1 + |tau|)^2 / (S - C kappa))`, so `sigma_min(H_P) >= c k r`, and no other zero of `grad f` lies
within scaled distance `rho_pin := c k / kappa` of `P`. The two conclusions are incompatible as soon as
`rho < rho_pin`, which holds when `S >= S_*` with the constant of (3.8). ∎

The lemma uses no planar reduction and no inverse of `D`; the only randomness it sees is through the deterministic
bound `K_4 <= kappa`. Note that on the singular slice `|sigma| <= |lambda_j|` for all `j`, so the hypothesis is exactly
`|s| >= S_*`.

### 3.6 The weight on the singular slice

**Lemma 3.4 (pin Hessians).** With `M_2(M) = [[-6k, -a_3/2], [-a_3/2, s - beta/2]]` and `M_2(S) = [[6k, a_3/2],
[a_3/2, s + beta/2]]` (exact second derivatives of `F_0` at the pins, control `NF`), the Hessians of `f~` at the pins are

    H~_i = [[ r M_2(i) + O(r^2 K_4), r c_i + O(r^2 K_4) ], [ r c_i^T + O(r^2 K_4), D + O(r K_4) ]],   i in {M, S},

with `c_i in R^(2 x n)` linear in `tau` (the entries `-+tau_xxw/2, -+tau_xzw/2`), and

    det H~_i = r^2 det D . det M_2(i) + O( r^3 K_4^d (1 + |tau|)^2 (1 + |s|)^2 ),                                 (3.9)

exactly as polynomials: the determinant of a matrix whose two soft rows are `O(r)` and whose stable block is `O(1)` is
`r^2` times the product of the soft-block and stable-block determinants plus `r^3` times a polynomial in the entries
(control `BD` checks this for `d = 3` and `d = 4` with exact rational arithmetic in `r`); the remainder involves no
inverse of `D`. The rotation `diag(1, U)` is orthogonal, so `det H_i = det H~_i` and the index is unchanged. Moreover,
by Hadamard's inequality applied to the two soft rows, the row of the least stable eigenvalue `lambda_2` (whose
entries are `O(r K_4)` off the diagonal and `lambda_2 + O(r K_4)` on it) and the remaining rows,

    r^(-4) W_r <= C (1 + |s| + |tau| + K_4)^4 ( |lambda_2| + r K_4 )^2 K_4^(2d - 6)      (d >= 3),                   (3.10)

and `r^(-4) W_r <= C (1 + |s| + |tau| + K_4)^4` for `d = 2`.

*Proof.* The entries follow from Lemma 3.1 by differentiating (3.4)–(3.5) once more at the pins:
`partial_X^2 F_0 = 12k X + a_3 Z`, `partial_X partial_Z F_0 = a_3 X + beta Z`, `partial_Z^2 F_0 = s + beta X + c_3 Z`,
evaluated at `(-+1/2, 0)`, and the mixed soft–stable entries are the derivatives of `Q`. The determinant expansion is
the Leibniz formula: every term contains at least two factors from the two soft rows, each `O(r)`; the terms with
exactly two such factors and the identity permutation on the stable block give `r^2 det M_2 det D`; the remaining terms
carry `r^3`. (3.10) is Hadamard's inequality with the stated row norms. ∎

**Lemma 3.5 (weight limit).** Write `w_0` as in (1.1). If `D` is nonsingular and `det M_2(M) != 0 != det M_2(S)`,
then under the conditional law of §3.1 with `sigma = r s`,

    r^(-4) W_r -> (det D)^2 w_0(s, a_3, beta)     in probability as r -> 0.

Here `w_0 > 0` exactly when `M_2(M)` is negative definite (index `d` at `M`, given `D < 0`) and `M_2(S)` has one negative
eigenvalue (index `d - 1` at `S`); if `D` is not negative definite the limit weight is `0` because the index conditions
fail.

*Proof.* (3.9) gives `r^(-2) det H~_M -> det D . det M_2(M)`, and the index of `H~_M` converges to `index(M_2(M)) +
index(D)` because eigenvalues are continuous and the limits are nonsingular. `F_d(H_M) = |det H_M| 1{index = d}` gives
the first factor; the second factor is the same at `S`. The condition `index M_2(M) = 2` is `det M_2(M) > 0` (its
`(1,1)` entry is `-6k < 0`), and `index M_2(S) = 1` is `det M_2(S) < 0` (its `(1,1)` entry is `6k > 0`); these are the
two positive parts in (1.1). ∎

### 3.7 The reduced weight

Let `p_0` be the `Q_0`-density of the raw jet `J = (A, T)` (§3.1). Define

    m(s, a_3, beta, c_3) := integral p_0( chart(0, U, D, tau) ) c_m |det D| Delta(D) (det D)^2 1{D < 0} dU dD dtau_rest,   (3.11)

where `tau_rest` collects the entries of `tau` other than `(a_3, beta, c_3)` (those involving a `w` index), `dU` is the
Haar measure, and `c_m |det D| Delta(D)` is the density (3.3) on the singular slice. The integrand does not depend on
`s`; `m` is a bounded, strictly positive, continuous function of `(a_3, beta, c_3)` with Gaussian decay (the density
`p_0` is Gaussian and nondegenerate, `|chart(0, U, D, tau)| >= c (|D| + |tau|)`, and the Weyl factor is polynomial), and
it depends continuously on `(b, k, R)`. In `d = 2`, `m(s, a_3, beta, c_3) = p_0(0, a_3, beta, c_3)`, the density of
`(f_zz, f_xxz, f_xzz, f_zzz)(0)` at `f_zz = 0`.

## 4. The near limit

### 4.1 Scaled representation

Fix `A >= 4` and write `N^A = N_near^A`. For every Borel set `E` of field configurations,

    Q_r^W(E) = Z_r^(-1) integral p_r(j) E[ W_r 1_E | U_r = v_r, J = j ] dj,                                        (4.1)

by the tower property with the regular conditional law of §3.1. Change variables to the spectral coordinates
`(sigma, U, D, tau)` with the density (3.3) and then `sigma = r s`:

    r^(-3) Q_r^W(E) = (r^2 / Z_r) integral ds dU dD dtau  p_r( chart(rs, U, D, tau) )  c_m prod_(j>=2) |rs - lambda_j| Delta(D)  r^(-4) E[ W_r 1_E | U_r = v_r, J = chart(rs, U, D, tau) ],   (4.2)

the `s`-integral being over `{ |rs| <= min_j |lambda_j| }`. The factor `r` from `dsigma = r ds` and the factor `r^(-4)`
from the weight combine with `r^2/Z_r -> 1/z_0` ([LP] (5.4)) to the exponent `1 + 4 - 2 = 3` of [LM] §6.

### 4.2 Convergence of the scaled count

**Lemma 4.1.** Fix `A >= 4`, fix `kappa > 0` outside the countable set of atoms of the law of `K_4` under the
conditional laws below, and fix `(s, U, D, tau)` with `D` nonsingular and `(s, a_3, beta, c_3)` outside `E_k` and outside
the null set of jets for which an extra critical point of `F_0` lies on the circle `X^2 + Z^2 = A^2`. Let
`j_r = chart(rs, U, D, tau)` and `j_0 = chart(0, U, D, tau)`. Then, under the conditional laws `Q_(r, j_r)` of §3.1,

    1{ N^A = n } 1{K_4 <= kappa}  ->  1{ N_oo^A = n } 1{K_4 <= kappa}     in probability,
    N_oo^A := #{ extra critical points (X, Z) of F_0 with X^2 + Z^2 < A^2 and F_0(X, Z) in (-k, 0) },

and likewise for the counts restricted to any fixed index, and for the scaled positions and heights (Corollary X).

*Proof.* Couple the conditional laws as in §3.1: `f = m_r(j_r) + Res_r` with `Res_r -> Res_0` in `C^4` in probability
and `m_r(j_r) -> m_0(j_0)` in `C^4`. Fix a realization with `K_4 < oo` along which the convergence holds; `1{K_4 <= kappa}`
converges because `kappa` is not an atom. By Lemma 3.1 (applicable since `D` is nonsingular and `r -> 0`), every
critical point of `f~` in the physical ball `X^2 + Z^2 + |rW|^2 <= A^2` has bounded `W`, so its `(X, Z)` lies in the disc
of radius `A + O(r)`, and by (3.6) `G_r -> F_0` in `C^2` on the disc of radius `A + 1`. The extra critical points of `F_0`
there are finitely many and nondegenerate, none lies on the circle of radius `A`, and none has height in `{-k, 0}`.
Ordinary `C^1` stability (the inverse function theorem on disjoint small discs around the nondegenerate zeros, plus
`|grad F_0| >= delta > 0` on the complement of those discs in the disc of radius `A + 1`, which `C^1`-closeness
preserves for small `r`) gives: for small `r`, `grad G_r` has exactly one zero in each small disc and none elsewhere;
each converges to the corresponding zero of `grad F_0`, its `G_r`-height converges (so its window membership
stabilizes), its planar Hessian converges (so its index stabilizes), and by the bijection of §3.3 these are exactly
the critical points of `f~` in the physical ball other than the pins, which are the two fixed nondegenerate zeros
`(-+1/2, 0)` of `grad F_0` and attract no other zero. Hence `N^A -> N_oo^A` for every such realization; convergence in
probability follows, together with the convergence of scaled positions and heights. ∎

### 4.3 The large-`K_4` tail

**Lemma 4.2.** For every `A >= 4` and every fixed `p >= 1` there is `C(A, p)` such that for `0 < r <= r_*` and
`kappa >= 1`,

    r^(-3) Q_r^W{ N^A >= 1, K_4 > kappa } <= C(A, p) kappa^(-p/2).                                                   (4.3)

*Proof.* `Q{N^A >= 1, K_4 > kappa} <= E[N^A 1{K_4 > kappa}]`, and the mark `1{K_4 > kappa}` is `sigma(f)`-measurable
and bounded, so [C6] Lemma 5.1 (gradient-only kernel, all heights, which dominates the windowed count) gives

    E[ N^A 1{K_4 > kappa} ] <= sum_j integral_(|X| <= A r) p_(grad f(X) | U_r)(0) E[ W_r F_j(H_X) 1{K_4 > kappa} | U_r, grad f(X) = 0 ] dX.

Cauchy–Schwarz on the conditional expectation and Markov's inequality with the conditional moments
`E[K_4^p | U_r, grad f(X) = 0] <= C_p (1 + beta_X)^p` ([C6] Lemma 4.4 with the frame parameter `beta_X` of [C6] Lemma 4.1)
bound the integrand by `p_(grad f(X)|U_r)(0) E[(W_r F_j(H_X))^2 | .]^(1/2) (C_p (1 + beta_X)^p)^(1/2) kappa^(-p/2)`. By
[C6] (5.3) and the §6 regime table, the density times the square root of the squared-weight conditional moment is
bounded, in every regime, by the source's unmarked intensity bound times `(1 + beta_X)^(3d)`, and the regime's
Gaussian penalty absorbs any fixed power of `(1 + beta_X)` by raising the index of the exponential series in [C6] §6.2
(the recorded ledgers `2n >= 6 + 3d`, `2n/3 >= 15 + 39d` are the choices for [C6]'s degree); the regional ledgers then
give `integral_(|X| <= A r) (...) dX <= C(A, p) r^3`. ∎

### 4.4 Domination on `{K_4 <= kappa}`

**Lemma 4.3.** Fix `A >= 4` and `kappa >= 1`. There is `r_0(kappa, A) > 0` and an integrable function
`g_(A, kappa)(s, U, D, tau)` on `R x O(m) x R^n x R^(dim tau)`, not depending on `r`, such that for `0 < r <= r_0` and all
`(s, U, D, tau)` in the domain of (4.2),

    p_r( chart(rs, U, D, tau) ) c_m prod_(j>=2) |rs - lambda_j| Delta(D)  r^(-4) E[ W_r 1{ N^A >= 1 } 1{K_4 <= kappa} | U_r = v_r, J = chart(rs, U, D, tau) ] <= g_(A, kappa)(s, U, D, tau),

and the same without the indicator `1{N^A >= 1}`, with `integral_(|s| <= S) g < oo` for every `S`.

*Proof.* Write `j = chart(rs, U, D, tau)`. On `{K_4 <= kappa}`, Lemma 3.3 gives `1{N^A >= 1} <= 1{ |s| <= S_*(tau, A, kappa, k) }`,
a deterministic indicator with `S_*` polynomial in `|tau|` by (3.8); note that it does not involve `D`. On the same
event, (3.10) gives `r^(-4) W_r <= C (1 + |s| + |tau| + kappa)^4 (|lambda_2| + r kappa)^2 kappa^(2d - 6)`, again with no
inverse of `D`. The density satisfies `p_r(j) <= C exp(-c |j|^2) <= C exp(-c (|D|^2 + |tau|^2))` (the chart is an
isometry in `T` and `|A| >= |D|`), and the Weyl factor is bounded by `C (1 + |s| r + |D|)^(m(m-1)/2)`. Hence the left
side is at most

    g_(A, kappa) := C(A, kappa) exp(-c (|D|^2 + |tau|^2)) (1 + |D|)^(m(m-1)/2) (1 + |s| + |tau|)^4 (|lambda_2| + 1)^2 1{ |s| <= S_*(tau, A, kappa, k) },

which is integrable: bounded `s`-range for each `tau`, polynomial growth in `|tau|` against the Gaussian factor
(`S_*` is polynomial in `|tau|`), Gaussian decay in `D`, and finite Haar measure in `U`. Without the count indicator,
the bound is the same without `1{|s| <= S_*}`, integrable over bounded `s`. ∎

Remark (nearly singular stable block). Nothing in `g_(A, kappa)` blows up as `lambda_2 -> 0`: the factor
`(|lambda_2| + r kappa)^2` in (3.10) only helps, and the exclusion range `S_*` is independent of `D`. The inverse
`D^(-1)` enters only the pointwise limits of Lemma 4.1, at fixed jets with `D` nonsingular, which is `Q_0`-almost
every jet.

### 4.5 The near limit for fixed `A`

**Proposition 4.4.** For every `A >= 4` and `n in {1, 2}`, uniformly on the compact parameter set,

    r^(-3) Q_r^W{ N^A = n } -> nu_near^A(n) := z_0^(-1) integral m w_0 1{ N_oo^A = n } ds da_3 dbeta dc_3,          (4.4)

and `r^(-3) Q_r^W{ N^A >= 3 } -> 0`. The integrals are finite. Moreover `nu_near^A(n) -> nu_near(n)` as `A -> oo`, and
`integral m w_0 N_oo ds da_3 dbeta dc_3 < oo`.

*Proof.* Fix `kappa` as in Lemma 4.1 and split `1 = 1{K_4 <= kappa} + 1{K_4 > kappa}`. By Lemma 4.2 the second part
contributes at most `C(A, 2) kappa^(-1)` to `r^(-3) Q{N^A = n}` for `n >= 1`, uniformly in `r`. For the first part apply
(4.2) with `E = {N^A = n, K_4 <= kappa}`. The integrand converges pointwise, for `Q_0`-almost every `(s, U, D, tau)`
(`D` nonsingular, jets off `E_k` and off the circle-null set), to
`p_0(chart(0, U, D, tau)) c_m |det D| Delta(D) (det D)^2 w_0(s, a_3, beta) E_0[1{K_4 <= kappa} | j_0] 1{N_oo^A = n}`, by
Lemma 4.1, Lemma 3.5 (convergence in probability of `r^(-4) W_r` together with the uniform `L^2` bound (3.1) gives
convergence of the conditional expectation of its product with a bounded functional) and the continuity of the
density and of the Weyl factor. Lemma 4.3 dominates uniformly in `r`. The prefactor `r^2/Z_r -> z_0^(-1)`. Hence

    limsup / liminf  r^(-3) Q{N^A = n}  are within  C(A, 2)/kappa  of  z_0^(-1) integral (...) E_0[1{K_4 <= kappa} | j_0] 1{N_oo^A = n};

let `kappa -> oo` (monotone convergence on the right, `E_0[1{K_4 <= kappa}|j_0] -> 1`) and integrate out `(U, D, tau_rest)`
to obtain (4.4) with the reduced weight (3.11). For `n >= 3` the pointwise limit of the indicator is zero by Lemma 3.2.
Finiteness: by Fatou's lemma applied along `r -> 0` to the `K_4 <= kappa` part of the representation (4.2) with `E = {N^A >= 1}` (the integrand converges pointwise to the (4.4)-integrand times `E_0[1{K_4 <= kappa}|j_0]`, and the latter tends to `1` as `kappa -> oo`), `integral m w_0 N_oo^A <= liminf r^(-3) E N^A <= C`, where the last bound is [DL] Theorems P_d, C_d and I_d (pin balls and collar of scaled radius `4`, shells from `4r` to `A r`: `C r^3 (1 + sum_j (r/s_j)^2 + s_j^2) <= C r^3` uniformly in `A <= 1/r`). The `A -> oo` limit: `N_oo^A` is nondecreasing in `A` with limit `N_oo`, so
`integral m w_0 N_oo < oo` by monotone convergence and the uniform bound; then `1{N_oo^A = n} -> 1{N_oo = n}`
pointwise (for every jet, once `A` exceeds the radius of the finitely many extra critical points) and is dominated by
`N_oo`, so dominated convergence gives `nu_near^A(n) -> nu_near(n)`. Uniformity on the compact parameter set: the
dominating functions and all limits depend continuously on `(b, k, R)`, so a sequence of parameters along which the
convergence fails would have a convergent subsequence contradicting the pointwise statement at the limit parameter,
exactly as in [LP] §5. ∎

'''
s = s[:start] + new + s[end:]

# adjust references elsewhere
reps = [
 ("q` is the shear parameter of the chart of §3.2\n(so that `|Z|(1+|q|^2)^(1/2)` is the scaled transverse distance); the right side is the weighted law of the scaled\nconfiguration. In `d = 2`, `q` is absent and the scaled transverse coordinate is `|Z|`.",
  "the right side is the weighted law of the scaled configuration, `X` being the scaled axial coordinate\nand `|Z|` the scaled transverse distance (§3.2: the rotated chart makes the physical ball a scaled ball)."),
 ("    r^(-3) E_(Q_r^W) sum_(X in near window critical points) phi( (X . u)/r, |X_perp|/r, (f(X) - b)/r^3 )\n       -> z_0^(-1) integral m w_0 sum_(critical points (X,Z) of F_0 in the window) phi( X, |Z| (1 + |q|^2)^(1/2), F_0(X,Z) ),\n\nwhere the inner sum runs over the extra critical points of `F_0` and ",
  "    r^(-3) E_(Q_r^W) sum_(X in near window critical points) phi( (X . u)/r, |X_perp|/r, (f(X) - b)/r^3 )\n       -> z_0^(-1) integral m w_0 sum_(critical points (X,Z) of F_0 in the window) phi( X, |Z|, F_0(X,Z) ),\n\nwhere the inner sum runs over the extra critical points of `F_0` and "),
 ("by integrating out the stable directions, (3.8)),", "by integrating out the stable directions, (3.11)),"),
  ("For `d = 2` there are no stable directions: `m = 1`, the chart of §3.2 is the identity, `m(s, a_3, beta, c_3)` is the",
  "For `d = 2` there are no stable directions: `m = 1`, the chart of §3.2 is trivial, `m(s, a_3, beta, c_3)` is the"),
 ("| [EDL] | `frontiers/elder_dimension_lift_20260928/PROOF.md` | §2 (complete third-order jets and their positive-definite joint covariance with `U_0`), §3 (the unit-Jacobian shear chart (A8)–(A10)), (A11)–(A16) (the rare set and the conditional `C^4` control), (A17) (the normal form on the sheared soft plane), (A26)–(A28) (the rescaled full gradient and the stable block), §9 (block-triangular stability of the extra critical points). Used as the source of the chart and of the normal form; the lower bound (A4) is not consumed. |",
  "| [EDL] | `frontiers/elder_dimension_lift_20260928/PROOF.md` | §2 (complete third-order jets and their positive-definite joint covariance with `U_0`), (A11)–(A16) (the rare set and the conditional `C^4` control), (A17) (the normal form on the soft plane; here written in a rotated rather than sheared frame), (A26)–(A28) (the rescaled full gradient and the stable block), §9 (block-triangular stability of the extra critical points). Used as the source of the normal form; the shear chart (A8)–(A10) is replaced by the spectral chart of §3.2; the lower bound (A4) is not consumed. |"),
]
for old, new in reps:
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
p.write_text(s)
print('sections 3.2-4 rewritten; total lines', s.count('\n'))