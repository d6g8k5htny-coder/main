# C6 sharp: the torus-wide factorial moments of the window count are `O(r^3)` in every fixed dimension

**Object:** CL-C6-PALM-20260929-v1.
**Version:** v1.4, 29 September 2026 (provenance refresh after Codex review 5358203608: the [DL] identity in §2 and the
status sentences point at the merged bytes on `main`; no mathematical text changed). v1.4: R3a (§6.2) states the axis-safe determinant-weight bound
`W_r |det H_X| / Z_r <= C K^(3d)` valid throughout the strip `|v| <= s^(1/8)`, in place of a bound cited on the axis
alone (OpenAI review 5357899713); the [DL] status sentences updated. v1.3: Lemma 5.1 rewritten with a canonical regression kernel for every `(X, h)`
and the two-measure argument, and the Morse genericity cited from [LP] §8 (OpenAI comment 5896041000). v1.1: clarifications after OpenAI review 5356233690 of §4 (the ratio
`zeta_0 <= eta'/8` in Lemma M' (b), the witness lift convention, `Phi = oo` at a vanishing floor, the finite-union
measurability convention of [C6L] v1.3, Proposition 4.5' and the remark after Lemma 4.3). v1.2: Corollary P restated
as a size-biased statement about the second factorial moment only, and the order of constants in §6, after OpenAI
comment 5895460246. No theorem or ledger change in either.
**Author:** Anthropic Claude (Claude Code session `session_015wNj8LPTKXsaT68G3DgPPh`), 29 September 2026.
**Disposition:** author-side proof candidate. **Nonauthor analytic review is required.** This note stacks on [C6L],
now merged on `main` with the OpenAI lane's complete nonauthor acceptance at its planar scope, and on [DL], now
merged on `main` via Math-#141 with a source-exposed OpenAI nonauthor acceptance of its continuum rows (§2); this note
itself carries OpenAI review 5358116559 (ACCEPT of §§5–7 and Theorem Q at `06bc0d6`), see `README.md`.
**Scientific effect:** NONE. No `STATUS`, `PROOF_INDEX` verdict, `GRAPH` node, claim, `lemma_closed` flag, catalog
entry, prize or source body changes.

## 0. Why this note exists

Catalog entry C6 (`reviews/candidates_pending_20260928/CANDIDATES.md`) asks for the optimal order of the torus-wide
second factorial moment of the window count, "with the pin-neighborhood collisions included". Two things are on the
table today.

- [C6L] (`frontiers/c6_factorial_moment_20260929/PROOF.md`, merged via Math-#140) proves
  `E_(Q_r^W)[N(N-1)] <= C r^3 log^2(1/r)` in `d = 2` through a pathwise Rouché–Bézout cap `Psi` with a uniform exponential tail (its Lemma M), and its §9 names
  the two statements that would remove the logarithm: `(M')`, the tail of `Psi` under the additional conditioning on
  one window witness, and `(H)`, a Hölder insertion into the reviewed first-moment proofs.
- [DL] (`frontiers/d5_dimension_lift_20260929/PROOF.md`, Math-#141) supplies the first moment `E_(Q_r^W) N <= C r^3`
  and each of its regional intensity bounds in every fixed `d >= 2`.

This note proves `(M')` and carries out `(H)`. The result is `E_(Q_r^W)[N(N-1)] <= C r^3` in every fixed `d >= 2`,
hence, with the lower bound (A4) of [EDL], `E_(Q_r^W)[N(N-1)] = Theta(r^3)`: the C6 order is `r^3`, and the
logarithm of [C6L] is removed. The same argument gives `E_(Q_r^W)[(N)_q] <= C_q r^3` for every fixed integer `q >= 2`.

The mechanism is the Palm route of [C6L] §9. Since `N(N-1) <= N Psi` pathwise, the marked Kac–Rice formula turns the
pair count into a first-moment integral whose mark is `Psi`. The conditional law at a witness is a Gaussian
regression on the pin frame plus the witness frame. Every regional first-moment proof already controls that regression
through one frame with an explicit inverse-covariance bound and an explicit Gaussian penalty. The two new facts are:

1. the tail of `Psi` under any such regression law is governed by two numbers, the complex sup of the mean's gradient
   and Hessian and a floor for the residual covariance of the complexified gradient on the test annuli, and both are
   polynomial in the same degeneracy variables that the penalty of that region already absorbs (§4, §5, §6);
2. a witness inside a test annulus forces a zero of the complexified gradient there, and the packing argument of
   [C6L] Lemma M survives the exclusion of a fixed slab of radii around it (§4).

Nothing about the pins, the weight, the normalizer or the Kac–Rice framework changes.

## 1. Setting and results

### 1.1 Model

Fix `d >= 2`, `m = d - 1`, and `L > 0`. `f` is the centered variance-one Gaussian field on `X = R^d/(L Z^d)` with the
exact normalized periodized Gaussian covariance of [LP] §1; [C6L] writes the period as `T`. Fix compact
`B = [b_-, b_+]` and `K = [k_-, k_+]` with `0 < k_-`. Frames `R` range over `O(d)`, `u = R e_1` is the axial direction,
`y in R^m` are the transverse coordinates. For small `r > 0` the pins are

    M = -(r/2) u,  S = (r/2) u,  f(M) = b,  f(S) = b - k r^3,  grad f(M) = grad f(S) = 0.

`Q_r` is Gaussian regression on these `2(d+1)` observations. `F_j(H) = |det H|` if `H` is nonsingular with `j`
negative eigenvalues and `0` otherwise. `W_r = F_d(H_M) F_(d-1)(H_S)`, `Z_r = E_(Q_r) W_r`, `dQ_r^W = (W_r/Z_r) dQ_r`,
`I_r = (b - k r^3, b)`. `N` is the number of critical points of `f` in `X minus {M, S}` with height in `I_r`, all
indices; `(N)_q = N(N-1)...(N-q+1)`. Constants `C, c, r_*` depend on `d, L, B, K` and on the fixed cover parameter
`eta` of §3; never on `r, b, k, R` or the witness position. No constant is evaluated numerically. Nothing is uniform as
`k -> 0`, as marks grow, or as `d` or `L` vary.

`K` also denotes a fixed multiple of `1 + ||f||_(C^6(X))`, as in the sources. The gradient `F = grad f` is extended to
`C^d` (§3); `|.|` is the Hermitian norm on `C^d`.

### 1.2 Statements

**Theorem Q (factorial moments, every fixed dimension).** There are `C_q, r_* > 0` such that for `0 < r <= r_*`, all
`b, k, R` and every fixed integer `q >= 2`,

    E_(Q_r^W)[ (N)_q ] <= C_q r^3.                                                    (1.1)

In particular `E_(Q_r^W)[N(N-1)] <= C r^3`.

**Corollary Theta.** In every fixed `d >= 2`,

    c r^3 <= E_(Q_r^W)[ N(N-1) ] <= C r^3,        c r^3 <= Q_r^W{ N >= 2 } <= C r^3.       (1.2)

The lower bounds are [EDL] (A4). The upper bound on the event is [DL] Corollary M_d, and it also follows from (1.1)
by `1{N >= 2} <= N(N-1)/2`. This settles the order asked for in catalog C6: it is `r^3`, with pin-neighbourhood pairs
included, and the logarithm of [C6L] Theorem C6-L is removed.

**Corollary P (pair intensity has order one relative to the singles).** `E_(Q_r^W)[N(N-1)] / E_(Q_r^W) N` is bounded
above and below by positive constants. Equivalently, under the size-biased (Palm) law `dP^sb = N dQ_r^W / E_(Q_r^W) N`,
the expected number of further window critical points, `E_sb[N - 1] = E_(Q_r^W)[N(N-1)] / E_(Q_r^W) N`, is of order
one. This is a statement about the size-biased law, not about `Q_r^W` conditioned on `N >= 1`. For `q >= 3` Theorem
Q gives only the upper bound `E_(Q_r^W)[(N)_q] <= C_q r^3`; no lower bound of order `r^3` is claimed for `q >= 3`
(it would need higher-cluster lower events, which are not supplied, and it can fail for a count that never exceeds
two).

### 1.3 What the results are not

They are moment bounds under the tilted pinned law with existential constants. They say nothing about the
distribution of the pairs' positions, nothing about elder selection, nothing outside the fixed torus and compact
marks, and nothing about numerical constants. §9 lists the non-claims. The note is conditional on [C6L] and [DL] at
the exact bytes of §2. [C6L] carries the OpenAI lane's complete nonauthor acceptance at its planar scope (its
`d`-dimensional restatement in §3 is this note's own obligation); [DL] has xAI acceptance of its algebraic identities
and, at Math-#141 head `c709854` and the wording-only successors, OpenAI source-exposed nonauthor acceptance of every
continuum row (reviews 5357858391, 5357882570: Theorems P_d, C_d, I_d, G_d), merged on `main` at `e1ca400` with
identical bytes. A defect in either blocks the corresponding step here.

### 1.4 The planar case depends on this note alone

For `d = 2`, every regional bound that §6 takes from [DL] is the planar statement it lifts, and each of those has a
nonauthor verdict on file: the pin ball (P10)–(P12), (P18)–(P20) (xAI 5894512272 and #111), the collar (C4)–(C19)
(`reviews/d5_collar_count_20260928/`), the shells (I9)–(I33) (`reviews/d5_i5_planar_f_20260928/`), the remote
region [RM] Theorem A (main#76), the normalizer [LP] (5.5) (Math-#106), and [C6L] (merged, review 5355682335). The
correspondence row by row is `frontiers/d5_dimension_lift_20260929/CONTINUUM_CROSSWALK.md`. Hence Theorem Q and
Corollary Theta at `d = 2`, that is the planar C6 order `E_(Q_r^W)[N(N-1)] = Theta(r^3)`, are conditional on the
steps of this note only (§§3–7), not on the `d >= 3` read of [DL]. The every-`d` statement additionally needs [DL].

## 2. Sources consumed, with the exact interface

Identities (bytes, sha256, git blob, branch head) are in `SOURCE_MAP.json`.

| Tag | Source | What is consumed |
|---|---|---|
| [C6L] | `frontiers/c6_factorial_moment_20260929/PROOF.md` v1.3 on `main` `bf7c45f` (merged via Math-#140; previously pinned at v1.2, see `SOURCE_MAP.json` for the byte comparison) | The entire extension and moments §4; the cover, `m_j*`, `Psi` and measurability §5; Lemma R (steps 1–4), Lemma M (steps (a)–(d) as a template), Lemma D (the reduction to `G = (Re F, Im F/t)` and its `t = 0` extension), Lemma E; the pathwise cap §2; the Palm route statement §9. All in `d = 2`; §3 below restates them in `d` dimensions with the two exponent changes. |
| [DL] | `frontiers/d5_dimension_lift_20260929/PROOF.md` on `main` `e1ca400` (merged via Math-#141; bytes identical to the reviewed branch head `ea35953`, blob `9d82c707`; every earlier pin and byte comparison in `SOURCE_MAP.json`) | Theorems P_d, C_d, I_d, G_d and the frames, densities, determinant bounds and ledgers of their proofs: §4 (pin ball: `Y`, (4.2)–(4.12)), §5 (collar: `G` of (5.2)–(5.7), the axial strip §5.4, the remainder §5.6), §6 (shells: `Z` of (6.3)–(6.15), the axial strip §6.5), §7 (tiling). The letter `G` of [DL] §5 is written `Gc` here. |
| [LP] | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | The model in every `d`; §2 distinct-site jet rank and positive spectrum; (4.1) uniform `C^q` moments; (5.5) `Z_r >= z_* r^2`. |
| [PP], [CP], [IW] | `reviews/d5_local_collar_20260928/PUNCTURED_PIN_PROOF.md`, `.../COLLAR_PROOF.md`, `frontiers/intermediate_window_20260928/PROOF.md` | The planar templates behind [DL], cited where [DL] cites them; in particular (P18) and (I33) for the weighted Kac–Rice formula with a field mark, and the degree-five frames of [CP] §4 and [IW] §3. |
| [RM] | `frontiers/remote_window_20260924/PROOF.md` | Theorem A in every `d` (reviewed, main#76); §2 uniform remote nondegeneracy of `(U_r, Y_x)`; §3 coupling and uniform moments under the extra witness conditioning (5); §4–§5 the weight and the intensity (12)–(14). |
| [EDL] | `frontiers/elder_dimension_lift_20260928/PROOF.md` | (A4): `Q_r^W(N_r >= 2) >= c r^3` and `E[N_r(N_r-1)] >= c r^3` in every fixed `d`. Used only in Corollary Theta. |

External framework: Armentano, Azaïs and León, arXiv:2304.07424v3, Theorems 2.2 and 7.1 with Remark 8, exactly as
cited in (P18), (I33), [RM] §5 and [DL] §4.7. Standard Gaussian-process facts used with a citation in the text:
the Sudakov–Fernique comparison inequality and the Borell–TIS inequality (Adler–Taylor, *Random Fields and Geometry*,
Theorems 2.2.3 and 2.1.1), and the refined Bézout inequality (Fulton, *Intersection Theory*, Example 12.3.1).

Steps of [C6L] whose proofs are dimension-free are cited and not reproved; the two places where the dimension enters
(Bézout's exponent, and the volume exponents in Lemma D and in the packing) are written out.

## 3. The cap `Psi` in `d` dimensions

### 3.1 Entire extension

By Poisson summation, `f = sum_n sigma_n (xi_n cos(n'.x) + zeta_n sin(n'.x))` with `n' = 2 pi n / L`, i.i.d. standard
`xi, zeta`, and `sigma_n^2 > 0` proportional to `exp(-|n'|^2/2)` ([LP] §2, [C6L] §4). Since
`sum_n sigma_n |n'|^k e^(|n'| Y) < oo` for every `k, Y`, `f` extends almost surely to an entire function on `C^d`,
real on `R^d`, with `f(conj z) = conj f(z)`, and for every `k, p, Y`

    E sup_( |Im z| <= Y ) |partial^k f(z)|^p < oo.                                     (3.1)

Write `F = grad f : C^d -> C^d` for the complexified gradient and `DF` for its complex Jacobian. Every Gaussian
regression of `f` on finitely many bounded linear functionals has an entire mean and an entire residual, because the
regression coefficients `z -> Cov(f(z), Lambda_i) = Lambda_i[K(z - .)]` are entire.

### 3.2 Lemma R_d (Rouché–Bézout cap)

For `c in C^d` and `rho > 0` let `Omega_rho(c) = { z in C^d : |z - c| < rho }`.

**Lemma R_d.** Let `F` be holomorphic near the closure of `Omega_rho(c)`, `c` real, and let `P: C^d -> C^d` be a
polynomial map of degree at most `m`. If `|F - P| < |P|` on `partial Omega_rho(c)`, then `F` has at most `m^d` zeros
in the real ball `B_R(c, rho)`.

*Proof.* Steps 1–4 of [C6L] Lemma R with two changes of exponent. Step 1: `Z(P)` meets the sphere nowhere, so
`Z(P) cap Omega_rho` is a compact analytic subset of an open set in `C^d` and is finite (a compact analytic set of
positive dimension in `C^d` is excluded by the maximum principle for the coordinate functions on its irreducible
components). Step 2: the refined Bézout inequality bounds the isolated zeros of `P`, counted with multiplicity, by the
product of the `d` degrees, at most `m^d`. Step 3: the homotopy `P + t(F - P)` has no zero on the sphere, so the
Brouwer degrees of `P` and `F` on `Omega_rho` agree; `|F| >= |P| - |F - P| > 0` on the sphere makes `Z(F) cap
Omega_rho` finite as in step 1; the local degree of a holomorphic map at an isolated zero is its multiplicity, at
least `1`. Step 4: real zeros in `B_R(c, rho)` are among the complex ones. ∎

### 3.3 The cover, the degrees, and `Psi`

Translate so that the midpoint `o = (M + S)/2` is the origin. Fix `eta in (0, L/20)`.

- `c_0 = o`, `eta_0 = eta`.
- `c_1, ..., c_J` form an `eta/16`-net of `{ |x| >= 7 eta / 8 }` in `X`, with `eta_j = eta/8`.

For any choice of `rho_j in [eta_j, 2 eta_j]` the real balls `B_R(c_j, rho_j)` cover `X`. The complex annuli

    A'_j = { z in C^d : eta_j / 2 <= |z - c_j| <= 3 eta_j }

have real slices at distance at least `eta/2` from `o` (for `j >= 1`, `|c_j| >= 7 eta/8` and `|x - c_j| <= 3 eta / 8`),
hence at distance at least `eta/4` from both pins when `r <= eta/2`. All complex balls used below have radius at most
`4 eta < L/5`, inside the injectivity range of the torus.

Let `T_m^(j)` be the degree-`m` Taylor polynomial of `F` at `c_j`, and

    m_j* = min{ m >= 1 : exists rho in [eta_j, 2 eta_j] with |F - T_m^(j)| < |T_m^(j)| on partial Omega_rho(c_j) },
    Psi = sum_(j=0)^J (m_j*)^d,                                                       (3.2)

with `m_j* = oo` if no degree works. Measurability of `m_j*` is [C6L] §5 in its v1.3 form: success at degree `k` is
not asserted monotone in `k`, `m_j*` is the first successful degree, and

    { m_j* <= m } = union_(k=1)^m  union_( rho in Q cap [eta_j, 2 eta_j] ) { min_(partial Omega_rho(c_j)) ( |T_k^(j)| - |F - T_k^(j)| ) > 0 },

a finite union over degrees of countable unions over rational radii of measurable events (the set of successful
radii at a fixed degree is relatively open by continuity of the sphere minimum in `rho`, so it is nonempty iff it
contains a rational radius). The tail argument of §4 uses only that `{m_j* > m}` implies failure at degree `m`.
By Lemma R_d applied to each ball `B_R(c_j, rho_j)` with an admissible `rho_j`,

    Psi >= #{ critical points of f on X } >= N + 2,                                   (3.3)

pathwise, the pins being critical points. Hence, for every integer `q >= 2`,

    N(N-1) <= N Psi,       (N)_q <= N Psi^(q-1).                                        (3.4)

### 3.4 Lemma E_d

**Lemma E_d.** Let `w_1, ..., w_K in C^d` be pairwise distinct modulo `L Z^d` and `P_1, ..., P_K` polynomials. If
`sum_k P_k(n) e^(2 pi i n . w_k / L) = 0` for all `n in Z^d`, then every `P_k = 0`.

*Proof.* [C6L] Lemma E with `Z^2` replaced by `Z^d` and the two shifts by the `d` coordinate shifts; the joint
generalized eigenspaces of the commuting shifts for distinct eigenvalue `d`-tuples are independent on the finite
shift-invariant span of `{ n^alpha chi_k(n) }`. ∎

Consequently ([RM] §2, [LP] §2, [C6L] Lemma D): any finite family of derivative functionals of `f`, at finitely many
sites of `C^d` that are pairwise distinct modulo `L Z^d`, and with independent symbols at each site, has positive
definite covariance; the covariance of a real-linear combination is `sum_n sigma_n^2 |ell(e^(i n'.))|^2`.

## 4. The parametric Rouché tail

### 4.1 Regression laws and their two invariants

Let `Lambda = (Lambda_1, ..., Lambda_l)` be a finite frame of bounded linear functionals of the field (values or
derivatives at finitely many points, or finite linear combinations of them) whose covariance under `Q_r` is
invertible, and let `tau in R^l`. Write

    Q' = Q_r( . | Lambda = tau )

for the Gaussian regression law, `m = E_(Q') f` for its mean (an entire function), and `g = f - m` for its residual
under `Q'` (a centered Gaussian field whose covariance is the `Q_r`-covariance minus the regression term). Every
conditional law used in a first-moment proof is of this form, with `Lambda` the frame of that proof.

Two numbers describe `Q'` for the purposes of this section. Fix a real point `X in X` (the witness) and
`zeta_0 = eta/64`, so that `zeta_0 <= eta_j / 8` for every `j` (the smallest radius is `eta_j = eta/8`; for the ball
around the pins `zeta_0 = eta_0/64`).

**Lifts of the witness.** Each ball `Omega_(4 eta_j)(c_j)` is read in the chart of `C^d` centred at a lift of
`c_j`; since `8 eta_j < 2L/5 < L`, at most one lift `X~_j in R^d` of `X` satisfies `|X~_j - c_j| <= 4 eta_j`. When
such a lift exists, `rho_X = |X~_j - c_j|` and `|z - X|` means `|z - X~_j|` for `z` in that chart; when none exists,
no radius is excluded in Lemma M' (b) and the floor (4.2) is taken over all of `A'_j`, because every lift of `X` is
then at distance more than `eta_j` from the annulus. Distances to the pins are read in the same charts; the cover
geometry of §3.3 keeps the real slices of the annuli at distance at least `eta/4` from every lift of `M` and `S`.

    mu(Q') = max_j sup_( z in Omega_(4 eta_j)(c_j) ) ( |grad m(z)| + ||D^2 m(z)|| ),                    (4.1)

    phi(Q'; X) = inf { det Cov_(Q')( G(z) ) : z in A'_j for some j, |z - X| >= zeta_0 / 2 },          (4.2)

where, for `z = x + i t u` with `u` a real unit vector and `t = |Im z| > 0`,

    G(z) = ( Re F(z), Im F(z) / t ) in R^(2d),

extended continuously to `t = 0` by `G = (grad f(x), D^2 f(x) u)` ([C6L] Lemma D). `Cov_(Q')(G(z))` is the residual
covariance of `G(z)`, which does not depend on `tau`. Put `Phi(Q'; X) = max(1, phi^(-1/2))`, with `Phi = oo` when
`phi = 0`; in that case every bound of this section that carries `Phi` is vacuous, and it is the applications in §6
that prove `phi > 0` with an explicit floor in each regime. Nothing here assumes `phi > 0` from the words "regression
law".

### 4.2 Three lemmas about regression laws

**Lemma 4.1 (mean).** For every `Y > 0` and multi-index `alpha` there is `C_(Y, alpha)`, depending only on the prior
field, such that for `|Im z| <= Y`

    |partial^alpha m(z)| <= C_(Y, alpha) ( 1 + beta ),
    beta := ( sum_i Var_(Q_r) Lambda_i )^(1/2) . ||Cov_(Q_r)(Lambda)^(-1)|| . |tau - E_(Q_r) Lambda|.        (4.3)

*Proof.* Regression is transitive: `m(z) = E_(Q_r) f(z) + Cov_(Q_r)(f(z), Lambda) Cov_(Q_r)(Lambda)^(-1)
(tau - E_(Q_r) Lambda)`, and the same identity holds for `partial^alpha` in place of `f`, real and imaginary parts
separately. The first term: `E_(Q_r) partial^alpha f(z) = Cov(partial^alpha f(z), E_r) Cov(E_r)^(-1) e` with
`Cov(E_r)^(-1)` and `e` bounded ([PP] §3, [DL] §4.1) and, by Cauchy–Schwarz,
`|Cov(Re partial^alpha f(z), E_(r,i))| <= Var(Re partial^alpha f(z))^(1/2) Var(E_(r,i))^(1/2) <= C_(Y,alpha)`, since
`Var(Re partial^alpha f(z)) <= E|partial^alpha f(z)|^2 <= sum_n sigma_n^2 |n'|^(2|alpha|) e^(2|n'|Y)` and the frame
variances are bounded. The second term: each entry of `Cov_(Q_r)(partial^alpha f(z), Lambda)` is bounded by
`C_(Y,alpha) Var_(Q_r)(Lambda_i)^(1/2)` by Cauchy–Schwarz again, residual variances being at most prior ones. ∎

**Lemma 4.2 (residual floor).** For every `z`,

    Cov_(Q')( G(z) ) >= lambda_min( Cov_(Q_r)( (G(z), Lambda) ) ) I_(2d).                             (4.4)

*Proof.* The residual covariance of `G(z)` given `Lambda` is the Schur complement of the block `Cov_(Q_r)(Lambda)` in
the joint covariance; for a symmetric matrix `N >= c I` and any partition, `min_y (x, y)^T N (x, y) = x^T (Schur) x
>= c |x|^2`. ∎

**Lemma 4.3 (residual sup moments).** For every compact complex domain `D` inside `{ |Im z| <= Y }`, every `p` and
every `|alpha| <= 6`,

    E_(Q') sup_D |partial^alpha g(z)|^p <= C_(p, D, alpha),                                            (4.5)

with `C` depending only on the prior field.

*Proof.* Fix a real component `X'(z) = Re partial^alpha g(z)` (or `Im`) and let `X(z)` be the same component of the
prior field. Since `g` is the regression residual of the prior on the total frame `(E_r, Lambda)` with standardized
coefficient vector `C(z)`, `E(X'(z) - X'(z'))^2 = E(X(z) - X(z'))^2 - |C(z) - C(z')|^2 <= E(X(z) - X(z'))^2` and
`Var X'(z) <= Var X(z)`. Sudakov–Fernique gives `E sup_D X' <= E sup_D X` and `E sup_D (-X') <= E sup_D (-X)`;
Borell–TIS turns the mean of the supremum and the maximal variance into all moments. `E sup_D |X| < oo` is (3.1). ∎

*Remark (a proof without comparison).* Each Fourier coordinate of the residual `g` after any finite Gaussian
regression is centered with variance at most its prior variance `sigma_n^2`, so its `L^p` norm is at most `C_p
sigma_n`; summing `sigma_n |n'|^k e^(|n'| Y)` by Minkowski, as in (3.1), bounds every derivative supremum on the tube
uniformly in the regressing frame, without independence between the residual coordinates and without any inverse
Gram bound (Math-#145 review 5356233690, item 1). Either proof gives (4.5).

**Lemma 4.4 (conditional moments of `K`).** `E_(Q')[ K^n ] <= C_n ( 1 + beta )^n` for every `n`, with `beta` from
(4.3). *Proof.* `K <= C(1 + ||m||_(C^6(X)) + ||g||_(C^6(X)))`; Lemma 4.1 with `Y = 0` bounds the first norm by
`C(1 + beta)`, Lemma 4.3 the moments of the second. ∎

### 4.3 Lemma D' (density on the annuli, forced zeros excluded)

**Lemma D'.** For `z in A'_j` with `|z - X| >= zeta_0 / 2` and `t = |Im z|`,

    Q'( |F(z)| <= 2 epsilon ) <= min( 1,  C_d Phi(Q'; X) epsilon^(2d) t^(-d) ),                       (4.6)

and for every `j`,

    integral_( {z in A'_j : |z - X| >= zeta_0/2} ) Q'( |F(z)| <= 2 epsilon ) d^(2d) z
        <= C_d Phi(Q'; X) epsilon^(2d) ( 1 + log(1/epsilon) ).                                       (4.7)

*Proof.* Under `Q'`, `G(z)` is Gaussian with covariance `Cov_(Q')(G(z))` of determinant at least `phi`, so its density
is at most `(2 pi)^(-d) phi^(-1/2)`. The event forces `|Re F| <= 2 epsilon` and `|Im F|/t <= 2 epsilon / t`, a set of
`R^(2d)`-volume at most `(4 epsilon)^d (4 epsilon / t)^d`. This is (4.6). For (4.7), with `A = C_d Phi epsilon^(2d)`,
integrate `min(1, A |y|^(-d))` over `y = Im z in R^d`, `|y| <= 3 eta_j`, and over `Re z` in a set of bounded volume:
the region `|y| <= A^(1/d)` contributes at most `omega_d A`, and the region `A^(1/d) <= |y| <= 3 eta_j` at most
`A |S^(d-1)| log( (3 eta_j)^d / A ) / d <= C A (1 + log(1/epsilon))`, because `A >= c epsilon^(2d)`; if
`A^(1/d) > 3 eta_j` the whole integral is at most the volume, which is at most `C A`. ∎

The radial exponent `(d - 1) - d = -1` is the same in every `d`: the single logarithm of [C6L] Lemma M (c) is not a
planar accident. Nothing else in Lemma D' depends on `d` beyond the volume exponents.

### 4.4 Lemma M' (parametric tail)

**Lemma M'.** There are `m_0`, depending only on `eta`, and `C`, depending only on `d`, `eta`, `L` and the prior
field, such that for every regression law `Q'` as in §4.1, every witness point `X`, every `j` and every `m >= m_0`,

    Q'( m_j* > m ) <= C ( 1 + mu(Q')^2 + Phi(Q'; X) ) . m . 2^(-m/(2d)).                              (4.8)

*Proof.* Fix `j`, write `c = c_j`, `eta' = eta_j`, `S = sup_(Omega_(4 eta')(c)) |F|`, `Lip = sup_(Omega_(3 eta')(c))
||DF||`, all under `Q'` (so `F = grad m + grad g`).

**(a) Remainder.** [C6L] (6.1) is pathwise and dimension-free: on `Omega_(2 eta')(c)`, `|F - T_m| <= S 2^(-m)`.

**(b) Failure means near-zeros on every admissible sphere, with a slab excluded.** Put

    lambda = 2^(m/(4d)),   epsilon = 2 lambda 2^(-m),   delta = epsilon / lambda = 2^(1-m),

and take `m_0` so large that `delta <= min( zeta_0 / 2, eta' / 8 )` for `m >= m_0`. If `m_j* > m` then for every
`rho in [eta', 2 eta']` some `z_rho in partial Omega_rho(c)` has `|T_m(z_rho)| <= |F - T_m|(z_rho) <= S 2^(-m)`, hence
`|F(z_rho)| <= 2 S 2^(-m)`. On the event `{ S <= lambda, Lip <= lambda }`, `|F| <= 2 epsilon` on each ball
`B(z_rho, delta)`.

Let `rho_X = |X~_j - c|` for the lift of §4.1 (if there is none, discard nothing). Among the radii
`rho_i = eta' + 2 i delta`, `0 <= i <= floor(eta'/(2 delta))`, discard those with `|rho_i - rho_X| < zeta_0`. At
most `zeta_0 / delta + 1` are discarded, so at least `eta'/(2 delta) - zeta_0/delta - 1` remain, and this is at
least `eta'/(4 delta)` exactly when `eta'/4 - zeta_0 >= delta`, which holds because `zeta_0 <= eta'/8` (§4.1) and
`delta <= eta'/8` give `eta'/4 - zeta_0 >= eta'/8 >= delta`. (The weaker `zeta_0 <= eta'/4` would not suffice.) For a
retained radius, every `z in B(z_(rho_i), delta)` satisfies `| |z - c| - rho_X | >= zeta_0 - delta >= zeta_0/2`, hence
`|z - X| >= zeta_0/2`; and `eta'/2 <= rho_i - delta <= |z - c| <= rho_i + delta <= 3 eta'`, so the ball lies in `A'_j`.
The retained balls are pairwise disjoint (centres on spheres `2 delta` apart). Therefore the volume

    V := Leb_(2d) { z in A'_j : |z - X| >= zeta_0/2, |F(z)| <= 2 epsilon }
       >= ( eta' / (4 delta) ) omega_(2d) delta^(2d) = ( omega_(2d) / 4 ) eta' delta^(2d-1).           (4.9)

**(c) Markov.** By (4.7), `E_(Q') V <= C_d Phi epsilon^(2d) (1 + log(1/epsilon))`. Hence

    Q'( m_j* > m, S <= lambda, Lip <= lambda )
        <= 4 E V / ( omega_(2d) eta' delta^(2d-1) )
        <= C Phi lambda^(2d-1) epsilon ( 1 + log(1/epsilon) ) / eta'
        =  C Phi lambda^(2d) 2^(1-m) ( 1 + log(1/epsilon) ) / eta'
        <= C Phi m 2^(-m/2),

since `lambda^(2d) = 2^(m/2)` and `log(1/epsilon) <= m log 2`.

**(d) Truncation.** `S <= mu + sup |grad g|` and `Lip <= mu + sup ||D^2 g||` on the relevant complex balls, so by
Lemma 4.3 with `p = 2`, `E_(Q') S^2 + E_(Q') Lip^2 <= C (1 + mu^2)`, and Markov gives
`Q'(S > lambda) + Q'(Lip > lambda) <= C (1 + mu^2) 2^(-m/(2d))`.

Adding (c) and (d), and using `2^(-m/2) <= 2^(-m/(2d))` and `m >= 1`, gives (4.8). ∎

For `d = 2` and the unconditioned `Q_r` (empty witness frame, `mu` and `Phi` bounded, `lambda = 2^(m/8)`), this is
[C6L] Lemma M with the slab exclusion, which costs a factor `2` in (4.9).

### 4.5 Moments of `Psi` under a regression law

**Proposition 4.5.** For every `p >= 1` there is `C_p`, depending only on `p, d, eta, L` and the prior field, such that
for every regression law `Q'` as in §4.1 and every witness point `X`,

    E_(Q')[ Psi^p ] <= C_p ( 1 + mu(Q')^2 + Phi(Q'; X) ).                                              (4.10)

*Proof.* `E[(m_j*)^(pd)] = sum_(m >= 0) ((m+1)^(pd) - m^(pd)) Q'(m_j* > m) <= m_0^(pd) + sum_(m >= m_0) pd (m+1)^(pd-1)
Q'(m_j* > m)`, and by (4.8) the series is at most `C (1 + mu^2 + Phi) sum_m pd (m+1)^(pd-1) m 2^(-m/(2d)) < oo`. Then
`Psi^p <= (J+1)^(p-1) sum_j (m_j*)^(pd)`. The finite control `TL` verifies that the summand ratio falls below
`2^(-1/(4d))` from an explicit index on, for `d = 2, ..., 6` and `p <= 4`. ∎

**Proposition 4.5' (logarithmic form).** With `A = 1 + mu(Q')^2 + Phi(Q'; X)` finite, for every `p >= 1`,

    E_(Q')[ Psi^p ] <= C'_p ( 1 + log A )^(pd).                                                         (4.11)

*Proof.* (4.8) and `Q' <= 1` give `Q'(m_j* > m) <= min(1, C A m 2^(-m/(2d)))`. Choose `M_0 = C_0 (1 + log A)` with
`C_0 = C_0(d, C)` so large that `C A m 2^(-m/(4d)) <= 1` for all `m >= M_0`; then `Q'(m_j* > m) <= 2^(-m/(4d))` for
`m >= M_0`. Split the sum of the proof of Proposition 4.5 at `M_0`: the part below `M_0` is at most `M_0^(pd)`, the
part above is at most `sum_(m >= M_0) pd (m+1)^(pd-1) 2^(-m/(4d)) <= C_p`. ∎

This form was pointed out by the OpenAI lane (Math-#145 comment 5895432912). It follows from the same premises as
Proposition 4.5 and turns the second factor of (5.3) into `C_q (1 + log(1 + mu^2 + Phi))^(d(q-1))`. The proof of
Theorem Q below uses only the polynomial form (4.10); the logarithmic form reduces the losses recorded in §6 but is
not needed for `O(r^3)`, since every regime's penalty already absorbs the polynomial factor.

## 5. Kac–Rice with a global mark, and the Hölder insertion

### 5.1 Marked formula

For a compact `B` in `X minus {M, S}`, a Borel set of heights `I`, an index `j`, and a nonnegative `sigma(f)`-measurable
random variable `Xi`, define the marked window count `N_j^Xi(B, I) = sum_X Xi 1{f(X) in I} 1{H_X in O_j}`, the sum over
critical points `X in B`, with `O_j` the open set of nonsingular symmetric matrices of index `j`.

**Lemma 5.1.** Under `Q_r`, for every such `B, I, j` and `Xi`,

    E_(Q_r) N_j^Xi(B, I) = integral_B integral_I p_( grad f(X), f(X) | pins )(0, h)
                              E_(Q_r)[ Xi F_j(H_X) | grad f(X) = 0, f(X) = h ] dh dX,                  (5.1)

both sides possibly `+oo`. The same holds without the height variable (all heights) with
`p_( grad f(X) | pins )(0) E[ Xi F_j(H_X) | grad f(X) = 0 ]`.

**Kernel convention.** The conditional expectation in (5.1) is taken with respect to the canonical Gaussian regression
kernel: for every `X in B` and every `h in R`, `Q_(X,h)` is the law of `f` given `(grad f(X), f(X)) = (0, h)` defined by
the explicit regression formulas (mean `E_(Q_r) f + Cov_(Q_r)(f, (grad f(X), f(X))) Sigma_X^(-1) ((0,h) - E_(Q_r)(grad
f(X), f(X)))`, residual covariance the Schur complement), which exist for every `(X, h)` because `Sigma_X` is
nondegenerate on the compact `B` off the pins ([LP] §2) and depend continuously on `(X, h)`. `Q_X` denotes the same
with the gradient alone. These kernels are fixed once; no almost-everywhere version of a conditional expectation is
evaluated at the single value `grad f(X) = 0`.

*Proof.* Fix `r`, a compact `B` off the pins, `I` and `j`. Define two measures on the field space (the Borel
`sigma`-field of `C(X)`):

    Lc(A) = E_(Q_r)[ 1_A(f) N_j(B, I) ],
    Rc(A) = integral_B integral_I p_( grad f(X), f(X) | pins )(0, h) E_(Q_(X,h))[ 1_A(f) F_j(H_X) ] dh dX.

Both are finite: `Lc(C(X)) = E_(Q_r) N_j(B, I) < oo` by the unweighted Kac–Rice formula on `B` ([LP] §9, [RM] §5,
(P18), (I33)), whose right side is `Rc(C(X))`; so the two total masses agree. For a bounded continuous function `g`
of finitely many jets of `f` at finitely many fixed sites, the weighted formula of the sources, with the auxiliary
Gaussian field of arXiv:2304.07424v3 Theorem 7.1 and Remark 8 taken constant in `X` and its conditional law read
through the kernel `Q_(X,h)`, gives `integral g dLc = integral g dRc`. These `g` form a vector space closed under
products which generates the Borel `sigma`-field of `C(X)` (a continuous field is determined by its values on a
countable dense set). Two finite measures whose integrals agree on such a class agree on the generated `sigma`-field
(monotone class theorem for functions: the bounded measurable `g` with `integral g dLc = integral g dRc` form a
vector space closed under bounded monotone limits containing the multiplicative class). Hence `Lc = Rc` on all Borel
`A`, which is (5.1) for indicators; simple functions by linearity, and nonnegative `Xi` by monotone convergence on
both sides (`Xi ∧ n` increasing to `Xi`). The all-height statement is the same construction with the kernels `Q_X`
and the unweighted gradient formula; the extension from compact `B` to `X minus {M, S}` is monotone convergence
along an exhaustion by compact punctures. ∎

Almost surely under `Q_r`, `f` is Morse with distinct critical values, hence every critical point off the pins is
nondegenerate: this is [LP] §8 ("Pinned genericity and global event measurability"), whose countable-exhaustion
argument for the pinned law transfers to `Q_r^W` by absolute continuity and `0 < Z_r < oo`. No density statement for
the nonlinear functional `det D^2 f` is used or needed. Hence `N = sum_j N_j(X minus {M,S}, I_r)` almost surely.

### 5.2 The Hölder insertion

Apply Lemma 5.1 with `Xi = W_r Psi^(q-1)` and, by (3.4),

    E_(Q_r^W)[ (N)_q ] <= Z_r^(-1) E_(Q_r)[ W_r N Psi^(q-1) ]
                      = Z_r^(-1) sum_j E_(Q_r) N_j^( W_r Psi^(q-1) )( X minus {M,S}, I_r ).           (5.2)

Inside (5.1), for the conditional law `Q' = Q_r( . | grad f(X) = 0, f(X) = h )` (or without the height in the
all-height regions), conditional Hölder gives

    E_(Q')[ W_r F_j(H_X) Psi^(q-1) ] <= E_(Q')[ (W_r F_j(H_X))^2 ]^(1/2) E_(Q')[ Psi^(2(q-1)) ]^(1/2).      (5.3)

The first factor is controlled exactly as the unmarked expectation in the source proofs: each of them bounds
`W_r F_j(H_X) <= |det H_M det H_S det H_X|` pathwise by a deterministic geometric factor times `K^(3d)`, so
`E_(Q')[(W_r F_j)^2]^(1/2)` is at most the same geometric factor times `E_(Q')[K^(6d)]^(1/2) <= C (1 + beta)^(3d)`
by Lemma 4.4 (the source's own bound is `C(1 + beta)^(3d)` for the unmarked `E_(Q')[K^(3d)]`, so the form is
unchanged). The second factor is at most `C_q (1 + mu + Phi^(1/2))` by Proposition 4.5. So the marked intensity in
each region is the unmarked intensity bound of the source with two extra polynomial factors in `beta`, `mu` and `Phi`,
which §6 computes region by region and the region's Gaussian penalty absorbs.

## 6. The regimes

The tiling is [DL] §7: the two punctured pin balls of scaled radius `1/4` (Theorem P_d, all heights), the compact
collar `C(1/4, 4)` in midpoint scaled coordinates (Theorem C_d, all heights), the dyadic shells `4 r <= |X| <= s_0`
(Theorem I_d, window), and the remote region `|X| >= s_0` ([RM] Theorem A, window).

**Order of constants.** `eta` is fixed first (§3.3), which fixes the cover, the annuli `A'_j`, `zeta_0 = eta/64` and
the joint-floor constant `c_J` of Lemma 6.1. The collar radius `R = 4` and the shell cutoff `s_0` are fixed next, with
`s_0 <= eta/4`, so that every witness of the local regimes lies inside the core of the ball around the pins and every
Taylor scale `sigma in {r, s}` of Lemma 6.3 is at most `s_0`. Lemma 6.3 then reduces `s_0`, and Lemma 6.2 and the
source proofs reduce `r_*`, until the remainder conditions (`C sigma^6 <= (1/2)(c sigma^10)^(1/2)` in Lemma 6.3, `r <=
c|v|^2` and `s <= c|v|^3` in Lemma 6.2) hold; these reductions depend on `d, L, B, K, eta` only, never on `r`, `s`, the
witness or the height. This is the same order as in [DL] §7 and [IW] §10, with `eta` added at the front. In the
remote regime the exclusion of the witness is read in the local periodic lifts of §4.1. In each region the source proof
fixes one frame `Lambda` and one target `tau`, bounds `||Cov_(Q_r)(Lambda)^(-1)||`, `Var_(Q_r) Lambda_i` and
`|tau - E_(Q_r) Lambda|`, and derives the density and the conditional moments from them. The table records, for each
regime, the parameter `beta` of (4.3) and a floor `lambda_J` for `lambda_min Cov_(Q_r)((G(z), Lambda))` over the
`z` of (4.2); then `mu <= C(1 + beta)` by Lemma 4.1 and `Phi <= max(1, lambda_J^(-d))` by Lemma 4.2.

| Regime | Frame `Lambda` (source) | `beta <=` | `lambda_J >=` | Penalty in the source |
|---|---|---|---|---|
| R1 pin ball, `X = M + r(p,q)`, all heights | `Y` of [DL] (4.2)–(4.5) | `C (1 + chi)`, `chi = |p|/Delta` | `c` | `e^(-c chi^2)` (region I); `e^(-c/(2r^2))` (region II) |
| R2a collar, `|v| <= r^(1/3)` | raw `grad f(X)`, [DL] §5.4 | `C r^(-10)` | `c r^10` | `exp[-c r^(-2/3)]` |
| R2b collar, `r^(1/3) <= |v| <= v_0` | `Gc` of [DL] (5.2)–(5.3) | `C |v|^(-4)` | `c |v|^4` | `exp[-c/|v|^2]` |
| R2c collar, `|v| >= v_0` | `Gc`, fixed floor, [DL] §5.6 | `C` | `c` | none needed |
| R3a shell, `|v| <= s^(1/8)` | raw `(grad f(X), f(X))`, [DL] §6.5 | `C s^(-10)` | `c s^10` | `exp[-c s^(-1/4)]` |
| R3b shell, `s^(1/8) <= |v| <= 2` | `Z` of [DL] (6.4), (6.11) | `C |v|^(-6)` | `c |v|^6` | `exp(-c/|v|^2)` (or a constant for `|v| >= v_0`) |
| R4 remote, `|X| >= s_0` | `Y_x = (grad f(x), f(x))`, [RM] §2–§3 | `C` | `c` | none needed |

### 6.1 The joint floors: a common lemma

Let `Jo` denote the vector of independent derivative monomials of `f` at `o` of orders `0` to `5` that are not
contact functionals, i.e. all monomials except `1, x, x^2, x^3, y_i, x y_i` ([IW] §3.1, [DL] §6.1); `Jo` contains the
jets `A_4, T_3, S_0, C_3, D_3` of [DL]. Let `J5` be the full vector of monomials of order `0` to `5` at `o`.

**Lemma 6.1.** There is `c_J > 0` such that, for all `z in A'_j` (every `j`), all `t in [0, 3 eta_j]`, all frames and
`r <= r_*`,

    Cov_(Q_r)( (G(z), Jo) ) >= c_J I,      Cov( (G(z), J5) ) >= c_J I  (prior law).                   (6.1)

The same holds with `Jo` taken at `M` instead of `o`.

*Proof.* At `t > 0` the functionals of `G(z)` are evaluations of `grad f` at the two non-real points `z, conj z`,
distinct from each other and from every real point modulo `L Z^d`; at `t = 0` they are `(grad f(x), D^2 f(x) u)` at
the real point `x` with `|x - o| >= eta/2`, whose symbols `i a.n' - (u.n')(b.n')` vanish identically only for
`a = b = 0`. The monomials at `o` (or `M`) have independent symbols. By Lemma E_d the prior covariance of
`(G(z), J5)` is positive definite at every parameter, and by continuity and compactness of `(z, u, t, frame)` it has
a uniform floor. Under `Q_r` the covariance of `(G(z), Jo)` is the Schur complement of `Cov(E_r)` in
`Cov((G(z), Jo, E_r))`; at `r = 0` the frame `E_r` is replaced by `E_0`, whose monomials are exactly the ones excluded
from `Jo`, so `(G(z), Jo, E_0)` is a family of the type just treated and its covariance is positive definite; `E_r ->
E_0` in `L^2` uniformly ([PP] §3, [DL] §4.1), so the Schur complement is continuous on `r in [0, r_*]` and compactness
gives the floor. ∎

**Lemma 6.2 (frames written on jets).** Suppose `Lambda = a + B Jo + Rm` with `a` deterministic under `Q_r`, `B` a
matrix with `sigma_min(B)^2 >= lambda_B > 0`, and `||Rm||_(L^2(Q_r)) <= rho_R`. If `rho_R <= (1/2) ( c_J min(1,
lambda_B) )^(1/2)`, then

    lambda_min Cov_(Q_r)( (G(z), Lambda) ) >= (1/4) c_J min(1, lambda_B).                             (6.2)

*Proof.* For unit `(alpha, gamma)`, `Var(alpha.G + gamma.Lambda)^(1/2) >= Var(alpha.G + gamma.B Jo)^(1/2) -
|gamma| rho_R`, and `Var(alpha.G + gamma.B Jo) = (alpha, B^T gamma)^T Cov((G, Jo)) (alpha, B^T gamma) >= c_J (|alpha|^2
+ |B^T gamma|^2) >= c_J min(1, lambda_B)`. ∎

**Lemma 6.3 (raw frames at a coalescing witness).** Let `V` be the degree-five observation vector of [CP] §4 (three
sites at scale `r`: `V_r = (f(ra), r grad f(ra), f(rb), r grad f(rb), f(rc), r grad f(rc))`) or of [IW] §3 / [DL] §6.1
(pins at scale `r`, witness at scale `s`: `V_(r,s,w) = (S_s U_r, f(sw), s grad f(sw))`), with scale `sigma = r` resp.
`sigma = s`. Then, under the prior law, `Cov((G(z), V)) >= c sigma^10 I`, and consequently, under `Q_r`,
`Cov_(Q_r)((G(z), grad f(X)))` resp. `Cov_(Q_r)((G(z), grad f(X), f(X)))` is at least `c sigma^10 I`.

*Proof.* The sources write `V = Bc D_sigma J5 + Rm` with `Bc` of uniformly positive smallest row singular value,
`D_sigma = diag(sigma^|alpha|)`, and `||Rm||_(L^p) <= C sigma^6`. For unit `(alpha, gamma)`, by (6.1),
`Var(alpha.G + gamma.Bc D_sigma J5) >= c_J (|alpha|^2 + |D_sigma Bc^T gamma|^2) >= c_J (|alpha|^2 + sigma^10
c' |gamma|^2) >= c sigma^10`, and the remainder `C sigma^6 |gamma|` is at most half of `(c sigma^10)^(1/2)` for
small `sigma`. Under `Q_r`, take the Schur complement in the pin rows of `V` (a principal submatrix keeps the floor, a
Schur complement of a matrix `>= c I` is `>= c I`), then unscale the gradient rows by `sigma^(-1) >= 1`. ∎

### 6.2 The regimes one by one

**R1 (pin ball).** `Lambda = Y`, `tau = 0`. [DL] (4.5): `c_0 I <= Cov_(Q_r)(Y) <= C_0 I`; mean `dvec + O(1)` with
`|dvec| <= C chi`; so `beta <= C(1 + chi)`. Joint floor: `Y = dvec + B J_M + R` with `det(B B^T) >= c_d` ([DL] (4.4)),
bounded entries, hence `sigma_min(B)^2 >= c'_d`, and `||R||_(L^2) <= C r`; Lemma 6.2 gives `lambda_J >= c` for small
`r`. The witness is inside the core of `A'_0` (`|X| <= r`), so the slab exclusion is vacuous.

Marked intensity. From [DL] (4.8): `(W_r F_j)^2 <= K^(6d) r^12 |q|^4 (1 + p^2/|q|^2)^6`; by Lemma 4.4,
`E_(Q')[K^(6d)]^(1/2) <= C(1 + chi^(3d))`; by Proposition 4.5, `E_(Q')[Psi^(2(q-1))]^(1/2) <= C_q (1 + chi)`. Region I
(`|q| >= r|p|`, `t = |p|/|q|`, `chi^2 in [t^2/2, t^2]`):

    rho_j^(W, Psi)(X) <= C r^(3-d) |q|^(2-d) sup_t (1 + t^2)^3 (1 + t^(3d)) (1 + t) e^(-c t^2/2) <= C' r^(3-d) |q|^(2-d).

Region II (`|q| < r|p|`, `chi <= 1/r`), from [DL] (4.9) and (4.12) with the extra factor `(1 + chi) <= 2/r`:

    rho_j^(W, Psi)(X) <= C r^(-3-4d) e^(-c/(2r^2)) (r|p|)^(2-d) <= C r^(3-d) (r|p|)^(2-d),

by the `n`-th term of the exponential series with `2n >= 6 + 3d`. Integration as in [DL] §4.8 with Lemma D5 there:

    E_(Q_r) N_j^(W_r Psi^(q-1))( M + rD, all heights ) / Z_r <= C r^3.                                (6.3)

The `S`-centred ball is identical ([DL] §4.9).

**R2a (collar, `|v| <= r^(1/3)`).** `Lambda = grad f(X)` raw, `tau = 0`; [DL] (5.1): `Sigma_X >= c r^10 I`, variances
bounded, mean bounded; `beta <= C r^(-10)`; Lemma 6.3 with `sigma = r`: `lambda_J >= c r^10`, so `Phi <= C r^(-10d)`.
[DL] §5.4 bounds the density by `C r^(-5d) exp[-c r^(-2/3)]` and `|det H_M det H_S det H_X| <= K^(3d)`. Marked:

    rho_j^(W, Psi)(X) <= C r^(-2) r^(-5d) exp[-c r^(-2/3)] . (1 + r^(-10))^(3d) . (r^(-10) + r^(-5d))
                      <= C r^(-12-40d) exp[-c r^(-2/3)] <= C r^(3-d),

by the `n`-th term of the exponential series with `2n/3 >= 15 + 39d`.

**R2b (collar, `r^(1/3) <= |v| <= v_0`).** `Lambda = Gc`, `tau = 0`; [DL] (5.3): `c|v|^4 I <= Cov_(Q_r)(Gc) <= C|v|^2
I`, mean bounded; `beta <= C |v|^(-4)`. Joint floor: `Gc = (6k D(u), 0) + B Jo + R` with `B B^T >= (|v|^4/4) I`
([DL] §5.5) and `||R||_(L^2) <= C r`; Lemma 6.2 applies because `C r <= (1/2)(c_J |v|^4/4)^(1/2)` for `|v| >= r^(1/3)`
and small `r`; `lambda_J >= c |v|^4`, `Phi <= C |v|^(-4d)`. Marked, from [DL] (5.4)–(5.6):

    rho_j^(W, Psi)(X) <= C r^(-2) . r^(-(d+1)) |v|^(-2d) e^(-c/|v|^2) . r^6 |v|^(-6) . |v|^(-12d) . (|v|^(-4) + |v|^(-2d))
                      <= C r^(3-d) |v|^(-(16d+10)) e^(-c/|v|^2) <= C' r^(3-d).

**R2c (collar, `|v| >= v_0`).** All of `beta, Phi` are bounded ([DL] §5.6: fixed singular-value floor of `B`); the
marked intensity is `C r^(3-d)` times a constant.

Integrating R2a–R2c over the collar (`dX = r^d du dv`) gives `Z_r^(-1) E_(Q_r) N_j^(W_r Psi^(q-1))(collar) <= C r^3`, and
with (6.3) the scaled ball of radius `4` minus the pins has marked count at most `C r^3` ([DL] (1.4)).

**R3a (shell, `|v| <= s^(1/8)`).** `Lambda = (grad f(X), f(X))` raw, `tau = (0, h)`; [DL] (6.1): `Sigma_X >= c s^10
I_(d+1)`, variances and means bounded, `|h - E f(X)| <= C`; `beta <= C s^(-10)`; Lemma 6.3 with `sigma = s`:
`lambda_J >= c s^10`, `Phi <= C s^(-10d)`. [DL] §6.5: joint density at most `C s^(-5(d+1)) exp[-c s^(-1/4)]`. For the determinant
weight use the axis-safe bound of [DL] §6.4, valid throughout the strip `|v| <= s^(1/8)` including `v = 0`: the endpoint
short columns `H_M e_x`, `H_S e_x`, of size at most `Lr/2` by [LP] (5.3), do not depend on the witness position, so
Hadamard ([DL] Lemma D2 with `e_x`) gives `|det H_M| |det H_S| <= C r^2 K^(2d)` for every `X` in the strip; with the
trivial `|det H_X| <= K^d` and `Z_r >= z_* r^2` ([LP] (5.5)), `W_r |det H_X| / Z_r <= C K^(3d)` on the whole strip,
with no division by `|v|`. (A bound stated only at `v = 0` would not control nonzero `v`; this is the reading of
[DL] §6.4 recorded in OpenAI review 5357882570.) The marked intensity per unit volume and height is at most
`C s^(-5(d+1)) exp[-c s^(-1/4)] (1 + s^(-10))^(3d) (s^(-10) + s^(-5d)) <= C`, and the strip contributes at most
`C r^3 s^2` to the marked shell count exactly as in [DL] §6.5.

**R3b (shell, `s^(1/8) <= |v| <= 2`).** `Lambda = Z`, `tau = (0, 0, (h - bbar)/s^3)` bounded; [DL] (6.11):
`Cov_(Q_r)(Z) >= c B B^T` with `sigma_min(B) >= c |v|^3`, `||Cov(Z)^(-1)|| <= C |v|^(-6)`, `Cov(Z) <= C |v|^2 I`, mean
bounded; `beta <= C |v|^(-6)`. Joint floor: `Z = (det.) + B Jo + O_(L^p)(s)` with `lambda_B >= c |v|^6` and remainder
`C s <= (1/2)(c_J c |v|^6)^(1/2)` for `|v| >= s^(1/8)` and small `s`; `lambda_J >= c |v|^6`, `Phi <= C |v|^(-6d)`.
Marked, from [DL] (6.10), (6.12), (6.13):

    Lambda_(r,j)^(W, Psi)(X, h) <= C s^(-(d+4)) |v|^(-(d+4)) e^(-c/|v|^2) . s^2 (r + s^2)^2 |v|^(-5) . |v|^(-18d) . (|v|^(-6) + |v|^(-3d))
                                <= C s^(-(d+2)) (r + s^2)^2 |v|^(-(22d+15)) e^(-c/|v|^2) <= C' s^(-(d+2)) (r + s^2)^2,

uniformly down to `|v| = s^(1/8)`. Integration over the shell and the window gives `C r^3 [(r/s)^2 + s^2]` as in [DL]
(6.15), and the dyadic sum `C r^3 (A_0^(-2) + s_0^2)` as in [DL] (1.6).

**R4 (remote, `|X| >= s_0`).** `Lambda = Y_x = (grad f(x), f(x))`, `tau = (0, h)`. [RM] §2: `Cov((U_r, Y_x))` and its
Schur complements are uniformly bounded and positive on `O(d) x D_rho x [0, r_*]`; so `beta <= C`. Joint floor: the
family `(G(z), Y_x, U_r)`, at sites `o` (contact limit `E_0` at `r = 0`), `x`, and `z, conj z` (or `x' = Re z` at
`t = 0`) with `|z - x| >= zeta_0/2` and `|x' - o| >= eta/2`, is positive definite at every parameter by Lemma E_d and
continuous on the compact parameter set `{ x in D_(s_0), z in union A'_j, |z - x| >= zeta_0/2, t, u, frame, r in [0,
r_*] }`, so `lambda_J >= c` and `Phi <= C`. Here the slab exclusion of Lemma M' (b) is what keeps `z` away from the
forced zero of `F` at the witness. [RM] §3 (5) gives uniform moments of the conditioned field, so
`E_(Q')[(W_r F_j(H_x))^2]^(1/2) <= C r^2` as in [RM] (10), (14); with `Z_r >= z_* r^2` and the window of length
`k r^3`,

    Z_r^(-1) E_(Q_r) N_j^(W_r Psi^(q-1))( D_(s_0), I_r ) <= C r^3.

### 6.3 Summary of §6

In every regime the marked intensity is at most a constant times the unmarked bound of the source, after absorbing
the two extra factors `(1 + beta)^(3d)` (in place of the source's `(1 + beta)^(3d)`, unchanged in form) and
`(1 + beta + lambda_J^(-d/2))` into the source's own Gaussian penalty. The radial ledgers are untouched: the pin ball
keeps `r^(3-d) omega_d`, the collar `r^(3-d)`, the shells `s^(-(d+2))(r + s^2)^2`, the remote region `r^3` per unit
volume. The finite control `LG` records the exponents of both extra factors and the resulting absorption powers for
`d = 2, ..., 6`.

## 7. Proof of Theorem Q and the corollaries

*Theorem Q.* By (3.4), (5.2) and Lemma 5.1 on each region of the tiling, with monotone convergence over compact
punctures of the pins and boundaries assigned once,

    E_(Q_r^W)[ (N)_q ] <= sum_j sum_(regions) Z_r^(-1) E_(Q_r) N_j^(W_r Psi^(q-1))( region, heights )
                      <= (d+1) [ 2 C r^3 + C r^3 + C r^3 (4^(-2) + s_0^2) + C r^3 ] = C_q r^3,

the four terms being the two pin balls (6.3), the collar (R2), the shells (R3, all-height counts dominate window
counts in the first two regions), and the remote region (R4). Every intensity bound of §6 is uniform in `b, k, R, h in
I_r`, and the constants depend on `q` only through Proposition 4.5. ∎

*Corollary Theta.* The lower bounds are [EDL] (A4), valid in every fixed `d`. The upper bound on `E N(N-1)` is (1.1)
with `q = 2`, and `Q_r^W{N >= 2} <= E N(N-1)/2`. ∎

*Corollary P.* `E_(Q_r^W) N >= Q_r^W(N >= 1) >= Q_r^W(N >= 2) >= c r^3` by [EDL] (A4), and `E N <= C r^3` by [DL] (1.7);
divide (1.2) by these. The size-biased identity `E[(N)_q] = E N . E_sb[(N-1)_(q-1)]` is the definition of `P^sb`
(Math-#146). The two-sided statement is for the second factorial moment only. ∎

## 8. Remarks on the mechanism

- The logarithm of [C6L] came from the pathwise inequality `N(N-1) <= lambda N + Psi^2 1{Psi > lambda}` and the
  stretched-exponential tail of `Psi`. Here `Psi` is placed inside the Kac–Rice integrand instead, and only its
  conditional second moment is needed. That moment is polynomial in the same degeneracy variables that each region's
  Gaussian penalty already absorbs; this is why no logarithm survives.
- The forced zero of `F` at the witness is handled geometrically, by discarding a fixed slab of test radii around
  `|X - c_j|` in the packing. Because the retained balls stay at distance `zeta_0/2` from `X`, the residual covariance
  of `G(z)` on them has a uniform floor by compactness, and no quantitative expansion of `F` near `X` is needed.
- The improvement route of Math-#140 comment 5893989971 (growing complex radius, `exp(-c m log m)` degree tail) is not
  used and not needed here; it would sharpen Lemma M' but the polynomial form (4.8) already suffices for (1.1).
- In `d = 2` the argument uses only the planar sources [PP], [CP], [IW], [RM] through [DL]'s restatement, and [C6L]'s
  planar lemmas directly; the dimension enters only through the exponents `m^d`, `epsilon^(2d) t^(-d)`,
  `delta^(2d-1)`, `lambda = 2^(m/(4d))`.

## 9. What this note does not establish

- Anything about the fixed-`eta` remote pair kernel of `frontiers/remote_collision_20260928/PROOF.md` beyond
  consistency: Theorem Q does not reprove its Corollary D (`O(r^5)` pairs with both witnesses remote) and does not
  sharpen it; that source is not consumed here.
- The distribution of `N` beyond its factorial moments, or a Poisson or compound-Poisson approximation with rate.
- Elder selection; uniformity in `L`, in `d`, as `k -> 0` or as marks grow; numerical constants; any register,
  catalog or status change. The catalog entry C6 and the GRAPH node `math.rn-region.witness-collision` are not moved
  by this file; a source-bound reconciliation after nonauthor review of [C6L], [DL] and this note is a separate act.
- Organizational independence: all lanes share one GitHub account. [C6L], [DL] and this note are all Claude-authored,
  in two sessions. [C6L] has its nonauthor verdict; [DL] and this note still need theirs before consumption.

## 10. Finite controls

`palm_exact_check.py` (standard library, exact rational arithmetic; `-O` mode identical) checks six groups and rejects
seven mutants:

- `PK` the slab-excluded packing of Lemma M' (b): `zeta_0 <= eta'/8`, the count inequality `eta'/4 - zeta_0 >= delta`,
  the retained count at least `eta'/(4 delta)`, every retained ball inside the annulus, and every point of a retained
  ball at distance at least `zeta_0/2` from every point at distance `rho_X` from the centre, on grids of `rho_X`, for
  `delta = 2^(1-m)` and both radii `eta_0, eta_j`;
- `EX` the exponent arithmetic of Lemma M' for `d = 2, ..., 6`: `lambda^(2d) 2^(-m) = 2^(-m/2)`, `lambda^(-2) =
  2^(-m/(2d))`, and the final exponent `1/(2d)`;
- `TL` the moment series of Proposition 4.5: an explicit index from which the summand ratio is below `2^(-1/(4d))`,
  for `d = 2, ..., 6` and `p <= 4`, as an exact rational inequality;
- `PW` the pathwise inequalities (3.4) on integer grids with `Psi >= N + 2`;
- `SC` the Schur-complement floor of Lemma 4.2 and the residual-metric domination of Lemma 4.3 on random rational
  covariance matrices (positivity certified by exact `LDL^T`);
- `LG` the regime ledger of §6: for each regime the `r`-exponent is that of [DL], and the absorption powers of R1 II
  and R2a are `2n >= 6 + 3d` and `2n/3 >= 15 + 39d`.

The mutants are `no-slab`, `weak-zeta` (the insufficient ratio `zeta_0 = eta'/4`), `lambda-too-large`,
`series-ratio`, `factorial-power`, `schur-upper`, `forget-mark-power`.
`RESULTS.json` is the exact output. These checks verify counting, exponents, identities and inequalities of finite
matrices only. They do not verify the Gaussian, compactness, regression, Rouché or Kac–Rice steps, which are the
written arguments above and in the sources.

## 11. Attribution and reconnaissance

The cap `Psi`, Lemma M and Lemma D are those of [C6L]; the regional frames, densities and penalties are those of the
planar D5 sources as restated in [DL]; the Kac–Rice framework is arXiv:2304.07424v3 as cited by every source. The
contribution of this note is the parametric form of the tail (Lemma M', Proposition 4.5), the slab exclusion, the
identification of the two invariants `mu` and `Phi` with the frame parameters that the first-moment proofs already
control (Lemmas 4.1–4.4, 6.1–6.3), and the assembly through the marked Kac–Rice formula (Lemma 5.1). Nothing is
claimed as new beyond this. `RECONNAISSANCE.md` records the external check and its failure mode.
