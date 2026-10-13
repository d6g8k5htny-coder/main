## 1. Coordinates and exact identities

In frame coordinates `(x, y) = (x, y_1, y_2)`, `M = (-h, 0, 0)` and `S = (h, 0, 0)` with `h = r/2`; the midpoint is `0`.

**Pins.** The eight preconditioned pins are those of [LP] (3.1):
- `U0 = (f(M) + f(S))/2`, `U1 = (f(S) - f(M))/r`, `U2 = (f_x(S) - f_x(M))/r`, `U3 = (6/r^2)[f_x(M) + f_x(S) - 2(f(S) - f(M))/r]`;
- for `j = 1, 2`: `V_j0 = (f_yj(M) + f_yj(S))/2` and `V_j1 = (f_yj(S) - f_yj(M))/r`.

Under `Q` their values are exactly `(b - k r^3/2, -k r^2, 0, 12k, 0, 0, 0, 0)` ([LP] (3.3)).

**Targets.** The 31 targets are:

    A_ij  = f_yiyj(M)                                         (3; B = -A is the transverse Hessian of [LP] section 7),
    v_j   = f_xyj(M)/r + (f_yj(M) - f_yj(S))/r^2               (2),
    O_ij  = (f_yiyj(S) - f_yiyj(M))/r                         (3; the matrix Omega'),
    Y_ijk = f_yiyjyk(0)                                       (4),
    T, tau as in [E] section 1                                (2),
    nu_j  = (f_xyj(M) + f_xyj(S))/r^2 + 2 (f_yj(M) - f_yj(S))/r^3   (2),
    X_alpha = partial^alpha f(0),   |alpha| = 4                (15).

Each is a sum of terms `c r^p partial_x^a partial_y1^e1 partial_y2^e2 f(x_0 r, 0, 0)` with a common **weight** `a - p`. Since the
pins hold exactly under `Q`, almost surely

    f_xx(M) = r^2 T - 6kr,  f_xx(S) = r^2 (T + r tau) + 6kr,  grad_y f_x(M) = r v,  grad_y f_x(S) = r (r nu - v),
    D_y^2 f(S) = -B + r Omega'.                                                                              (I1)

Write `c1 = 6k - rT`, `c2 = 6k + rT + r^2 tau`, `e = r nu - v` (a vector) and `B_S = B - r Omega'`. The Hessians are
`H_M = [[-r c1, r v^T], [r v, -B]]` and `H_S = [[r c2, r e^T], [r e, -B_S]]`, and Schur complements give

    det H_M = -r P1,   P1 = c1 det B - r v^T adj(B) v,
    det H_S =  r P2,   P2 = c2 det B_S + r e^T adj(B_S) e.                                                  (I2)

(For `2 x 2` matrices `det(-B) = det B` and `det(B) B^-1 = adj B`.)

**The typed support.** `H_M < 0` holds exactly when `B > 0` and `c1 > r v^T B^-1 v`; then `c1 > 0` and `P1 > 0`. Index two
for `H_S` forces `det H_S > 0`, so `P2 > 0`. Hence, in every case,

    W_r <= r^2 P1 P2 1{B > 0, P1 > 0, P2 > 0}.                                                                (I3)

Unlike the plane, (I3) is only an inequality: `det H_S > 0` also allows index zero. Conversely, `B > 0`, `B_S > 0`,
`c1 > r v^T B^-1 v` and `c2 > 0` imply the typed event with `W_r = r^2 P1 P2` (section 6).

## 2. The Gaussian law on an `r`-band (Lemma 1)

**(a) Exact series.** Let `F`, `F'` be functionals of weights `w_F`, `w_F'`, with `w = w_F + w_F'`. For the reference kernel
`G(z) = exp(-|z|^2/2)`,

    Cov(F, F') = sum_gamma h(g_x) h(g_1) h(g_2) pi_gamma r^(g_x - w),        gamma = (g_x, g_1, g_2).

- `h(n) = (d/dt)^n exp(-t^2/2)` at `t = 0`, which is `(-1)^(n/2) (n-1)!!` for even `n` and `0` for odd `n`.
- `pi_gamma` is the exact rational `sum c c' (-1)^(a' + e1' + e2') (x_0 - x_0')^m / m!` over pairs of terms with
  `g_1 = e1 + e1'`, `g_2 = e2 + e2'` and `m = g_x - a - a' >= 0`.
- **Regularity:** `pi_gamma = 0` for every `g_x < w`. This is checked exactly for all 780 unordered pairs of the 39 functionals,
  so no negative power of `r` occurs.

**(b) Truncation at `g_x <= NX = 60`.** As in [E] section 2(b), with `|h(g_y)|` replaced by `|h(g_1) h(g_2)|`: for `r <= 1/64`
the terms with `g_x > 60` of one pair sum to at most `2 |c c'| |h(g_1) h(g_2)| 61^A r^(61 - w)`, `A = a + a'`.

**(c) Torus remainder, uniform in `L >= 10`, the frame and `r`.**
- **Decomposition.** `K_L = (G + rho)/Theta` with `rho(z) = sum_{n != 0} G(z + p_n)`, `p_n` in the rotated lattice `L R^T Z^3`,
  and `Theta = 1 + sum_{n != 0} exp(-|p_n|^2/2)` in `[1, 1 + theta_1(L)]`.
- **Bound on `rho`.** `rho` is entire. On the complex polydisc `{|z_1|, |z_2|, |z_3| <= 1}`:
  - `|exp(-(z + p).(z + p)/2)| <= e^(3/2) exp(-|Re z + p|^2/2)`, since `|Im z|^2 <= 3`;
  - exactly `24 j^2 + 2` lattice points have sup-norm index `j`, each at distance `>= jL`;
  - `|Re z| <= sqrt 3 < 7/4`.

  Hence `|rho| <= M_1(L) := e^(3/2) sum_{j >= 1} (24 j^2 + 2) exp(-(jL - 7/4)^2/2)`, independently of the frame.
- **Cauchy estimates** give `|partial^gamma rho(0)| <= g_x! g_1! g_2! M_1`, and the `rho`-part of a covariance is bounded by
  `sum |c c'| g_1! g_2! max(w, 1)^A M_1 / (1 - e r)` exactly as in [E] section 2(c).
- **Values.** `M_1(10) < 1.94 x 10^-13` and `theta_1(10) < 5.1 x 10^-21`. Both decrease in `L`, so the enclosures computed at
  `L = 10` hold for every `L >= 10`, every frame, and the reference kernel (`rho = 0`, `Theta = 1`).

**(d) Regression on the pins.** Interval Gaussian elimination on the eight pins gives, uniformly over `r in [r_0, r_1]`:
- the `Q`-means `mu_F = b alpha_F + k beta_F` of the 31 targets;
- their `Q`-covariance, scaled by `Theta^-1 in [1/(1 + theta_1), 1]`;
- the pin energy `E = e_bb b^2 + 2 e_bk b k + e_kk k^2`, scaled by `Theta`.

At contact (reference kernel) these reduce to the law of [G3] Lemma 1:
- `B ~ b I + GOE_2`;
- `v ~ N(0, I/2)`, `Omega' ~ GOE_2`;
- `Y` with independent entries of variances `6, 2, 2, 6`;
- all of them independent.

The certificate's band laws reproduce this to `10^-9` on the last band.

**(e) Regression on the transverse Hessian.** Use the Frobenius-isometric coordinates `b~ = (B_11, B_22, sqrt2 B_12)`, so that
`||b~ - b~'|| = ||B - B'||_F`, with `Q`-mean `mu~` and covariance `C~` (at contact `2 I`).
- For every other target, `beta~_F = Cov(F, b~) C~^-1` and `F_g = F - beta~_F . (b~ - mu~)`. Then `F_g` is independent of
  `B` under `Q`, with the same mean and the covariance `C - Cov(., b~) C~^-1 Cov(b~, .)`.
- Gershgorin bounds give `lambda_min(C~) >= sigma_B^2` and `lambda_max(C~) <= sigma^2`, with `sigma` rational.
- By the `y_1`- and `y_2`-parities of the reference kernel, `beta~_v = beta~_Y = beta~_nu = 0` there; the certificate
  carries their torus-sized enclosures.

**(f) The free blocks.** In the coordinates `v`, `omega = (O_11, O_22, sqrt2 O_12)` and
`Y~ = (Y_111, sqrt3 Y_112, sqrt3 Y_122, Y_222)`, Euclidean norms are `||v||`, `||Omega'||_F` and `||Y||_F` (the Frobenius norm of
the symmetric 3-tensor).
- An interval Cholesky factor `L` of the `g`-covariance of these nine coordinates has blocks `L_vv`, `L_ov`, `L_oo`, `L_Yv`,
  `L_Yo` and `L_YY`.
- Their operator norms are bounded by `max |diagonal|` plus the Frobenius norm of the strictly lower part (square blocks),
  and by the Frobenius norm otherwise.
- At contact `L_vv = I/sqrt2`, `L_oo = sqrt2 I`, `L_YY = sqrt6 I`, and the off-diagonal blocks vanish.

## 3. Third and fourth derivatives on `D` (Lemmas 2 and 3)

Write `A_(a,c) = sup_D ||partial_x^a D_y^c f||_op` for `a + c = 4`, so that `M_4 = max A_(a,c)`.

**Lemma 2 (averaging identities, blockwise).** For every `C^4` function satisfying the pins, and every `z in D`:

    |f_xxx(z)|            <= 12k        + r Delta_1,    Delta_1 = (5/2) A_(4,0) + 2 A_(3,1),
    ||grad_y f_xx(z)||    <= 2 ||v||     + r Delta_2,    Delta_2 = (5/2) A_(3,1) + 2 A_(2,2),
    ||D_y^2 f_x(z)||_op   <= ||Omega'||  + r Delta_3,    Delta_3 = (5/2) A_(2,2) + 2 A_(1,3),
    ||D_y^3 f(z)||_op     <= ||Y||_op    + r Delta_4,    Delta_4 = 2 A_(1,3) + 2 A_(0,4),

so `M_3 <= max_i (B_i + r Delta_i)` with `B = (12k, 2||v||, ||Omega'||_op, ||Y||_op) <= (12k, 2||v||, ||omega||, ||Y~||)`.

*Proof.* Let `g(t) = f(t, 0, 0)` and `q(t) = grad_y f(t, 0, 0)`. The four averages are:
- `U3 = integral K(t) g'''(t) dt` with the Peano kernel `K(t) = 3(h^2 - t^2)/(4h^3) >= 0`, `integral K = 1`, and `U3 = 12k`;
- `-2v = integral (2(h - t)/r^2) q''(t) dt` (each component; `q(+-h) = 0` by the pins), weights `>= 0` with total `1`;
- `Omega' = (1/r) integral D_y^2 f_x(t, 0, 0) dt`;
- `Y = D_y^3 f(0)`.

The norm of an average is at most `sup_t` of the norm at `(t, 0, 0)`. Along the segment from `(t, 0, 0)` to `z`, the
derivative of `partial_x^a D_y^c f` is `(z_x - t) partial_x^(a+1) D_y^c f + D_y^(c+1) partial_x^a f [z_y, ...]`. Its operator
norm is at most `|z_x - t| A_(a+1,c) + ||z_y|| A_(a,c+1)`, for multilinear operator norms with independently chosen unit
vectors. Use `|z_x - t| <= 5r/2` and `||z_y|| <= 2r` (and `|z_x| <= 2r` for `Y`). The operator norm of a symmetric tensor is at
most its Frobenius norm (Cauchy-Schwarz).

**Lemma 3 (fourth derivatives).**
- **Taylor bound.** The field is almost surely entire under `Q` ([E] Lemma 3). Hence, with `X_(a,c) = partial_x^a D_y^c f(0)`,

      A_(a,c) <= ||X_(a,c)||_F + T_(a,c),      T_(a,c) = sum_{gamma != 0} ||partial_x^a D_y^c partial^gamma f(0)||_F (2r)^|gamma| / gamma!,

  where `||X||_F^2 = sum_beta (c!/beta!) X_beta^2` for a symmetric `c`-tensor in two variables.
- **Gaussian bounds.** For every `delta`, `s_delta^2 = prod_i (2 delta_i - 1)!! + (2 delta)! M_1` bounds the unconditioned
  variance of `partial^delta f(0)`, and `|mean| <= s_delta sqrt(E)`.
- **Norm bound.** For a Gaussian vector, `|| ||X - EX|| ||_p <= sqrt(tr Cov) max(1, ||N(0,1)||_p)` (Minkowski in `L^(p/2)` on
  the eigen-decomposition). Hence `||T_(a,c)||_p <= S_(a,c) (sqrt(E) + max(1, ||N||_p))` with

      S_(a,c)(r_1) = sum_{1 <= |gamma| <= 24} sigma_(a,c),gamma (2 r_1)^|gamma| / gamma!  +  (tail),
      sigma_(a,c),gamma^2 = sum_beta (c!/beta!) s^2_(a + gamma_x, beta + gamma_y).

  The tail uses `s_delta <= 2^|delta| delta! sqrt(1 + M_1)`, `(alpha + gamma)!/gamma! <= (|gamma| + 4)^4` and
  `(n + 1)(n + 2)/2` multi-indices of order `n`. It requires `4 r_1 <= 1/16`.
- **Dependence on `B`.** The regression coefficient of `partial^delta f(0)` on `b~` has norm at most `s_delta / sigma_B`. So
  `A_(a,c) <= A^g_(a,c) + B_(a,c) ||B - mu_B||_F`, where
  - `B_(a,c) = (sum_beta (c!/beta!) ||beta~_X(a,beta)||^2)^(1/2) + S_(a,c)/sigma_B`;
  - `A^g_(a,c)` is the same bound for the `g`-field: same means, smaller variances.

## 4. Regression on `B` and the failure set (Lemma 4)

On the typed support `B > 0`. With `lambda_1 <= lambda_2` the eigenvalues of `B`,

    ||B - mu_B||_F <= ||B||_F + ||mu_B||_F <= lambda_1 + lambda_2 + ||mu_B||_F.

**The `g`-measurable bound on `M_3`.** By Lemmas 2 and 3 and the regression, `M_3 <= max_i (y_i + bar-beta_i ||B - mu_B||_F)`, with

    y_1 = 12k + r Delta_1^g,   y_2 = 2 ||v_g|| + r Delta_2^g,   y_3 = ||omega_g|| + r Delta_3^g,   y_4 = ||Y~_g|| + r Delta_4^g,
    bar-beta_1 = r ((5/2) B_(4,0) + 2 B_(3,1)),
    bar-beta_2 = 2 ||beta_v|| + r ((5/2) B_(3,1) + 2 B_(2,2)),
    bar-beta_3 = ||beta_omega|| + r ((5/2) B_(2,2) + 2 B_(1,3)),
    bar-beta_4 = ||beta_Y~|| + r (2 B_(1,3) + 2 B_(0,4)),

where `||beta_v||^2 = sum_j ||beta~_vj||^2`, and similarly for `omega` and `Y~` with the Frobenius weights.

**Lemma 4.** Put `y_i'' = y_i + bar-beta_i (||mu_B||_F + lambda_2)`, which is measurable in the `g`-variables and `lambda_2`. Put
`a = 4/(3k)` and `x_i = a r bar-beta_i y_i''`. If `lambda_1 <= a r M_3^2`, then for some `i`,
`lambda_1 <= a r (y_i'' + bar-beta_i lambda_1)^2`. For `x_i <= 1/4` the roots of this quadratic satisfy the bounds of [E] Lemma 4:

    lambda_-^(i) <= a r y_i''^2 / (1 - 3 x_i),        lambda_+^(i) >= 1 / (4 a r bar-beta_i^2).

**The cover.** With `X_0 = 1/64` and `kappa = 3/(1 - 3 X_0)`, on the typed support

    G_r^c  subset  N  u  F  u  X  u  A,
    N = {lambda_1 <= r U,  x_i <= X_0 for all i},    U = max_i a y_i''^2 (1 + kappa x_i),
    F = {lambda_1 >= 1/(4 a r bar-beta^2)},  bar-beta = max_i bar-beta_i,
    X = {x_i > X_0 for some i},    A = {r M_4 > 3k/10}.

The threshold `U` is a function of the `g`-variables and of `lambda_2`. The `g`-variables are independent of `B`. The failure
window in `lambda_1` is `[0, r U]`.

## 5. The main term

**Eigenvalue coordinates.** Write `B = R_phi diag(lambda_1, lambda_2) R_phi^T` with `phi in [0, pi)`. Then
`dB_11 dB_22 dB_12 = (lambda_2 - lambda_1) d lambda_1 d lambda_2 d phi` (the Jacobian determinant is `-(lambda_2 - lambda_1)`).

**An angle-free density majorant.**
- The density of `(B_11, B_22, B_12)` is `sqrt2` times that of `b~`, which is at most
  `(2 pi)^(-3/2) det(C~)^(-1/2) exp(-||b~ - mu~||^2 / (2 sigma^2))`.
- Let `epsilon` be the supremum over the box of `||mu_B - b I||_F`. Since `||B - mu_B||_F >= ||B - b I||_F - epsilon`, we have
  `||B - bI||_F^2 = (lambda_1 - b)^2 + (lambda_2 - b)^2` and `||B - bI||_F <= |lambda_1 - b| + |lambda_2 - b|`.
- Integrating over `phi` therefore bounds the joint density of `(lambda_1, lambda_2)` by

      p(lambda_1, lambda_2) <= K_B (lambda_2 - lambda_1) g(lambda_1) g(lambda_2),
      K_B = pi sqrt2 (2 pi)^(-3/2) det(C~)^(-1/2),   g(lambda) = exp(-(lambda - b)^2/(2 sigma^2) + epsilon |lambda - b| / sigma^2).

- At contact (`epsilon = 0`, `C~ = 2I`) this is the exact `GOE_2` eigenvalue density
  `(lambda_2 - lambda_1) exp(-((lambda_1 - b)^2 + (lambda_2 - b)^2)/4) / (4 sqrt(2 pi))` of [G3] section 1.

**The weight on `N`.** Write `Omega = ||Omega'||_op`. On the typed support:
- `P1 <= |c1| lambda_1 lambda_2`, since `adj B >= 0`;
- `|det B_S| <= (lambda_1 + r Omega)(lambda_2 + r Omega)`, by Weyl's inequalities;
- `|e^T adj(B_S) e| <= ||e||^2 (lambda_2 + r Omega)`, since `adj B_S` has the eigenvalues of `B_S`.

Hence

    W_r <= r^2 |c1| lambda_1 lambda_2 (lambda_2 + r Omega) [|c2| (lambda_1 + r Omega) + r ||e||^2],

and with `lambda_1 = r u`, `W_r <= r^4 |c1| u lambda_2 (lambda_2 + r Omega) [|c2| (u + Omega) + ||e||^2]`.

**Majorants.** On `N`, since `lambda_1 <= lambda_2`, every target satisfies
`|F| <= bar-F = |F_g| + ||beta~_F|| (||mu_B||_F + 2 lambda_2)`. In particular
`Omega <= bar-Omega = ||omega_g|| + ||beta_omega|| (||mu_B||_F + 2 lambda_2)` and `||e||^2 <= bar-e = (r ||bar-nu|| + ||bar-v||)^2`.

**Integration.** Integrate `lambda_1 = r u` over `[0, r U]`, extend `lambda_2` to `(0, infinity)`, and use
`lambda_2 - lambda_1 <= lambda_2`:

    E_Q[W_r 1_N] <= K_B r^5 E_g integral_0^inf lambda_2^2 (lambda_2 + r bar-Omega) g(lambda_2)
                                     integral_0^U g(r u) |bar-c1| u [|bar-c2| (u + bar-Omega) + bar-e] du d lambda_2.

Write `lambda_2^2 (lambda_2 + r bar-Omega) g(lambda_2) d lambda_2 = I_3 nu(d lambda_2) + r bar-Omega I_2 nu'(d lambda_2)` with
`I_j = integral_0^inf lambda^j g(lambda) d lambda`, and the probability measures `nu ∝ lambda^3 g` and `nu' ∝ lambda^2 g`.
- Under `g x nu` and `g x nu'`, `lambda_2` is independent of the `g`-variables.
- `E_nu[lambda_2^p] = I_(3+p)/I_3`, and it dominates `E_nu'[lambda_2^p]` (Chebyshev's association inequality for the increasing
  functions `lambda^p` and `lambda` under `nu'`).

**Bounds on `I_j`.** For `lambda >= 0`, `epsilon |lambda - b| <= epsilon (lambda + b)`, so

    g(lambda) <= exp((4 b epsilon + epsilon^2)/(2 sigma^2)) exp(-(lambda - b - epsilon)^2/(2 sigma^2)),

and `I_j <= exp((4 b_hi epsilon + epsilon^2)/(2 sigma^2)) J_j(b_hi + epsilon, sigma)`.
- Here `J_j(m, sigma) = integral_0^inf lambda^j exp(-(lambda - m)^2/(2 sigma^2)) d lambda` increases in `m`.
- It satisfies `J_0 = sigma sqrt(2 pi) Phi(m/sigma)`, `J_1 = m J_0 + sigma^2 exp(-m^2/(2 sigma^2))` and
  `J_(j+1) = m J_j + j sigma^2 J_(j-1)`, by integration by parts.
- Also `I_3 >= J_3(b_lo, sigma)`.

**Density in `lambda_1`.** Two bounds hold on the failure window:
- `g(lambda_1) <= bar-g = exp(epsilon^2/(2 sigma^2))`, the supremum of `g`;
- `g(lambda_1) <= g_0 + lambda_1 L`, where
  - `g_0` is the supremum of `g(0)` over the box;
  - `L = min(exp(-1/2) bar-g / sigma, g_0 c (x - 1)/log x)`, with `c = (b_hi + epsilon)/sigma^2` and `log x = log(bar-g/g_0)`.

  The second slope is the chord of `min(bar-g, g_0 exp(c lambda))`, valid because `g(lambda) <= g(0) exp((b + epsilon) lambda / sigma^2)`
  for `lambda >= 0`. It matters at larger `b`, where `g` rises on the failure window.

**Normalization by `K = 12k`.** As in [E] section 5, `U = a K^2 tilde-U` with `a K^2 = 192k`, and
`|c1 c2|/(36k^2) <= 1 + eps_T` with `eps_T <= (2r|bar-T| + r^2|bar-tau|)/(6k) + r|bar-T|(r|bar-T| + r^2|bar-tau|)/(36k^2)`. Then

    E_Q[W_r 1_N] / (36 k^2 r^5)  <=  K_B min( bar-g Lambda,  g_0 Lambda + r L Lambda' ),
    Lambda  = I_3 (A + B + C) + r I_2 D,       Lambda' = I_3 (A' + B' + C') + r I_2 D'.

- `A, B, C, A', B', C'` are the terms of [E] section 5, with the factor `bar-om` replaced by `bar-Omega` and the factor
  `bar-e` by its vector form.
- `D` and `D'` bound `E[bar-Omega (integrand of A + B + C)]` and the same with one more power of `tilde-U`, under `g x nu'`.
  - The part of `bar-Omega` sharing the variable `chi_3` of `u_3` is integrated exactly against `tilde-U^3` (resp.
    `tilde-U^4`).
  - The rest goes through Hölder's inequality.

**Moments of `tilde-U`.** These follow [E] section 5, with three changes.
- **Chi variables.** The free variables are `u_2 = (2 ||L_vv||/K) chi_2`, `u_3 = (||L_oo||/K) chi_3` and
  `u_4 = (||L_YY||/K) chi_4`, independent. Their positive-part moments are in closed form:
  - `E[u^e (u^n - c^n)_+] = s^(n+e) E[chi^(n+e) 1{chi > c/s}] - c^n s^e E[chi^e 1{chi > c/s}]`;
  - `E[chi_nu^j 1{chi_nu > x}] = C_nu sqrt(2 pi) M_(j+nu-1)(x)`, with `C_nu = 1/(2^(nu/2 - 1) Gamma(nu/2))` and the
    incomplete Gaussian moments `M_j` of [E].
- **Cross terms.** The cross-Cholesky terms `||L_ov|| chi_2` and `||L_Yv|| chi_2 + ||L_Yo|| chi_3` enter `delta_3` and
  `delta_4`.
- **`lambda_2`.** It enters `tilde-y_1` through `(bar-beta_1/K) lambda_2` and each `delta_i` through `(bar-beta_i/K) lambda_2`,
  with `L^p` norms under `nu`. Since `bar-beta_1 = O(r^2)`, this is invisible in the leading term.

## 6. The normalizer floor (Lemma 5')

**Pointwise bound.** Let `G_0 = {r|T| <= 3k/2, r^2|tau| <= 3k/2}`. On `G_0`:
- `c1 >= 9k/2`, `3k <= c2 <= 9k`, and `c1 c2 >= 36k^2 (1 - eps_T)` with `eps_T = (2r|T| + r^2|tau|)/(6k)`, as in [E].
- Write `ell = (lambda_1)_+` and `Omega = ||Omega'||_op`. Pointwise,

      W_r / r^2  >=  1_{G_0} { 36k^2 [ (1 - eps_T) ell^2 lambda_2^2 - r Omega ell lambda_2 (ell + lambda_2) ] - c2 r ||v||^2 ell lambda_2^2 }.

*Proof.* Suppose `B > 0`, `lambda_1 > r Omega`, `c1 lambda_1 > r ||v||^2` and `c2 > 0`. Then the event is typed and
`W_r = r^2 P1 P2`:
- `lambda_1 > r Omega` gives `B_S > 0` by Weyl;
- `c1 lambda_1 > r ||v||^2` gives `c1 > r v^T B^-1 v`;
- then `P1 >= lambda_2 (c1 lambda_1 - r ||v||^2)`, since `v^T adj(B) v <= lambda_2 ||v||^2`;
- and `P2 >= c2 (lambda_1 - r Omega)(lambda_2 - r Omega)`, since `adj(B_S) >= 0`.

So `W_r / r^2 >= 1_{G_0} c2 lambda_2 (lambda_2 - r Omega)_+ (c1 lambda_1 - r ||v||^2)_+ (lambda_1 - r Omega)_+` in every case.
Then use:
- `(x - y)_+ z >= xz - yz`;
- `lambda_1 (lambda_1 - r Omega)_+ lambda_2 (lambda_2 - r Omega)_+ >= ell^2 lambda_2^2 - r Omega ell lambda_2 (ell + lambda_2)`;
- `c2 <= 9k`.

**Taking expectations.** With `F = (det B)^2 1{B > 0} = ell^2 lambda_2^2` and `DT = det B tr B 1{B > 0}` (note
`ell lambda_2 (ell + lambda_2) = DT` and `ell lambda_2^2 <= DT`):

    Z_r / (36 k^2 r^2)  >=  E[F] - ||F||_2 P(G_0^c)^(1/2) - ||eps_T||_2 ||F||_2 - r E[Omega DT] - (r/(4k)) E[||v||^2 DT].

The regression splits the last two terms:
- `E[Omega DT] <= E||omega_g|| E[DT] + ||beta_omega|| E[(tr B + ||mu_B||_F) DT]`;
- `E[||v||^2 DT] <= (17/16) E||v_g||^2 E[DT] + 17 ||beta_v||^2 E[(tr B + ||mu_B||_F)^2 DT]`.

**Contact moments in closed form.** For `B = m I + GOE_2`:
- The eigenvalues are `s -+ rho`, where `s = m + (G_11 + G_22)/2 ~ N(m, 1)` and
  `rho = (((G_11 - G_22)/2)^2 + G_12^2)^(1/2)` is Rayleigh and independent of `s`.
- Hence `det B = s^2 - rho^2`, `tr B = 2s`, and `B > 0` exactly when `s > rho`.
- With `u = rho^2/2`, `integral_0^s (s^2 - rho^2)^a rho e^(-rho^2/2) d rho = P_a(s) - e^(-s^2/2) Q_a(s)` for explicit
  polynomials `P_a`, `Q_a`.
- `E[s^n e^(-s^2/2) 1{s > 0}] = (e^(-m^2/4)/sqrt2) E[(m/2 + U/sqrt2)^n 1{U > -m/sqrt2}]` with `U ~ N(0, 1)`.

So every `E[det(B)^a tr(B)^c 1{B > 0}]` is a finite combination of incomplete Gaussian moments. In particular

    m_(3,m) = E[(det B)^2 1{B > 0}] = E[(s^4 - 4 s^2 + 8) 1{s > 0}] - 4 sqrt2 e^(-m^2/4) Phi(m/sqrt2),

which gives `m_(3,0) = (7 - 4 sqrt2)/2`, the value of [G3] section 1. The certificate checks this value.

**Loewner coupling.**
- Write `b~ = mu~ + L~ z`, with `L~` the Cholesky factor of `C~` and `z` standard in `R^3`, and put `G = mat(sqrt2 z) ~ GOE_2`.
- Then `B - [(b - epsilon) I + G] >= -eta ||z|| I`, where `eta = ||L~ - sqrt2 I||`, because `||mat(y)||_op <= ||y||`.
- Every `h(B) = (det B)^a (tr B)^c 1{B > 0}` is monotone in the Loewner order.
- So on `{||z|| <= R_0 = 12}`, `h((b_lo - epsilon - eta R_0) I + G) <= h(B) <= h((b_hi + epsilon + eta R_0) I + G)`.

Hence:
- `E[F] >= m_(3, b'_lo) - E[F((b'_lo) I + G)^2]^(1/2) P(chi_3 > 12)^(1/2)`;
- the correction moments are bounded at `b'' = b_hi + epsilon + eta R_0`, plus outside-ball terms;
- `m_(3,b)` and the correction moments increase in `b`.

## 7. Rare branches

**Weight bound.** On the typed support, with `D = det B 1{B > 0}`,

    W_r / r^2 <= |c1| D [ |c2| (D + r Omega tr B + r^2 Omega^2) + r ||e||^2 (tr B + r Omega) ].

Hölder gives `||W_r 1_typed||_q <= r^2 g_1(q) g_2(q)`.
- `||D||_p` (for `p = 4q <= 64`) comes from the contact moments at `b''` (Loewner coupling).
- The outside-ball part is bounded by `2^-p E[(||mu_B||_F + sigma chi_3)^(2p) 1{chi_3 > 12}]`, using `D <= ||B||_F^2/2`.

Then `E_Q[W_r 1_E] <= ||W_r 1_typed||_q P(E)^(1 - 1/q)` for `E = F, X, A` and `q in {2, 4, 8, 16}`, the smallest bound being
used.

**`P(A)`.** For each block `(a, c)`, the threshold `t = 3k/(10r)` is split as `f t + (1 - f) t`, `f in {7/8, 15/16}`, and the
smaller bound is used:
- `c = 0`: a Gaussian tail of `X_(4,0,0)`.
- `c = 1`: `||X_(3,.)|| <= ||mean|| + lambda_max^(1/2) chi_2`, a chi tail.
- `c >= 2`: `||X_(a,c)||_op = max_theta |Q(theta)|`, where `Q(theta) = partial_x^a partial_(u_theta)^c f(0)` is a trigonometric
  polynomial of degree `c`.
  - Bernstein's inequality gives `|Q'| <= c max|Q|`.
  - A net of 258 rational unit vectors (`((1 - t^2), 2t)/(1 + t^2)`, `t = j/128`, and their rotations by `pi/2`) has angular
    gaps at most `2/128`.
  - Together these give `max|Q| <= max_j |Q(theta_j)| / (1 - c/128)`.
  - The certificate uses the union bound with the largest net variance of the `Q`-law and `|mean| <= ||mean||_F`.
- In every case, plus Markov's inequality at `p = 32` on `T_(a,c)`.

The Frobenius norm would be far too generous here. The bi-Laplacian direction of the quartic block carries Frobenius variance
`~ 144` against `~ 96` for any single direction.

**`P(F)`.** `P(F) <= P(B_11 >= lambda_far)`, since `lambda_1 <= B_11`.

**`P(X)`.** `P(X) <= sum_i P(y_i' + bar-beta_i lambda_2 > X_0/(a r bar-beta_i))`, split into:
- Markov's inequality on `r Delta_1` (`i = 1`), and chi tails of `u_i` plus Markov's inequality on `delta_i` (`i >= 2`);
- `P(lambda_2 > s) <= P(sigma chi_3 > s - ||mu_B||_F)`.

## 8. Bands, boxes, and the last band

**Bands.** `(0, 1/64]` is covered by `[0, 2^-29]` and 120 bands: the octaves `[2^-(j+1), 2^-j]` are split into 8 equal
sub-bands for `j = 6, ..., 12` and into 4 for `j = 13, ..., 28`. On each band the law is enclosed uniformly (Lemma 1), and every
`r`-monotone quantity is taken at its worst endpoint.

**Boxes.** There are 39 boxes `(b, k)`, listed as `GRID` in `theorem_a3.py` and in `RESULTS.json`:
- the `k`-boxes are `[1/2, 3/4], [3/4, 1], [1, 5/4], [5/4, 3/2], [3/2, 13/8], [13/8, 7/4], [7/4, 15/8], [15/8, 31/16], [31/16, 63/32]` and
  `[63/32, 2]`;
- each `k`-box has its own `b`-grid, refined toward the corner `(b, k) = (0, 2)`; the top `k`-box has 11 `b`-boxes, starting
  with `[0, 1/512]`.

The grids were chosen so that every box stays below the corner box at `r <= 1/4096` and `r <= 1/512`. This matters because
the bound grows with `b_hi` through `I_3` while `c_G^(3)` decreases in `b`. A box only affects how close `C3` comes to
`c_G^(3)`, never validity.

`C3(r_*)` is the maximum of the box bounds over the bands with `r_1 <= r_*`. `z3(r_*)` is the minimum over the same boxes of
`36 k_lo^2` times the floor of Lemma 5'.

**The last band `[0, R]`, `R = 2^-29`.** The argument of [E] section 8 carries over:
- every Gaussian or chi tail argument is `>= 3` at `R` (checked), and it scales at least like `1/r`;
- no tail probability reaches `1` at `R` (checked);
- for a `chi_nu` tail with `nu <= 5`, `P(chi_nu > 2x) <= 2^nu e^(-3x^2/2) P(chi_nu > x) < P(chi_nu > x)/64` when `x >= 3`;
- every Markov piece scales like `r^32` or faster.

So the tail term over `r^3` attains its supremum over `(0, R]` on `[R/2, R]`.

## 9. Arithmetic

As in [E] section 9:
- exact rationals;
- `exp`, `Phi`, `pi` and `sqrt` enclosed with outward rounding to `2^-160`;
- upper bounds rounded upward to 200 significant bits;
- interval Gaussian elimination and Cholesky decompositions with positive pivots.

In addition:
- The `chi` constants `C_nu sqrt(2 pi)` come from `Gamma` at integers and half-integers.
- The contact moments use exact polynomial coefficients.
- The `J_j` come from the stable three-term recursion. All its terms are positive for `m >= 0`.
