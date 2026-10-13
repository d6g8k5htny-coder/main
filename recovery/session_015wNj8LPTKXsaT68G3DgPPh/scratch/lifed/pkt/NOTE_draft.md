# Theorem B of [LP] with explicit constants on the C8 band (`d = 2, 3`): the radial ledger to second order, two-sided lifetime densities, and the difference constant `C_{B,K}`

**Object:** `CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1`. **Author lane:** Anthropic / Claude.

**Kind.** An author-side theorem with explicit constants, certified by exact rational interval arithmetic (standard library
only): Taylor models in the radius for the pin law, and the box-bound engines of Math-#206 and Math-#212 for the cap route.

**Scientific effect:** NONE. Catalog entry C8 (`reviews/candidates_pending_20260928/CANDIDATES.md`) stays OPEN. No STATUS,
PROOF_INDEX, GRAPH or catalog edit. Same GitHub account as every lane; zero organizational-independence credit. The author will
not merge. Scope claim: Math-#206 comment 5924133127.

**Sources** (pinned in `SOURCE_MAP.json`):
- [LP] `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`:
  - section 1 (the model in every `d`, Theorem B and (1.2));
  - section 3 (the preconditioned pins (3.1)–(3.3));
  - section 8 (`1 - p_r <= Q^W(G_r^c)`);
  - sections 9–11 (the weighted Kac–Rice interface, the radial ledger (10.1)–(10.2), the lifetime pushforward (11.1)–(11.3));
  - section 12 ((12.1), (12.3)).
  Sections 8–12 are consumed as statements; nothing else of [LP] is used.
- [CAP] `imports/lifetime_parent_20260925/MARKED_CYLINDER_CAP_PROOF.md`, section 1 (the good event `G_r`), through the
  engines.
- [E] Math-#206 at `0cb8ad3`: the planar engine `theorem_a.py`, carried here byte for byte as `engine_e2.py`.
- [E3] Math-#212 at `c4b8ec3`: the `d = 3` engine `theorem_a3.py`, carried here byte for byte as `engine_e3.py`.
- Read-only cross-checks: [W] Math-#197 (`c_{B,K}` in the plane), [W3] Math-#205 (`c_{B,K}` in `d = 3`), [G] Math-#203
  (`c_G = R(b) J(k)`), [G3] Math-#208 (`c_G^(3)`).

## 0. Statement

**Setting** ([LP] sections 1, 10, 11 with `d in {2, 3}`, `m = d - 1`). The torus `X = (R / L Z)^d` with the periodized
Gaussian kernel `K_L`, or the reference kernel `exp(-|z|^2/2)`. For an orthonormal frame `R` with axial vector `u`, the pins
`M = -(r/2) u`, `S = (r/2) u`, `f(M) = b`, `f(S) = b - k r^3`, `grad f(M) = grad f(S) = 0`, the regression law `Q`, the typed
weight `W_r` and `Z_r = E_Q W_r`. [LP] (10.2) writes the candidate intensity per unit volume as `r A_r(b, k, u) dr db dk dsigma(u)`
with

    A_r(b, k, u) = 12 pi_r(R; v_r) Z_r / r^2,

where `pi_r(R; v_r)` is the density of the preconditioned pins (3.1) at their prescribed value (3.3). Let `B = [0, 1]`,
`K = [1/2, 2]`. The **reference contact integrand** is the closed form

    A_0^ref(b, k) = 432 (2 pi)^-(d+1) 12^(-1/2) k^2 exp(-12 k^2) exp(-3 b^2/4) m_d(b),

    m_2(b) = (b^2 + 2) Phi(b/sqrt2) + sqrt2 b phi(b/sqrt2),          m_3(b) = E[det(B)^2 1{B > 0}],  B = b I + GOE_2,

the integrand of [LP] (11.3) for the reference kernel (`pi_0 = (2 pi)^-(d+1) 12^(-1/2) exp(-3b^2/4 - 12k^2)`,
`z_0 = 36 k^2 m_d(b)`; `m_3` in closed form in [E3] section 6).

**Theorem L (the radial ledger to second order).** For `d = 2, 3`, every torus side `L >= 10`, every frame, every `b in B`,
`k in K`, every `R` in the table and every `0 < r <= R`,

    -(a_dn(R) r^2 + eps_d)  <=  A_r(b, k, u) / A_0^ref(b, k) - 1  <=  a_up(R) r^2 + eps_d.

The same holds for the reference kernel. The constants are exact rationals (`RESULTS.json`), shown rounded up:

| `R` | `a_up`, `d = 2` | `a_dn`, `d = 2` | `a_up`, `d = 3` | `a_dn`, `d = 3` |
|---|---|---|---|---|
| `1/4096` | `15.397488` | `5.393515` | `119.982042` | `117.810137` |
| `1/2048` | `15.400170` | `5.399557` | `120.570949` | `118.398982` |
| `1/1024` | `15.405607` | `5.411650` | `121.759239` | `119.587053` |
| `1/512` | `15.416775` | `5.435867` | `124.178061` | `122.005035` |
| `1/256` | `15.440299` | `5.484427` | `129.187376` | `127.010960` |
| `1/128` | `15.492143` | `5.582080` | `139.915037` | `137.724030` |
| `1/64` | `15.615439` | `5.779703` | `164.396424` | `162.135918` |

with `eps_2 <= 3.96 x 10^-10` and `eps_3 <= 1.80 x 10^-8` (torus and truncation remainders only).

So `A_r -> A_0` uniformly with the explicit rate `O(r^2)`. [LP] proves the convergence `A_r -> A_0` ((10.3)) and states
"Convergence of `A_r` to `A_0` was not assigned a numerical rate" (section 11). There is no first-order term: section 4 shows
that the first-order parts of `Z_r / r^2` cancel exactly, and Lemma 1 that for the Gaussian kernel the law of the transverse
Hessian at `M` is the contact law for every `r`.

**Corollary L1 (lifetime densities to second order).** Let `nu_cand`, `nu_eld` be the compact-window densities of [LP]
Theorem B for pairs at distance at most `r_pop`, with the versions (11.2). For `0 < ell < min(r_pop, R)^3 / 2`:

    | ell^(1/3) nu_cand(ell) - c^ref_{B,K} |  <=  c'(R) ell^(2/3) + eps'_d,
    c^ref_{B,K} (1 - eps_d) - c'_dn(R) ell^(2/3) - C_{B,K}(R) ell  <=  ell^(1/3) nu_eld(ell)  <=  ell^(1/3) nu_cand(ell).

- `c^ref_{B,K} = |S^(d-1)| int_{B x K} A_0^ref / (3 k^(2/3))` is certified:
  - `d = 2`: `[0.002386938000079198, 0.002386938059531751]`, containing Math-#197's `0.0023869380211529483`;
  - `d = 3`: `[0.000974382342936858, 0.000974382390951253]`, containing Math-#205's `0.00097438236602425`.
- `c'(R)` is the weighted second-order constant `|S^(d-1)| sum_boxes a_up,box int_box A_0^ref / (3 k^(4/3))`, and `c'_dn` the
  same with `a_dn`:
  - `d = 2`: `c'/c^ref = 9.6557` at `R = 1/4096` and `10.0364` at `1/64`; `c'_dn/c^ref = 6.7351` and `7.1041`;
  - `d = 3`: `c'/c^ref = 99.891` at `1/4096` and `143.83` at `1/64`.
- `eps'_2 <= 1.2 x 10^-13` and `eps'_3 <= 2.5 x 10^-12`.

Hence, with explicit constants, both compact-window densities have the second-order form

    nu_cand(ell) = c_{B,K} ell^(-1/3) + O(ell^(1/3)),        nu_eld(ell) = c_{B,K} ell^(-1/3) + O(ell^(1/3)),

where `c_{B,K}` (torus, any `L >= 10`) lies within `eps'_d` of `c^ref_{B,K}`. [LP] notes that (1.2) "is an O(ell^(2/3))
difference, not a claim that either density separately has a second-order expansion with that remainder". For example in the
plane, `|ell^(1/3) nu_cand / c_{B,K} - 1| <= 10.04 ell^(2/3) + 5 x 10^-11` for `ell < 1.9 x 10^-6`, which is at most
`1.6 x 10^-3`.

**Corollary L2 (explicit [LP] (1.2), (12.1), (12.3)).** For `0 < ell, t < min(r_pop, r_*)^3 / 2` and `q > -5/3`:

    0 <= nu_cand(ell) - nu_eld(ell) <= C_{B,K}(r_*) ell^(2/3),
    0 <= E[N_cand(0, t] - N_eld(0, t]] <= (3/5) C_{B,K}(r_*) t^(5/3),
    E sum_{candidate, not selected, ell <= t} ell^q <= C_{B,K}(r_*) t^(q + 5/3) / (q + 5/3),

    (3/2) c^ref (1 - eps) t^(2/3) - (3/5) c'_dn t^(4/3) - (3/5) C_{B,K} t^(5/3)  <=  E N_eld(0, t]  <=  (3/2) c^ref t^(2/3) + (3/5) c' t^(4/3) + (3/2) eps' t^(2/3).

@@CBK_TABLE@@

[LP] gives (1.2) with "`C ell^(2/3) integral A^*/(3 k^(5/3))`" and no value for either factor.

@@CBK_COMPARE@@

## 1. The radial ledger and what is consumed

[LP] sections 9–11 are consumed as statements:
- the weighted Kac–Rice identity on the near-diagonal pair domain (section 9);
- the ledger (10.1)–(10.2): the candidate intensity is `r A_r dr db dk dsigma(u)`, the selected intensity carries the extra
  factor `p_r`, and `A_r` does not depend on the transverse frame;
- the pushforward (11.1)–(11.2): for `ell < k_- r_pop^3`,
  `nu_cand(ell) = ell^(-1/3) int_{B x K x S^(d-1)} A_{(ell/k)^(1/3)}(b, k, u) / (3 k^(2/3)) db dk dsigma(u)`, and `nu_eld` has the
  extra factor `p_r`;
- section 8: `0 <= 1 - p_r <= Q^W(G_r^c)`.

Theorem L is a statement about `A_r` alone. The corollaries follow by integration:
- **L1.** At fixed `ell < R^3/2` every `k in K` has `r = (ell/k)^(1/3) <= (2 ell)^(1/3) < R`, and `r^2 = ell^(2/3) k^(-2/3)`.
  Integrating Theorem L box by box against `A_0^ref / (3 k^(2/3))` gives `c^ref + ell^(2/3) c' + eps'` above and the mirror
  image below. `A_0^ref` does not depend on `u`, so the sphere contributes `|S^(d-1)|`.
- **L2.** `nu_cand - nu_eld = ell^(-1/3) int A_r (1 - p_r) / (3 k^(2/3))`, and on each cap box `1 - p_r <= Q^W(G_r^c) <= C_box r^3
  = C_box ell / k`. So the difference is at most `ell^(2/3) sum_box C_box int_box A_r / (3 k^(5/3))`. Theorem L bounds `A_r` by
  `(1 + a_up r_*^2 + eps) A_0^ref`. Integrating over `(0, t]` gives (12.1) and (12.3).

## 2. The laws: Taylor models in `r` (Lemmas 1–2)

**Taylor models.** Each covariance of two preconditioned functionals is the exact series of [E] / [E3] section 2,
`sum_gamma kappa_gamma pi_gamma r^(g_x - w)`, plus the truncation tail beyond `g_x = 60` and the torus remainder (Cauchy
estimates on the polydisc, `M_1(10) < 4.46 x 10^-15` in the plane and `< 1.94 x 10^-13` in `d = 3`). The certificate keeps
each one on `[0, 1/64]` as a **Taylor model**:
- `f(r) = sum_{j <= 8} c_j r^j + theta e`, `|theta| <= 1`;
- the coefficients are fixed-point integers with `288` fractional bits;
- every rounding, every truncated power `r^j <= 64^-j`, the tails and the torus remainder go into `e`.

Products, sums and reciprocals (a geometric series with an explicit tail) are closed in this class. Gaussian elimination without
pivoting then gives Taylor models for the following, all exact in their polynomial parts up to `2^-288`:
- `det Sigma_r`;
- the pin energies `e_bb`, `e_bk`, `e_kk` (`v_r' Sigma_r^-1 v_r = e_bb b^2 + 2 e_bk b k + e_kk k^2`);
- the regression coefficients `alpha`, `beta` (mean `= b alpha + k beta`) of the transverse Hessian `A = D_y^2 f(M)` and of
  `Omega' = (D_y^2 f(S) - D_y^2 f(M))/r`;
- their covariances.

This removes the dependence loss of interval elimination. For example, the band laws of [E] give the `k`-coefficient of the
mean of `A` only as `+-3 r` on `[0, r]`, while the Taylor model gives `0 + O(10^-11)`.

**Lemma 1 (the transverse Hessian is exactly contact-distributed).** For the reference kernel and every `r > 0`, `b`, `k`,
the `Q`-law of `A = D_y^2 f(M)` is that of `-b I_m + G`, with `G` the `GOE_m` of [G3] (`N(0, 2)` diagonal, `N(0, 1)`
off-diagonal), independent of `r` and `k`.

*Proof.* The kernel factorizes, `exp(-|z|^2/2) = exp(-x^2/2) exp(-|y|^2/2)`. Every pin is either axial (`partial_x^a f` on the
axis) or a first transverse derivative on the axis.
- For an axial functional `g`, `Cov(f_{y_i y_j}(x, 0), g) = -delta_ij Cov(f(x, 0), g)`, since `partial_{y_i} partial_{y_j}
  exp(-|y|^2/2) = -delta_ij` at `y = 0`.
- For a transverse pin, `Cov(f_{y_i y_j}(x, 0), f_{y_l}(x', 0)) = Cov(f(x, 0), f_{y_l}(x', 0)) = 0`, since odd `y`-derivatives
  of the kernel vanish at `y = 0`.

So `A + f(M) I` is uncorrelated with every pin, hence independent of them. Under `Q`, `f(M) = b`, so `A = -b I + (A + f(M) I)`,
and `A + f(M) I` keeps its unconditional law:
- `Var(f_yy + f) = 3 - 2 + 1 = 2`;
- `Cov(f_{y_1 y_1} + f, f_{y_2 y_2} + f) = 1 - 1 - 1 + 1 = 0`;
- `Var f_{y_1 y_2} = 1`.

∎

The certificate verifies Lemma 1 through the Taylor models: every polynomial coefficient of `alpha_A + 1`, `beta_A` and
`Cov(A) - Cov(GOE)` vanishes to within `10^-60` (the code requires it), and only the remainder survives:
- `d = 2`: at most `1.9 x 10^-12`;
- `d = 3`: at most `8.1 x 10^-11`.

So on the torus the law is the contact law up to these remainders.

**Lemma 2 (pin law and energies).** For the reference kernel the Taylor models give, with all printed coefficients exact up
to `2^-288`:

    d = 2:  det Sigma_r = 12 - 18 r^2 + 14.4 r^4 + O(r^6),       d = 3:  det Sigma_r = 12 - 24 r^2 + 25.4 r^4 + O(r^6),
    both:   e_bb = 3/2 + r^2/8 - r^4/96 + ...,   e_bk = -(3/4) r^3 + ...,   e_kk = 24 - 6 r^2 + (6/5) r^4 + ...

The `Omega'` block, in `d = 2` (`om`) and `d = 3` (`O_11`, `O_22`, `O_12`):
- the mean has `alpha = 0` and `beta = r^2` on the diagonal entries (`beta_{O_12} = 0`);
- `Cov(om, w) = -r + r^3/4`, and in `d = 3` `Cov(A_ii, O_ii) = -r + r^3/4` and `Cov(A_12, O_12) = -r/2 + r^3/8`;
- `Var om = 2 - r^2/2 + r^4/12`.

At `r = 0` this is the contact law of [LP] section 15:
- `det = 12 = det Cov(f, f_xx) det Cov(f_x, f_xxx) = 2 . 6`;
- the energy is `3b^2/2 + 24 k^2`, which gives `pi_0 ∝ exp(-3b^2/4 - 12 k^2)`.

## 3. The pin-density ratio

`pi_r / pi_0^ref = (12 Theta^n / det S_r)^(1/2) exp(-(E_r - E_0)/2)`, with:
- `n = 2(d + 1)`;
- `S_r` the unnormalized pin covariance, `Theta in [1, 1 + theta_1(10)]` the kernel normalization;
- `E_r = Theta (e_bb b^2 + 2 e_bk b k + e_kk k^2)` and `E_0 = 3 b^2/2 + 24 k^2`.

The Taylor models give `det S_r - 12 in [d_lo r^2 - e, d_hi r^2 + e]` and the same for each energy coefficient. On a box
`b in [b_0, b_1]`, `k in [k_0, k_1]` (`b, k >= 0`), the certificate bounds:
- `log(pi_r/pi_0^ref)` above by `max(0, -d_lo) r^2 / (24 (1 - x_m)) + max(0, -E_lo/2) r^2 + (remainders)`;
- `log(pi_r/pi_0^ref)` below by `-max(0, d_hi) r^2/24 - max(0, E_hi/2) r^2 - (remainders)`;

using `log(1 + x) >= x/(1 + x)` and `log(1 + x) <= x`. The dominant genuine term is `+3 k^2 r^2` from `e_kk = 24 - 6 r^2`. At
`k = 2` it is `12 r^2`, which is the largest part of `a_up` in the plane.

## 4. The normalizer: the first-order cancellation (Lemma 3)

**Plane.** In the notation of [E] section 1 (`lambda = -w`, `c1 = 6k - rT`, `c2 = 6k + rT + r^2 tau`, `e = (r nu - v)^2`), the
typed event is `{c1 > 0, P1 > 0, P2 > 0}`, which lies inside `{lambda > 0}`, and there `W_r = r^2 P1 P2`. Expanding exactly,

    P1 P2 = c1 c2 lambda^2 - r c1 c2 lambda om + r^2 lambda xi + r^2 v^2 (c2 om - e),

    c1 c2 = 36 k^2 + r^2 kappa,        kappa = 6 k tau - T^2 - r T tau,
    xi    = 6k (r nu^2 - 2 nu v) - T (r^2 nu^2 - 2 r nu v + 2 v^2) - r tau v^2.

Three cancellations remove every first-order term:
- the first-order parts of `c1` and `c2` cancel in `c1 c2`;
- the first-order terms `r c1 lambda e` (from `P2`) and `-r c2 lambda v^2` (from `P1`) cancel, because `e - v^2 = r^2 nu^2 - 2 r nu v`;
- the remaining first-order term `-36 k^2 r E[lambda_+ om]` has an `O(r)` coefficient.

For the last one, regress `om` on `w`: `om = om_g + bw (w - mu_w)` with `om_g` independent of `w`, so

    E[lambda_+ om] = mu_om E[lambda_+] - bw (E[lambda_+^2] - m E[lambda_+]),

and Lemma 2 gives `mu_om = k r^2` and `bw = -r/2 + O(r^3)`.

So `Z_r / (36 k^2 r^2) = E[lambda_+^2] + O(r^2)`, and `E[lambda_+^2] = m_2(b)` exactly for the reference kernel (Lemma 1). Every
second-order term is bounded by Hölder's inequality with the Gaussian norms of the band law of [E] on `[0, R]`; the coefficients
`mu_om`, `bw` come from the Taylor models in the form `s r + e`.

**`d = 3`.** With `D = det B` (`B = -A`), `A* = adj B`, `B_S = B - r Omega'` and `e = r nu - v` (a vector), [E3] (I2) gives
`P1 = c1 D - r v' A* v` and `P2 = c2 det B_S + r e' adj(B_S) e`. In two dimensions `det B_S = D - r tr(A* Omega') + r^2 det Omega'`
and `adj(B_S) = A* - r adj(Omega')`, so

    P1 P2 = c1 c2 D^2 - r c1 c2 D tr(A* Omega') + r^2 c1 c2 D det Omega' + r^2 D xi_3 - r^2 c1 D e' adj(Omega') e
            + r^2 c2 tr(A* Omega') v' A* v - r^3 c2 det Omega' v' A* v - r^2 (v' A* v)(e' A* e) + r^3 (v' A* v)(e' adj(Omega') e),

    xi_3 = -(2T + r tau) v' A* v - 2 c1 nu' A* v + r c1 nu' A* nu.

Again `c1 e' A* e - c2 v' A* v = r xi_3`. The only first-order term is `-36 k^2 r E[D tr(A* Omega') 1{B > 0}]`. Regressing
`Omega'` on the three entries of `B` gives

    |E[D tr(A* Omega') 1{B > 0}]| <= ||mu_Omega'||_F E[D tr B 1{B > 0}] + ||beta_Omega'|| E[D tr B ||b~ - mu~|| 1{B > 0}],

where `||A*||_F = ||B||_F <= tr B` on `B > 0`, and both `||mu_Omega'||` and `||beta_Omega'||` are `O(r)` (Lemma 2). The main term
is `36 k^2 E[D^2 1{B > 0}]`, which equals `36 k^2 m_3(b)` for the reference kernel (Lemma 1). On the torus the Loewner coupling
of [E3] section 6, at the same `b` and with shift `eps + 12 eta` (remainders only), compares it with `m_3`. The moments of `B`
are the closed-form contact moments `E[det^a tr^c 1{B > 0}]` at the shifted mean, plus the outside-ball parts.

## 5. The thin typed shell (Lemma 4)

The typed event differs from `{lambda > 0}` (plane) or `{B > 0}` (`d = 3`) only where the smallest eigenvalue is `O(r)`. On
`G_0 = {r |T| <= 3k/2, r^2 |tau| <= 3k/2}`, where `c1 >= 9k/2` and `3k <= c2 <= 9k`:
- **Plane.** `{lambda > 0}` minus the typed event lies in `{0 < lambda <= r h}` with `h = 2 v^2/(9k) + |om|`, and there
  `|P1 P2| <= r^2 (15 k h/2 + v^2)(9k (h + |om|) + e)`.
  - Given `(v, om, nu)`, `lambda` is Gaussian. Its variance is at least
    `1/(1/Var w + bw' Cg^-1 bw)` (Schur complement), with `Cg^-1 <= D^-1/(1 - ||E||_F)` by diagonal scaling.
  - Hence `E[G 1{0 < lambda <= r h}] <= r pbar E[G h]`, and the shell contributes `O(r^3)`.
- **`d = 3`.** The shell is `{0 < lambda_1 <= r h}` with `h = 2 ||v||^2/(9k) + ||Omega'||_F`, and there
  `|P1 P2| <= r^2 lambda_2 (lambda_2 + r ||Omega'||)(15 k h/2 + ||v||^2)(9k (h + ||Omega'||) + ||e||^2)`.
  - Given `Y = (v, Omega', nu)`, `B` is Gaussian with covariance `C' >= cl I`, where `1/cl <= 1/lambda_min(Cb) + ||Bt' Cg^-1 Bt||`.
  - In eigenvalue coordinates (`dB = sqrt2 (lambda_2 - lambda_1) d lambda_1 d lambda_2 d theta`, `theta in [0, pi)`),
    `E[lambda_2 (lambda_2 + r Omega) 1{0 < lambda_1 <= s} | Y] <= s KB' J(||mu'||, r Omega)`.
    - `KB' = pi sqrt2 (2 pi)^(-3/2) cl^(-3/2)`.
    - `J(m, x) = int_0^inf lambda^2 (lambda + x) exp(-(lambda - m)_+^2/(2 c_hi)) d lambda`, a polynomial in `m` and `x` with
      half-Gaussian moments.
    - `||mu' - mu_B|| <= ||C_BY C_YY^(-1/2)|| chi_7`.
  - The rows of `Bt` for `Omega'` come from the Taylor models (`O(r)`); those for `v` and `nu` are below `10^-11`.

The complement of `G_0` has probability below `10^-50` for `R <= 1/64` and enters through `||P1 P2||_2 P(G_0^c)^(1/2)`.

The sweep evaluates these bounds on `96` boxes (`b`-step `1/8`, `k`-step `1/8`) for each `R`. The box constants feed the
weighted `c'`, and their maxima are Theorem L's `a_up`, `a_dn`.

## 6. Reference integrals

`A_0^ref / (3 k^p) = K_d k^(2-p) exp(-12 k^2) exp(-3b^2/4) m_d(b)` with `K_d = |S^(d-1)| 144 (2 pi)^-(d+1) / sqrt12`. The `k`-integrals are incomplete gamma functions:

    int_{k0}^{k1} k^q exp(-12 k^2) dk = (1/2) 12^(-(q+1)/2) [gamma((q+1)/2, 12 k1^2) - gamma((q+1)/2, 12 k0^2)].

Their powers `x^a` come from exact integer roots, and the series has a geometric tail. The `b`-integrals use the composite
midpoint rule on `4096` cells:
- the error per cell is at most `h^3 |g''|_max / 24`;
- `|g''|` is bounded per block of `16` cells through `m, m', m''`, all increasing in `b`;
- `d = 2`: `m' = 2 E[(b + sqrt2 z)_+]` and `m'' = 2 Phi(b/sqrt2) <= 2`;
- `d = 3`: `m_3' = 2 E[det tr 1]` and `m_3'' = 2 E[tr^2 1] + 4 E[det 1]` (contact moments).

**Cross-checks** (exact, in the certificate): the enclosures of `c^ref_{B,K}` contain the values of Math-#197
(`0.0023869380211529483`, Owen-T closed form) and Math-#205 (`0.00097438236602425`).

## 7. The cap sweep and `C_{B,K}`

The cap route needs `sup_{r <= r_*} Q^W(G_r^c)/r^3` on boxes that resolve `k` near `1/2`. There the weight
`A_0^ref / k^(5/3) ∝ k^(1/3) e^(-12 k^2)` concentrates, and `c_G` grows like `k^3`. The engines of [E], [E3] (`box_bound`, with
their first parameter pair `(X_0, theta)`) are run on:
- every band of [E] (`153`) and of [E3] (`121`);
- `56` boxes: `b`-boxes `[i/4, (i+1)/4]`, and `k`-boxes of width `1/32` on `[1/2, 3/4]`, `1/16` on `[3/4, 1]`, then `[1, 3/2]` and
  `[3/2, 2]`.

The certified per-box bounds are stored to four decimals, rounded up. For each `r_*`, `C_box(r_*)` is their maximum over the
bands with `r_1 <= r_*`, and

    C_{B,K}(r_*) = (1 + a_up(r_*) r_*^2 + eps) sum_box C_box(r_*) |S^(d-1)| int_box A_0^ref / (3 k^(5/3)).

## 8. Arithmetic

- Exact rationals throughout. The Taylor models are fixed-point integers with explicit rounding.
- Interval enclosures of `exp`, `Phi` and the moments come from the engines.
- Integer `p`-th roots are used for fractional powers.
- Every stored number is rounded outward to a decimal and replayed exactly.

## 9. Controls (floating point; not part of the certificate)

@@CONTROLS@@

## 10. What this does not do

- **`d = 2, 3` and the band `B x K` only.** The method needs a compact `K` away from `0` (the shell bound uses `k >= 1/2`). The
  unrestricted densities of [LP] Theorem C are not addressed. There the constant term `B_{d,L}` (Math-#191, Math-#211) comes
  from pairs this window excludes.
- **The Kac–Rice interface is consumed, not reviewed.** [LP] sections 8–12 are used as statements.
- **The cap route only for `C_{B,K}`.** `C_{B,K}` bounds `nu_cand - nu_eld` through `1 - p_r <= Q^W(G_r^c)`. By [G] the cap
  criterion overstates the pairing failure by about `10^5` at `k = 1/2`, so the true difference coefficient is far smaller. No
  lower bound for it is given.
- **Precision.**
  - `a_up`, `a_dn` in `d = 3` are dominated by Hölder bounds of second-order terms whose expectations nearly vanish (for example
    `E[tr(A* Omega') v' A* v]`). Sign-aware bounds would reduce them.
  - The plane's `a_up` is within a few units of the genuine coefficient `3 k^2 + 3/4` at `k = 2`.
- **Not a review** of [LP], [CAP], [E] or [E3]. The engines are used as certified box-bound functions; their correctness is the
  subject of Math-#206 and Math-#212.
- **Status.** C8 stays OPEN.
- **Independence.** Same GitHub account as every lane; zero organizational-independence credit. The author will not merge.

## 11. Verification

- `python3 -B -S theorem_b.py --check --procs N` replays exactly:
  - the Taylor-model laws and their recorded coefficients;
  - the rate sweep (`96` boxes, seven `R`, both dimensions);
  - the reference integrals and the cross-checks;
  - a sample of the cap bands (the band ending at each `r_*`, at `2^-20`, and the last band);
  - the assembled tables.
- `--check-full` replays every cap band.
- Mutants `no-om-term`, `no-bad-set`, `no-det`, `no-torus` and `cap-half` make the replay fail.
- The workflow checks:
  - the manifest;
  - the main-resident pins ([LP], [CAP]);
  - the engine copies against `SOURCE_MAP.json`;
  - `--check-full` and `--check` in the two interpreter modes;
  - the mutants;
  - a clean tree.
